"""MCP tool: ``reasonix_approve`` (R10).

R10 (this file):
  Thin MCP tool entry point that wraps
  :meth:`adapter.tools.reasonix_delegate.DelegateDispatcher.on_approve`.
  Takes a dispatcher instance (set at MCP server registration
  time) and resolves a pending ``plan_mode="approve"`` session.

Contract (locked 2026-06-15, phase2-spec.md §P2.2 base MCP tools):
  - ``reasonix_approve(sid, decision, feedback=None) -> dict``
  - ``decision`` must be ``"approve"`` or ``"reject"`` (validated
    at the tool layer; the underlying dispatcher's contract is
    the same, but we surface friendlier error dicts).
  - Returns a 5-field result shape:
      - ``sid`` (str): echo of input
      - ``decision`` (str): echo of input
      - ``accepted`` (bool): True iff the session was unblocked
      - ``feedback_provided`` (bool): True iff ``feedback`` was
        a non-empty string
      - ``reason`` (str): outcome tag — one of
        ``"session_unblocked"`` / ``"no_pending_approval"`` /
        ``"invalid_decision"`` / ``"no_dispatcher"``
  - When the call is rejected before reaching the dispatcher
    (no dispatcher, invalid decision), the result dict also
    carries ``error`` + ``message`` fields for the LLM caller.

The MCP server layer (registered separately) injects the live
``DelegateDispatcher`` into the module-level ``_dispatcher`` slot.
This split keeps the module importable without a Supervisor (the
tool layer can register/unregister dispatchers at boot/shutdown).

Pattern mirrors R7/R9 (see :mod:`adapter.tools.reasonix_delegate`,
:mod:`adapter.tools.reasonix_status`).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

log = logging.getLogger(__name__)


# Valid decision values. Tool layer enforces strict enum
# (mirror R7's persona-priority strictness at the MCP boundary).
_VALID_DECISIONS = frozenset({"approve", "reject"})


# Module-level dispatcher slot. Set via set_dispatcher() at MCP
# server boot, cleared via clear_dispatcher() at shutdown. Guarded
# by a lock so concurrent MCP tool calls don't race on the
# registration step.
_dispatcher: Optional[Any] = None
_dispatcher_lock = threading.Lock()


def set_dispatcher(dispatcher: Any) -> None:
    """Register the live ``DelegateDispatcher`` for the MCP tool layer.

    The MCP server calls this once at boot. Subsequent calls REPLACE
    the current dispatcher (no error) — useful for hot-reload tests
    or graceful supervisor restart.

    Thread-safe: held under :data:`_dispatcher_lock` for the duration
    of the swap. Tool calls in flight get the old dispatcher (atomic
    read) — no torn reference.
    """
    global _dispatcher
    with _dispatcher_lock:
        _dispatcher = dispatcher
        log.info("reasonix_approve: dispatcher registered (%r)", dispatcher)


def clear_dispatcher() -> None:
    """Unregister the dispatcher (MCP server shutdown).

    Safe to call multiple times. After this, ``reasonix_approve``
    returns the ``no_dispatcher`` error dict instead of touching
    the dispatcher.
    """
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is not None:
            log.info("reasonix_approve: dispatcher cleared")
        _dispatcher = None


def _read_dispatcher() -> Optional[Any]:
    """Atomic read of the current dispatcher slot.

    Returns None if no dispatcher is registered.
    """
    with _dispatcher_lock:
        return _dispatcher


def reasonix_approve(
    sid: str,
    decision: str,
    feedback: Optional[str] = None,
) -> dict:
    """Resolve a pending ``plan_mode="approve"`` session.

    See module docstring for the full contract.

    Args:
        sid: session id returned by a previous
            :meth:`DelegateDispatcher.dispatch` call with
            ``plan_mode="approve"``.
        decision: ``"approve"`` continues to the executor phase;
            ``"reject"`` stops after the planner phase.
        feedback: optional natural-language feedback to inject
            into the next turn. Empty string is normalized to
            ``feedback_provided=False`` but still forwarded.

    Returns:
        dict with the 5-field result shape described in the
        module docstring. Error dicts add ``error`` + ``message``
        fields.
    """
    # 1. Decision validation (tool-layer strict; mirrors R7
    #    persona-priority strictness at the MCP boundary).
    if decision not in _VALID_DECISIONS:
        return {
            "sid": sid,
            "decision": decision,
            "accepted": False,
            "feedback_provided": False,
            "reason": "invalid_decision",
            "error": "invalid_decision",
            "message": (
                f"decision must be 'approve' or 'reject', "
                f"got {decision!r}"
            ),
        }

    # 2. Dispatcher presence check.
    dispatcher = _read_dispatcher()
    if dispatcher is None:
        return {
            "sid": sid,
            "decision": decision,
            "accepted": False,
            "feedback_provided": False,
            "reason": "no_dispatcher",
            "error": "no_dispatcher",
            "message": (
                "reasonix_approve called before MCP server boot or "
                "after shutdown; no DelegateDispatcher registered."
            ),
        }

    # 3. Forward to dispatcher's on_approve(). The dispatcher
    #    returns False when no pending approval exists for this
    #    sid (raced, already resolved, or not in approve mode).
    accepted = dispatcher.on_approve(sid, decision, feedback)

    return {
        "sid": sid,
        "decision": decision,
        "accepted": accepted,
        "feedback_provided": bool(feedback),
        "reason": "session_unblocked" if accepted else "no_pending_approval",
    }
