"""MCP tool: ``reasonix_status`` + ``reasonix_replay`` (R9).

R9 (this file):
  Thin MCP tool entry points that wrap
  :class:`adapter.tools.reasonix_delegate.DelegateDispatcher.status` /
  :meth:`replay`. Both functions take a dispatcher instance (set at
  MCP server registration time) and a session id.

Contract (locked 2026-06-15, phase2-spec.md §P2.2 base MCP tools):
  - ``reasonix_status(sid) -> dict`` with fields: status, persona,
    plan_mode, cost (5-field), last_event, queue_len, created_at,
    prompt_preview, log_path.
  - ``reasonix_replay(sid, since_seq=0) -> list[dict]`` returning
    raw NDJSON events from the session's transcript.

The MCP server layer (registered separately) injects the live
``DelegateDispatcher`` into the module-level ``_dispatcher`` slot.
This split keeps the module importable without a Supervisor (the
tool layer can register/unregister dispatchers at boot/shutdown).

Pattern mirrors R7/R8 (see :mod:`adapter.tools.reasonix_delegate`).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

log = logging.getLogger(__name__)


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
        log.info("reasonix_status: dispatcher registered (%r)", dispatcher)


def clear_dispatcher() -> None:
    """Unregister the dispatcher (MCP server shutdown).

    Safe to call multiple times. After this, ``reasonix_status`` /
    ``reasonix_replay`` return error dicts instead of touching the
    dispatcher.
    """
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is not None:
            log.info("reasonix_status: dispatcher cleared")
        _dispatcher = None


def reasonix_status(sid: str) -> dict:
    """Return a status snapshot for ``sid``.

    Thin wrapper over
    :meth:`adapter.tools.reasonix_delegate.DelegateDispatcher.status`.
    If no dispatcher is registered (e.g. called before MCP server
    boot, or after shutdown), returns a structured error dict
    instead of raising — the LLM caller can decide whether to
    retry or surface the error to the user.

    Args:
        sid: session id returned by a previous ``reasonix_delegate``
            call.

    Returns:
        dict with the 5-field cost shape + session metadata. See
        :meth:`DelegateDispatcher.status` for the full schema. On
        missing-dispatcher, returns ``{"error": "no_dispatcher", ...}``.
    """
    with _dispatcher_lock:
        dispatcher = _dispatcher
    if dispatcher is None:
        return {
            "sid": sid,
            "status": "error",
            "error": "no_dispatcher",
            "message": (
                "reasonix_status called before MCP server boot or "
                "after shutdown; no DelegateDispatcher registered."
            ),
        }
    try:
        return dispatcher.status(sid)
    except Exception as e:  # pragma: no cover - defensive
        log.exception("reasonix_status: dispatcher.status failed for sid=%s", sid)
        return {
            "sid": sid,
            "status": "error",
            "error": "dispatcher_failure",
            "message": f"{type(e).__name__}: {e}",
        }


def reasonix_replay(
    sid: str,
    since_seq: int = 0,
    *,
    mode: str = "conversation",
) -> list[dict]:
    """Return the session transcript for ``sid`` in the requested shape.

    Thin wrapper over
    :meth:`adapter.tools.reasonix_delegate.DelegateDispatcher.replay`.
    Same error-handling shape as :func:`reasonix_status`.

    R13.3: 3-mode API. ``mode`` selects the output shape:

      - ``"raw"``: raw NDJSON records from the binary log only (no
        user prompts; user prompts live in the parallel
        ``<sid>.dispatcher.jsonl``).
      - ``"conversation"`` (default): unified timeline merging
        user prompts (dispatcher log) + assistant text (binary
        log, chunks concatenated per turn) + tool events
        (``tool_call`` + ``tool_call_update`` merged by
        ``toolCallId``). See the dispatcher docstring for the
        full algorithm.
      - ``"summary"``: placeholder. Returns ``[]`` for R13.x;
        R14 will fill in aggregated counts / durations / tokens.

    Args:
        sid: session id.
        since_seq: minimum event seq to return. Default 0 = all
            events. Used by the LLM to poll for incremental
            updates (track the highest seq it has seen, pass
            ``since_seq = last_seen_seq + 1`` next time). Only
            applies to the underlying raw events; for
            ``mode="conversation"`` this filters binary events
            before transformation.
        mode: one of ``"raw"`` / ``"conversation"`` /
            ``"summary"``. Default ``"conversation"`` (changed
            from raw in R13.3 — most callers want the merged
            dialog, not the wire dump).

    Returns:
        list of message dicts (shape depends on ``mode``). ``[]``
        if the log(s) don't exist. On missing-dispatcher, returns
        a single-item list with an error dict so the LLM still
        sees the failure shape. On invalid mode, returns a
        single-item list with ``error="invalid_mode"``.
    """
    with _dispatcher_lock:
        dispatcher = _dispatcher
    if dispatcher is None:
        return [{
            "sid": sid,
            "error": "no_dispatcher",
            "message": (
                "reasonix_replay called before MCP server boot or "
                "after shutdown; no DelegateDispatcher registered."
            ),
        }]
    try:
        return dispatcher.replay(sid, since_seq=since_seq, mode=mode)
    except ValueError as e:
        # R13.3: invalid mode surfaces as a single-item error list
        # (NOT a raised exception — LLM caller can read the error
        # without try/except).
        return [{
            "sid": sid,
            "error": "invalid_mode",
            "message": str(e),
        }]
    except Exception as e:  # pragma: no cover - defensive
        log.exception("reasonix_replay: dispatcher.replay failed for sid=%s", sid)
        return [{
            "sid": sid,
            "error": "dispatcher_failure",
            "message": f"{type(e).__name__}: {e}",
        }]
