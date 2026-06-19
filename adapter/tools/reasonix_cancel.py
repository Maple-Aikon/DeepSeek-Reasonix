"""MCP tool: ``reasonix_cancel`` (R3 — new tool for P2.0 MCP server).

Cancels a running Reasonix session. The corresponding ACP method
on the wire is ``session/cancel`` (P2.1, exists in v1.9.1 binary —
verified at internal/acp/service.go sessionCancel). The wire
method stops the in-flight turn; the LLM can then either:

  - dispatch a new prompt on the SAME session (preserves session
    state + log continuity), OR
  - leave the session idle (it can be cleaned up later)

Contract (locked 2026-06-19, this PR):
  - ``reasonix_cancel(sid) -> dict``
  - ``sid`` must be a non-empty string (whitespace rejected).
  - Returns ``{sid, cancelled, status_after}`` on success.
    ``cancelled`` is True iff the session was known and the
    cancel was acknowledged by the wire.
    ``status_after`` is the post-cancel snapshot (best-effort,
    None if it fails — same partial-success policy as steer).
  - Error dicts (no_dispatcher, dispatcher_failure, invalid_sid)
    carry ``sid`` + ``error`` + ``message`` fields.

Difference vs. reasonix_steer: steer *appends* guidance to the
queue (preserves the in-flight turn); cancel *stops* the in-flight
turn and unblocks the LLM. They are complementary, not
interchangeable.

Pattern mirrors R11 (steer) and R10 (approve) — module-level
dispatcher slot guarded by a lock.
"""
from __future__ import annotations

import asyncio
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
    """Register the live ``TaskDispatcher`` for the MCP tool layer.

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
        log.info("reasonix_cancel: dispatcher registered (%r)", dispatcher)


def clear_dispatcher() -> None:
    """Unregister the dispatcher (MCP server shutdown).

    Safe to call multiple times. After this, ``reasonix_cancel``
    returns the ``no_dispatcher`` error dict instead of touching
    the dispatcher.
    """
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is not None:
            log.info("reasonix_cancel: dispatcher cleared")
        _dispatcher = None


def _read_dispatcher() -> Optional[Any]:
    """Atomic read of the current dispatcher slot.

    Returns None if no dispatcher is registered.
    """
    with _dispatcher_lock:
        return _dispatcher


def reasonix_cancel(sid: str) -> dict:
    """Cancel the in-flight turn on session ``sid``.

    Thin wrapper over
    :meth:`adapter.task_dispatcher.TaskDispatcher.cancel`. The
    underlying ``session/cancel`` ACP method (P2.1, v1.9.1 binary
    internal/acp/service.go sessionCancel) signals the executor
    to stop the current turn. The session itself remains queryable
    and reusable — a subsequent :func:`reasonix_delegate` call on
    the same sid will dispatch a new prompt.

    This is the "hard stop" tool. If you want to GUIDE the current
    turn (add context) without stopping it, use
    :func:`reasonix_steer` instead.

    Args:
        sid: session id returned by a previous
            :meth:`DelegateDispatcher.dispatch` call.

    Returns:
        dict with keys:
            - ``sid`` (str): echo of input
            - ``cancelled`` (bool): True iff ``dispatcher.cancel()``
              returned without raising. ``TaskDispatcher.cancel``
              is best-effort and returns ``None`` for both
              "successfully cancelled" and "session was already
              idle" — we cannot distinguish those cases from the
              tool layer, so we surface cancelled=True for any
              successful return. The LLM can call
              :func:`reasonix_status` afterwards to confirm the
              session is actually idle.
            - ``status_after`` (dict or None): post-cancel status
              snapshot from :meth:`TaskDispatcher.status` (best
              effort; None on failure).

        On missing-dispatcher / invalid-sid / dispatcher-error,
        returns a dict with ``sid`` + ``error`` + ``message``
        fields and the cancel is NOT performed.
    """
    # 1. Input validation (strict sid shape; sid must be a
    #    non-empty string. Whitespace-only is rejected as a sanity
    #    check — empty sids would always be unknown anyway, but
    #    catching them early keeps the error message clear.)
    if not isinstance(sid, str) or not sid.strip():
        return {
            "sid": sid,
            "cancelled": False,
            "status_after": None,
            "error": "invalid_sid",
            "message": "sid must be a non-empty string (whitespace-only is rejected)",
        }

    # 2. Atomic dispatcher read.
    dispatcher = _read_dispatcher()
    if dispatcher is None:
        return {
            "sid": sid,
            "cancelled": False,
            "status_after": None,
            "error": "no_dispatcher",
            "message": (
                "reasonix_cancel called before MCP server boot or "
                "after shutdown; no TaskDispatcher registered."
            ),
        }

    # 3. Forward to dispatcher.cancel().
    #    The dispatcher is async, so we run it in a fresh event loop
    #    when called from a sync MCP tool context. This mirrors how
    #    reasonix_approve / reasonix_steer wrap their async calls.
    #    TaskDispatcher.cancel() returns None on success
    #    (best-effort — does not raise even if the session was
    #    already idle), so any non-exception return is treated
    #    as "cancelled=True" at the tool layer. The LLM caller
    #    can use reasonix_status() to verify the actual state.
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(dispatcher.cancel(sid))
        finally:
            loop.close()
        cancelled = True
    except Exception as e:  # pragma: no cover - defensive
        log.exception("reasonix_cancel: dispatcher.cancel failed for sid=%s", sid)
        return {
            "sid": sid,
            "cancelled": False,
            "status_after": None,
            "error": "dispatcher_failure",
            "message": f"{type(e).__name__}: {e}",
        }

    # 4. Snapshot status_after (best-effort; if it fails, return None).
    #    Same partial-success policy as reasonix_steer: don't punish
    #    success with a total error dict.
    try:
        loop = asyncio.new_event_loop()
        try:
            status_after = loop.run_until_complete(dispatcher.status(sid))
        finally:
            loop.close()
    except Exception as e:  # pragma: no cover - defensive
        log.warning("reasonix_cancel: status_after snapshot failed (sid=%s): %s", sid, e)
        status_after = None

    return {
        "sid": sid,
        "cancelled": cancelled,
        "status_after": status_after,
    }
