"""MCP tool: ``reasonix_steer`` (P2.1 wire + R7 wrapper).

Thin MCP tool entry point that wraps
:meth:`adapter.task_dispatcher.TaskDispatcher.steer`. The dispatcher
calls the ``session/steer`` ACP method (added in P2.1, commit
575a6c7d), which lets the LLM inject mid-turn guidance into a
running Reasonix session *without* cancelling the current turn.

Contract (locked 2026-06-15, phase2-spec.md §P2.2 base MCP tools):
  - ``reasonix_steer(sid, text) -> dict``
  - ``text`` must be a non-empty string (whitespace-only is
    rejected at the tool layer; the underlying dispatcher's
    protocol would also reject it, but we surface a friendlier
    error dict).
  - Returns: ``{sid, queued, queue_len, status_after}`` on success.
    ``status_after`` is the snapshot from
    :meth:`TaskDispatcher.status` taken immediately after the steer
    (lets the LLM caller see queue_len + session state in one call).
  - Error dicts (no_dispatcher, dispatcher_failure, invalid_text)
    carry ``sid`` + ``error`` + ``message`` fields.

Pattern mirrors R9 (status) and R10 (approve) — module-level
dispatcher slot guarded by a lock. See
:mod:`adapter.tools.reasonix_status` for the canonical pattern.
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
        log.info("reasonix_steer: dispatcher registered (%r)", dispatcher)


def clear_dispatcher() -> None:
    """Unregister the dispatcher (MCP server shutdown).

    Safe to call multiple times. After this, ``reasonix_steer``
    returns the ``no_dispatcher`` error dict instead of touching
    the dispatcher.
    """
    global _dispatcher
    with _dispatcher_lock:
        if _dispatcher is not None:
            log.info("reasonix_steer: dispatcher cleared")
        _dispatcher = None


def _read_dispatcher() -> Optional[Any]:
    """Atomic read of the current dispatcher slot.

    Returns None if no dispatcher is registered.
    """
    with _dispatcher_lock:
        return _dispatcher


def reasonix_steer(sid: str, text: str) -> dict:
    """Queue ``text`` as mid-turn guidance on session ``sid``.

    Thin wrapper over
    :meth:`adapter.task_dispatcher.TaskDispatcher.steer`. The
    underlying ``session/steer`` ACP method (P2.1, commit 575a6c7d)
    appends ``text`` to a FIFO queue that the executor consumes
    after the current step completes — it does NOT cancel the
    in-flight turn, so the agent sees both the original prompt
    and the steered text in order.

    This is the proper "steer" semantics. If you need to restart
    the turn with a different task instead, cancel first and then
    dispatch a new prompt::

        await td.cancel(sid)
        await td.dispatch("entirely new task")

    Args:
        sid: session id returned by a previous
            :meth:`DelegateDispatcher.dispatch` call.
        text: guidance text to inject. Must be a non-empty string
            (whitespace-only counts as empty).

    Returns:
        dict with keys:
            - ``sid`` (str): echo of input
            - ``queued`` (bool): True iff the text was accepted into
              the queue (False if the session was unknown / dead).
            - ``queue_len`` (int): length of the steer queue after
              this text was appended (0 = session unknown).
            - ``status_after`` (dict): status snapshot from
              :meth:`TaskDispatcher.status` taken immediately after
              the steer (so the LLM caller sees queue state + cost
              + session state in one call).

        On missing-dispatcher / invalid-text / dispatcher-error,
        returns a dict with ``sid`` + ``error`` + ``message``
        fields and the steer is NOT performed.
    """
    # 1. Input validation (tool-layer strict; mirrors R7
    #    persona-priority strictness at the MCP boundary).
    if not isinstance(text, str) or not text.strip():
        return {
            "sid": sid,
            "queued": False,
            "queue_len": 0,
            "status_after": None,
            "error": "invalid_text",
            "message": "text must be a non-empty string (whitespace-only is rejected)",
        }

    # 2. Atomic dispatcher read.
    dispatcher = _read_dispatcher()
    if dispatcher is None:
        return {
            "sid": sid,
            "queued": False,
            "queue_len": 0,
            "status_after": None,
            "error": "no_dispatcher",
            "message": (
                "reasonix_steer called before MCP server boot or "
                "after shutdown; no TaskDispatcher registered."
            ),
        }

    # 3. Forward to dispatcher.steer().
    #    The dispatcher is async, so we run it in a fresh event loop
    #    when called from a sync MCP tool context. This mirrors how
    #    the existing reasonix_approve tool calls dispatcher methods.
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        try:
            steer_result = loop.run_until_complete(dispatcher.steer(sid, text))
        finally:
            loop.close()
    except Exception as e:  # pragma: no cover - defensive
        log.exception("reasonix_steer: dispatcher.steer failed for sid=%s", sid)
        return {
            "sid": sid,
            "queued": False,
            "queue_len": 0,
            "status_after": None,
            "error": "dispatcher_failure",
            "message": f"{type(e).__name__}: {e}",
        }

    # 4. Snapshot status_after (best-effort; if it fails, return None).
    #    LLM caller can decide to call reasonix_status() separately.
    try:
        loop = asyncio.new_event_loop()
        try:
            status_after = loop.run_until_complete(dispatcher.status(sid))
        finally:
            loop.close()
    except Exception as e:  # pragma: no cover - defensive
        log.warning("reasonix_steer: status_after snapshot failed (sid=%s): %s", sid, e)
        status_after = None

    return {
        "sid": sid,
        "queued": bool(steer_result.get("queued")),
        "queue_len": int(steer_result.get("queue_len", 0)),
        "status_after": status_after,
    }
