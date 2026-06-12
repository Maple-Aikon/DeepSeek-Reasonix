"""High-level dispatch API for the Reasonix subagent adapter.

TaskDispatcher is the public surface PicoClaw's tools layer talks to.
It composes a :class:`adapter.supervisor.Supervisor` and a
:class:`adapter.log_capture.LogCapture` into five operations:

    dispatch(task)        — start a session, prompt, return stop payload
    cancel(sid)           — best-effort cancel of an in-flight session
    steer(sid, new_task)  — cancel + re-prompt with the new task
    status(sid, n=20)     — return the last N events from the NDJSON log
    replay(sid)           — async-iterate over the entire NDJSON log

Why a wrapper?
--------------
A Supervisor already exposes ``start()``, ``new_session()``, ``prompt()``,
``cancel()``, ``close()``. The wrapper earns its keep by:

* **Wiring log_capture to the supervisor's notification channel** so
  callers don't have to remember to do it.
* **Serializing prompt content** (task string → ``ContentBlock`` list)
  so the dispatcher API takes a plain string.
* **Reading log files for status/replay** so the caller never has to
  reach into the log directory.
* **Caching the cwd + supervisor per TaskDispatcher** so a single
  dispatcher is one session (Phase 1). Phase 2 will replace this with
  a session pool.

Public surface
--------------
- :class:`TaskDispatcher` — the dispatcher
- :class:`TaskDispatcherError` — raised on dispatcher-level failures

Locking model
-------------
None. The underlying Supervisor serializes its own writes through the
Conn reader loop, and LogCapture serializes its own per-file writes.
TaskDispatcher adds no concurrency control; concurrent dispatches must
use *different* TaskDispatcher instances (each with its own Supervisor
and LogCapture), which is exactly what Phase 2's pool will provide.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Optional, Protocol, Sequence

log = logging.getLogger("adapter.task_dispatcher")


class TaskDispatcherError(RuntimeError):
    """Raised for dispatcher-level failures (e.g. log file unreadable)."""


class SupervisorLike(Protocol):
    """Structural type for the Supervisor we compose with.

    We don't import ``adapter.supervisor.Supervisor`` to keep this
    module leaf-level — easier to test, easier to mock. The Protocol
    documents the surface we actually use.
    """

    @property
    def state(self) -> str: ...
    @property
    def pid(self) -> Optional[int]: ...
    @property
    def current_sid(self) -> Optional[str]: ...
    @property
    def auto_approve(self) -> bool: ...
    async def start(self) -> dict: ...
    async def new_session(self, cwd: str) -> dict: ...
    async def prompt(self, sid: str, content: list) -> dict: ...
    async def cancel(self, sid: str) -> None: ...
    async def close(self) -> None: ...
    @property
    def on_notification(self) -> Any: ...
    @on_notification.setter
    def on_notification(self, fn: Any) -> None: ...


class LogCaptureLike(Protocol):
    """Structural type for the LogCapture we compose with."""

    def handler(self) -> Any: ...
    async def close(self) -> None: ...


# Map the well-known reasonix ``stopReason`` values we propagate as-is.
# Anything we don't know is forwarded verbatim to the caller.
_KNOWN_STOP_REASONS = frozenset({
    "end_turn", "cancelled", "refused", "error", "stop",
})


class TaskDispatcher:
    """Compose Supervisor + LogCapture into a high-level dispatch API.

    Parameters
    ----------
    supervisor:
        Object satisfying :class:`SupervisorLike`. Usually a
        :class:`adapter.supervisor.Supervisor` instance.
    log_capture:
        Object satisfying :class:`LogCaptureLike`. Usually a
        :class:`adapter.log_capture.LogCapture` instance.
    cwd:
        Working directory for the new session. Must be a directory that
        Reasonix can ``os.Stat``.
    log_dir, pid_dir:
        Optional overrides for the log and PID directories. Default to
        LogCapture's own defaults (typically
        ``~/.picoclaw/logs/reasonix-subagent/`` and
        ``~/.picoclaw/state/reasonix/``).
    auto_approve:
        Forwarded to the Supervisor constructor. When True, the
        supervisor auto-approves every ``session/request_permission``
        request the server sends. Default False (the dispatcher is
        a pass-through wrapper; trust decisions belong to the caller).
        This parameter is only honored when ``supervisor`` is an
        actual :class:`adapter.supervisor.Supervisor` instance — the
        :class:`SupervisorLike` protocol is read-only for it.
    """

    def __init__(
        self,
        *,
        supervisor: SupervisorLike,
        log_capture: LogCaptureLike,
        cwd: str | os.PathLike,
        log_dir: Optional[str | os.PathLike] = None,
        pid_dir: Optional[str | os.PathLike] = None,
        auto_approve: bool = False,
    ) -> None:
        # If the supervisor is a real ``adapter.supervisor.Supervisor``
        # and the caller asked for auto_approve, set it on the live
        # instance. The real Supervisor accepts the kwarg in its
        # constructor, but TaskDispatcher composes a pre-built
        # supervisor (it does not own its construction), so we
        # post-configure it here.
        if auto_approve and hasattr(supervisor, "auto_approve"):
            try:
                supervisor.auto_approve = True  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                log.warning(
                    "could not set auto_approve on supervisor %r; "
                    "continuing without it",
                    supervisor,
                )
        self._sup = supervisor
        self._cap = log_capture
        self._cwd = str(cwd)
        # If the caller passed log_dir / pid_dir overrides, build a new
        # log_capture that uses them. (Cheaper than monkey-patching.)
        if log_dir is not None or pid_dir is not None:
            try:
                from adapter.log_capture import LogCapture as _LC
            except ImportError as e:  # pragma: no cover
                raise TaskDispatcherError(
                    "log_dir/pid_dir overrides require adapter.log_capture"
                ) from e
            self._cap = _LC(
                supervisor,
                log_dir=Path(log_dir) if log_dir is not None else None,
                pid_dir=Path(pid_dir) if pid_dir is not None else None,
            )
        # Wire the log_capture's handler into the supervisor.
        # A real Supervisor exposes ``on_notification`` as a settable
        # property; we always set it through the setter so subclasses
        # can override how dispatch works.
        self._sup.on_notification = self._cap.handler()

        self._started = False
        self._sid: Optional[str] = None

    # ---------------- public API ----------------

    async def dispatch(self, task: str) -> dict:
        """Start a session (idempotent), prompt with ``task``, and
        return the final stop notification payload.

        On success: returns a dict with at least ``stopReason`` (one of
        ``end_turn``, ``cancelled``, ``refused``, ``error``, ``stop``).
        On failure: raises :class:`TaskDispatcherError`.
        """
        if not self._started:
            await self._start()
        assert self._sid is not None
        content = _string_to_content(task)
        try:
            stop = await self._sup.prompt(self._sid, content)
        except Exception as e:
            raise TaskDispatcherError(f"dispatch failed: {e}") from e
        return stop

    async def cancel(self, sid: str) -> None:
        """Cancel a session. Best-effort — does not raise if the
        supervisor was already idle."""
        await self._sup.cancel(sid)

    async def steer(self, sid: str, new_task: str) -> dict:
        """Cancel ``sid`` and immediately re-prompt with ``new_task``.

        Why not a true fork/rewind? Phase 1 reasonix doesn't expose
        ``session/fork`` or ``session/rewind`` over ACP (see plan §"Goal
        & scope" §"Critical scope"). The soft-steer (cancel + new
        prompt) is the closest equivalent.
        """
        await self.cancel(sid)
        return await self.dispatch(new_task)

    async def status(self, sid: str, n: int = 20) -> list[dict]:
        """Return the last ``n`` events from ``<log_dir>/<sid>.jsonl``.

        Returns ``[]`` if the log file doesn't exist (e.g. session
        never produced any notifications). Lines that fail to parse
        are skipped with a warning rather than aborting the call.
        """
        log_path = self._log_path(sid)
        if not log_path.exists():
            return []
        try:
            # Off-load blocking I/O to a thread so we don't stall the
            # event loop on large logs.
            lines = await asyncio.to_thread(self._tail_lines, log_path, n)
        except OSError as e:
            raise TaskDispatcherError(
                f"could not read log file {log_path}: {e}"
            ) from e
        events: list[dict] = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("status: skipping malformed line in %s: %s",
                            log_path, e)
        return events

    async def replay(self, sid: str) -> AsyncIterator[dict]:
        """Async-iterate over all events in ``<log_dir>/<sid>.jsonl``.

        Yields parsed event dicts in write order. If the log file
        doesn't exist, yields nothing.
        """
        log_path = self._log_path(sid)
        if not log_path.exists():
            return
        try:
            # Read whole file in a thread, then yield line-by-line.
            content = await asyncio.to_thread(log_path.read_text)
        except OSError as e:
            raise TaskDispatcherError(
                f"could not read log file {log_path}: {e}"
            ) from e
        for line in content.splitlines():
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("replay: skipping malformed line in %s: %s",
                            log_path, e)

    async def close(self) -> None:
        """Release the supervisor and close the log_capture.

        Idempotent: safe to call multiple times.
        """
        if self._started:
            try:
                await self._sup.close()
            except Exception as e:  # pragma: no cover - defensive
                log.warning("supervisor close failed: %s", e)
        try:
            await self._cap.close()
        except Exception as e:  # pragma: no cover - defensive
            log.warning("log_capture close failed: %s", e)
        self._started = False

    # ---------------- internals ----------------

    async def _start(self) -> None:
        """Initialize the supervisor, open a session, and remember its sid."""
        await self._sup.start()
        result = await self._sup.new_session(self._cwd)
        # Reasonix returns ``{"sessionId": "..."}`` for session/new.
        sid = result.get("sessionId") if isinstance(result, Mapping) else None
        if not sid:
            raise TaskDispatcherError(
                f"new_session did not return a sessionId: {result!r}"
            )
        self._sid = sid
        self._started = True

    def _log_path(self, sid: str) -> Path:
        """Resolve the NDJSON log path for ``sid``.

        We ask the log_capture (or, in tests, a real LogCapture) for
        its log_dir. If log_capture doesn't expose a log_dir (e.g. a
        duck-typed test double), fall back to the canonical default.
        """
        log_dir = getattr(self._cap, "_log_dir", None)
        if log_dir is None:
            try:
                from adapter.log_capture import SESSION_LOG_DIR
                log_dir = SESSION_LOG_DIR
            except ImportError:
                log_dir = Path("~/.picoclaw/logs/reasonix-subagent").expanduser()
        return Path(log_dir) / f"{sid}.jsonl"

    @staticmethod
    def _tail_lines(path: Path, n: int) -> list[str]:
        """Return the last ``n`` lines of a text file (blocking helper)."""
        with open(path, "r", encoding="utf-8") as f:
            # deque with maxlen keeps memory bounded even for huge files.
            from collections import deque
            tail = deque(f, maxlen=n)
        return list(tail)


def _string_to_content(task: str) -> list[dict]:
    """Wrap a plain string into the ACP ``ContentBlock`` shape.

    Reasonix takes a list of content blocks; the only one we need for
    Phase 1 is ``{"type": "text", "text": ...}``.
    """
    return [{"type": "text", "text": task}]


__all__ = [
    "TaskDispatcher",
    "TaskDispatcherError",
    "SupervisorLike",
    "LogCaptureLike",
]
