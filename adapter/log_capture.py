"""Event log capture for Reasonix ACP sessions.

Subscribes to ``session/update`` notifications dispatched by
:class:`adapter.acp_client.Conn`. Each notification is enriched with
a ``ts`` (wall-clock UTC ISO8601) and a ``kind`` (echo of
``sessionUpdate`` for grep convenience), then written as NDJSON to
``<log_dir>/<sessionId>.jsonl``.

Why NDJSON?
    - One record per line → tail -f, ``grep``, and ``jq`` friendly.
    - Append-only → safe with concurrent writers (we serialize per file).
    - Trivial to backfill / replay; no schema migration needed.

Public surface
--------------
- :class:`LogCapture` — the subscriber. Construct with a Supervisor-like
  object that exposes ``pid`` and ``current_sid``. Call ``.handler()`` to
  get the coroutine that :class:`Conn` will dispatch notifications to.
- :func:`default_log_dir` / :func:`default_pid_dir` — resolve the canonical
  ``~/.picoclaw/logs/reasonix-subagent/`` and ``~/.picoclaw/state/reasonix/``
  paths so R4 (task_dispatcher) can wire the same defaults.

Locking model
-------------
We use one ``asyncio.Lock`` per (session_id, file_path) tuple, lazily
created and cached in :attr:`_file_locks`. This means writes to *different*
sessions never block each other, but writes to the *same* session are
serialized within a single event loop turn. Reasonix dispatches
notifications serially per connection, so in practice the lock is rarely
contended; it exists only to defend against future parallel dispatch
(e.g. R4 multi-session supervisor).

Rotation
--------
When a session's NDJSON file grows past :data:`MAX_FILE_BYTES` (50 MB),
the active file is renamed to ``<sid>.jsonl.<UTC-timestamp>.rotated``
and a new active file is opened. We keep an unlimited number of
rotated files; the 50 MB cap is per-active-file, not total. Disk
pressure is deferred to ``state/reasonix/``'s own logrotate config
(Phase 2 concern).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional

log = logging.getLogger("adapter.log_capture")

# Default locations (canonical for R4 / task_dispatcher).
# Kept here, not in __init__, so subprocess tooling can read them too.
SESSION_LOG_DIR = Path("~/.picoclaw/logs/reasonix-subagent").expanduser()
PID_DIR = Path("~/.picoclaw/state/reasonix").expanduser()

# Per-active-file rotation cap.
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

# Rotated file suffix — see ``_maybe_rotate``.
ROTATED_SUFFIX = ".rotated"

# sessionUpdate kinds that get pretty-printed to stdout when
# ``pretty_stdout=True``. Thoughts are noisy and rarely useful at
# a glance; we keep them in the log but suppress the live echo.
PRETTY_KINDS = frozenset({
    "agent_message_chunk",
    "tool_call",
    "stop",
    "ask",
    "permission",
})

# Shape of what a Supervisor-like object must expose. We don't import
# the Supervisor class to avoid a circular dep (supervisor -> log_capture
# is a one-way edge; LogCapture is the *leaf*).
SupervisorLike = Any  # protocol: {pid: int|None, current_sid: str|None, state: str}


class LogCaptureError(RuntimeError):
    """Raised when the log_capture can't write a notification
    (permission denied, disk full, etc.)."""


def default_log_dir() -> Path:
    """Resolve the default per-session log directory."""
    return SESSION_LOG_DIR


def default_pid_dir() -> Path:
    """Resolve the default per-session PID file directory."""
    return PID_DIR


def _now_iso() -> str:
    """UTC ISO8601 with millisecond precision + Z suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _now_compact() -> str:
    """Compact UTC timestamp for rotated-file names: YYYYMMDDTHHMMSSZ."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class LogCapture:
    """Capture session/update notifications to NDJSON + PID files.

    Parameters
    ----------
    supervisor:
        Object exposing ``pid`` (int|None), ``current_sid`` (str|None),
        and ``state`` (str). Reads only — LogCapture never mutates the
        supervisor.
    log_dir:
        Directory where ``<sid>.jsonl`` files live. Created lazily.
        Defaults to :data:`SESSION_LOG_DIR`.
    pid_dir:
        Directory where ``<sid>.pid`` files live. Created lazily.
        Defaults to :data:`PID_DIR`.
    pretty_stdout:
        If ``True``, certain kinds are also written to stdout in a
        human-friendly format (one line per chunk). Default ``False``
        because PicoClaw is a Telegram bot — extra stdout noise is
        anti-feature in production.
    """

    def __init__(
        self,
        supervisor: SupervisorLike,
        log_dir: Optional[Path] = None,
        pid_dir: Optional[Path] = None,
        pretty_stdout: bool = False,
    ) -> None:
        self._sup = supervisor
        self._log_dir: Path = Path(log_dir) if log_dir is not None else SESSION_LOG_DIR
        self._pid_dir: Path = Path(pid_dir) if pid_dir is not None else PID_DIR
        self._pretty_stdout = pretty_stdout

        # Per-file async locks. Keyed by the *Path* of the active log
        # file. We don't expect collisions on different sessions, but
        # we also don't trust the absence of weird re-entrancy.
        self._file_locks: dict[Path, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()  # protects _file_locks dict itself
        # Cached set of (sid, file) for which we already wrote a PID file.
        # We use this to avoid an mtime() syscall on every event.
        self._pid_written: set[str] = set()
        # Whether close() has been called. Idempotent no-op.
        self._closed = False

    # ---------------- public API ----------------

    def handler(self) -> Callable[[dict], Awaitable[None]]:
        """Return the coroutine that :class:`Conn` will dispatch to.

        Captures ``self`` by closure; the returned function takes a
        single ``frame`` dict (the parsed JSON-RPC notification).
        """
        async def _handler(frame: dict) -> None:
            await self._dispatch(frame)
        return _handler

    async def close(self) -> None:
        """Idempotent cleanup. We don't keep persistent file handles
        (every write opens+closes), so there's nothing to flush."""
        self._closed = True

    # ---------------- internals ----------------

    async def _dispatch(self, frame: dict) -> None:
        """Decide whether to log this notification. Silent no-op for
        notifications we don't care about.

        Wire format (per protocol.go:113-200, ``SessionUpdate`` is a
        tagged union wrapped under ``params.update``)::

            {
              "jsonrpc": "2.0",
              "method": "session/update",
              "params": {
                "sessionId": "...",
                "update": {"sessionUpdate": "stop", "stopReason": "..."}
              }
            }
        """
        if frame.get("method") != "session/update":
            return
        params = frame.get("params") or {}
        sid = params.get("sessionId")
        if not sid:
            # Notification without a session id — we can't route it
            # to a per-session file, so drop it. The supervisor's own
            # on_notification will still see it.
            return

        record = self._enrich(sid, params)
        try:
            await self._write(sid, record)
        except LogCaptureError:
            raise
        except Exception as e:
            raise LogCaptureError(f"write failed for sid={sid}: {e}") from e

    def _enrich(self, sid: str, params: dict) -> dict:
        """Stamp the record with ts, kind, and a few supervisor hints.

        Reads ``sessionUpdate`` and the rest of the update payload from
        ``params.update`` (nested), not ``params`` (flat). See
        ``_dispatch`` for the wire format.
        """
        update = params.get("update") or {}
        kind = update.get("sessionUpdate", params.get("sessionUpdate", "unknown"))
        rec = {
            "ts": _now_iso(),
            "sessionId": sid,
            "sessionUpdate": kind,
            "kind": kind,
        }
        # Pass-through the rest of update (content, stopReason, toolCallId, …)
        for k, v in update.items():
            if k not in rec:
                rec[k] = v
        # Top-level fields that aren't ``sessionId`` (already used) and
        # aren't ``update`` (handled above).
        for k, v in params.items():
            if k not in rec and k not in ("sessionId", "update"):
                rec[k] = v
        return rec

    async def _write(self, sid: str, record: dict) -> None:
        """Append the record to ``<log_dir>/<sid>.jsonl`` and ensure
        a PID file exists. Wraps the actual I/O in the per-file lock."""
        file_path = self._log_path(sid)
        lock = await self._lock_for(file_path)
        async with lock:
            # Off-load the blocking I/O to a thread so we don't stall
            # the event loop on slow disks.
            await asyncio.to_thread(self._append, file_path, record)

        # Pretty stdout is opt-in and never blocks on I/O.
        if self._pretty_stdout and record["kind"] in PRETTY_KINDS:
            self._print_pretty(record)

        # PID file is best-effort: missing supervisor.pid means we
        # don't write anything. (E.g. during a graceful close after
        # the process has already exited.)
        await asyncio.to_thread(self._maybe_write_pid, sid)

    # ---- blocking helpers (run via to_thread) ----

    def _append(self, file_path: Path, record: dict) -> None:
        """Append one NDJSON line. Handles lazy dir creation + rotation."""
        self._maybe_rotate(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _maybe_rotate(self, file_path: Path) -> None:
        """If the active file is at/over the cap, rename it aside."""
        try:
            size = file_path.stat().st_size
        except FileNotFoundError:
            return
        if size < MAX_FILE_BYTES:
            return
        rotated = file_path.with_name(
            f"{file_path.name}.{_now_compact()}{ROTATED_SUFFIX}"
        )
        try:
            os.replace(file_path, rotated)
            log.info("rotated log %s -> %s (%d bytes)", file_path, rotated, size)
        except OSError as e:
            # If rotation fails (e.g. permission denied), we keep
            # appending; the cap is a soft limit, not a hard wall.
            log.warning("log rotation failed for %s: %s", file_path, e)

    def _maybe_write_pid(self, sid: str) -> None:
        """Write ``<pid_dir>/<sid>.pid`` once per session, never rewrite."""
        if sid in self._pid_written:
            return
        pid = getattr(self._sup, "pid", None)
        if pid is None:
            return
        self._pid_dir.mkdir(parents=True, exist_ok=True)
        pid_path = self._pid_dir / f"{sid}.pid"
        try:
            pid_path.write_text(str(pid), encoding="utf-8")
        except OSError as e:
            # PID file is informational; never break the capture
            # path because we couldn't write it.
            log.warning("could not write PID file %s: %s", pid_path, e)
            return
        self._pid_written.add(sid)

    @staticmethod
    def _print_pretty(record: dict) -> None:
        """Emit a short, single-line pretty view to stdout."""
        kind = record["kind"]
        sid = record["sessionId"]
        # Keep it terse — stdout is for humans, NDJSON is the source of truth.
        if kind == "agent_message_chunk":
            content = record.get("content") or {}
            text = content.get("text", "")
            print(f"[{sid[:8]}] {text}", flush=True)
        elif kind == "tool_call":
            content = record.get("content") or {}
            print(f"[{sid[:8]}] 🔧 {content.get('toolName', '?')}", flush=True)
        elif kind == "stop":
            reason = record.get("stopReason", "")
            print(f"[{sid[:8]}] ⏹  {reason}", flush=True)
        elif kind == "ask":
            content = record.get("content") or {}
            print(f"[{sid[:8]}] ❓ {content.get('question', '?')}", flush=True)
        elif kind == "permission":
            content = record.get("content") or {}
            print(f"[{sid[:8]}] 🛂  {content.get('tool', '?')}", flush=True)

    # ---- path helpers ----

    def _log_path(self, sid: str) -> Path:
        return self._log_dir / f"{sid}.jsonl"

    async def _lock_for(self, path: Path) -> asyncio.Lock:
        # We need to allocate the lock outside of the per-file lock
        # itself, hence the meta-lock.
        async with self._locks_guard:
            lock = self._file_locks.get(path)
            if lock is None:
                lock = asyncio.Lock()
                self._file_locks[path] = lock
            return lock


__all__ = [
    "LogCapture",
    "LogCaptureError",
    "SESSION_LOG_DIR",
    "PID_DIR",
    "MAX_FILE_BYTES",
    "ROTATED_SUFFIX",
    "PRETTY_KINDS",
    "default_log_dir",
    "default_pid_dir",
]
