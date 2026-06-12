"""Process supervisor for one Reasonix ACP session.

Owns a single subprocess (``bin/reasonix acp``) and its stdio. Wraps the
pipes as a :class:`adapter.acp_client.Conn` and exposes the four
lifecycle methods a Phase 1 caller needs:

    1. ``start()``        — spawn + ``initialize``
    2. ``new_session()``  — open a session at a workspace root
    3. ``prompt()``       — send user content, await end-of-turn
    4. ``cancel()`` / ``close()`` — stop work, tear down

Design rules (locked 2026-06-11):

* **One Supervisor = one process**. No pooling; the Phase 2 plan adds a
  pool if/when we need concurrency.
* **State machine is explicit** (see :attr:`Supervisor.state`). The
  supervisor never auto-recovers a crashed process — the caller decides
  whether to spawn a fresh one.
* **No blocking I/O outside asyncio**. All reads/writes go through
  :class:`Conn` which uses ``asyncio.StreamReader``/``StreamWriter``.
* **Stderr is captured**, not ignored. Each stderr line is forwarded to
  a Python logger and (optionally) appended to a file. Reasonix only
  writes diagnostics to stderr; the JSON-RPC channel is stdout-only.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Optional, Union

from adapter.acp_client import Conn, ConnError, JSONRPCError, ServerRequestHandler

log = logging.getLogger("adapter.supervisor")

State = Literal["new", "starting", "ready", "prompting", "closed", "dead"]

# Default binary resolution: <adapter_dir>/../bin/reasonix  (the
# symlink at cli-bin/reasonix points to the same file; we prefer the
# repo-local copy so a `make build` in DeepSeek-Reasonix is picked up
# without an extra deploy step).
DEFAULT_BINARY = Path(__file__).resolve().parent.parent / "bin" / "reasonix"

# Content block for session/prompt — a thin alias; Reasonix uses the
# ACP ``ContentBlock`` shape, but the adapter only needs ``text`` for
# Phase 1.
ContentBlock = Mapping[str, Any]

# Notification handler signature (matches Conn.on_notification).
NotificationHandler = Callable[[dict], Awaitable[None]]

class SupervisorError(RuntimeError):
    """Raised for supervisor-level failures (spawn, crash, double-start)."""


class Supervisor:
    """Owns one ``bin/reasonix acp`` subprocess.

    Parameters
    ----------
    binary:
        Path to the reasonix binary. Falls back to
        :data:`DEFAULT_BINARY`, then to ``$REASONIX_BIN`` if set.
    cwd:
        Working directory for the subprocess. If ``None``, the
        subprocess inherits the current process's cwd. Reasonix uses
        this as the default workspace root when a session does not
        pass one.
    model:
        Provider/model name passed as ``--model``. ``None`` means
        "use reasonix's config default".
    env:
        Extra environment variables merged on top of ``os.environ``.
    stderr_log:
        Optional file path. If set, every stderr line from the child
        is appended (one line, no timestamp). If ``None``, stderr is
        only forwarded to the ``adapter.supervisor`` logger.
    on_notification:
        Optional coroutine that receives every ``session/update``
        notification. The supervisor forwards it to :class:`Conn`.
    init_timeout:
        Seconds to wait for the ``initialize`` response.
    prompt_timeout:
        Seconds to wait for a ``session/prompt`` to deliver a stop
        notification. ``None`` disables the cap (use with care).
    auto_approve:
        If True, the supervisor auto-approves any server-initiated
        ``session/request_permission`` request by replying with
        ``{"outcome": "approved"}``. Phase 1 debug #5 default is
        False (pass-through) so tests can inspect the server request
        stream. PicoClaw's tools layer is expected to set this True
        for trusted sessions.
    """

    def __init__(
        self,
        binary=None,
        *,
        cwd=None,
        model: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
        stderr_log=None,
        on_notification: Optional[NotificationHandler] = None,
        init_timeout: float = 10.0,
        prompt_timeout: Optional[float] = 120.0,
        auto_approve: bool = False,
    ) -> None:
        self._binary = self._resolve_binary(binary)
        self._cwd = Path(cwd) if cwd is not None else Path.cwd()
        self._model = model
        self._env_extra = dict(env) if env else {}
        self._stderr_log: Optional[Path] = Path(stderr_log) if stderr_log else None
        self._on_notification = on_notification
        self._init_timeout = init_timeout
        self._prompt_timeout = prompt_timeout
        self._auto_approve = auto_approve

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._conn: Optional[Conn] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._proc_watch_task: Optional[asyncio.Task] = None
        self._state: State = "new"
        self._exit_code: Optional[int] = None
        self._last_error: Optional[BaseException] = None

    # ---------------- public API ----------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def on_notification(self) -> Optional[NotificationHandler]:
        """Return the current notification handler.

        Phase 1 (debug #5) wired ``Conn``'s notification callback only
        at ``start()`` time, reading the value of ``self._on_notification``
        in the constructor. TaskDispatcher (and any post-construction
        wiring) needs to swap the handler in *after* ``__init__`` but
        *before* the first ``session/prompt`` fires. Without this
        setter, ``sup.on_notification = new_handler`` silently created
        a separate instance attribute, leaving ``Conn`` wired to the
        no-op fallback and dropping every notification on the floor.

        If the supervisor is already started, we also propagate the
        change to ``self._conn._on_notification`` so the live reader
        loop picks it up immediately (no need to wait for the next
        session). Writes are O(1) and synchronous.
        """
        return self._on_notification

    @on_notification.setter
    def on_notification(self, fn: Optional[NotificationHandler]) -> None:
        self._on_notification = fn
        if self._conn is not None:
            self._conn._on_notification = fn or _noop  # type: ignore[attr-defined]

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc is not None else None

    @property
    def conn(self) -> Conn:
        if self._conn is None:
            raise SupervisorError("conn not available: supervisor not started")
        return self._conn

    @property
    def exit_code(self) -> Optional[int]:
        return self._exit_code

    async def start(self) -> dict:
        """Spawn the subprocess and run ``initialize``. Returns the init result.

        Raises :class:`SupervisorError` if already started, or if the
        binary is missing / crashes / fails to respond in time.
        """
        if self._state != "new":
            raise SupervisorError(f"cannot start: state is {self._state!r}, expected 'new'")
        if not self._binary.exists():
            raise SupervisorError(f"reasonix binary not found: {self._binary}")
        self._state = "starting"

        argv = [str(self._binary), "acp"]
        if self._model:
            argv.extend(["--model", self._model])

        env = {**os.environ, **self._env_extra}
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd),
                env=env,
                # Detach from our process group so a Ctrl-C in the
                # parent does not propagate. We control termination
                # via close() instead.
                start_new_session=True,
            )
        except (FileNotFoundError, OSError) as e:
            self._state = "dead"
            self._last_error = e
            raise SupervisorError(f"spawn failed: {e}") from e

        log.info("spawned reasonix pid=%s argv=%s", self._proc.pid, argv)
        assert self._proc.stdout is not None and self._proc.stderr is not None
        self._conn = Conn(
            self._proc.stdout,
            self._proc.stdin,
            self._on_notification or _noop,
            on_server_request=self._handle_server_request if self._auto_approve else None,
        )
        # Start background tasks: reader (Conn.run) + stderr drainer
        self._reader_task = asyncio.create_task(
            self._conn.run(), name="reasonix-conn-run"
        )
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(self._proc.stderr), name="reasonix-stderr-drain"
        )
        # Also monitor the process itself so we can flip to "dead"
        # if it exits unexpectedly.
        self._proc_watch_task = asyncio.create_task(
            self._watch_process(), name="reasonix-proc-watch"
        )

        try:
            result = await asyncio.wait_for(
                self._conn.request("initialize", params={}),
                timeout=self._init_timeout,
            )
        except (asyncio.TimeoutError, JSONRPCError, ConnError) as e:
            self._last_error = e
            self._state = "dead"
            await self._terminate_quickly()
            raise SupervisorError(f"initialize failed: {e}") from e
        self._state = "ready"
        return result

    async def new_session(self, cwd: str) -> dict:
        """Open a new ACP session rooted at ``cwd``."""
        self._require_state("ready")
        result = await self._conn.request(
            "session/new", params={"cwd": cwd}
        )
        return result

    async def prompt(self, sid: str, content: list) -> dict:
        """Send ``session/prompt`` and block until the agent signals
        end-of-turn.

        Resolution rules (R5 — discovered against reasonix v1.4.0):

        1. Reasonix v1.4.0 returns ``stopReason`` INLINE in the
           ``session/prompt`` response (``SessionPromptResult``),
           not as a separate ``session/update`` with
           ``sessionUpdate=stop``. The response alone is sufficient
           to resolve the call.
        2. Some Reasonix versions / scenarios DO emit a stop
           notification (e.g. mid-prompt cancel). We race a stop
           future against the response and resolve on whichever
           arrives first.
        3. Whichever wins, we return a dict shaped like a stop
           notification (``{"sessionId": sid, "update": {"sessionUpdate":
           "stop", "stopReason": ...}}``) so the rest of the adapter
           (TaskDispatcher, callers) can treat both signals uniformly.

        Returns the normalized stop payload, or raises
        :class:`SupervisorError` on transport failure /
        :class:`asyncio.TimeoutError` if neither signal arrives
        within ``prompt_timeout``.
        """
        self._require_state("ready")
        self._state = "prompting"
        stop_future, original_handler = self._install_stop_wrapper(sid)
        try:
            request_task: asyncio.Task = asyncio.create_task(
                self._conn.request(
                    "session/prompt",
                    params={"sessionId": sid, "prompt": list(content)},
                ),
                name="reasonix-prompt-request",
            )

            async def _wait_stop() -> dict:
                return await stop_future

            stop_task: asyncio.Task = asyncio.create_task(
                _wait_stop(), name="reasonix-stop-watcher"
            )

            async def _normalize_response(resp: dict) -> dict:
                """Wrap an inline ``SessionPromptResult`` so it looks
                like a stop notification for downstream callers."""
                return {
                    "sessionId": sid,
                    "update": {
                        "sessionUpdate": "stop",
                        "stopReason": resp.get("stopReason", "end_turn"),
                    },
                    "_source": "response",
                    "transcriptPath": resp.get("transcriptPath"),
                }

            try:
                done, pending = await asyncio.wait(
                    {request_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=self._prompt_timeout,
                )
            except asyncio.TimeoutError:
                request_task.cancel()
                stop_task.cancel()
                raise

            if not done:
                # Shouldn't happen because asyncio.wait with timeout
                # raises on timeout, but be defensive.
                request_task.cancel()
                stop_task.cancel()
                raise asyncio.TimeoutError(
                    f"prompt timeout (no signal in {self._prompt_timeout}s)"
                )

            # Cancel the loser so we don't leak tasks.
            for t in pending:
                t.cancel()

            if request_task in done:
                try:
                    resp = request_task.result()
                except (JSONRPCError, ConnError) as e:
                    raise SupervisorError(f"prompt failed: {e}") from e
                # Stop notification may have raced ahead — prefer
                # it if it already landed, otherwise normalize the
                # response.
                if stop_task.done() and not stop_task.cancelled():
                    try:
                        return stop_task.result()
                    except Exception:
                        pass
                return await _normalize_response(resp if isinstance(resp, dict) else {})

            # stop_task won the race
            if stop_task.cancelled():
                # request still pending — await it for the response
                # payload, then normalize.
                try:
                    resp = await request_task
                except (JSONRPCError, ConnError) as e:
                    raise SupervisorError(f"prompt failed: {e}") from e
                return await _normalize_response(resp if isinstance(resp, dict) else {})
            try:
                return stop_task.result()
            except (JSONRPCError, ConnError) as e:
                raise SupervisorError(f"stop watcher failed: {e}") from e
        finally:
            self._restore_notification_handler(original_handler)
            if self._state != "dead":
                self._state = "ready"

    async def cancel(self, sid: str) -> None:
        """Best-effort cancel. Returns when the cancel notification
        is sent (does not wait for the server to acknowledge)."""
        if self._state not in ("ready", "prompting"):
            return
        try:
            await self._conn.notify("session/cancel", params={"sessionId": sid})
        except ConnError as e:
            log.warning("cancel notify failed (process likely already gone): %s", e)

    async def close(self) -> None:
        """Graceful shutdown: cancel any in-flight prompt, drain
        pipes, terminate. Idempotent — calling twice is a no-op."""
        if self._state in ("closed", "dead", "new"):
            return
        log.info("closing supervisor pid=%s state=%s", self.pid, self._state)
        if self._conn is not None:
            await self._conn.close()
        await self._terminate_quickly()
        for t in (self._reader_task, self._stderr_task, self._proc_watch_task):
            if t is not None and not t.done():
                try:
                    await asyncio.wait_for(t, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    t.cancel()
        self._state = "closed"

    async def wait_closed(self) -> int:
        """Wait for the subprocess to exit and return its return code.
        Returns 0 if the process was never spawned."""
        if self._proc is None:
            return 0
        return await self._proc.wait()

    # ---------------- internals ----------------

    def _handle_server_request(self, frame: dict) -> dict:
        """Handle a server-initiated JSON-RPC request.

        Phase 1 (debug #5): we only auto-approve
        ``session/request_permission``. Other server-initiated methods
        (e.g. ``session/fork``, ``session/rewind``) are rejected with
        ``METHOD_NOT_FOUND`` via Conn's no-handler fallback, which the
        Reasonix server treats as "client doesn't support this method".

        Returns
        -------
        dict
            The result payload to send back as a JSON-RPC response.
            For ``session/request_permission`` the auto-approve result
            is ``{"outcome": "approved"}`` (per protocol.go:402-450).
        """
        method = frame.get("method", "")
        params = frame.get("params") or {}
        if method == "session/request_permission":
            tool = params.get("toolName") or params.get("name") or "<unknown>"
            log.info("auto-approving session/request_permission tool=%s", tool)
            return {"outcome": "approved"}
        # Unknown server-initiated method. Returning ``None`` would
        # cause Conn to send METHOD_NOT_FOUND, which is the correct
        # behavior. We log a warning so misconfigured clients show up
        # in the supervisor's stderr.
        log.warning("unhandled server-initiated request: %s", method)
        return None  # type: ignore[return-value]

    def _require_state(self, expected: State) -> None:
        if self._state != expected:
            raise SupervisorError(
                f"invalid state: have {self._state!r}, need {expected!r}"
            )

    def _install_stop_wrapper(self, sid: str) -> tuple[asyncio.Future, Optional[Callable]]:
        """Install a notification wrapper that captures the next
        ``session/update`` with ``sessionUpdate='stop'`` for ``sid``.

        Returns ``(future, original_handler)``:
        - ``future`` resolves to the stop params payload when stop arrives.
        - ``original_handler`` is whatever was on
          ``self._conn._on_notification`` before; callers should
          restore it (passing it back via ``_restore_notification_handler``)
          in a ``finally`` block.

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

        This MUST be called BEFORE ``session/prompt`` is sent: Conn
        re-reads ``_on_notification`` for every frame, so the wrapper
        must be live by the time Reasonix emits its first stop
        notification (which on a tiny prompt can arrive before
        ``session/prompt``'s own response frame).
        """
        assert self._conn is not None
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        original = self._conn._on_notification  # type: ignore[attr-defined]

        async def wrapper(frame: dict) -> None:
            params = frame.get("params") or {}
            update = params.get("update") or {}
            if (
                frame.get("method") == "session/update"
                and params.get("sessionId") == sid
                and update.get("sessionUpdate") == "stop"
            ):
                if not fut.done():
                    fut.set_result(params)
            if original is not None:
                await original(frame)

        self._conn._on_notification = wrapper  # type: ignore[attr-defined]
        return fut, original

    def _restore_notification_handler(self, original: Optional[Callable]) -> None:
        """Restore the previous ``Conn._on_notification`` handler.

        Best-effort: if the conn was torn down, the attribute may
        already be gone — that's fine, no restoration needed.
        """
        if self._conn is None:
            return
        try:
            self._conn._on_notification = original  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive
            log.warning("could not restore _on_notification handler", exc_info=True)

    async def _drain_stderr(self, stream: asyncio.StreamReader) -> None:
        """Read stderr line-by-line, log + tee to file."""
        fh = None
        try:
            if self._stderr_log is not None:
                self._stderr_log.parent.mkdir(parents=True, exist_ok=True)
                fh = open(self._stderr_log, "a", encoding="utf-8")
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                log.info("reasonix-stderr: %s", line)
                if fh is not None:
                    fh.write(line + "\n")
                    fh.flush()
        except Exception as e:  # pragma: no cover - defensive
            log.warning("stderr drainer crashed: %s", e)
        finally:
            if fh is not None:
                fh.close()

    async def _watch_process(self) -> None:
        """Watch the subprocess and flip state to 'dead' if it exits
        unexpectedly."""
        assert self._proc is not None
        rc = await self._proc.wait()
        self._exit_code = rc
        if self._state not in ("closed",):
            log.warning("reasonix pid=%s exited unexpectedly rc=%s state=%s",
                        self._proc.pid, rc, self._state)
            self._state = "dead"
            if self._conn is not None:
                await self._conn.close()

    async def _terminate_quickly(self) -> None:
        """Best-effort terminate with a short grace period."""
        if self._proc is None or self._proc.returncode is not None:
            return
        try:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                log.warning("reasonix did not exit on SIGTERM; sending SIGKILL")
                self._proc.kill()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        except ProcessLookupError:
            pass

    @staticmethod
    def _resolve_binary(supplied) -> Path:
        if supplied is not None:
            return Path(supplied)
        env = os.environ.get("REASONIX_BIN")
        if env:
            return Path(env)
        return DEFAULT_BINARY


async def _noop(_frame: dict) -> None:
    return None
