"""JSON-RPC 2.0 transport over NDJSON (used by Reasonix's Agent Client Protocol).

Wire format
-----------
- One JSON object per line, lines separated by ``\\n``, UTF-8 encoded.
- Request:       ``{"jsonrpc": "2.0", "id": <int>, "method": <str>, "params": <obj>}``
- Response ok:   ``{"jsonrpc": "2.0", "id": <int>, "result": <any>}``
- Response err:  ``{"jsonrpc": "2.0", "id": <int>, "error": {"code": <int>, "message": <str>, "data"?: <any>}}``
- Notification:  ``{"jsonrpc": "2.0", "method": <str>, "params": <obj>}``  (no ``id``)

This module owns the bytes, the id counter, and the request/response
correlation. It does NOT know about Reasonix's specific methods or events —
that's the job of ``supervisor`` and ``log_capture``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger("adapter.acp_client")

# JSON-RPC 2.0 standard error codes (subset; Reasonix may use others).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Reasonix's wire cap (per protocol.go: maxFrameSize = 32 MiB).
MAX_FRAME_BYTES = 32 * 1024 * 1024

# Type aliases
NotificationHandler = Callable[[dict], Awaitable[None]]

# Server-initiated request handler. Receives the parsed frame (which carries
# ``id``, ``method``, and ``params``) and must RETURN a result dict (which
# the connection will wrap into a JSON-RPC response). Returning ``None``
# causes the connection to send back a ``METHOD_NOT_FOUND`` error, which
# the server treats as the method being unsupported by this client.
ServerRequestHandler = Callable[[dict], "asyncio.Future[Any] | Any"]


class JSONRPCError(Exception):
    """Raised when the server (Reasonix) returns an error response.

    Attributes:
        code: integer error code (per JSON-RPC 2.0 or Reasonix extension).
        message: human-readable description.
        data: optional structured context (may be None).
    """

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"JSONRPCError({code}): {message}")
        self.code = code
        self.message = message
        self.data = data


class ConnError(Exception):
    """Raised for transport-level errors (e.g. closed pipe, malformed frame)."""


class Conn:
    """Asyncio JSON-RPC 2.0 connection over an arbitrary reader/writer pair.

    Lifecycle:
        1. Construct with the reader, writer, and notification callback.
        2. ``await conn.run()`` once (typically as a background task). It
           loops reading NDJSON frames and dispatching responses/notifications.
        3. ``request()`` / ``notify()`` from anywhere.
        4. ``close()`` to stop the read loop and detach pending futures.

    The ``writer`` may be ``None`` if the caller only needs to read (e.g.
    notification-only streams). In that case ``request()`` and ``notify()``
    raise ``ConnError``.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: Optional[asyncio.StreamWriter],
        on_notification: NotificationHandler,
        on_server_request: Optional[ServerRequestHandler] = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._on_notification = on_notification
        self._on_server_request = on_server_request
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._closed = False
        self._write_lock = asyncio.Lock()
        self._run_task: Optional[asyncio.Task] = None

    # ---------------- public API ----------------

    async def run(self) -> None:
        """Block reading frames until the reader returns EOF or ``close()`` is called.

        Wire loop:
            - Read one line (``\\n``-delimited).
            - Parse JSON; if parse fails, log + skip (do NOT crash).
            - If frame has ``id`` → resolve the matching future with result or error.
            - Else if frame has ``method`` → invoke ``on_notification`` (await it).
            - Else log + skip (malformed).
        """
        try:
            while not self._closed:
                try:
                    line = await self._reader.readline()
                except (asyncio.IncompleteReadError, ConnectionError):
                    break
                if not line:
                    # EOF
                    break

                if len(line) > MAX_FRAME_BYTES:
                    log.warning(
                        "frame exceeds %d bytes, truncating and dropping",
                        MAX_FRAME_BYTES,
                    )
                    continue

                try:
                    frame = json.loads(line.decode("utf-8", errors="replace"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    log.warning("malformed frame skipped: %s", exc)
                    continue

                if not isinstance(frame, dict):
                    log.warning("non-object frame skipped: %r", frame)
                    continue

                await self._dispatch_frame(frame)
        except asyncio.CancelledError:
            # Normal shutdown
            raise
        except Exception:
            log.exception("unexpected error in read loop")
            raise
        finally:
            self._closed = True
            # Fail any in-flight requests so callers don't hang
            for fid, fut in self._pending.items():
                if not fut.done():
                    fut.set_exception(
                        ConnError(f"connection closed (id={fid})")
                    )
            self._pending.clear()

    async def request(self, method: str, params: Any = None) -> Any:
        """Send a request and await the response.

        Raises:
            ConnError: writer is None (read-only conn) or conn is closed.
            JSONRPCError: server returned an error frame.
        """
        if self._writer is None:
            raise ConnError("cannot request on a read-only conn (writer=None)")
        if self._closed:
            raise ConnError("conn is closed")

        self._id += 1
        fid = self._id
        frame = {
            "jsonrpc": "2.0",
            "id": fid,
            "method": method,
        }
        if params is not None:
            frame["params"] = params

        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[fid] = fut

        try:
            await self._send_frame(frame)
        except Exception:
            self._pending.pop(fid, None)
            if not fut.done():
                fut.cancel()
            raise

        return await fut

    async def notify(self, method: str, params: Any = None) -> None:
        """Send a notification (no id, no response expected).

        Raises:
            ConnError: writer is None or conn is closed.
        """
        if self._writer is None:
            raise ConnError("cannot notify on a read-only conn (writer=None)")
        if self._closed:
            raise ConnError("conn is closed")

        frame = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            frame["params"] = params
        await self._send_frame(frame)

    async def close(self) -> None:
        """Idempotent shutdown. Cancels read loop, closes writer if owned."""
        if self._closed:
            return
        self._closed = True
        # Closing the writer triggers EOF on the reader, breaking the loop.
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            try:
                await self._writer.wait_closed()
            except Exception:
                pass
        # If run() is being awaited externally, cancel it.
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()

    # ---------------- internals ----------------

    async def _dispatch_frame(self, frame: dict) -> None:
        """Route a single frame to a pending request or to the notification cb."""
        if "id" in frame and "method" in frame:
            # Server-initiated request. Phase 1 ignored these; Phase 2 (R4
            # auto-approve) routes them to ``on_server_request`` which
            # returns a result we echo back as a JSON-RPC response. If no
            # handler is registered we fall back to METHOD_NOT_FOUND so
            # callers don't hang waiting for a reply.
            fid = frame["id"]
            method = frame.get("method")
            if self._on_server_request is None or self._writer is None:
                log.warning(
                    "received server-initiated request (not supported): %s", method
                )
                if self._writer is not None:
                    err_frame = {
                        "jsonrpc": "2.0",
                        "id": fid,
                        "error": {"code": METHOD_NOT_FOUND, "message": f"method {method!r} not supported by client"},
                    }
                    try:
                        await self._send_frame(err_frame)
                    except Exception:
                        log.exception("failed to send METHOD_NOT_FOUND reply for %s", method)
                return
            try:
                result = self._on_server_request(frame)
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception as exc:  # handler raised — surface to server
                log.exception("on_server_request raised for %s", method)
                err_frame = {
                    "jsonrpc": "2.0",
                    "id": fid,
                    "error": {"code": INTERNAL_ERROR, "message": f"client handler raised: {exc}"},
                }
                try:
                    await self._send_frame(err_frame)
                except Exception:
                    log.exception("failed to send error reply for %s", method)
                return
            resp_frame = {"jsonrpc": "2.0", "id": fid, "result": result}
            try:
                await self._send_frame(resp_frame)
            except Exception:
                log.exception("failed to send reply for %s", method)
            return

        if "id" in frame:
            # Response to a previous request
            fid = frame["id"]
            fut = self._pending.pop(fid, None)
            if fut is None or fut.done():
                log.warning("response for unknown id=%s ignored", fid)
                return
            if "error" in frame:
                err = frame["error"]
                if not isinstance(err, dict):
                    err = {"code": INTERNAL_ERROR, "message": str(err)}
                fut.set_exception(JSONRPCError(
                    code=err.get("code", INTERNAL_ERROR),
                    message=err.get("message", "<no message>"),
                    data=err.get("data"),
                ))
            else:
                fut.set_result(frame.get("result"))
            return

        if "method" in frame:
            # Notification
            try:
                await self._on_notification(frame)
            except Exception:
                log.exception("on_notification handler raised")
            return

        log.warning("frame has neither id nor method: %r", frame)

    async def _send_frame(self, frame: dict) -> None:
        """Serialize ``frame`` to NDJSON and write it to the writer.

        Uses an asyncio lock so concurrent ``request()`` calls don't interleave
        bytes (they would still be valid JSON lines, but ordering matters for
        request correlation in the test harness).
        """
        assert self._writer is not None  # for type checkers
        data = (json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8")
        if len(data) > MAX_FRAME_BYTES:
            raise ConnError(
                f"outbound frame {len(data)} bytes exceeds {MAX_FRAME_BYTES}"
            )
        async with self._write_lock:
            self._writer.write(data)
            await self._writer.drain()
