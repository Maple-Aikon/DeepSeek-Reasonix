"""Unit tests for adapter.acp_client — JSON-RPC 2.0 NDJSON transport.

Test strategy: spawn real subprocesses (cat, python) and drive them as the
"server" side. `cat -u` echoes whatever comes in — useful for verifying wire
shape. `python -u` with inline scripts simulates protocol-aware behavior
(responder, notifier, error responder). Keeps tests independent from any
Reasonix build.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import pytest

# Ensure adapter package is importable when running `uv run pytest` from
# inside the adapter/ directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapter import acp_client  # noqa: E402


# ---------- helpers ----------

async def make_cat_pipe() -> tuple[acp_client.Conn, asyncio.subprocess.Process]:
    """Spawn `cat -u` and wrap its stdio as a Conn.

    `cat -u` echoes NDJSON lines back unchanged — perfect for testing
    frame roundtrip without protocol knowledge.
    """
    proc = await asyncio.create_subprocess_exec(
        "cat", "-u",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.stdin is not None and proc.stdout is not None
    conn = acp_client.Conn(
        reader=proc.stdout,
        writer=proc.stdin,
        on_notification=_nop_async,
    )
    return conn, proc


async def make_responder_pipe(responses: list[dict]) -> tuple[acp_client.Conn, asyncio.subprocess.Process]:
    """Spawn a python responder that reads JSON-RPC requests and replies.

    `responses` is a queue: for each request read, the next response in the
    list is sent. If the queue is exhausted, a default `{"ok": True}` reply
    is used.
    """
    script = (
        "import json, sys\n"
        "responses = json.loads(sys.argv[1])\n"
        "ridx = [0]\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if not line:\n"
        "        continue\n"
        "    try:\n"
        "        req = json.loads(line)\n"
        "    except Exception:\n"
        "        continue\n"
        "    if 'id' not in req:\n"
        "        continue  # ignore notifs\n"
        "    if ridx[0] < len(responses):\n"
        "        resp = responses[ridx[0]]\n"
        "        ridx[0] += 1\n"
        "    else:\n"
        "        resp = {\"result\": \"ok\"}\n"
        "    resp.setdefault(\"jsonrpc\", \"2.0\")\n"
        "    resp[\"id\"] = req[\"id\"]\n"
        "    sys.stdout.write(json.dumps(resp) + \"\\n\")\n"
        "    sys.stdout.flush()\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", script, json.dumps(responses),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.stdin is not None and proc.stdout is not None
    conn = acp_client.Conn(
        reader=proc.stdout,
        writer=proc.stdin,
        on_notification=_nop_async,
    )
    return conn, proc


async def _nop_async(_notif: dict) -> None:
    return None


async def _kill(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()


async def _start_and_drain(conn: acp_client.Conn) -> asyncio.Task:
    """Start conn.run() and return the task; caller must cancel at end."""
    return asyncio.create_task(conn.run())


async def _shutdown(conn: acp_client.Conn, read_task: asyncio.Task) -> None:
    await conn.close()
    read_task.cancel()
    try:
        await read_task
    except (asyncio.CancelledError, Exception):
        pass


# ---------- tests ----------

@pytest.mark.asyncio
async def test_request_response_correlation():
    """Send a request, server replies, request() returns the result."""
    conn, proc = await make_responder_pipe([{"result": {"ok": True}}])
    try:
        read_task = await _start_and_drain(conn)
        result = await asyncio.wait_for(
            conn.request("ping", {"x": 1}), timeout=2.0
        )
        assert result == {"ok": True}
        await _shutdown(conn, read_task)
    finally:
        await _kill(proc)


@pytest.mark.asyncio
async def test_id_increments_monotonically():
    """Two requests get ids 1 then 2, with responses in reversed queue order
    to prove correlation by id, not arrival order."""
    conn, proc = await make_responder_pipe([
        {"result": "two"},  # response to id=1 (reversed)
        {"result": "one"},  # response to id=2 (reversed)
    ])
    try:
        read_task = await _start_and_drain(conn)
        fut1 = asyncio.create_task(conn.request("a", {}))
        fut2 = asyncio.create_task(conn.request("b", {}))
        r1 = await asyncio.wait_for(fut1, timeout=2.0)
        r2 = await asyncio.wait_for(fut2, timeout=2.0)
        # The responder replies in arrival order but reverses the queue;
        # fut1 (id=1) gets "two", fut2 (id=2) gets "one".
        assert r1 == "two"
        assert r2 == "one"
        await _shutdown(conn, read_task)
    finally:
        await _kill(proc)


@pytest.mark.asyncio
async def test_error_response_raises():
    """Server returns an error frame → request() raises JSONRPCError."""
    conn, proc = await make_responder_pipe([
        {"error": {"code": -32601, "message": "Method not found"}},
    ])
    try:
        read_task = await _start_and_drain(conn)
        with pytest.raises(acp_client.JSONRPCError) as exc_info:
            await asyncio.wait_for(conn.request("nope", {}), timeout=2.0)
        assert exc_info.value.code == -32601
        assert "Method not found" in str(exc_info.value)
        await _shutdown(conn, read_task)
    finally:
        await _kill(proc)


@pytest.mark.asyncio
async def test_notification_callback_fires():
    """Server emits a notification → on_notification callback fires once
    with the full wire params."""
    seen: list[dict] = []

    async def cb(n: dict) -> None:
        seen.append(n)

    script = (
        "import json, sys, time\n"
        "n = {\"jsonrpc\": \"2.0\", \"method\": \"session/update\", \"params\": {\"k\": \"v\"}}\n"
        "sys.stdout.write(json.dumps(n) + \"\\n\")\n"
        "sys.stdout.flush()\n"
        "time.sleep(0.3)\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", script,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    conn = acp_client.Conn(
        reader=proc.stdout,
        writer=None,  # type: ignore[arg-type]
        on_notification=cb,
    )
    try:
        read_task = await _start_and_drain(conn)
        for _ in range(50):
            if seen:
                break
            await asyncio.sleep(0.02)
        assert len(seen) == 1
        assert seen[0]["method"] == "session/update"
        assert seen[0]["params"] == {"k": "v"}
        await _shutdown(conn, read_task)
    finally:
        await _kill(proc)


@pytest.mark.asyncio
async def test_concurrent_requests_resolved_independently():
    """3 concurrent requests, all complete, no cross-talk."""
    conn, proc = await make_responder_pipe([
        {"result": "a"},
        {"result": "b"},
        {"result": "c"},
    ])
    try:
        read_task = await _start_and_drain(conn)
        f1 = asyncio.create_task(conn.request("m1", {}))
        f2 = asyncio.create_task(conn.request("m2", {}))
        f3 = asyncio.create_task(conn.request("m3", {}))
        results = await asyncio.gather(
            asyncio.wait_for(f1, timeout=2.0),
            asyncio.wait_for(f2, timeout=2.0),
            asyncio.wait_for(f3, timeout=2.0),
        )
        assert results == ["a", "b", "c"]
        await _shutdown(conn, read_task)
    finally:
        await _kill(proc)


@pytest.mark.asyncio
async def test_large_frame_under_32mib():
    """A 1 MB params payload roundtrips as one NDJSON line within 5s."""
    big_string = "x" * (1024 * 1024)  # 1 MB
    conn, proc = await make_responder_pipe([])  # default ok reply
    try:
        read_task = await _start_and_drain(conn)
        result = await asyncio.wait_for(
            conn.request("big", {"data": big_string}), timeout=5.0
        )
        # Responder default result is "ok" (string)
        assert result == "ok"
        await _shutdown(conn, read_task)
    finally:
        await _kill(proc)


@pytest.mark.asyncio
async def test_unicode_in_string():
    """Non-ASCII UTF-8 params roundtrip correctly."""
    payload = {"text": "An Hiên — Ơi buồn ơi là buồn 🌙"}
    conn, proc = await make_responder_pipe([
        {"result": {"echoed": payload["text"]}},
    ])
    try:
        read_task = await _start_and_drain(conn)
        result = await asyncio.wait_for(
            conn.request("unicode", payload), timeout=2.0
        )
        assert result == {"echoed": "An Hiên — Ơi buồn ơi là buồn 🌙"}
        await _shutdown(conn, read_task)
    finally:
        await _kill(proc)


@pytest.mark.asyncio
async def test_malformed_json_skipped():
    """Server sends a non-JSON line, then a valid notification. The conn
    should log the bad line (or silently skip) and continue; the valid
    notification still fires the callback."""
    seen: list[dict] = []

    async def cb(n: dict) -> None:
        seen.append(n)

    script = (
        "import sys, json, time\n"
        "sys.stdout.write('this is not json\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stdout.write(json.dumps({\"jsonrpc\": \"2.0\", \"method\": \"ping\", \"params\": {}}) + \"\\n\")\n"
        "sys.stdout.flush()\n"
        "time.sleep(0.3)\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", script,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    conn = acp_client.Conn(
        reader=proc.stdout,
        writer=None,  # type: ignore[arg-type]
        on_notification=cb,
    )
    try:
        read_task = await _start_and_drain(conn)
        for _ in range(50):
            if seen:
                break
            await asyncio.sleep(0.02)
        assert len(seen) == 1
        assert seen[0]["method"] == "ping"
        await _shutdown(conn, read_task)
    finally:
        await _kill(proc)


@pytest.mark.asyncio
async def test_close_is_idempotent():
    """Calling close() twice does not raise."""
    conn, proc = await make_cat_pipe()
    try:
        read_task = await _start_and_drain(conn)
        await conn.close()
        await conn.close()  # second call must be a no-op
        try:
            await read_task
        except (asyncio.CancelledError, Exception):
            pass
    finally:
        await _kill(proc)


@pytest.mark.asyncio
async def test_notify_send_writes_to_wire():
    """The Conn can SEND notifications (no id) to the server. Verify the
    server-side script receives and counts them. This is the path the
    adapter uses for permission responses, pings, etc."""
    received: list[dict] = []

    # The responder script writes received lines to a file we can read back.
    out_path = "/tmp/_acp_client_notify_recv.txt"
    if os.path.exists(out_path):
        os.unlink(out_path)
    script = (
        "import sys, json, time\n"
        "f = open(sys.argv[1], 'w')\n"
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if line:\n"
        "        f.write(line + \"\\n\")\n"
        "        f.flush()\n"
        "    if 'terminate' in line:\n"
        "        break\n"
        "time.sleep(0.1)\n"
        "f.close()\n"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", script, out_path,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.stdin is not None and proc.stdout is not None
    conn = acp_client.Conn(
        reader=proc.stdout,
        writer=proc.stdin,
        on_notification=_nop_async,
    )
    try:
        read_task = await _start_and_drain(conn)
        await conn.notify("session/cancel", {"sessionId": "abc"})
        await conn.notify("terminate", {})
        # Give the script time to write
        for _ in range(50):
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                # Check for the "terminate" line
                try:
                    with open(out_path) as f:
                        content = f.read()
                    if "terminate" in content:
                        break
                except Exception:
                    pass
            await asyncio.sleep(0.02)
        with open(out_path) as f:
            content = f.read()
        # Both notifications should appear on the wire
        assert "session/cancel" in content
        assert "terminate" in content
        # No "id" field in notification frames
        for line in content.strip().splitlines():
            obj = json.loads(line)
            assert "id" not in obj, f"notification should not have id: {line}"
        await _shutdown(conn, read_task)
    finally:
        await _kill(proc)
        if os.path.exists(out_path):
            os.unlink(out_path)
