"""Tests for adapter.log_capture.

Strategy:
- Unit tests use ``tmp_path`` for both NDJSON log dir and PID file dir
  so they never touch the real ``~/.picocclaw/logs/...`` or
  ``~/.picoclaw/state/...`` trees.
- LogCapture is a callback factory: it returns an ``on_notification``
  handler that the Supervisor hands to Conn. Tests feed the handler
  raw notification dicts (as Conn would dispatch them) and assert the
  NDJSON file + PID file + pretty stdout (where applicable).
- A small `FakeSupervisor` stands in for a real Supervisor: LogCapture
  only needs `pid`, `current_sid`, and `state`, all read at write
  time, so a thin duck-typed shim is enough.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

import pytest

from adapter.log_capture import (
    LogCapture,
    LogCaptureError,
    SESSION_LOG_DIR,
    PID_DIR,
    MAX_FILE_BYTES,
    PRETTY_KINDS,
    ROTATED_SUFFIX,
)

# ---------------- fixtures ----------------

pytestmark = pytest.mark.asyncio


class FakeSupervisor:
    """Duck-typed shim for Supervisor — LogCapture only reads these attrs."""

    def __init__(self, pid: int = 12345, sid: Optional[str] = "test-sid-1"):
        self.pid = pid
        self._sid = sid
        self.state = "prompting"

    @property
    def current_sid(self) -> Optional[str]:
        return self._sid


def _make_notification(
    method: str = "session/update",
    session_id: str = "test-sid-1",
    session_update: str = "agent_message_chunk",
    content=None,
    stop_reason: Optional[str] = None,
) -> dict:
    params = {
        "sessionId": session_id,
        "sessionUpdate": session_update,
    }
    if session_update == "stop" and stop_reason is not None:
        params["stopReason"] = stop_reason
    if content is not None:
        params["content"] = content
    return {"jsonrpc": "2.0", "method": method, "params": params}


# ---------------- Test 1: handler factory ----------------

async def test_handler_returns_async_callable(tmp_path):
    sup = FakeSupervisor()
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    assert asyncio.iscoroutinefunction(handler)
    await handler(_make_notification(content={"type": "text", "text": "hi"}))


# ---------------- Test 2: NDJSON file written per session ----------------

async def test_ndjson_written_per_session(tmp_path):
    sup = FakeSupervisor(sid="s-A")
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    await handler(_make_notification(session_id="s-A",
                                     content={"type": "text", "text": "hello"}))
    log_file = tmp_path / "logs" / "s-A.jsonl"
    assert log_file.exists()
    line = log_file.read_text().strip()
    rec = json.loads(line)
    assert rec["sessionId"] == "s-A"
    assert rec["sessionUpdate"] == "agent_message_chunk"
    assert rec["content"]["text"] == "hello"
    assert "ts" in rec
    assert "kind" in rec  # convenience tag for downstream consumers


# ---------------- Test 3: stop notification captured with stopReason ----------------

async def test_stop_notification_captures_stop_reason(tmp_path):
    sup = FakeSupervisor()
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    await handler(_make_notification(
        session_update="stop",
        stop_reason="end_turn",
    ))
    log_file = tmp_path / "logs" / "test-sid-1.jsonl"
    rec = json.loads(log_file.read_text().strip())
    assert rec["sessionUpdate"] == "stop"
    assert rec["stopReason"] == "end_turn"
    assert rec["kind"] == "stop"


# ---------------- Test 4: file per session (one per sid) ----------------

async def test_separate_files_per_session(tmp_path):
    sup = FakeSupervisor()
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    await handler(_make_notification(session_id="s-A", content={"type": "text", "text": "a1"}))
    await handler(_make_notification(session_id="s-B", content={"type": "text", "text": "b1"}))
    await handler(_make_notification(session_id="s-A", content={"type": "text", "text": "a2"}))
    a = (tmp_path / "logs" / "s-A.jsonl").read_text().strip().splitlines()
    b = (tmp_path / "logs" / "s-B.jsonl").read_text().strip().splitlines()
    assert len(a) == 2 and len(b) == 1
    assert json.loads(a[0])["content"]["text"] == "a1"
    assert json.loads(a[1])["content"]["text"] == "a2"
    assert json.loads(b[0])["content"]["text"] == "b1"


# ---------------- Test 5: NDJSON one record per line (parseable line-by-line) ----------------

async def test_ndjson_is_line_delimited_and_parseable(tmp_path):
    sup = FakeSupervisor()
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    for i in range(5):
        await handler(_make_notification(content={"type": "text", "text": f"msg-{i}"}))
    raw = (tmp_path / "logs" / "test-sid-1.jsonl").read_text()
    lines = [ln for ln in raw.split("\n") if ln]
    assert len(lines) == 5
    for i, ln in enumerate(lines):
        rec = json.loads(ln)
        assert rec["content"]["text"] == f"msg-{i}"


# ---------------- Test 6: PID file written on first event ----------------

async def test_pid_file_written_on_first_event(tmp_path):
    sup = FakeSupervisor(pid=4242)
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    assert not (tmp_path / "pids" / "test-sid-1.pid").exists()
    await handler(_make_notification())
    pid_file = tmp_path / "pids" / "test-sid-1.pid"
    assert pid_file.exists()
    assert pid_file.read_text().strip() == "4242"


# ---------------- Test 7: PID file NOT rewritten on subsequent events ----------------

async def test_pid_file_idempotent(tmp_path):
    sup = FakeSupervisor(pid=99)
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    await handler(_make_notification())
    pid_file = tmp_path / "pids" / "test-sid-1.pid"
    first_mtime = pid_file.stat().st_mtime
    # second event should NOT rewrite (cheap O(1) check)
    time.sleep(0.02)
    await handler(_make_notification())
    assert pid_file.stat().st_mtime == first_mtime


# ---------------- Test 8: skip when no session id ----------------

async def test_skip_when_no_session_id(tmp_path):
    sup = FakeSupervisor(sid=None)
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    # Notification with no sessionId in params — should be a no-op
    notif = {"jsonrpc": "2.0", "method": "session/update",
             "params": {"sessionUpdate": "agent_message_chunk"}}
    await handler(notif)
    # Nothing should be written anywhere
    assert not (tmp_path / "logs").exists() or not list((tmp_path / "logs").iterdir())


# ---------------- Test 9: skip non-session/update notifications ----------------

async def test_ignores_non_session_update_notifications(tmp_path):
    sup = FakeSupervisor()
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    # Some other method (e.g. cancel ack notification)
    await handler({"jsonrpc": "2.0", "method": "exit", "params": {}})
    assert not (tmp_path / "logs" / "test-sid-1.jsonl").exists()


# ---------------- Test 10: 50MB rotation ----------------

async def test_rotation_at_50mb(tmp_path, monkeypatch):
    # Monkeypatch the cap constant so we don't have to write 50MB
    monkeypatch.setattr("adapter.log_capture.MAX_FILE_BYTES", 1024)  # 1 KB
    sup = FakeSupervisor(sid="s-rotate")
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    # Write enough events to exceed 1 KB
    big_text = "x" * 200
    for i in range(10):
        await handler(_make_notification(session_id="s-rotate",
                                         content={"type": "text", "text": big_text}))
    # The active file should exist AND a rotated sibling should exist
    files = sorted((tmp_path / "logs").iterdir())
    names = [f.name for f in files]
    assert "s-rotate.jsonl" in names
    # Rotated names are like ``s-rotate.jsonl.20260611T060027Z.rotated``
    assert any(
        n.startswith("s-rotate.jsonl.") and n.endswith(ROTATED_SUFFIX)
        for n in names
    ), f"expected rotated file, got {names}"


# ---------------- Test 11: pretty stdout disabled by default ----------------

async def test_pretty_stdout_disabled_by_default(tmp_path, capsys):
    sup = FakeSupervisor()
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    await handler(_make_notification(content={"type": "text", "text": "should not print"}))
    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------- Test 12: pretty stdout enabled ----------------

async def test_pretty_stdout_enabled_for_supported_kinds(tmp_path, capsys):
    sup = FakeSupervisor()
    cap = LogCapture(
        sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids",
        pretty_stdout=True,
    )
    handler = cap.handler()
    await handler(_make_notification(
        session_update="agent_message_chunk",
        content={"type": "text", "text": "world"},
    ))
    captured = capsys.readouterr()
    assert "world" in captured.out


# ---------------- Test 13: pretty stdout skips non-message kinds ----------------

async def test_pretty_stdout_skips_thought_chunks(tmp_path, capsys):
    sup = FakeSupervisor()
    cap = LogCapture(
        sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids",
        pretty_stdout=True,
    )
    handler = cap.handler()
    await handler(_make_notification(
        session_update="agent_thought_chunk",
        content={"type": "text", "text": "internal thought"},
    ))
    captured = capsys.readouterr()
    assert "internal thought" not in captured.out


# ---------------- Test 14: tag 'kind' is set for grep convenience ----------------

async def test_kind_tag_set_correctly(tmp_path):
    sup = FakeSupervisor()
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    await handler(_make_notification(session_update="agent_message_chunk"))
    await handler(_make_notification(session_update="agent_thought_chunk"))
    await handler(_make_notification(session_update="tool_call",
                                     content={"toolName": "read_file"}))
    await handler(_make_notification(session_update="tool_call_update",
                                     content={"status": "ok"}))
    await handler(_make_notification(session_update="stop", stop_reason="end_turn"))
    await handler(_make_notification(session_update="ask",
                                     content={"question": "Proceed?"}))
    await handler(_make_notification(session_update="permission",
                                     content={"tool": "write_file"}))
    raw = (tmp_path / "logs" / "test-sid-1.jsonl").read_text().strip().splitlines()
    kinds = [json.loads(ln)["kind"] for ln in raw]
    assert kinds == [
        "agent_message_chunk",
        "agent_thought_chunk",
        "tool_call",
        "tool_call_update",
        "stop",
        "ask",
        "permission",
    ]


# ---------------- Test 15: explicit close() no-op (file handles closed by append) ----------------

async def test_close_is_idempotent_noop(tmp_path):
    sup = FakeSupervisor()
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")
    handler = cap.handler()
    await handler(_make_notification())
    # close() should not raise even if called multiple times
    await cap.close()
    await cap.close()


# ---------------- Test 16: log_dir / pid_dir created lazily ----------------

async def test_dirs_created_lazily(tmp_path):
    sup = FakeSupervisor()
    nested = tmp_path / "deep" / "nested" / "logs"
    cap = LogCapture(sup, log_dir=nested, pid_dir=tmp_path / "pids")
    handler = cap.handler()
    assert not nested.exists()
    await handler(_make_notification())
    assert nested.is_dir()
    assert (tmp_path / "pids").is_dir()


# ---------------- Test 17: defensive — on write error, raise LogCaptureError ----------------

async def test_write_error_raises_log_capture_error(tmp_path, monkeypatch):
    sup = FakeSupervisor()
    cap = LogCapture(sup, log_dir=tmp_path / "logs", pid_dir=tmp_path / "pids")

    # Replace _append to raise
    def boom(*_a, **_kw):
        raise OSError("disk full")
    monkeypatch.setattr(cap, "_append", boom)
    handler = cap.handler()
    with pytest.raises(LogCaptureError, match="write failed"):
        await handler(_make_notification())
