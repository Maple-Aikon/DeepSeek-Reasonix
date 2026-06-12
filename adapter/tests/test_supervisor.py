"""Tests for adapter.supervisor.

Strategy:
- **Unit tests** use the fake_reasonix.py fixture (a python script that
  reads NDJSON and emits scripted responses). They are fast (<1s each)
  and cover: state transitions, error paths, lifecycle, stderr capture.
- **Integration test** spawns the real ``bin/reasonix acp`` binary and
  is marked with ``@pytest.mark.integration`` so it can be skipped with
  ``-m "not integration"`` during normal runs. It verifies the wire
  shape matches what the actual binary emits.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

# Mark every test in this module as asyncio (pytest-asyncio strict mode)
pytestmark = pytest.mark.asyncio

from adapter.supervisor import Supervisor, SupervisorError

# Path to the fake reasonix script (a regular python file we invoke as
# if it were a binary).
FAKE_BIN = Path(__file__).parent / "fixtures" / "fake_reasonix.py"

# Path to the real reasonix binary (only present if the repo is built).
REAL_BIN = Path(__file__).resolve().parent.parent.parent / "bin" / "reasonix"


@pytest.fixture
def fake_bin():
    """Skip the suite if fake_reasonix.py is missing."""
    if not FAKE_BIN.exists():
        pytest.skip(f"fake_reasonix fixture not found at {FAKE_BIN}")
    return FAKE_BIN


@pytest.fixture
def real_bin():
    """Skip integration tests if the real binary is missing."""
    if not REAL_BIN.exists():
        pytest.skip(f"real reasonix binary not found at {REAL_BIN}")
    return REAL_BIN


def _env_for_fake(mode="happy", **extra):
    """Build an env dict that points fake_reasonix into a chosen mode."""
    env = {
        "FAKE_REASONIX_MODE": mode,
        "FAKE_REASONIX_VERBOSE": "0",
        "PATH": os.environ.get("PATH", ""),
    }
    for k, v in extra.items():
        env[k] = v
    return env


def _real_env():
    """Return a minimal env for the real binary. We deliberately drop
    everything Reasonix might interpret, leaving only what it needs to
    locate its config (HOME, XDG_CONFIG_HOME) and PATH for subtools."""
    keep = ("PATH", "HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "LANG", "LC_ALL", "USER")
    return {k: v for k, v in os.environ.items() if k in keep}


async def _cleanup(sup):
    """Idempotent close helper for tests."""
    try:
        await sup.close()
    except Exception:  # pragma: no cover
        pass


# ---------------- Test 1: start() happy path ----------------

async def test_start_happy_returns_init_result(fake_bin, tmp_path):
    sup = Supervisor(binary=fake_bin, env=_env_for_fake("happy"),
                     stderr_log=tmp_path / "err.log")
    try:
        result = await sup.start()
        assert sup.state == "ready"
        assert sup.pid is not None and sup.pid > 0
        assert result["agentInfo"]["name"] == "fake-reasonix"
        assert result["agentInfo"]["version"] == "test-fixture-0.0.1"
        assert "sessionCapabilities" in result["agentCapabilities"]
    finally:
        await _cleanup(sup)


# ---------------- Test 2: start() raises on missing binary ----------------

async def test_start_raises_when_binary_missing(tmp_path):
    sup = Supervisor(binary=tmp_path / "does-not-exist", env={})
    with pytest.raises(SupervisorError, match="binary not found"):
        await sup.start()
    assert sup.state == "new"  # never transitioned


# ---------------- Test 3: start() raises on crashing binary ----------------

async def test_start_raises_on_crash(fake_bin):
    sup = Supervisor(binary=fake_bin, env=_env_for_fake("crash"),
                     init_timeout=2.0)
    with pytest.raises(SupervisorError, match="initialize failed"):
        await sup.start()
    assert sup.state == "dead"


# ---------------- Test 4: state transitions ----------------

async def test_state_transitions_new_to_ready_to_closed(fake_bin, tmp_path):
    sup = Supervisor(binary=fake_bin, env=_env_for_fake("happy"),
                     stderr_log=tmp_path / "err.log")
    assert sup.state == "new"
    await sup.start()
    assert sup.state == "ready"
    await sup.close()
    assert sup.state == "closed"
    # Calling close again is a no-op
    await sup.close()
    assert sup.state == "closed"


# ---------------- Test 5: prompt() blocks until stop ----------------

async def test_prompt_blocks_until_stop_notification(fake_bin, tmp_path):
    notifications = []

    async def on_notif(frame):
        notifications.append(frame)

    sup = Supervisor(
        binary=fake_bin, env=_env_for_fake("happy"),
        on_notification=on_notif,
        stderr_log=tmp_path / "err.log",
        prompt_timeout=5.0,
    )
    try:
        await sup.start()
        sess = await sup.new_session(cwd="/tmp")
        sid = sess["sessionId"]
        stop = await sup.prompt(sid, [{"type": "text", "text": "hi"}])
        # _await_stop returns the full ``params`` dict; the tagged union
        # lives under params.update per internal/acp/protocol.go.
        assert stop["update"]["sessionUpdate"] == "stop"
        assert stop["update"]["stopReason"] == "end_turn"
        assert sup.state == "ready"
        # We should have seen the agent_message_chunk in the stream.
        # Wire format: params.update.sessionUpdate (nested wrapper).
        msg_chunks = [
            f for f in notifications
            if (f.get("params") or {}).get("update", {}).get("sessionUpdate")
               == "agent_message_chunk"
        ]
        assert len(msg_chunks) == 1
        update = msg_chunks[0]["params"]["update"]
        assert "hello from fake-reasonix" in update["content"]["text"]
    finally:
        await _cleanup(sup)


# ---------------- Test 6: cancel() mid-prompt (no-op for fake, but must not crash) ----------------

async def test_cancel_mid_prompt_does_not_crash(fake_bin, tmp_path):
    """With a slow session, cancel() should return promptly without crashing."""
    sup = Supervisor(
        binary=fake_bin,
        env=_env_for_fake("happy", FAKE_REASONIX_PROMPT_DELAY="2.0"),
        stderr_log=tmp_path / "err.log",
        prompt_timeout=10.0,
    )
    try:
        await sup.start()
        sess = await sup.new_session(cwd="/tmp")
        sid = sess["sessionId"]

        async def run_prompt():
            return await sup.prompt(sid, [{"type": "text", "text": "x"}])

        prompt_task = asyncio.create_task(run_prompt())
        # Let the prompt start, then cancel
        await asyncio.sleep(0.3)
        assert sup.state == "prompting"
        await sup.cancel(sid)  # must not raise
        # Cancel is best-effort: the fake ignores it, so the prompt will
        # still complete after FAKE_REASONIX_SESSION_DELAY. Wait for it.
        # Then close.
        try:
            await asyncio.wait_for(prompt_task, timeout=8.0)
        except asyncio.TimeoutError:
            pass  # Fake may stall; close() will tear down.
    finally:
        await _cleanup(sup)


# ---------------- Test 7: close() is idempotent ----------------

async def test_close_idempotent(fake_bin, tmp_path):
    sup = Supervisor(binary=fake_bin, env=_env_for_fake("happy"),
                     stderr_log=tmp_path / "err.log")
    await sup.start()
    await sup.close()
    await sup.close()  # second call should be a no-op, no exception
    await sup.close()  # third for good measure
    assert sup.state == "closed"


# ---------------- Test 8: stderr capture ----------------

async def test_stderr_lines_written_to_log_file(fake_bin, tmp_path, caplog):
    log_path = tmp_path / "err.log"
    # Force the fake to emit a stderr line by enabling verbose
    env = _env_for_fake("happy", FAKE_REASONIX_VERBOSE="1")
    sup = Supervisor(binary=fake_bin, env=env, stderr_log=log_path)
    try:
        with caplog.at_level("INFO", logger="adapter.supervisor"):
            await sup.start()
            sess = await sup.new_session(cwd="/tmp")
            await sup.prompt(sess["sessionId"], [{"type": "text", "text": "x"}])
        # The fake writes "recv session/prompt (id=N)" to stderr
        assert log_path.exists()
        contents = log_path.read_text()
        assert "session/prompt" in contents
        # Python logger also picked it up
        stderr_logs = [r for r in caplog.records if "reasonix-stderr" in r.getMessage()]
        assert len(stderr_logs) >= 1
    finally:
        await _cleanup(sup)


# ---------------- Test 9: process crash mid-session flips to dead ----------------

async def test_process_crash_flips_to_dead(fake_bin, tmp_path):
    sup = Supervisor(binary=fake_bin, env=_env_for_fake("happy"),
                       stderr_log=tmp_path / "err2.log")
    try:
        await sup.start()
        assert sup.state == "ready"
        # Kill the subprocess directly (simulate crash)
        assert sup._proc is not None
        sup._proc.kill()
        await sup._proc.wait()
        # Give the watch task a moment to flip state
        for _ in range(50):
            if sup.state == "dead":
                break
            await asyncio.sleep(0.05)
        assert sup.state == "dead"
        assert sup.exit_code is not None
    finally:
        await _cleanup(sup)


# ---------------- Test 10: REASONIX_BIN env var overrides default ----------------

async def test_env_var_overrides_default_binary(fake_bin, monkeypatch):
    monkeypatch.setenv("REASONIX_BIN", str(fake_bin))
    sup = Supervisor(env=_env_for_fake("happy"))
    try:
        result = await sup.start()
        assert result["agentInfo"]["name"] == "fake-reasonix"
    finally:
        await _cleanup(sup)


# ---------------- Test 11: prompt() with no content (empty list) ----------------

async def test_prompt_empty_content_still_works(fake_bin, tmp_path):
    sup = Supervisor(binary=fake_bin, env=_env_for_fake("happy"),
                     stderr_log=tmp_path / "err.log")
    try:
        await sup.start()
        sess = await sup.new_session(cwd="/tmp")
        # Fake ignores content shape, just emits agent_message + stop.
        # Nested ``update.sessionUpdate`` per internal/acp/protocol.go.
        stop = await sup.prompt(sess["sessionId"], [])
        assert stop["update"]["sessionUpdate"] == "stop"
    finally:
        await _cleanup(sup)


# ---------------- Test 12: new_session() returns sessionId ----------------

async def test_new_session_returns_session_id(fake_bin, tmp_path):
    sup = Supervisor(binary=fake_bin, env=_env_for_fake("happy"),
                     stderr_log=tmp_path / "err.log")
    try:
        await sup.start()
        sess = await sup.new_session(cwd="/home/maple")
        assert "sessionId" in sess
        assert sess["sessionId"].startswith("fake-sid-")
    finally:
        await _cleanup(sup)


# ---------------- Integration test: real binary (skipped unless available) ----------------

@pytest.mark.integration
async def test_integration_real_binary_initialize(real_bin, tmp_path):
    """Spawn the real reasonix v1.4.0 binary and verify initialize works.

    This is the smoke test that proves our wire format assumptions are
    correct against the real server. Marked as integration so it can be
    skipped with ``-m "not integration"`` when running unit tests in CI.
    """
    sup = Supervisor(
        binary=real_bin,
        cwd=tmp_path,
        stderr_log=tmp_path / "real-stderr.log",
        init_timeout=15.0,
    )
    try:
        result = await sup.start()
        assert sup.state == "ready"
        assert result["agentInfo"]["name"] == "reasonix"
        # The version string includes the tag — accept any semver-like value
        version = result["agentInfo"]["version"]
        assert version and ("v" in version or "desktop" in version), f"unexpected version: {version}"
    finally:
        await _cleanup(sup)


@pytest.mark.integration
async def test_integration_real_binary_new_session(real_bin, tmp_path):
    """Verify session/new works against the real binary."""
    sup = Supervisor(
        binary=real_bin,
        cwd=tmp_path,
        stderr_log=tmp_path / "real-stderr.log",
        init_timeout=15.0,
        prompt_timeout=60.0,
    )
    try:
        await sup.start()
        sess = await sup.new_session(cwd=str(tmp_path))
        assert "sessionId" in sess
        sid = sess["sessionId"]
        # Try a real prompt. This may fail in sandboxed env (no API key),
        # so we catch SupervisorError and only require that session/new
        # worked — proves the wire is intact.
        try:
            stop = await sup.prompt(sid, [{"type": "text", "text": "say hi"}])
            # R5 fix: Supervisor.prompt() normalizes the response so the
            # caller sees the canonical session/update shape regardless
            # of whether the stop signal arrived as a notification or
            # inline in the session/prompt response (Reasonix v1.4.0
            # returns stopReason INLINE; no separate stop notification).
            assert stop["update"]["sessionUpdate"] == "stop"
        except SupervisorError as e:
            # If prompt fails (e.g. missing API key), the wire is still good
            # — we just cannot test the LLM layer in CI.
            pytest.skip(f"prompt failed in this env (likely missing API key): {e}")
    finally:
        await _cleanup(sup)
