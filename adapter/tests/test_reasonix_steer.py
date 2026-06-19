"""R2 tests: ``reasonix_steer`` MCP tool.

The :func:`adapter.tools.reasonix_steer.reasonix_steer` wrapper
mirrors the R9 + R10 + R11 pattern (module-level dispatcher slot
guarded by a lock). These tests pin the contract:

  - Returns the structured error dict ``{sid, queued=False, ...,
    error: "invalid_text", message: ...}`` when ``text`` is empty
    or whitespace-only. Validation happens BEFORE the dispatcher
    read, so we don't need a registered dispatcher for this case.
  - Returns ``{sid, queued=False, ..., error: "no_dispatcher", ...}``
    when no dispatcher is registered. Same as R9/R10/R11.
  - When the dispatcher IS registered, the wrapper:
      1. Calls ``dispatcher.steer(sid, text)`` (async), gets back
         ``{queued, queue_len}``.
      2. Calls ``dispatcher.status(sid)`` (async) to snapshot the
         session state (best-effort; on failure, returns ``None``).
      3. Returns ``{sid, queued, queue_len, status_after}``.
  - If the dispatcher's ``steer()`` raises, the wrapper catches
    and returns ``{sid, queued=False, error: "dispatcher_failure", ...}``.
  - If the dispatcher's ``status()`` raises, the wrapper still
    returns the steer result (with ``status_after=None``) — don't
    lose the queued=True confirmation just because status failed.

All tests use ``AsyncMock``-based fake dispatchers; no real binary
required. Runs in <100ms.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from adapter.tools import reasonix_steer as steer_tool


@pytest.fixture(autouse=True)
def _isolate_dispatcher():
    """Save/restore the module-level dispatcher slot around each test.

    Defensive: tests run in any order; we never want one test's
    registered dispatcher to leak into another.
    """
    saved = steer_tool._dispatcher
    steer_tool.clear_dispatcher()
    yield
    steer_tool.clear_dispatcher()
    if saved is not None:
        steer_tool.set_dispatcher(saved)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_text_returns_invalid_text_error():
    """Empty string text → structured error, NO dispatcher call.

    The wrapper validates input BEFORE touching the dispatcher slot,
    so we don't need a registered dispatcher for this case.
    """
    result = steer_tool.reasonix_steer("sid-1", "")
    assert result["sid"] == "sid-1"
    assert result["queued"] is False
    assert result["queue_len"] == 0
    assert result["status_after"] is None
    assert result["error"] == "invalid_text"
    assert "non-empty" in result["message"]


def test_whitespace_only_text_returns_invalid_text_error():
    """``"   "`` (whitespace) is rejected the same as empty string."""
    result = steer_tool.reasonix_steer("sid-2", "   \n\t  ")
    assert result["queued"] is False
    assert result["error"] == "invalid_text"


def test_non_string_text_returns_invalid_text_error():
    """``text=None`` or ``text=42`` is also rejected as invalid_text
    (defensive — LLMs can accidentally pass non-strings)."""
    for bad in (None, 42, ["x"], {"y": 1}):
        result = steer_tool.reasonix_steer("sid-3", bad)
        assert result["error"] == "invalid_text", f"failed for {bad!r}"
        assert result["queued"] is False


# ---------------------------------------------------------------------------
# No dispatcher
# ---------------------------------------------------------------------------


def test_no_dispatcher_returns_error_dict():
    """Without a registered dispatcher, return error dict (not raise)."""
    result = steer_tool.reasonix_steer("sid-x", "any text")
    assert result["sid"] == "sid-x"
    assert result["queued"] is False
    assert result["queue_len"] == 0
    assert result["status_after"] is None
    assert result["error"] == "no_dispatcher"
    assert "MCP server boot" in result["message"] or "shutdown" in result["message"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_steer_happy_path_returns_steer_result_and_status():
    """Registered dispatcher → wrapper calls steer() then status(),
    returns combined result with queued=True and a real status dict.
    """
    fake = MagicMock()
    fake.steer = AsyncMock(return_value={"queued": True, "queue_len": 2})
    fake.status = AsyncMock(return_value={
        "sid": "sid-ok",
        "status": "running",
        "queue_len": 2,
        "cost": {"prompt": 10, "completion": 5, "total": 15, "currency": "USD", "rate": "flat"},
    })
    steer_tool.set_dispatcher(fake)

    result = steer_tool.reasonix_steer("sid-ok", "please also check X")

    # Steer was called with the right args
    fake.steer.assert_awaited_once_with("sid-ok", "please also check X")
    # Status was called for the snapshot
    fake.status.assert_awaited_once_with("sid-ok")

    assert result["sid"] == "sid-ok"
    assert result["queued"] is True
    assert result["queue_len"] == 2
    assert result["status_after"] is not None
    assert result["status_after"]["status"] == "running"
    assert result["status_after"]["queue_len"] == 2
    assert "error" not in result


def test_steer_with_queued_false_still_snapshots_status():
    """If dispatcher's steer() returns queued=False (e.g. unknown sid),
    the wrapper still snapshots status (so the LLM can see the actual
    session state) and propagates queued=False.
    """
    fake = MagicMock()
    fake.steer = AsyncMock(return_value={"queued": False, "queue_len": 0})
    fake.status = AsyncMock(return_value={
        "sid": "sid-unknown",
        "status": "unknown",
    })
    steer_tool.set_dispatcher(fake)

    result = steer_tool.reasonix_steer("sid-unknown", "some text")

    assert result["queued"] is False
    assert result["queue_len"] == 0
    assert result["status_after"]["status"] == "unknown"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_dispatcher_steer_failure_returns_error_dict():
    """If dispatcher.steer() raises, the wrapper catches + returns
    structured error dict (status_after is None, status() not called).
    """
    fake = MagicMock()
    fake.steer = AsyncMock(side_effect=RuntimeError("connection lost"))
    fake.status = AsyncMock(return_value={"status": "running"})
    steer_tool.set_dispatcher(fake)

    result = steer_tool.reasonix_steer("sid-fail", "guidance")

    assert result["queued"] is False
    assert result["queue_len"] == 0
    assert result["status_after"] is None
    assert result["error"] == "dispatcher_failure"
    assert "RuntimeError" in result["message"]
    assert "connection lost" in result["message"]


def test_dispatcher_status_failure_does_not_lose_steer_result():
    """If dispatcher.steer() succeeded but status() failed, the
    wrapper still returns the steer result (queued=True preserved)
    with status_after=None. We don't punish success with a total
    error dict.
    """
    fake = MagicMock()
    fake.steer = AsyncMock(return_value={"queued": True, "queue_len": 1})
    fake.status = AsyncMock(side_effect=RuntimeError("status timeout"))
    steer_tool.set_dispatcher(fake)

    result = steer_tool.reasonix_steer("sid-partial", "more guidance")

    assert result["queued"] is True
    assert result["queue_len"] == 1
    assert result["status_after"] is None
    # No error field — partial success is not an error
    assert "error" not in result


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_set_dispatcher_replaces_existing():
    """Second set_dispatcher() call replaces the first (no error,
    useful for hot-reload)."""
    fake1 = MagicMock()
    fake2 = MagicMock()
    steer_tool.set_dispatcher(fake1)
    steer_tool.set_dispatcher(fake2)

    # The second call wins
    assert steer_tool._dispatcher is fake2
