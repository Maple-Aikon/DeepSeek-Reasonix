"""R3 tests: ``reasonix_cancel`` MCP tool.

The :func:`adapter.tools.reasonix_cancel.reasonix_cancel` wrapper
mirrors the R10/R11 pattern (module-level dispatcher slot guarded
by a lock). These tests pin the contract:

  - Returns ``{sid, cancelled=False, error: "invalid_sid", ...}``
    when ``sid`` is empty, whitespace, or non-string.
  - Returns ``{sid, cancelled=False, error: "no_dispatcher", ...}``
    when no dispatcher is registered.
  - Happy path: dispatcher.cancel() returns ``None`` (best-effort
    — does not raise even if the session was already idle). The
    wrapper treats any non-exception return as cancelled=True.
  - dispatcher.cancel() raises → error dict, status_after=None.
  - dispatcher.status() raises → cancel result preserved,
    status_after=None (partial-success policy).

All tests use ``AsyncMock``-based fake dispatchers; no real binary
required. Runs in <100ms.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from adapter.tools import reasonix_cancel as cancel_tool


@pytest.fixture(autouse=True)
def _isolate_dispatcher():
    """Save/restore the module-level dispatcher slot around each test."""
    saved = cancel_tool._dispatcher
    cancel_tool.clear_dispatcher()
    yield
    cancel_tool.clear_dispatcher()
    if saved is not None:
        cancel_tool.set_dispatcher(saved)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_sid_returns_invalid_sid_error():
    """Empty string sid → structured error, NO dispatcher call."""
    result = cancel_tool.reasonix_cancel("")
    assert result["sid"] == ""
    assert result["cancelled"] is False
    assert result["status_after"] is None
    assert result["error"] == "invalid_sid"
    assert "non-empty" in result["message"]


def test_whitespace_only_sid_returns_invalid_sid_error():
    """``"   "`` is rejected as invalid_sid (defensive)."""
    result = cancel_tool.reasonix_cancel("   \n\t  ")
    assert result["cancelled"] is False
    assert result["error"] == "invalid_sid"


def test_non_string_sid_returns_invalid_sid_error():
    """``sid=None`` or ``sid=42`` is also rejected as invalid_sid
    (defensive — LLM callers can pass anything)."""
    for bad in (None, 42, ["x"], {"y": 1}):
        result = cancel_tool.reasonix_cancel(bad)
        assert result["error"] == "invalid_sid", f"failed for {bad!r}"
        assert result["cancelled"] is False


# ---------------------------------------------------------------------------
# No dispatcher
# ---------------------------------------------------------------------------


def test_no_dispatcher_returns_error_dict():
    """Without a registered dispatcher, return error dict (not raise)."""
    result = cancel_tool.reasonix_cancel("sid-x")
    assert result["sid"] == "sid-x"
    assert result["cancelled"] is False
    assert result["status_after"] is None
    assert result["error"] == "no_dispatcher"
    assert "MCP server boot" in result["message"] or "shutdown" in result["message"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_cancel_happy_path_returns_cancelled_true_and_status():
    """Registered dispatcher → wrapper calls cancel() (returns None)
    then status(), returns combined result with cancelled=True and
    a real status dict.

    TaskDispatcher.cancel() is best-effort — it returns ``None``
    for both "successfully cancelled" and "session was already
    idle". The tool wrapper treats any non-exception return as
    cancelled=True. LLM caller can use reasonix_status() to verify.
    """
    fake = MagicMock()
    fake.cancel = AsyncMock(return_value=None)
    fake.status = AsyncMock(return_value={
        "sid": "sid-ok",
        "status": "cancelled",
        "cost": {"prompt": 10, "completion": 5, "total": 15, "currency": "USD", "rate": "flat"},
    })
    cancel_tool.set_dispatcher(fake)

    result = cancel_tool.reasonix_cancel("sid-ok")

    fake.cancel.assert_awaited_once_with("sid-ok")
    fake.status.assert_awaited_once_with("sid-ok")

    assert result["sid"] == "sid-ok"
    assert result["cancelled"] is True
    assert result["status_after"] is not None
    assert result["status_after"]["status"] == "cancelled"
    assert "error" not in result


def test_cancel_idle_session_still_returns_cancelled_true():
    """If the session was already idle (cancel() returns None without
    raising), the wrapper still returns cancelled=True. The LLM can
    use reasonix_status() to inspect the actual state — the tool
    layer cannot distinguish "cancelled a running turn" from
    "session was already idle" because the dispatcher contract
    is best-effort.
    """
    fake = MagicMock()
    fake.cancel = AsyncMock(return_value=None)
    fake.status = AsyncMock(return_value={
        "sid": "sid-already-idle",
        "status": "idle",
    })
    cancel_tool.set_dispatcher(fake)

    result = cancel_tool.reasonix_cancel("sid-already-idle")

    assert result["cancelled"] is True
    assert result["status_after"]["status"] == "idle"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_dispatcher_cancel_failure_returns_error_dict():
    """If dispatcher.cancel() raises, the wrapper catches + returns
    structured error dict (status_after is None, status() not called).
    """
    fake = MagicMock()
    fake.cancel = AsyncMock(side_effect=RuntimeError("wire closed"))
    fake.status = AsyncMock(return_value={"status": "running"})
    cancel_tool.set_dispatcher(fake)

    result = cancel_tool.reasonix_cancel("sid-fail")

    assert result["cancelled"] is False
    assert result["status_after"] is None
    assert result["error"] == "dispatcher_failure"
    assert "RuntimeError" in result["message"]
    assert "wire closed" in result["message"]


def test_dispatcher_status_failure_does_not_lose_cancel_result():
    """If dispatcher.cancel() succeeded but status() failed, the
    wrapper still returns the cancel result (cancelled=True preserved)
    with status_after=None. Partial success is not an error.
    """
    fake = MagicMock()
    fake.cancel = AsyncMock(return_value=None)
    fake.status = AsyncMock(side_effect=RuntimeError("status timeout"))
    cancel_tool.set_dispatcher(fake)

    result = cancel_tool.reasonix_cancel("sid-partial")

    assert result["cancelled"] is True
    assert result["status_after"] is None
    assert "error" not in result


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_set_dispatcher_replaces_existing():
    """Second set_dispatcher() call replaces the first (no error,
    useful for hot-reload)."""
    fake1 = MagicMock()
    fake2 = MagicMock()
    cancel_tool.set_dispatcher(fake1)
    cancel_tool.set_dispatcher(fake2)

    assert cancel_tool._dispatcher is fake2
