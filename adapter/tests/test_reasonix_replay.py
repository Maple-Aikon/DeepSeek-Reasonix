"""R1 tests: ``reasonix_replay`` MCP tool (lives in reasonix_status.py).

The :func:`adapter.tools.reasonix_status.reasonix_replay` wrapper
mirrors the R9 + R10 pattern (module-level dispatcher slot guarded
by a lock). These tests pin the contract:

  - Returns ``[error_dict]`` (single-item list) when no dispatcher is
    registered, NOT a bare list (LLM caller must be able to read the
    error).
  - Forwards ``sid`` + ``since_seq`` to the dispatcher's ``replay()``
    method verbatim.
  - When the dispatcher returns ``[event, event, ...]``, the tool
    wrapper returns the same list.
  - ``since_seq=0`` (default) is explicit; tests confirm the wrapper
    doesn't accidentally pass ``None`` or omit the kwarg.

These tests complement ``test_reasonix_status.py::TestReplay`` (which
covers the dispatcher's own ``replay()`` method); this file covers
the *tool wrapper* layer above it.

All tests use mock dispatchers; no real binary required.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from adapter.tools import reasonix_status as status_tool


@pytest.fixture(autouse=True)
def _isolate_dispatcher():
    """Save/restore the module-level dispatcher slot around each test.

    Defensive: tests run in any order; we never want one test's
    registered dispatcher to leak into another.
    """
    saved = status_tool._dispatcher
    status_tool.clear_dispatcher()
    yield
    status_tool.clear_dispatcher()
    if saved is not None:
        status_tool.set_dispatcher(saved)


def test_no_dispatcher_returns_error_list():
    """Without a registered dispatcher, return a single-item error list.

    The wrapper returns ``[{...error dict...}]`` (NOT ``[]``) so the
    LLM caller can distinguish "no events yet" from "dispatcher
    not wired" — both must produce different responses.
    """
    result = status_tool.reasonix_replay("any-sid")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["sid"] == "any-sid"
    assert result[0]["error"] == "no_dispatcher"
    assert "message" in result[0]


def test_dispatcher_called_with_sid_only_when_since_seq_default():
    """Default ``since_seq=0`` is forwarded to dispatcher.replay()."""
    fake = MagicMock()
    fake.replay = MagicMock(return_value=[{"seq": 0, "kind": "user"}])
    status_tool.set_dispatcher(fake)

    result = status_tool.reasonix_replay("sid-abc")

    # Verify the call shape
    fake.replay.assert_called_once_with("sid-abc", since_seq=0)
    assert result == [{"seq": 0, "kind": "user"}]


def test_dispatcher_called_with_explicit_since_seq():
    """Explicit ``since_seq=5`` is forwarded verbatim."""
    fake = MagicMock()
    fake.replay = MagicMock(return_value=[{"seq": 5, "kind": "agent_message_chunk"}])
    status_tool.set_dispatcher(fake)

    result = status_tool.reasonix_replay("sid-xyz", since_seq=5)

    fake.replay.assert_called_once_with("sid-xyz", since_seq=5)
    assert result == [{"seq": 5, "kind": "agent_message_chunk"}]


def test_dispatcher_exception_surfaces_as_error_list():
    """If dispatcher.replay() raises, the wrapper catches + returns
    ``[error_dict]`` (not the raised exception). The LLM caller must
    be able to read the error without try/except.
    """
    fake = MagicMock()
    fake.replay = MagicMock(side_effect=RuntimeError("log file missing"))
    status_tool.set_dispatcher(fake)

    result = status_tool.reasonix_replay("sid-err")

    assert isinstance(result, list)
    assert len(result) == 1
    err = result[0]
    assert err["sid"] == "sid-err"
    assert err["error"] == "dispatcher_failure"
    assert "RuntimeError" in err["message"]
    assert "log file missing" in err["message"]


def test_empty_replay_passes_through():
    """Dispatcher returning ``[]`` means "no events" — wrapper
    forwards as-is (no synthetic error dict injected)."""
    fake = MagicMock()
    fake.replay = MagicMock(return_value=[])
    status_tool.set_dispatcher(fake)

    result = status_tool.reasonix_replay("sid-empty")

    assert result == []
    fake.replay.assert_called_once_with("sid-empty", since_seq=0)
