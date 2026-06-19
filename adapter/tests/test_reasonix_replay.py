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
    """Default ``since_seq=0`` + default ``mode="conversation"`` are
    forwarded to dispatcher.replay() (R13.3: mode is keyword-only)."""
    fake = MagicMock()
    fake.replay = MagicMock(return_value=[{"role": "user", "turn": 0, "text": "hi"}])
    status_tool.set_dispatcher(fake)

    result = status_tool.reasonix_replay("sid-abc")

    # Verify the call shape (R13.3: mode is keyword-only kwarg)
    fake.replay.assert_called_once_with("sid-abc", since_seq=0, mode="conversation")
    assert result == [{"role": "user", "turn": 0, "text": "hi"}]


def test_dispatcher_called_with_explicit_since_seq():
    """Explicit ``since_seq=5`` is forwarded verbatim (default mode)."""
    fake = MagicMock()
    fake.replay = MagicMock(return_value=[{"role": "assistant", "turn": 5, "text": "reply"}])
    status_tool.set_dispatcher(fake)

    result = status_tool.reasonix_replay("sid-xyz", since_seq=5)

    fake.replay.assert_called_once_with("sid-xyz", since_seq=5, mode="conversation")
    assert result == [{"role": "assistant", "turn": 5, "text": "reply"}]


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
    fake.replay.assert_called_once_with("sid-empty", since_seq=0, mode="conversation")


# === R13.3: 3-mode API tests (6 new tests) ===
# These pin the new mode=raw|conversation|summary contract added in
# R13.3. The dispatcher is mocked; we only care that the wrapper
# passes mode through correctly and shapes errors consistently.


def test_mode_raw_forwards_to_dispatcher():
    """Explicit ``mode="raw"`` is forwarded to dispatcher.replay()."""
    fake = MagicMock()
    fake.replay = MagicMock(return_value=[{"kind": "agent_message_chunk", "text": "hi"}])
    status_tool.set_dispatcher(fake)

    result = status_tool.reasonix_replay("sid-r", mode="raw")

    fake.replay.assert_called_once_with("sid-r", since_seq=0, mode="raw")
    assert result == [{"kind": "agent_message_chunk", "text": "hi"}]


def test_mode_summary_forwards_to_dispatcher():
    """``mode="summary"`` is forwarded (R13.x returns ``[]``; R14
    will fill in real summary data)."""
    fake = MagicMock()
    fake.replay = MagicMock(return_value=[])
    status_tool.set_dispatcher(fake)

    result = status_tool.reasonix_replay("sid-s", mode="summary")

    fake.replay.assert_called_once_with("sid-s", since_seq=0, mode="summary")
    assert result == []


def test_invalid_mode_returns_error_list():
    """If dispatcher raises ``ValueError`` (invalid mode), the
    wrapper returns ``[error_dict]`` with ``error="invalid_mode"`` —
    NOT a raised exception. LLM caller reads the error directly.
    """
    fake = MagicMock()
    fake.replay = MagicMock(side_effect=ValueError("mode must be raw|conversation|summary"))
    status_tool.set_dispatcher(fake)

    result = status_tool.reasonix_replay("sid-bad", mode="garbage")

    assert isinstance(result, list)
    assert len(result) == 1
    err = result[0]
    assert err["sid"] == "sid-bad"
    assert err["error"] == "invalid_mode"
    assert "mode must be raw" in err["message"]


def test_default_mode_is_conversation():
    """When caller doesn't pass ``mode``, wrapper defaults to
    ``"conversation"`` (R13.3 contract: most LLM callers want the
    merged dialog, not the wire dump)."""
    fake = MagicMock()
    fake.replay = MagicMock(return_value=[{"role": "user", "turn": 0, "text": "hi"}])
    status_tool.set_dispatcher(fake)

    status_tool.reasonix_replay("sid-d")

    # The dispatched mode must be "conversation" by default
    _args, kwargs = fake.replay.call_args
    assert kwargs["mode"] == "conversation"


def test_mode_conversation_explicit():
    """Explicit ``mode="conversation"`` is identical to default."""
    fake = MagicMock()
    fake.replay = MagicMock(return_value=[
        {"role": "user", "turn": 0, "text": "hi"},
        {"role": "assistant", "turn": 0, "text": "hello"},
    ])
    status_tool.set_dispatcher(fake)

    result = status_tool.reasonix_replay("sid-c", mode="conversation")

    fake.replay.assert_called_once_with("sid-c", since_seq=0, mode="conversation")
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"


def test_mode_with_explicit_since_seq():
    """All three args (sid, since_seq, mode) are forwarded correctly."""
    fake = MagicMock()
    fake.replay = MagicMock(return_value=[])
    status_tool.set_dispatcher(fake)

    status_tool.reasonix_replay("sid-tri", since_seq=10, mode="raw")

    fake.replay.assert_called_once_with("sid-tri", since_seq=10, mode="raw")
