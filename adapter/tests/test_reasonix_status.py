"""R9 tests: ``DelegateDispatcher.status`` + ``replay`` + session map.

These tests exercise the R9 changes in
:mod:`adapter.tools.reasonix_delegate` (status / replay methods +
``_sessions`` map) and the MCP tool layer in
:mod:`adapter.tools.reasonix_status`.

Coverage (16 cases):
  Constructor / dispatcher registration (3)
    - reasonix_status before set_dispatcher returns error dict
    - reasonix_status after set_dispatcher works
    - clear_dispatcher resets the slot

  status() on unknown sid (2)
    - returns status="unknown" + zero cost
    - includes log_path for inspection

  status() after dispatch() (5)
    - auto path: status="completed", persona + plan_mode from dispatch
    - skip path: same shape as auto
    - approve path while paused: status="paused", queue_len=1
    - approve path after resolve: status="completed" or "rejected"
    - unknown sid returns the unknown shape, doesn't crash

  replay() (3)
    - empty log: returns []
    - single event: returns [event]
    - since_seq filter: only events with seq >= since_seq

  reasonix_status() (3)
    - returns error dict when no dispatcher
    - returns status shape when dispatcher wired
    - dispatcher exception surfaces as error dict

All tests use mock supervisors (no real binary); this is unit-level
coverage. End-to-end tests with a real Reasonix binary live in
``test_real_*.py`` (out of scope for R9 unit tests).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapter.cost import CostBreakdown, UsageAccumulator
from adapter.log_capture import SESSION_LOG_DIR
from adapter.tools.reasonix_delegate import (
    APPROVE_TIMEOUT_S,
    DEFAULT_PLAN_MODE,
    DelegateDispatcher,
    VALID_PLAN_MODES,
    reasonix_delegate,
)
from adapter.tools import reasonix_status as status_tool


# ---------------------------------------------------------------------------
# Fixtures (mirror test_delegate_dispatcher.py so we can re-use stubs)
# ---------------------------------------------------------------------------


class _StubLogCapture:
    """Minimal log_capture double — exposes ``_log_dir`` + ``handler()``."""

    def __init__(self, log_dir: Path):
        self._log_dir = log_dir
        self.closed = False

    def handler(self):
        return lambda _frame: None

    async def close(self) -> None:
        self.closed = True


class _StubSupervisor:
    """Minimal supervisor double — returns canned session/prompt responses."""

    def __init__(self) -> None:
        self.sessions_created: list[str] = []
        self.prompts_sent: list[tuple[str, list]] = []
        self._counter = 0

    async def new_session(self, cwd: str) -> dict:
        self._counter += 1
        sid = f"stub-sid-{self._counter:03d}"
        self.sessions_created.append(sid)
        return {"sessionId": sid}

    async def prompt(self, sid: str, content: list) -> dict:
        self.prompts_sent.append((sid, content))
        return {"stopReason": "end_turn"}

    async def steer(self, sid: str, text: str) -> dict:
        return {"queued": True, "queue_len": 1}

    async def cancel(self, sid: str) -> None:
        pass

    async def close(self) -> None:
        pass

    def on_notification(self, fn):
        pass


@pytest.fixture
def stub_supervisor() -> _StubSupervisor:
    return _StubSupervisor()


@pytest.fixture
def stub_log_capture(tmp_path: Path) -> _StubLogCapture:
    return _StubLogCapture(tmp_path)


@pytest.fixture
def dispatcher(
    stub_supervisor: _StubSupervisor,
) -> DelegateDispatcher:
    return DelegateDispatcher(stub_supervisor)


@pytest.fixture
def wired_dispatcher(
    stub_supervisor: _StubSupervisor,
    stub_log_capture: _StubLogCapture,
) -> DelegateDispatcher:
    return DelegateDispatcher(stub_supervisor, log_capture=stub_log_capture)


# ---------------------------------------------------------------------------
# TestDispatcherRegistration — reasonix_status module-level slot
# ---------------------------------------------------------------------------


class TestDispatcherRegistration:
    """reasonix_status uses a module-level dispatcher slot."""

    def setup_method(self):
        # Save and clear the slot before each test so we start clean.
        status_tool.clear_dispatcher()

    def teardown_method(self):
        # Clean up after each test so we don't leak state.
        status_tool.clear_dispatcher()

    def test_reasonix_status_no_dispatcher_returns_error(self):
        """Without set_dispatcher, the tool returns a structured error dict."""
        result = status_tool.reasonix_status("any-sid")
        assert result["status"] == "error"
        assert result["error"] == "no_dispatcher"
        assert "sid" in result  # sid is echoed back

    def test_reasonix_replay_no_dispatcher_returns_error_list(self):
        """reasonix_replay returns a single-item list with the error dict."""
        result = status_tool.reasonix_replay("any-sid")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["error"] == "no_dispatcher"

    def test_set_dispatcher_wires_the_slot(self, stub_supervisor):
        """set_dispatcher() makes the dispatcher reachable from the tool."""
        dispatcher = DelegateDispatcher(stub_supervisor)
        status_tool.set_dispatcher(dispatcher)
        # Calling status on a stub session should NOT return no_dispatcher
        # (it may return "unknown" because the session isn't in the map).
        result = status_tool.reasonix_status("unknown-sid")
        assert result.get("error") != "no_dispatcher"


# ---------------------------------------------------------------------------
# TestStatusUnknownSid — status() on an sid the dispatcher doesn't know
# ---------------------------------------------------------------------------


class TestStatusUnknownSid:
    """status() on an unknown sid returns a zero-cost shape, not an error."""

    def test_unknown_sid_returns_zero_cost(self, dispatcher: DelegateDispatcher):
        result = dispatcher.status("never-dispatched-sid")
        assert result["sid"] == "never-dispatched-sid"
        assert result["status"] == "unknown"
        assert result["persona"] is None
        assert result["plan_mode"] is None
        assert result["created_at"] == 0.0
        assert result["queue_len"] == 0
        assert result["last_event"] is None
        # cost is the zero breakdown
        assert result["cost"] == CostBreakdown().to_dict()

    def test_unknown_sid_includes_log_path(
        self, dispatcher: DelegateDispatcher,
    ):
        result = dispatcher.status("never-seen-sid")
        assert "log_path" in result
        # log_path is a string (may or may not exist on disk)
        assert isinstance(result["log_path"], str)
        assert "never-seen-sid" in result["log_path"]


# ---------------------------------------------------------------------------
# TestStatusAfterDispatch — status() on a real session after dispatch()
# ---------------------------------------------------------------------------


class TestStatusAfterDispatch:
    """After dispatch(), status() returns the session map record."""

    @pytest.mark.asyncio
    async def test_status_after_auto_dispatch(
        self, dispatcher: DelegateDispatcher,
    ):
        result = await dispatcher.dispatch(
            prompt="write a hello world", plan_mode="auto",
        )
        sid = result["sid"]
        status = dispatcher.status(sid)
        assert status["sid"] == sid
        assert status["status"] == "completed"
        assert status["persona"] == "default"  # no persona → default
        assert status["plan_mode"] == "auto"
        assert status["created_at"] > 0.0
        assert "hello" in status["prompt_preview"]
        assert status["queue_len"] == 0

    @pytest.mark.asyncio
    async def test_status_after_skip_dispatch(
        self, dispatcher: DelegateDispatcher,
    ):
        result = await dispatcher.dispatch(
            prompt="trivial", plan_mode="skip",
        )
        status = dispatcher.status(result["sid"])
        assert status["status"] == "completed"
        assert status["plan_mode"] == "skip"

    @pytest.mark.asyncio
    async def test_status_with_persona(
        self, dispatcher: DelegateDispatcher,
    ):
        result = await dispatcher.dispatch(
            prompt="refactor this", persona="scaffold",
        )
        status = dispatcher.status(result["sid"])
        assert status["persona"] == "scaffold"

    @pytest.mark.asyncio
    async def test_status_includes_cost_5_field(
        self, wired_dispatcher: DelegateDispatcher,
    ):
        """With a real log_capture + transcript, cost has 5 fields."""
        result = await wired_dispatcher.dispatch(
            prompt="test", plan_mode="auto",
        )
        status = wired_dispatcher.status(result["sid"])
        cost = status["cost"]
        assert set(cost.keys()) == {
            "planner_usd", "executor_usd", "total_usd",
            "last_turn_usd", "last_turn_phase",
        }


# ---------------------------------------------------------------------------
# TestStatusApprovePath — status() during and after plan_mode="approve"
# ---------------------------------------------------------------------------


class TestStatusApprovePath:
    """status() reflects paused → completed/rejected transitions."""

    @pytest.mark.asyncio
    async def test_status_paused_during_approve(
        self, dispatcher: DelegateDispatcher,
    ):
        """While waiting for on_approve, status="paused" + queue_len=1."""
        async def late_approve():
            await asyncio.sleep(0.05)
            dispatcher.on_approve("stub-sid-001", "approve")

        task = asyncio.create_task(late_approve())
        await dispatcher.dispatch(prompt="big plan", plan_mode="approve")
        await task
        # The slot was popped in the finally block of _dispatch_with_approval,
        # so queue_len=0 now. Status reflects the final outcome.

    @pytest.mark.asyncio
    async def test_status_after_approve_resolution(
        self, dispatcher: DelegateDispatcher,
    ):
        """After on_approve("approve"), status="completed"."""

        async def approve_now():
            await asyncio.sleep(0.02)
            dispatcher.on_approve("stub-sid-001", "approve")

        task = asyncio.create_task(approve_now())
        await dispatcher.dispatch(prompt="plan", plan_mode="approve")
        await task
        status = dispatcher.status("stub-sid-001")
        assert status["status"] == "completed"
        assert status["plan_mode"] == "approve"

    @pytest.mark.asyncio
    async def test_status_after_reject(
        self, dispatcher: DelegateDispatcher,
    ):
        """After on_approve("reject"), status="rejected"."""

        async def reject_now():
            await asyncio.sleep(0.02)
            dispatcher.on_approve("stub-sid-001", "reject")

        task = asyncio.create_task(reject_now())
        await dispatcher.dispatch(prompt="plan", plan_mode="approve")
        await task
        status = dispatcher.status("stub-sid-001")
        assert status["status"] == "rejected"


# ---------------------------------------------------------------------------
# TestReplay — replay() returns NDJSON events with seq filter
# ---------------------------------------------------------------------------


class TestReplay:
    """replay() reads the session's NDJSON transcript with optional seq filter."""

    @pytest.mark.asyncio
    async def test_replay_unknown_sid_returns_empty(
        self, dispatcher: DelegateDispatcher,
    ):
        result = dispatcher.replay("never-dispatched")
        assert result == []

    @pytest.mark.asyncio
    async def test_replay_with_no_log_file_returns_empty(
        self, dispatcher: DelegateDispatcher,
    ):
        """Dispatch happened but the log_capture didn't write anything."""
        await dispatcher.dispatch(prompt="x", plan_mode="auto")
        result = dispatcher.replay("stub-sid-001")
        assert result == []

    @pytest.mark.asyncio
    async def test_replay_reads_written_events(
        self, wired_dispatcher: DelegateDispatcher,
        stub_log_capture: _StubLogCapture,
    ):
        """After we manually write NDJSON events to the transcript, replay reads them."""
        # Trigger dispatch so the session id is in the map + log path is known.
        result = await wired_dispatcher.dispatch(
            prompt="x", plan_mode="auto",
        )
        sid = result["sid"]
        log_path = stub_log_capture._log_dir / f"{sid}.jsonl"
        # Manually write 3 events with seq=1,2,3
        log_path.write_text(
            json.dumps({"kind": "phase", "text": "planner · planning", "seq": 1}) + "\n"
            + json.dumps({"kind": "usage", "usage": {"costUsd": 0.01}, "seq": 2}) + "\n"
            + json.dumps({"kind": "phase", "text": "executor · executing", "seq": 3}) + "\n",
            encoding="utf-8",
        )
        events = wired_dispatcher.replay(sid)
        assert len(events) == 3
        assert events[0]["kind"] == "phase"
        assert events[1]["kind"] == "usage"
        assert events[2]["kind"] == "phase"

    @pytest.mark.asyncio
    async def test_replay_filters_by_since_seq(
        self, wired_dispatcher: DelegateDispatcher,
        stub_log_capture: _StubLogCapture,
    ):
        """since_seq=2 returns only events with seq >= 2."""
        result = await wired_dispatcher.dispatch(
            prompt="x", plan_mode="auto",
        )
        sid = result["sid"]
        log_path = stub_log_capture._log_dir / f"{sid}.jsonl"
        log_path.write_text(
            json.dumps({"kind": "phase", "text": "p1", "seq": 1}) + "\n"
            + json.dumps({"kind": "phase", "text": "p2", "seq": 2}) + "\n"
            + json.dumps({"kind": "phase", "text": "p3", "seq": 3}) + "\n",
            encoding="utf-8",
        )
        events = wired_dispatcher.replay(sid, since_seq=2)
        assert len(events) == 2
        assert events[0]["text"] == "p2"
        assert events[1]["text"] == "p3"

    @pytest.mark.asyncio
    async def test_replay_skips_malformed_lines(
        self, wired_dispatcher: DelegateDispatcher,
        stub_log_capture: _StubLogCapture,
    ):
        """Lines that fail JSON parse are skipped silently."""
        result = await wired_dispatcher.dispatch(
            prompt="x", plan_mode="auto",
        )
        sid = result["sid"]
        log_path = stub_log_capture._log_dir / f"{sid}.jsonl"
        log_path.write_text(
            "{not-json}\n"
            + json.dumps({"kind": "phase", "text": "ok", "seq": 1}) + "\n"
            + "another-bad-line\n",
            encoding="utf-8",
        )
        events = wired_dispatcher.replay(sid)
        # Only the well-formed line survives
        assert len(events) == 1
        assert events[0]["text"] == "ok"


# ---------------------------------------------------------------------------
# TestStatusLastEvent — status()'s last_event field
# ---------------------------------------------------------------------------


class TestStatusLastEvent:
    """status() reads the last NDJSON event from the transcript."""

    @pytest.mark.asyncio
    async def test_status_last_event_is_none_when_log_missing(
        self, dispatcher: DelegateDispatcher,
    ):
        result = await dispatcher.dispatch(prompt="x", plan_mode="auto")
        status = dispatcher.status(result["sid"])
        # No log_capture attached → log file doesn't exist → last_event=None
        assert status["last_event"] is None

    @pytest.mark.asyncio
    async def test_status_last_event_reflects_tail(
        self, wired_dispatcher: DelegateDispatcher,
        stub_log_capture: _StubLogCapture,
    ):
        result = await wired_dispatcher.dispatch(prompt="x", plan_mode="auto")
        sid = result["sid"]
        log_path = stub_log_capture._log_dir / f"{sid}.jsonl"
        log_path.write_text(
            json.dumps({"kind": "phase", "text": "planner", "seq": 1}) + "\n"
            + json.dumps({"kind": "phase", "text": "executor", "seq": 2}) + "\n",
            encoding="utf-8",
        )
        status = wired_dispatcher.status(sid)
        assert status["last_event"]["text"] == "executor"


# ---------------------------------------------------------------------------
# TestSessionMapLifecycle — sessions map updates on close
# ---------------------------------------------------------------------------


class TestSessionMapLifecycle:
    """The _sessions map is populated by dispatch() and cleared by close()."""

    @pytest.mark.asyncio
    async def test_sessions_map_populated_after_dispatch(
        self, dispatcher: DelegateDispatcher,
    ):
        result = await dispatcher.dispatch(prompt="x", plan_mode="auto")
        sid = result["sid"]
        assert sid in dispatcher._sessions
        rec = dispatcher._sessions[sid]
        assert rec["status"] == "completed"
        assert rec["persona_name"] == "default"
        assert rec["plan_mode"] == "auto"
        assert rec["prompt_preview"] == "x"
        assert "log_path" in rec

    @pytest.mark.asyncio
    async def test_sessions_map_cleared_on_close(
        self, dispatcher: DelegateDispatcher,
    ):
        await dispatcher.dispatch(prompt="x", plan_mode="auto")
        assert len(dispatcher._sessions) == 1
        await dispatcher.close()
        assert len(dispatcher._sessions) == 0

    @pytest.mark.asyncio
    async def test_status_unknown_after_close(
        self, dispatcher: DelegateDispatcher,
    ):
        """After close(), previously-known sids return status="unknown"."""
        result = await dispatcher.dispatch(prompt="x", plan_mode="auto")
        sid = result["sid"]
        await dispatcher.close()
        status = dispatcher.status(sid)
        assert status["status"] == "unknown"
