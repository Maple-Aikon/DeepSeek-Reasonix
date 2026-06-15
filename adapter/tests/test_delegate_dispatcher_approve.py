"""R11 tests: ``DelegateDispatcher`` approve flow e2e (race + timeout).

R8 added the approve flow plumbing (slot map, asyncio.Event, race against
``APPROVE_TIMEOUT_S``). R10 wrapped ``on_approve`` in the MCP tool layer.

This file covers the e2e behavior the unit tests in
:mod:`test_delegate_dispatcher` deliberately defer (T1-T10 there use
sequential ``await`` + a fast-approving supervisor, which can't
exercise the actual race). Each test here drives the real async
race via:

  * a single-threaded mock supervisor (no real Reasonix binary)
  * short ``approve_timeout_s`` (0.1-0.5s) so timeout paths fire
  * explicit ``asyncio.sleep`` + concurrent tasks to model the race

Test scope:
  1. Happy approve race — manual approval lands before timeout,
     cost snapshot has planner spend, follow-up prompt IS sent
  2. Manual reject — explicit reject decision, no follow-up prompt
  3. Auto-reject timeout — no manual approval within window,
     follow-up prompt is NOT sent, ``decision="reject"``
  4. Concurrent sessions — 2 sids in parallel, decisions don't
     cross-contaminate
  5. Idempotent on_approve — calling twice returns False the 2nd
     time, slot is single-shot
  6. End-to-end via R10 MCP tool — ``set_dispatcher`` then call
     ``reasonix_approve(sid, "approve")`` from the MCP layer; the
     dispatcher's pending session unblocks correctly
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapter.tools import reasonix_approve as approve_mod
from adapter.tools.reasonix_approve import (
    clear_dispatcher,
    reasonix_approve,
    set_dispatcher,
)
from adapter.tools.reasonix_delegate import (
    APPROVE_TIMEOUT_S,
    DelegateDispatcher,
    VALID_PLAN_MODES,
)


# ---------------------------------------------------------------------------
# Test doubles — fuller than the unit tests' stubs
# ---------------------------------------------------------------------------


class _RaceSupervisor:
    """Mock supervisor that exposes the approve-race surface.

    Records every ``prompt`` call (so we can assert that the
    follow-up was/wasn't sent), lets the test schedule the
    approval at a precise moment, and never blocks on its own —
    the *real* race is the one between ``asyncio.wait_for(event)``
    and the test's ``on_approve()`` call.
    """

    def __init__(self) -> None:
        self.sid_counter = 0
        self.new_session_calls: list[dict] = []
        self.prompt_calls: list[dict] = []
        self.on_notification_set_count = 0
        self._on_notification_value = None

    @property
    def on_notification(self):
        return self._on_notification_value

    @on_notification.setter
    def on_notification(self, value):
        self.on_notification_set_count += 1
        self._on_notification_value = value

    async def new_session(self, cwd: str | None = None) -> dict:
        self.sid_counter += 1
        sid = f"race-sid-{self.sid_counter:03d}"
        self.new_session_calls.append({"cwd": cwd, "sid": sid})
        return {"sessionId": sid}

    async def prompt(self, sid: str, content: list[dict]) -> dict:
        self.prompt_calls.append({"sid": sid, "content": content})
        return {"stopReason": "end_turn"}


class _NoopLogCapture:
    """Log-capture double — no transcript, no threading.

    Exposes ``_log_dir`` and a ``handler()`` that returns a no-op
    async function, so :class:`DelegateDispatcher` thinks it has
    transcripts to read (they just don't exist, leading to zero
    cost — which is fine for race tests, where we only care about
    the race outcome, not the cents).
    """

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self.handler_calls = 0

    def handler(self):
        self.handler_calls += 1

        async def _h(*_args: Any, **_kwargs: Any) -> None:
            return None

        return _h

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def race_supervisor() -> _RaceSupervisor:
    return _RaceSupervisor()


@pytest.fixture
def noop_log_capture(tmp_path: Path) -> _NoopLogCapture:
    return _NoopLogCapture(log_dir=tmp_path)


@pytest.fixture
def race_dispatcher(
    race_supervisor: _RaceSupervisor,
    noop_log_capture: _NoopLogCapture,
) -> DelegateDispatcher:
    return DelegateDispatcher(
        race_supervisor,
        log_capture=noop_log_capture,
        approve_timeout_s=0.3,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestApproveHappyRace:
    """T1: manual approval lands before timeout → follow-up prompt sent."""

    @pytest.mark.asyncio
    async def test_manual_approve_unblocks_dispatch(
        self,
        race_dispatcher: DelegateDispatcher,
        race_supervisor: _RaceSupervisor,
    ) -> None:
        # Schedule the approval to land 50ms into the 300ms window.
        async def _approver() -> None:
            await asyncio.sleep(0.05)
            ok = race_dispatcher.on_approve("race-sid-001", "approve")
            assert ok is True

        approver = asyncio.create_task(_approver())
        result = await race_dispatcher.dispatch(
            prompt="plan then execute",
            plan_mode="approve",
        )
        await approver

        # The dispatch flow:
        #   prompt 1: the user's task ("plan then execute")
        #   prompt 2: the auto-injected follow-up "Plan approved — continue."
        assert len(race_supervisor.prompt_calls) == 2
        assert race_supervisor.prompt_calls[0]["content"][0]["text"] == "plan then execute"
        assert "Plan approved" in race_supervisor.prompt_calls[1]["content"][0]["text"]

        # Return shape: decision="approve" propagated.
        assert result["decision"] == "approve"
        assert result["plan_mode"] == "approve"
        assert result["sid"] == "race-sid-001"
        # No transcript exists → cost is zero breakdown.
        assert result["cost"]["planner_usd"] == 0.0
        assert result["cost"]["executor_usd"] == 0.0


class TestApproveManualReject:
    """T2: explicit reject decision → no follow-up prompt."""

    @pytest.mark.asyncio
    async def test_manual_reject_returns_early(
        self,
        race_dispatcher: DelegateDispatcher,
        race_supervisor: _RaceSupervisor,
    ) -> None:
        async def _rejecter() -> None:
            await asyncio.sleep(0.02)
            ok = race_dispatcher.on_approve(
                "race-sid-001", "reject", feedback="plan is too aggressive",
            )
            assert ok is True

        rejecter = asyncio.create_task(_rejecter())
        result = await race_dispatcher.dispatch(
            prompt="plan then execute",
            plan_mode="approve",
        )
        await rejecter

        # Only the initial prompt, NO follow-up.
        assert len(race_supervisor.prompt_calls) == 1
        assert race_supervisor.prompt_calls[0]["content"][0]["text"] == "plan then execute"

        # Return shape: decision="reject" propagated, no executor phase.
        assert result["decision"] == "reject"
        # The session map (R9) records final state as "rejected".
        assert race_dispatcher._sessions["race-sid-001"]["status"] == "rejected"
        assert race_dispatcher._sessions["race-sid-001"]["decision"] == "reject"


class TestApproveAutoRejectTimeout:
    """T3: no manual approval → auto-reject after timeout window."""

    @pytest.mark.asyncio
    async def test_timeout_triggers_auto_reject(
        self,
        race_supervisor: _RaceSupervisor,
        noop_log_capture: _NoopLogCapture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Use a very short timeout so the test runs in <1s.
        d = DelegateDispatcher(
            race_supervisor,
            log_capture=noop_log_capture,
            approve_timeout_s=0.1,
        )

        with caplog.at_level(logging.INFO, logger="adapter.tools.reasonix_delegate"):
            result = await d.dispatch(
                prompt="plan then execute",
                plan_mode="approve",
            )

        # Auto-reject logged.
        assert any("auto-reject" in r.message for r in caplog.records)

        # decision="reject" propagated, NO follow-up prompt.
        assert result["decision"] == "reject"
        assert len(race_supervisor.prompt_calls) == 1
        assert race_supervisor.prompt_calls[0]["content"][0]["text"] == "plan then execute"

        # Status map reflects the auto-reject.
        assert d._sessions["race-sid-001"]["status"] == "rejected"
        assert d._sessions["race-sid-001"]["decision"] == "reject"

        # Slot was cleaned up in the finally block.
        assert "race-sid-001" not in d._pending_approves


class TestConcurrentApproveSessions:
    """T4: 2 sids pending in parallel → decisions don't cross-contaminate."""

    @pytest.mark.asyncio
    async def test_two_sids_approve_independently(
        self,
        race_dispatcher: DelegateDispatcher,
        race_supervisor: _RaceSupervisor,
    ) -> None:
        # Launch both dispatches concurrently.
        async def _approve_late(d: DelegateDispatcher, sid: str, decision: str) -> None:
            await asyncio.sleep(0.05)
            assert d.on_approve(sid, decision) is True

        async def _run_dispatch(prompt: str) -> dict:
            return await race_dispatcher.dispatch(
                prompt=prompt, plan_mode="approve",
            )

        t_a = asyncio.create_task(_run_dispatch("task A"))
        t_b = asyncio.create_task(_run_dispatch("task B"))

        # Approvers race after a delay, approve A, reject B.
        await asyncio.sleep(0.02)  # let both dispatches reach the wait
        # Get the assigned sids (counters incremented in order).
        # race-sid-001 → task A, race-sid-002 → task B
        approver_a = asyncio.create_task(
            _approve_late(race_dispatcher, "race-sid-001", "approve"),
        )
        approver_b = asyncio.create_task(
            _approve_late(race_dispatcher, "race-sid-002", "reject"),
        )

        result_a, result_b = await asyncio.gather(t_a, t_b)
        await asyncio.gather(approver_a, approver_b)

        # Cross-check: each dispatch got its own decision.
        assert result_a["decision"] == "approve"
        assert result_b["decision"] == "reject"

        # Each session has its own follow-up (or lack thereof).
        # A: 2 prompts (initial + "Plan approved — continue.")
        # B: 1 prompt (initial only)
        sid_a_prompts = [c for c in race_supervisor.prompt_calls if c["sid"] == "race-sid-001"]
        sid_b_prompts = [c for c in race_supervisor.prompt_calls if c["sid"] == "race-sid-002"]
        assert len(sid_a_prompts) == 2
        assert len(sid_b_prompts) == 1


class TestIdempotentOnApprove:
    """T5: on_approve after the slot has resolved returns False."""

    @pytest.mark.asyncio
    async def test_second_approve_call_returns_false(
        self,
        race_dispatcher: DelegateDispatcher,
    ) -> None:
        async def _first() -> None:
            await asyncio.sleep(0.02)
            assert race_dispatcher.on_approve("race-sid-001", "approve") is True

        async def _second() -> None:
            # Wait for the first to win the slot, then try again.
            await asyncio.sleep(0.05)
            ok = race_dispatcher.on_approve("race-sid-001", "approve")
            assert ok is False  # already resolved

        approver1 = asyncio.create_task(_first())
        approver2 = asyncio.create_task(_second())

        result = await race_dispatcher.dispatch(
            prompt="plan then execute",
            plan_mode="approve",
        )
        await asyncio.gather(approver1, approver2)

        assert result["decision"] == "approve"

    @pytest.mark.asyncio
    async def test_unknown_sid_returns_false(
        self, race_dispatcher: DelegateDispatcher,
    ) -> None:
        # No pending approval for an unknown sid — defensive return False.
        assert race_dispatcher.on_approve("nonexistent-sid", "approve") is False
        assert race_dispatcher.on_approve("nonexistent-sid", "reject") is False


class TestMCPApproveEndToEnd:
    """T6: R10 ``reasonix_approve`` MCP tool unblocks a real R8 dispatch."""

    @pytest.mark.asyncio
    async def test_reasonix_approve_mcp_tool_unblocks_dispatch(
        self,
        race_dispatcher: DelegateDispatcher,
        race_supervisor: _RaceSupervisor,
    ) -> None:
        # Wire the dispatcher into the MCP tool's module-level slot.
        set_dispatcher(race_dispatcher)
        try:
            async def _approver() -> None:
                await asyncio.sleep(0.05)
                # Call the MCP tool layer (R10) — it routes to
                # DelegateDispatcher.on_approve via the slot.
                result = reasonix_approve(
                    sid="race-sid-001", decision="approve", feedback=None,
                )
                # R10 returns a 5-field dict.
                assert result["accepted"] is True
                assert result["decision"] == "approve"
                assert result["sid"] == "race-sid-001"

            approver = asyncio.create_task(_approver())
            dispatch_result = await race_dispatcher.dispatch(
                prompt="plan then execute",
                plan_mode="approve",
            )
            await approver

            # The MCP tool's approval DID unblock the dispatch.
            assert dispatch_result["decision"] == "approve"
            assert len(race_supervisor.prompt_calls) == 2
        finally:
            clear_dispatcher()

    @pytest.mark.asyncio
    async def test_reasonix_approve_unknown_sid_via_mcp(
        self, race_dispatcher: DelegateDispatcher,
    ) -> None:
        set_dispatcher(race_dispatcher)
        try:
            result = reasonix_approve(sid="unknown-sid", decision="approve")
            # R10 returns accepted=False for unknown sid.
            assert result["accepted"] is False
            assert result["reason"] in ("no_pending_approval", "session_unblocked")
        finally:
            clear_dispatcher()
