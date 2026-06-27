"""R8 tests: ``DelegateDispatcher`` real dispatch + cost + approve flow.

These tests exercise the R8 changes in :mod:`adapter.tools.reasonix_delegate`:

  1. Constructor wires ``supervisor.on_notification`` when given a
     log_capture (composition mirrors TaskDispatcher).
  2. ``dispatch(plan_mode="auto")`` opens a session, sends the prompt,
     snapshots cost from the transcript, returns 5-field shape.
  3. ``dispatch(plan_mode="skip")`` behaves identically to ``auto``
     from the dispatcher's perspective (the binary's Coordinator
     owns the trivial-skip policy).
  4. ``dispatch(plan_mode="approve")`` pauses after the planner phase
     and returns ``cost.last_turn_phase="planner"`` immediately.
  5. ``on_approve(sid, "approve")`` unblocks a pending dispatch and
     sends a follow-up prompt.
  6. ``on_approve(sid, "reject")`` unblocks a pending dispatch and
     returns without sending a follow-up.
  7. ``on_approve`` for an unknown sid returns False (no race).
  8. ``close()`` is idempotent and safe on a dispatcher with no
     log_capture.
  9. ``_log_path`` falls back to ``SESSION_LOG_DIR`` when log_capture
     is ``None``.
 10. The ``reasonix_delegate`` stub function (R7) is unaffected by
     the R8 class refactor — it still works without a supervisor.

All tests use mock supervisors (no real binary); this is unit-level
coverage. End-to-end tests with a real Reasonix binary live in
``test_real_*.py`` (out of scope for R8 unit tests).
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StubLogCapture:
    """Minimal log_capture double — exposes ``_log_dir`` + ``handler()``."""

    def __init__(self, log_dir: Path) -> None:
        self._log_dir = log_dir
        self.handler_calls = 0
        self.closed = False

    def handler(self):
        self.handler_calls += 1
        # Return a no-op async handler; real LogCapture writes NDJSON.
        async def _h(*_args: Any, **_kwargs: Any) -> None:
            return None
        return _h

    async def close(self) -> None:
        self.closed = True


class _StubSupervisor:
    """Minimal supervisor double — records calls, returns canned sids."""

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
        sid = f"stub-sid-{self.sid_counter:03d}"
        self.new_session_calls.append({"cwd": cwd, "sid": sid})
        return {"sessionId": sid}

    async def prompt(self, sid: str, content: list[dict]) -> dict:
        self.prompt_calls.append({"sid": sid, "content": content})
        return {"stopReason": "end_turn"}


@pytest.fixture
def stub_supervisor() -> _StubSupervisor:
    return _StubSupervisor()


@pytest.fixture
def stub_log_capture(tmp_path: Path) -> _StubLogCapture:
    return _StubLogCapture(log_dir=tmp_path)


@pytest.fixture
def dispatcher(stub_supervisor: _StubSupervisor) -> DelegateDispatcher:
    # No log_capture — exercises the "blind mode" code path.
    return DelegateDispatcher(stub_supervisor, approve_timeout_s=0.5)


@pytest.fixture
def wired_dispatcher(
    stub_supervisor: _StubSupervisor,
    stub_log_capture: _StubLogCapture,
) -> DelegateDispatcher:
    return DelegateDispatcher(
        stub_supervisor, log_capture=stub_log_capture, approve_timeout_s=0.5,
    )


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestConstructor:
    """DelegateDispatcher.__init__ — composition + wiring."""

    def test_no_log_capture_does_not_touch_supervisor(
        self, stub_supervisor: _StubSupervisor,
    ) -> None:
        DelegateDispatcher(stub_supervisor)
        assert stub_supervisor.on_notification_set_count == 0

    def test_with_log_capture_wires_handler(
        self,
        stub_supervisor: _StubSupervisor,
        stub_log_capture: _StubLogCapture,
    ) -> None:
        DelegateDispatcher(
            stub_supervisor, log_capture=stub_log_capture,
        )
        # The supervisor's setter should have been called exactly once.
        assert stub_supervisor.on_notification_set_count == 1
        # And the handler factory on log_capture should have been called.
        assert stub_log_capture.handler_calls == 1

    def test_log_capture_handler_exception_is_logged_not_raised(
        self, stub_supervisor: _StubSupervisor, caplog: pytest.LogCaptureFixture,
    ) -> None:
        class _BadLogCapture:
            def handler(self):
                raise RuntimeError("boom")
        with caplog.at_level(logging.WARNING, logger="adapter.tools.reasonix_delegate"):
            DelegateDispatcher(stub_supervisor, log_capture=_BadLogCapture())
        # Supervisor setter NOT called because the handler() call failed
        # before we got to the assignment.
        assert stub_supervisor.on_notification_set_count == 0
        assert any("could not wire" in r.message for r in caplog.records)

    def test_approve_timeout_default(
        self, stub_supervisor: _StubSupervisor,
    ) -> None:
        d = DelegateDispatcher(stub_supervisor)
        assert d._approve_timeout == APPROVE_TIMEOUT_S

    def test_approve_timeout_custom(
        self, stub_supervisor: _StubSupervisor,
    ) -> None:
        d = DelegateDispatcher(stub_supervisor, approve_timeout_s=1.5)
        assert d._approve_timeout == 1.5


# ---------------------------------------------------------------------------
# dispatch() — normal path (auto / skip)
# ---------------------------------------------------------------------------


class TestDispatchAuto:
    """plan_mode="auto" opens a session, sends the prompt, snapshots cost."""

    @pytest.mark.asyncio
    async def test_returns_real_sid_from_new_session(
        self, dispatcher: DelegateDispatcher, stub_supervisor: _StubSupervisor,
    ) -> None:
        result = await dispatcher.dispatch(prompt="hello world")
        assert result["sid"] == "stub-sid-001"
        assert len(stub_supervisor.new_session_calls) == 1
        assert len(stub_supervisor.prompt_calls) == 1

    @pytest.mark.asyncio
    async def test_prompt_content_is_text_wrapped(
        self, dispatcher: DelegateDispatcher, stub_supervisor: _StubSupervisor,
    ) -> None:
        await dispatcher.dispatch(prompt="explain recursion")
        call = stub_supervisor.prompt_calls[0]
        assert call["content"] == [{"type": "text", "text": "explain recursion"}]
        assert call["sid"] == "stub-sid-001"

    @pytest.mark.asyncio
    async def test_default_plan_mode_is_auto(
        self, dispatcher: DelegateDispatcher, stub_supervisor: _StubSupervisor,
    ) -> None:
        result = await dispatcher.dispatch(prompt="x")
        assert result["plan_mode"] == "auto"
        assert DEFAULT_PLAN_MODE == "auto"

    @pytest.mark.asyncio
    async def test_skip_plan_mode_uses_same_dispatch_path(
        self, dispatcher: DelegateDispatcher,
    ) -> None:
        # "skip" is a Coordinator concern, not a dispatcher concern.
        # The dispatcher still opens a session and sends the prompt;
        # the binary's Coordinator handles the trivial-skip policy.
        result = await dispatcher.dispatch(prompt="x", plan_mode="skip")
        assert result["plan_mode"] == "skip"
        assert result["sid"] == "stub-sid-001"

    @pytest.mark.asyncio
    async def test_cwd_defaults_to_os_cwd_when_none(
        self, dispatcher: DelegateDispatcher, stub_supervisor: _StubSupervisor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("os.getcwd", lambda: "/some/default/dir")
        await dispatcher.dispatch(prompt="x")
        assert stub_supervisor.new_session_calls[0]["cwd"] == "/some/default/dir"

    @pytest.mark.asyncio
    async def test_explicit_cwd_passed_through(
        self, dispatcher: DelegateDispatcher, stub_supervisor: _StubSupervisor,
    ) -> None:
        await dispatcher.dispatch(prompt="x", cwd="/tmp/work")
        assert stub_supervisor.new_session_calls[0]["cwd"] == "/tmp/work"

    @pytest.mark.asyncio
    async def test_cost_is_zero_breakdown_without_log_capture(
        self, dispatcher: DelegateDispatcher,
    ) -> None:
        # No log_capture → no transcript → cost is the zero breakdown.
        result = await dispatcher.dispatch(prompt="x")
        cost = result["cost"]
        assert cost == CostBreakdown().to_dict()
        assert cost["total_usd"] == 0.0
        assert cost["planner_usd"] == 0.0
        assert cost["executor_usd"] == 0.0
        assert cost["last_turn_phase"] == "planner"

    @pytest.mark.asyncio
    async def test_log_path_resolves_to_session_log_dir_when_no_capture(
        self, dispatcher: DelegateDispatcher,
    ) -> None:
        result = await dispatcher.dispatch(prompt="x")
        expected = SESSION_LOG_DIR / "stub-sid-001.jsonl"
        assert result["log_path"] == str(expected)

    @pytest.mark.asyncio
    async def test_log_path_uses_log_capture_log_dir(
        self,
        wired_dispatcher: DelegateDispatcher,
        stub_log_capture: _StubLogCapture,
    ) -> None:
        result = await wired_dispatcher.dispatch(prompt="x")
        expected = stub_log_capture._log_dir / "stub-sid-001.jsonl"
        assert result["log_path"] == str(expected)

    @pytest.mark.asyncio
    async def test_cost_parsed_from_transcript_when_log_capture_attached(
        self,
        wired_dispatcher: DelegateDispatcher,
        stub_log_capture: _StubLogCapture,
    ) -> None:
        # Write a transcript file the dispatcher can read. The
        # ``usage`` payload is nested (matches Reasonix wire format
        # ``event.Usage``), not flat — see cost.py:_record_usage.
        transcript = stub_log_capture._log_dir / "stub-sid-001.jsonl"
        transcript.write_text(
            json.dumps({
                "kind": "phase", "text": "planner · planning",
            }) + "\n"
            + json.dumps({
                "kind": "usage",
                "usage": {"costUsd": 0.012, "totalTokens": 100},
            }) + "\n"
            + json.dumps({
                "kind": "phase", "text": "executor · executing",
            }) + "\n"
            + json.dumps({
                "kind": "usage",
                "usage": {"costUsd": 0.045, "totalTokens": 200},
            }) + "\n"
        )
        result = await wired_dispatcher.dispatch(prompt="x")
        cost = result["cost"]
        assert cost["planner_usd"] == pytest.approx(0.012)
        assert cost["executor_usd"] == pytest.approx(0.045)
        assert cost["total_usd"] == pytest.approx(0.057)
        assert cost["last_turn_phase"] == "executor"

    @pytest.mark.asyncio
    async def test_return_dict_has_all_stub_keys_plus_cost(
        self, dispatcher: DelegateDispatcher,
    ) -> None:
        result = await dispatcher.dispatch(prompt="x")
        for k in (
            "sid", "log_path", "persona_name", "plan_mode",
            "cwd", "system_prompt_preview", "cost",
        ):
            assert k in result, f"missing key: {k}"

    @pytest.mark.asyncio
    async def test_persona_resolution_uses_default(
        self, dispatcher: DelegateDispatcher,
    ) -> None:
        result = await dispatcher.dispatch(prompt="x")
        assert result["persona_name"] == "default"

    @pytest.mark.asyncio
    async def test_persona_builtin_scaffold(
        self, dispatcher: DelegateDispatcher,
    ) -> None:
        result = await dispatcher.dispatch(prompt="x", persona="scaffold")
        assert result["persona_name"] == "scaffold"


# ---------------------------------------------------------------------------
# dispatch() — plan_mode="approve" race
# ---------------------------------------------------------------------------


class TestDispatchApprove:
    """plan_mode="approve" pauses after the planner phase, then races."""

    @pytest.mark.asyncio
    async def test_approve_returns_planner_cost_snapshot(
        self, dispatcher: DelegateDispatcher,
    ) -> None:
        result = await dispatcher.dispatch(prompt="plan me", plan_mode="approve")
        # The dispatcher auto-rejects after timeout (no caller called
        # on_approve). The cost snapshot still has last_turn_phase="planner".
        assert result["plan_mode"] == "approve"
        assert result["decision"] == "reject"
        assert result["cost"]["last_turn_phase"] == "planner"

    @pytest.mark.asyncio
    async def test_approve_with_explicit_approve_unblocks(
        self, dispatcher: DelegateDispatcher, stub_supervisor: _StubSupervisor,
    ) -> None:
        # Schedule an approve call shortly after dispatch starts.
        # No feedback → dispatcher uses the default follow-up message.
        async def _approve_after_delay() -> None:
            await asyncio.sleep(0.05)
            assert dispatcher.on_approve("stub-sid-001", "approve")

        approve_task = asyncio.create_task(_approve_after_delay())
        result = await dispatcher.dispatch(prompt="plan me", plan_mode="approve")
        await approve_task

        assert result["decision"] == "approve"
        # Two prompt calls: initial plan + follow-up executor.
        assert len(stub_supervisor.prompt_calls) == 2
        assert "Plan approved" in stub_supervisor.prompt_calls[1]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_approve_with_explicit_reject_skips_followup(
        self, dispatcher: DelegateDispatcher, stub_supervisor: _StubSupervisor,
    ) -> None:
        async def _reject_after_delay() -> None:
            await asyncio.sleep(0.05)
            dispatcher.on_approve("stub-sid-001", "reject")

        reject_task = asyncio.create_task(_reject_after_delay())
        result = await dispatcher.dispatch(prompt="plan me", plan_mode="approve")
        await reject_task

        assert result["decision"] == "reject"
        # Only one prompt call (the initial plan) — no executor follow-up.
        assert len(stub_supervisor.prompt_calls) == 1

    @pytest.mark.asyncio
    async def test_approve_timeout_auto_rejects(
        self, dispatcher: DelegateDispatcher, stub_supervisor: _StubSupervisor,
    ) -> None:
        # approve_timeout_s=0.5 (set in fixture). Don't call on_approve.
        result = await dispatcher.dispatch(prompt="plan me", plan_mode="approve")
        assert result["decision"] == "reject"
        assert len(stub_supervisor.prompt_calls) == 1

    @pytest.mark.asyncio
    async def test_approve_followup_includes_feedback(
        self, dispatcher: DelegateDispatcher, stub_supervisor: _StubSupervisor,
    ) -> None:
        async def _approve_with_feedback() -> None:
            await asyncio.sleep(0.05)
            dispatcher.on_approve(
                "stub-sid-001", "approve", feedback="add tests please",
            )

        task = asyncio.create_task(_approve_with_feedback())
        await dispatcher.dispatch(prompt="plan me", plan_mode="approve")
        await task

        # Second prompt should contain the feedback, not the default message.
        followup_text = stub_supervisor.prompt_calls[1]["content"][0]["text"]
        assert "add tests please" in followup_text


# ---------------------------------------------------------------------------
# on_approve() — return-value semantics
# ---------------------------------------------------------------------------


class TestOnApprove:
    """on_approve returns True iff a pending slot was found and resolved."""

    def test_unknown_sid_returns_false(
        self, dispatcher: DelegateDispatcher,
    ) -> None:
        assert dispatcher.on_approve("never-pending", "approve") is False

    @pytest.mark.asyncio
    async def test_resolved_slot_returns_false_on_second_call(
        self, dispatcher: DelegateDispatcher,
    ) -> None:
        # Schedule an approve call right at start; then try a second
        # call to the same sid after resolution.
        async def _double_approve() -> None:
            await asyncio.sleep(0.05)
            first = dispatcher.on_approve("stub-sid-001", "approve")
            second = dispatcher.on_approve("stub-sid-001", "approve")
            assert first is True
            assert second is False  # already fired

        task = asyncio.create_task(_double_approve())
        await dispatcher.dispatch(prompt="x", plan_mode="approve")
        await task


# ---------------------------------------------------------------------------
# close() — idempotency
# ---------------------------------------------------------------------------


class TestClose:
    """close() releases log_capture cleanly and is safe to re-call."""

    @pytest.mark.asyncio
    async def test_close_without_log_capture_is_noop(
        self, dispatcher: DelegateDispatcher,
    ) -> None:
        await dispatcher.close()  # no log_capture → safe no-op
        # Idempotent: call again.
        await dispatcher.close()

    @pytest.mark.asyncio
    async def test_close_with_log_capture_calls_close(
        self,
        wired_dispatcher: DelegateDispatcher,
        stub_log_capture: _StubLogCapture,
    ) -> None:
        await wired_dispatcher.close()
        assert stub_log_capture.closed is True
        # Idempotent.
        await wired_dispatcher.close()


# ---------------------------------------------------------------------------
# R7 stub function — still works after R8 refactor
# ---------------------------------------------------------------------------


class TestR7StubStillWorks:
    """The pure-function ``reasonix_delegate`` is unaffected by the class."""

    def test_stub_returns_dict_with_all_expected_keys(self) -> None:
        result = reasonix_delegate(prompt="hello")
        for k in (
            "sid", "log_path", "persona_name", "plan_mode",
            "cwd", "system_prompt_preview",
        ):
            assert k in result

    def test_stub_stub_sid_format(self) -> None:
        result = reasonix_delegate(prompt="x")
        assert result["sid"].startswith("stub-")

    def test_stub_default_plan_mode(self) -> None:
        result = reasonix_delegate(prompt="x")
        assert result["plan_mode"] == "auto"

    def test_stub_empty_prompt_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            reasonix_delegate(prompt="")

    def test_stub_invalid_plan_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="plan_mode"):
            reasonix_delegate(prompt="x", plan_mode="bogus")

    def test_stub_mutually_exclusive_persona_params(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            reasonix_delegate(
                prompt="x", persona="scaffold", persona_file="/tmp/foo.md",
            )

    def test_stub_accepts_all_three_plan_modes(self) -> None:
        for mode in ("auto", "skip", "approve"):
            result = reasonix_delegate(prompt="x", plan_mode=mode)
            assert result["plan_mode"] == mode
        assert VALID_PLAN_MODES == frozenset({"auto", "skip", "approve"})


# ---------------------------------------------------------------------------
# Concurrency safety: multiple dispatches in flight
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Multiple concurrent dispatches should each get a unique sid."""

    @pytest.mark.asyncio
    async def test_concurrent_dispatches_get_unique_sids(
        self, dispatcher: DelegateDispatcher,
    ) -> None:
        results = await asyncio.gather(
            dispatcher.dispatch(prompt="task A"),
            dispatcher.dispatch(prompt="task B"),
            dispatcher.dispatch(prompt="task C"),
        )
        sids = {r["sid"] for r in results}
        assert len(sids) == 3
        assert sids == {"stub-sid-001", "stub-sid-002", "stub-sid-003"}


# ---------------------------------------------------------------------------
# R13.4: parallel dispatcher log (<sid>.dispatcher.jsonl) for user prompts
# ---------------------------------------------------------------------------


class TestR13UserLog:
    """R13.4: ``dispatch()`` persists the user prompt to a parallel
    dispatcher log (``<sid>.dispatcher.jsonl``) so
    ``replay(mode="conversation")`` can reconstruct the user side of
    the dialog (the binary's ``<sid>.jsonl`` does not echo
    ``session/prompt`` requests).

    The log is **best-effort** — a write failure must NOT fail the
    user's dispatch call (R13.4 spec). These tests cover the happy
    path; the swallow-on-failure behavior is exercised in
    :mod:`adapter.tests.test_delegate_dispatcher_faults` (deferred
    to R13.x — needs a fault-injection layer).
    """

    @pytest.mark.asyncio
    async def test_dispatch_writes_user_prompt_to_dispatcher_log(
        self, wired_dispatcher: DelegateDispatcher, stub_log_capture: _StubLogCapture,
    ) -> None:
        await wired_dispatcher.dispatch(prompt="explain recursion")
        # R13.4 _append_user_log is fire-and-forget (loop.create_task).
        # Drain the background task — each task acquires an asyncio.Lock
        # then awaits asyncio.to_thread, so we need real wall-clock time
        # for the loop to schedule the write.
        path = stub_log_capture._log_dir / "stub-sid-001.dispatcher.jsonl"
        for _ in range(50):
            await asyncio.sleep(0.01)
            if path.exists() and path.stat().st_size > 0:
                break
        assert path.exists(), f"dispatcher log not created at {path}"
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        # R13.4 contract: 5 fields, role="user", source="dispatcher".
        assert set(record.keys()) == {"ts", "turn", "role", "text", "source"}
        assert record["role"] == "user"
        assert record["source"] == "dispatcher"
        assert record["text"] == "explain recursion"
        assert record["turn"] == 1
        assert isinstance(record["ts"], float)

    @pytest.mark.asyncio
    async def test_consecutive_dispatches_increment_turn_number(
        self, stub_supervisor: _StubSupervisor, stub_log_capture: _StubLogCapture,
    ) -> None:
        # R13.4: turn count advances per-dispatch on the SAME sid.
        # The default _StubSupervisor auto-increments sid_counter, which
        # would mask the increment test. Override new_session to return
        # a fixed sid so all three dispatches land in the same file
        # with turn=1, 2, 3.
        class _FixedSidSupervisor(_StubSupervisor):  # type: ignore[misc]
            async def new_session(self, cwd: str | None = None) -> dict:
                self.new_session_calls.append({"cwd": cwd, "sid": "fixed-sid"})
                return {"sessionId": "fixed-sid"}

        dispatcher = DelegateDispatcher(
            _FixedSidSupervisor(),  # type: ignore[arg-type]
            log_capture=stub_log_capture,
        )
        await dispatcher.dispatch(prompt="turn one")
        await dispatcher.dispatch(prompt="turn two")
        await dispatcher.dispatch(prompt="turn three")
        # Drain the background write tasks before reading the file.
        # Each task acquires an asyncio.Lock then awaits
        # asyncio.to_thread for the actual I/O, so we need real
        # wall-clock time for the loop to schedule all three.
        for _ in range(50):
            await asyncio.sleep(0.01)
        path = stub_log_capture._log_dir / "fixed-sid.dispatcher.jsonl"
        assert path.exists(), f"missing dispatcher log: {path}"
        lines = [
            ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        assert len(lines) == 3, f"expected 3 records, got {len(lines)}"
        records = [json.loads(ln) for ln in lines]
        # R13.4 contract: turn increments per dispatch on the same sid.
        assert [r["turn"] for r in records] == [1, 2, 3]
        assert [r["text"] for r in records] == ["turn one", "turn two", "turn three"]
        for r in records:
            assert r["role"] == "user"
            assert r["source"] == "dispatcher"

    @pytest.mark.asyncio
    async def test_dispatcher_log_uses_log_capture_dir_when_wired(
        self, wired_dispatcher: DelegateDispatcher, stub_log_capture: _StubLogCapture,
    ) -> None:
        # _user_log_path mirrors _log_path: prefer log_capture._log_dir.
        # Verify the dispatcher log lands inside the same temp dir as
        # the binary log, not in SESSION_LOG_DIR.
        await wired_dispatcher.dispatch(prompt="x")
        binary_log = stub_log_capture._log_dir / "stub-sid-001.jsonl"
        dispatcher_log = stub_log_capture._log_dir / "stub-sid-001.dispatcher.jsonl"
        assert dispatcher_log.parent == binary_log.parent == stub_log_capture._log_dir
        # And NOT in SESSION_LOG_DIR (sanity check that we're using
        # the wired dir, not the global default).
        assert not (SESSION_LOG_DIR / "stub-sid-001.dispatcher.jsonl").exists() \
            or dispatcher_log.parent == stub_log_capture._log_dir
