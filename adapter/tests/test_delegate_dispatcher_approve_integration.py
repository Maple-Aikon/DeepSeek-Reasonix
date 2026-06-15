"""R12 — end-to-end integration tests for the ``plan_mode="approve"`` flow.

These tests spawn a *real* ``Supervisor`` (against the
``fake_reasonix.py`` fixture script) and drive
``DelegateDispatcher.dispatch(plan_mode="approve")`` end-to-end. They
complement the R8/R10/R11 unit tests by exercising the full stack:

  1. ``Supervisor`` spawn + ACP ``session/new`` + ``session/prompt``
  2. ``DelegateDispatcher`` race between ``on_approve`` callback and
     the auto-reject timeout (we override ``approve_timeout_s`` to a
     short window so the tests stay fast)
  3. MCP tool layer (``reasonix_approve``) — the ``set_dispatcher`` /
     ``clear_dispatcher`` slot wiring through to a real ``Supervisor``

We use the ``fake_reasonix.py`` fixture (not the real binary) so the
tests run hermetically — no API key, no LLM cost, no network. The
fake binary's transcript does not emit ``phase`` / ``usage`` records,
so ``UsageAccumulator`` returns a zero breakdown; we therefore assert
the *shape* and the *state-machine transitions* (status field,
decision field) but not the dollar amounts (those are R8 unit-test
territory, see ``test_delegate_dispatcher_approve.py``).

These tests are marked ``@pytest.mark.integration`` so they can be
skipped with ``-m "not integration"`` during normal unit-test runs.

Skip conditions
---------------
- ``fake_reasonix.py`` fixture missing
- ``DelegateDispatcher`` or ``reasonix_approve`` module missing
  (defensive guard for early CI before R8/R10 ship)
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Conditional imports — skip cleanly if R8/R10 haven't shipped yet
# ---------------------------------------------------------------------------

try:
    from adapter.supervisor import Supervisor  # noqa: E402
    from adapter.tools.reasonix_delegate import DelegateDispatcher  # noqa: E402
    from adapter.tools.reasonix_approve import (  # noqa: E402
        reasonix_approve,
        set_dispatcher,
        clear_dispatcher,
    )
    _IMPORTS_OK = True
except ImportError as e:  # pragma: no cover
    _IMPORTS_OK = False
    _IMPORT_ERR = repr(e)


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

FAKE_BIN = Path(__file__).parent / "fixtures" / "fake_reasonix.py"


@pytest.fixture
def fake_bin():
    """Skip the suite if fake_reasonix.py is missing or imports fail."""
    if not FAKE_BIN.exists():
        pytest.skip(f"fake_reasonix fixture not found at {FAKE_BIN}")
    if not _IMPORTS_OK:
        pytest.skip(f"adapter modules unavailable: {_IMPORT_ERR}")
    return FAKE_BIN


def _env_for_fake(**extra):
    """Build an env dict that points fake_reasonix into ``happy`` mode.

    ``happy`` mode emits 1 ``agent_message_chunk`` + 1 ``stop`` per
    ``session/prompt`` call. With ``FAKE_REASONIX_PROMPT_DELAY=0``
    the whole cycle takes a few ms, well under our test timeouts.
    """
    env = {
        "FAKE_REASONIX_MODE": "happy",
        "FAKE_REASONIX_VERBOSE": "0",
        "FAKE_REASONIX_PROMPT_DELAY": "0",
        "PATH": os.environ.get("PATH", ""),
    }
    env.update(extra)
    return env


async def _cleanup_sup(sup) -> None:
    """Idempotent close helper. Never raises."""
    try:
        await sup.close()
    except Exception:  # pragma: no cover
        pass


def _make_cost_zero_check(cost: dict) -> None:
    """Shared helper: assert the 5-field cost shape, allow all-zero.

    The fake binary doesn't emit usage records so all monetary
    fields are 0.0 — the *shape* is what we verify here.
    """
    assert isinstance(cost, dict), f"cost must be dict, got {type(cost)}"
    for key in ("planner_usd", "executor_usd", "total_usd",
                "last_turn_usd", "last_turn_phase"):
        assert key in cost, f"cost missing field {key!r}: {cost}"
    assert cost["planner_usd"] == pytest.approx(0.0)
    assert cost["executor_usd"] == pytest.approx(0.0)
    assert cost["total_usd"] == pytest.approx(0.0)


async def _wait_for_pending_slot(
    dispatcher: DelegateDispatcher, *, timeout_s: float = 1.0,
) -> Optional[str]:
    """Poll ``_pending_approves`` until exactly one sid is registered.

    Returns the sid, or ``None`` if the timeout elapses. The fake
    binary takes a few ms per prompt cycle, so we poll every 5ms.
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        pending = dispatcher._pending_approves  # type: ignore[attr-defined]
        if len(pending) == 1:
            return next(iter(pending))
        if len(pending) > 1:
            # Multiple sessions pending — caller should disambiguate
            pytest.fail(
                f"expected exactly 1 pending slot, got {len(pending)}: "
                f"{list(pending.keys())}",
            )
        await asyncio.sleep(0.005)
    return None


# ===========================================================================
# Test 1: happy path — manual approve lands before timeout
# ===========================================================================

@pytest.mark.integration
async def test_integration_approve_happy_path(fake_bin, tmp_path):
    """End-to-end: dispatch(plan_mode=approve) + on_approve("approve")
    in 50ms, executor follow-up prompt completes, status=completed.
    """
    sup = Supervisor(
        binary=fake_bin,
        env=_env_for_fake(),
        stderr_log=tmp_path / "stderr.log",
        init_timeout=5.0,
        prompt_timeout=10.0,
    )
    try:
        await sup.start()
        dispatcher = DelegateDispatcher(sup, approve_timeout_s=2.0)

        dispatch_task = asyncio.create_task(dispatcher.dispatch("say hi", plan_mode="approve"))
        sid = await _wait_for_pending_slot(dispatcher)
        assert sid is not None, "dispatch did not register a pending slot"

        accepted = dispatcher.on_approve(sid, "approve")
        assert accepted is True

        result = await dispatch_task
        assert result["sid"] == sid
        assert result["plan_mode"] == "approve"
        assert result["decision"] == "approve"
        _make_cost_zero_check(result["cost"])

        status = dispatcher.status(sid)
        assert status["status"] == "completed"
        assert status["plan_mode"] == "approve"
    finally:
        await _cleanup_sup(sup)


# ===========================================================================
# Test 2: manual reject — no executor phase runs
# ===========================================================================

@pytest.mark.integration
async def test_integration_approve_manual_reject(fake_bin, tmp_path):
    """Drive reject: dispatch returns with decision="reject",
    status=rejected, executor follow-up prompt is NOT sent.
    """
    sup = Supervisor(
        binary=fake_bin,
        env=_env_for_fake(),
        stderr_log=tmp_path / "stderr.log",
        init_timeout=5.0,
        prompt_timeout=10.0,
    )
    try:
        await sup.start()
        dispatcher = DelegateDispatcher(sup, approve_timeout_s=2.0)

        dispatch_task = asyncio.create_task(dispatcher.dispatch("say hi", plan_mode="approve"))
        sid = await _wait_for_pending_slot(dispatcher)
        assert sid is not None

        accepted = dispatcher.on_approve(sid, "reject")
        assert accepted is True

        result = await dispatch_task
        assert result["decision"] == "reject"
        _make_cost_zero_check(result["cost"])

        status = dispatcher.status(sid)
        assert status["status"] == "rejected"
        # Slot must be cleaned up after dispatch returns
        assert sid not in dispatcher._pending_approves  # type: ignore[attr-defined]
    finally:
        await _cleanup_sup(sup)


# ===========================================================================
# Test 3: auto-timeout — no approval within window → auto-reject
# ===========================================================================

@pytest.mark.integration
async def test_integration_approve_auto_timeout_rejects(fake_bin, tmp_path):
    """Drive auto-timeout: no on_approve call within the window,
    dispatcher auto-rejects and returns with decision="reject".
    """
    sup = Supervisor(
        binary=fake_bin,
        env=_env_for_fake(),
        stderr_log=tmp_path / "stderr.log",
        init_timeout=5.0,
        prompt_timeout=10.0,
    )
    try:
        await sup.start()
        # Tight 100ms timeout — the fake binary's initial prompt
        # takes ~5-10ms, so 100ms gives the slot time to register
        # AND the timeout enough headroom to fire reliably.
        dispatcher = DelegateDispatcher(sup, approve_timeout_s=0.1)

        result = await dispatcher.dispatch("say hi", plan_mode="approve")

        assert result["decision"] == "reject"
        _make_cost_zero_check(result["cost"])
        status = dispatcher.status(result["sid"])
        assert status["status"] == "rejected"
    finally:
        await _cleanup_sup(sup)


# ===========================================================================
# Test 4: feedback passthrough — reject with feedback string is recorded
# ===========================================================================

@pytest.mark.integration
async def test_integration_approve_reject_with_feedback(fake_bin, tmp_path):
    """Reject with a non-None feedback string. The dispatcher must
    accept the call and return decision="reject" (the follow-up is
    not sent on reject, so feedback is not used here, but the API
    surface should accept it without error).
    """
    sup = Supervisor(
        binary=fake_bin,
        env=_env_for_fake(),
        stderr_log=tmp_path / "stderr.log",
        init_timeout=5.0,
        prompt_timeout=10.0,
    )
    try:
        await sup.start()
        dispatcher = DelegateDispatcher(sup, approve_timeout_s=2.0)

        dispatch_task = asyncio.create_task(dispatcher.dispatch("say hi", plan_mode="approve"))
        sid = await _wait_for_pending_slot(dispatcher)
        assert sid is not None

        accepted = dispatcher.on_approve(sid, "reject", feedback="please retry")
        assert accepted is True

        result = await dispatch_task
        assert result["decision"] == "reject"
    finally:
        await _cleanup_sup(sup)


# ===========================================================================
# Test 5: MCP slot wiring — reasonix_approve drives real dispatcher e2e
# ===========================================================================

@pytest.mark.integration
async def test_integration_mcp_approve_slot_drives_dispatcher(fake_bin, tmp_path):
    """End-to-end through the MCP tool layer: register a real
    dispatcher via ``set_dispatcher``, call ``reasonix_approve``,
    verify the slot fires and the underlying session resolves.
    """
    sup = Supervisor(
        binary=fake_bin,
        env=_env_for_fake(),
        stderr_log=tmp_path / "stderr.log",
        init_timeout=5.0,
        prompt_timeout=10.0,
    )
    try:
        await sup.start()
        dispatcher = DelegateDispatcher(sup, approve_timeout_s=2.0)
        set_dispatcher(dispatcher)

        try:
            dispatch_task = asyncio.create_task(dispatcher.dispatch("say hi", plan_mode="approve"))
            sid = await _wait_for_pending_slot(dispatcher)
            assert sid is not None

            # Drive the approval through the MCP tool entry point
            result = reasonix_approve(sid, "approve")
            assert result["accepted"] is True
            assert result["decision"] == "approve"
            assert result["sid"] == sid

            final = await dispatch_task
            assert final["decision"] == "approve"
            assert final["sid"] == sid
        finally:
            clear_dispatcher()
    finally:
        await _cleanup_sup(sup)
