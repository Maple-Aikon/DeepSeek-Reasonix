"""Tests for R10 — ``reasonix_approve`` MCP tool.

R10 exposes :meth:`DelegateDispatcher.on_approve` as a top-level
MCP tool. Mirrors the R7/R9 pattern: thin wrapper over the
dispatcher method, with a module-level slot for the live
dispatcher that the MCP server registers at boot.

Test scope:
  1. Module-level slot lifecycle (set/clear/thread-safety)
  2. MCP tool happy path: approve / reject
  3. MCP tool error paths:
     - no dispatcher registered
     - unknown sid
     - invalid decision (not in {"approve","reject"})
     - sid without pending approval (already resolved / not in approve mode)
  4. Backward compat: tool layer accepts decision via param (not
     the dispatcher's ``on_approve(sid, decision, feedback)`` order)
  5. Feedback passthrough (None default, custom string)
  6. Return shape contract (5-field result: {sid, decision, accepted,
     feedback_provided, reason})
"""
from __future__ import annotations

import threading

import pytest

from adapter.tools import reasonix_approve as approve_mod
from adapter.tools.reasonix_approve import (
    clear_dispatcher,
    reasonix_approve,
    set_dispatcher,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeDispatcher:
    """Stand-in for :class:`DelegateDispatcher`.

    Records ``on_approve`` calls and returns a programmable
    result. Lighter than mocking the real class.
    """

    def __init__(self, result: bool = True):
        self.calls: list[tuple[str, str, str | None]] = []
        self._result = result

    def on_approve(
        self,
        sid: str,
        decision: str,
        feedback: str | None = None,
    ) -> bool:
        self.calls.append((sid, decision, feedback))
        return self._result


@pytest.fixture
def dispatcher():
    """Yield a fresh FakeDispatcher + register it for the test.

    Auto-clears the slot after the test so a leaked dispatcher
    cannot pollute the next test.
    """
    clear_dispatcher()
    fd = FakeDispatcher()
    set_dispatcher(fd)
    yield fd
    clear_dispatcher()


# ---------------------------------------------------------------------------
# 1. Module-level slot lifecycle
# ---------------------------------------------------------------------------


class TestSlotLifecycle:
    def test_set_dispatcher_stores_instance(self, dispatcher):
        with approve_mod._dispatcher_lock:
            assert approve_mod._dispatcher is dispatcher

    def test_clear_dispatcher_removes_instance(self, dispatcher):
        clear_dispatcher()
        with approve_mod._dispatcher_lock:
            assert approve_mod._dispatcher is None

    def test_set_replaces_existing_dispatcher(self):
        clear_dispatcher()
        a = FakeDispatcher()
        b = FakeDispatcher()
        set_dispatcher(a)
        set_dispatcher(b)  # no error, replaces
        with approve_mod._dispatcher_lock:
            assert approve_mod._dispatcher is b
        clear_dispatcher()

    def test_clear_when_already_none_is_noop(self):
        clear_dispatcher()
        # Second clear should not raise
        clear_dispatcher()
        with approve_mod._dispatcher_lock:
            assert approve_mod._dispatcher is None

    def test_set_dispatcher_is_thread_safe(self):
        """Concurrent ``set_dispatcher`` calls from N threads must
        leave the slot referencing one of the supplied dispatchers
        (no torn write, no crash)."""
        clear_dispatcher()
        dispatchers = [FakeDispatcher() for _ in range(8)]
        threads = [
            threading.Thread(target=set_dispatcher, args=(d,))
            for d in dispatchers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        with approve_mod._dispatcher_lock:
            assert approve_mod._dispatcher in dispatchers
        clear_dispatcher()


# ---------------------------------------------------------------------------
# 2. Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_approve_returns_accepted_dict(self, dispatcher):
        result = reasonix_approve(sid="sid-1", decision="approve")
        assert result == {
            "sid": "sid-1",
            "decision": "approve",
            "accepted": True,
            "feedback_provided": False,
            "reason": "session_unblocked",
        }

    def test_reject_returns_accepted_dict(self, dispatcher):
        result = reasonix_approve(sid="sid-2", decision="reject")
        assert result == {
            "sid": "sid-2",
            "decision": "reject",
            "accepted": True,
            "feedback_provided": False,
            "reason": "session_unblocked",
        }

    def test_dispatcher_on_approve_invoked_with_correct_args(
        self, dispatcher
    ):
        reasonix_approve(
            sid="sid-3",
            decision="approve",
            feedback="please add error handling",
        )
        assert dispatcher.calls == [
            ("sid-3", "approve", "please add error handling"),
        ]

    def test_feedback_none_default(self, dispatcher):
        reasonix_approve(sid="sid-4", decision="approve")
        assert dispatcher.calls == [("sid-4", "approve", None)]

    def test_feedback_provided_flag_set_when_string(self, dispatcher):
        result = reasonix_approve(
            sid="sid-5", decision="approve", feedback="ok go"
        )
        assert result["feedback_provided"] is True

    def test_feedback_empty_string_treated_as_not_provided(
        self, dispatcher
    ):
        """Empty feedback is normalized to None and flagged False.

        The dispatcher's on_approve still receives the empty string
        (preserves caller's intent) but the tool layer reports
        ``feedback_provided=False`` so the LLM caller can tell.
        """
        result = reasonix_approve(
            sid="sid-6", decision="approve", feedback=""
        )
        assert result["feedback_provided"] is False
        assert dispatcher.calls == [("sid-6", "approve", "")]


# ---------------------------------------------------------------------------
# 3. Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_no_dispatcher_returns_error_dict(self):
        clear_dispatcher()
        result = reasonix_approve(sid="sid-x", decision="approve")
        assert result == {
            "sid": "sid-x",
            "decision": "approve",
            "accepted": False,
            "feedback_provided": False,
            "reason": "no_dispatcher",
            "error": "no_dispatcher",
            "message": (
                "reasonix_approve called before MCP server boot or "
                "after shutdown; no DelegateDispatcher registered."
            ),
        }

    def test_unknown_sid_returns_not_pending(self, dispatcher):
        dispatcher._result = False  # dispatcher says no pending
        result = reasonix_approve(sid="sid-unknown", decision="approve")
        assert result == {
            "sid": "sid-unknown",
            "decision": "approve",
            "accepted": False,
            "feedback_provided": False,
            "reason": "no_pending_approval",
        }
        # Dispatcher was still called (it has the final say)
        assert dispatcher.calls == [("sid-unknown", "approve", None)]

    def test_invalid_decision_rejected_at_tool_layer(
        self, dispatcher
    ):
        """Tool layer enforces decision ∈ {"approve","reject"}.

        Mirrors R7's strict mutual-exclusivity at the tool boundary
        (PersonaResolver is permissive underneath, but the MCP tool
        layer validates). This prevents silent typos like
        ``"aprove"`` from leaking into the dispatcher.
        """
        result = reasonix_approve(
            sid="sid-bad", decision="aprove"  # typo
        )
        assert result == {
            "sid": "sid-bad",
            "decision": "aprove",
            "accepted": False,
            "feedback_provided": False,
            "reason": "invalid_decision",
            "error": "invalid_decision",
            "message": (
                "decision must be 'approve' or 'reject', got 'aprove'"
            ),
        }
        # Dispatcher was NOT called (validation happened first)
        assert dispatcher.calls == []

    def test_empty_decision_rejected(self, dispatcher):
        result = reasonix_approve(sid="sid-empty", decision="")
        assert result["reason"] == "invalid_decision"
        assert result["accepted"] is False
        assert dispatcher.calls == []

    def test_race_resolved_dispatcher_returns_false(self, dispatcher):
        """If the caller already resolved via another channel, the
        dispatcher returns False; the tool surfaces ``no_pending_approval``.
        """
        dispatcher._result = False
        result = reasonix_approve(sid="sid-raced", decision="approve")
        assert result["accepted"] is False
        assert result["reason"] == "no_pending_approval"


# ---------------------------------------------------------------------------
# 4. Return shape contract
# ---------------------------------------------------------------------------


class TestReturnShape:
    def test_keys_present_on_success(self, dispatcher):
        result = reasonix_approve(sid="sid-shape", decision="approve")
        expected = {
            "sid", "decision", "accepted", "feedback_provided", "reason"
        }
        assert set(result.keys()) == expected

    def test_keys_present_on_no_dispatcher(self):
        clear_dispatcher()
        result = reasonix_approve(sid="sid-nd", decision="approve")
        expected = {
            "sid", "decision", "accepted", "feedback_provided",
            "reason", "error", "message",
        }
        assert set(result.keys()) == expected

    def test_keys_present_on_invalid_decision(self, dispatcher):
        result = reasonix_approve(sid="sid-x", decision="reject!")
        expected = {
            "sid", "decision", "accepted", "feedback_provided",
            "reason", "error", "message",
        }
        assert set(result.keys()) == expected

    def test_sid_always_echoed(self, dispatcher):
        """Every error/success path preserves the caller's sid."""
        for sid in ["", "s", "sid-with-dashes", "x" * 256]:
            result = reasonix_approve(sid=sid, decision="approve")
            assert result["sid"] == sid
