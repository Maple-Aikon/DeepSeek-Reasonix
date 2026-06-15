"""Tests for :mod:`adapter.cost` (R8: 5-field cost shape).

These tests are unit-level: they construct synthetic NDJSON
records (matching the wire shape from Reasonix upstream
`internal/serve/wire.go:70-86`) and feed them to a
``UsageAccumulator``. They do NOT spawn a real Reasonix binary.

End-to-end coverage (real binary, real transcript) lives in
``test_reasonix_status_tool.py`` (R8 part 2) and the supervisor
integration suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapter.cost import (
    CostBreakdown,
    DEFAULT_PHASE,
    UsageAccumulator,
)


# ---- helpers ----


def _phase(text: str) -> dict:
    """Build a synthetic ``phase`` record matching log_capture shape."""
    return {"kind": "phase", "text": text}


def _usage(cost_usd: float, **extra) -> dict:
    """Build a synthetic ``usage`` record matching wire.go shape."""
    rec = {"kind": "usage", "usage": {"costUsd": cost_usd, "cost": cost_usd}}
    rec["usage"].update(extra)
    return rec


def _turn_started() -> dict:
    return {"kind": "turn_started"}


def _write_ndjson(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


# ---- summary shape ----


class TestCostBreakdownShape:
    def test_default_zeros(self):
        bd = CostBreakdown()
        d = bd.to_dict()
        assert d == {
            "planner_usd": 0.0,
            "executor_usd": 0.0,
            "total_usd": 0.0,
            "last_turn_usd": 0.0,
            "last_turn_phase": DEFAULT_PHASE,
        }

    def test_total_is_sum(self):
        bd = CostBreakdown(planner_usd=0.012, executor_usd=0.045)
        assert bd.total_usd == pytest.approx(0.057)
        assert bd.to_dict()["total_usd"] == pytest.approx(0.057)

    def test_round_to_6dp(self):
        bd = CostBreakdown(planner_usd=0.0000001234)
        # Rounded to 6 decimal places (matches currency precision).
        assert bd.to_dict()["planner_usd"] == 0.0


# ---- observe: phase attribution ----


class TestPhaseAttribution:
    def test_default_phase_is_planner(self):
        acc = UsageAccumulator()
        acc.observe(_usage(0.01))
        s = acc.summary()
        assert s.planner_usd == 0.01
        assert s.executor_usd == 0.0
        assert s.last_turn_phase == "planner"

    def test_executor_phase_attribution(self):
        acc = UsageAccumulator()
        acc.observe(_phase("planner · planning"))
        acc.observe(_usage(0.01))  # planner
        acc.observe(_phase("executor · executing"))
        acc.observe(_usage(0.05))  # executor
        s = acc.summary()
        assert s.planner_usd == 0.01
        assert s.executor_usd == 0.05
        assert s.last_turn_phase == "executor"

    def test_phase_label_case_insensitive(self):
        acc = UsageAccumulator()
        acc.observe(_phase("EXECUTOR · executing"))
        acc.observe(_usage(0.02))
        assert acc.summary().executor_usd == 0.02

    def test_unknown_phase_label_keeps_current(self):
        acc = UsageAccumulator()
        acc.observe(_phase("executor · executing"))
        acc.observe(_usage(0.01))
        acc.observe(_phase("mystery phase"))  # unknown
        acc.observe(_usage(0.02))
        s = acc.summary()
        # Both records attributed to executor (unknown label
        # does NOT change the current phase).
        assert s.executor_usd == 0.03
        assert s.planner_usd == 0.0

    def test_planner_then_executor_accumulates_separately(self):
        acc = UsageAccumulator()
        # Turn 1: planner only.
        acc.observe(_phase("planner · planning"))
        acc.observe(_usage(0.005))
        # Turn 2: executor.
        acc.observe(_phase("executor · executing"))
        acc.observe(_usage(0.020))
        # Turn 3: planner again (e.g. mid-session re-plan).
        acc.observe(_phase("planner · replanning"))
        acc.observe(_usage(0.003))
        s = acc.summary()
        assert s.planner_usd == pytest.approx(0.008)
        assert s.executor_usd == pytest.approx(0.020)


# ---- observe: turn tracking ----


class TestTurnTracking:
    def test_turn_started_resets_last_turn(self):
        acc = UsageAccumulator()
        acc.observe(_turn_started())
        acc.observe(_usage(0.10))
        assert acc.summary().last_turn_usd == 0.10

        # New turn: last_turn should be 0.05, NOT 0.15.
        acc.observe(_turn_started())
        acc.observe(_usage(0.05))
        s = acc.summary()
        assert s.last_turn_usd == 0.05
        # Cumulative totals are NOT reset by turn_started.
        assert s.total_usd == pytest.approx(0.15)

    def test_last_turn_carries_current_phase(self):
        acc = UsageAccumulator()
        acc.observe(_turn_started())
        acc.observe(_phase("executor · executing"))
        acc.observe(_usage(0.07))
        s = acc.summary()
        assert s.last_turn_usd == 0.07
        assert s.last_turn_phase == "executor"


# ---- observe: cost extraction ----


class TestCostExtraction:
    def test_prefers_costusd_over_cost(self):
        acc = UsageAccumulator()
        rec = {"kind": "usage", "usage": {"costUsd": 0.05, "cost": 999.0}}
        acc.observe(rec)
        assert acc.summary().planner_usd == 0.05

    def test_falls_back_to_cost_when_costusd_missing(self):
        acc = UsageAccumulator()
        rec = {"kind": "usage", "usage": {"cost": 0.04}}
        acc.observe(rec)
        assert acc.summary().planner_usd == 0.04

    def test_zero_cost_record_is_harmless(self):
        acc = UsageAccumulator()
        acc.observe(_usage(0.0))
        s = acc.summary()
        assert s.planner_usd == 0.0
        assert s.last_turn_usd == 0.0

    def test_negative_cost_is_ignored(self):
        acc = UsageAccumulator()
        acc.observe(_usage(-0.05))
        acc.observe(_usage(0.03))
        s = acc.summary()
        # Negative skipped, positive kept.
        assert s.planner_usd == 0.03

    def test_usage_without_cost_fields_uses_zero(self):
        acc = UsageAccumulator()
        acc.observe({"kind": "usage", "usage": {"promptTokens": 100}})
        # No cost field → treated as zero.
        assert acc.summary().planner_usd == 0.0

    def test_unknown_kind_is_ignored(self):
        acc = UsageAccumulator()
        acc.observe({"kind": "tool_call", "toolName": "read_file"})
        acc.observe({"kind": "message", "text": "hello"})
        acc.observe(_usage(0.01))
        s = acc.summary()
        assert s.planner_usd == 0.01
        # Unrecognized kinds did not crash, did not affect totals.


# ---- load_from_transcript ----


class TestLoadFromTranscript:
    def test_missing_file_returns_zeros(self, tmp_path):
        acc = UsageAccumulator(transcript_path=tmp_path / "nope.jsonl")
        bd = acc.load_from_transcript()
        assert bd.to_dict()["total_usd"] == 0.0

    def test_empty_file_returns_zeros(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        acc = UsageAccumulator(transcript_path=p)
        assert acc.load_from_transcript().total_usd == 0.0

    def test_load_full_session(self, tmp_path):
        p = tmp_path / "s-1.jsonl"
        _write_ndjson(p, [
            _turn_started(),
            _phase("planner · planning"),
            _usage(0.012),
            _turn_started(),
            _phase("executor · executing"),
            _usage(0.045),
            {"kind": "message", "text": "tool result"},
            _turn_started(),
            _usage(0.003),  # still executor (no new phase change)
        ])
        acc = UsageAccumulator(transcript_path=p)
        bd = acc.load_from_transcript()
        d = bd.to_dict()
        assert d["planner_usd"] == 0.012
        assert d["executor_usd"] == pytest.approx(0.048)
        assert d["total_usd"] == pytest.approx(0.060)
        assert d["last_turn_usd"] == 0.003
        assert d["last_turn_phase"] == "executor"

    def test_load_skips_malformed_lines(self, tmp_path):
        p = tmp_path / "broken.jsonl"
        p.write_text(
            json.dumps(_usage(0.01)) + "\n"
            + "{ this is not valid json\n"
            + json.dumps(_usage(0.02)) + "\n",
            encoding="utf-8",
        )
        acc = UsageAccumulator(transcript_path=p)
        bd = acc.load_from_transcript()
        # The 2 valid records are still counted.
        assert bd.planner_usd == pytest.approx(0.03)

    def test_load_skips_blank_lines(self, tmp_path):
        p = tmp_path / "blanks.jsonl"
        p.write_text(
            "\n"
            + json.dumps(_usage(0.01)) + "\n"
            + "\n"
            + json.dumps(_usage(0.02)) + "\n"
            + "\n",
            encoding="utf-8",
        )
        acc = UsageAccumulator(transcript_path=p)
        assert acc.load_from_transcript().planner_usd == pytest.approx(0.03)
