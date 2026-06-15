"""R8: Cost accumulator for Reasonix sessions.

Computes the 5-field cost shape (`planner_usd`, `executor_usd`,
`total_usd`, `last_turn_usd`, `last_turn_phase`) from a session's
transcript JSONL. Designed to back the `reasonix_status` MCP tool.

Path A (chosen for R8): parse the session's transcript file (NDJSON
written by `log_capture.LogCapture`) and extract `usage` records.
Path B (deferred, requires upstream PR): consume `event.Usage`
notifications live from the ACP connection. We pick A for R8 ship
speed; B is the right long-term shape but needs Reasonix upstream
changes (currently `event.Usage` is dropped from the ACP v2
session/update stream — see `internal/acp/dispatch.go:38-40`).

Cost formula
------------
The wire shape (Reasonix upstream `internal/serve/wire.go:70-86`)
emits `usage.costUsd` and `usage.cost` per turn when `Pricing` is
attached to the Usage event. We prefer `costUsd` because the
`symbol`/`currency` field is model-specific (some models use
custom units). Both are per-turn; the accumulator sums them.

Phase attribution
-----------------
`event.Phase` notifications carry a label like ``"planner · planning"``
or ``"executor · executing"`` (see coordinator.go). We track the last
phase seen and tag each `usage` record with the current phase. This
gives the caller a clean split between planner cost and executor
cost without per-tool attribution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


#: Default phase when a usage record arrives before any phase
#: notification (e.g. the very first turn of a session).
DEFAULT_PHASE: str = "planner"


@dataclass
class CostBreakdown:
    """The 5-field cost shape returned by :meth:`UsageAccumulator.summary`.

    All monetary fields are in USD (the wire field is ``costUsd``).
    Per-turn fields are reset on each call to :meth:`reset_last_turn`.
    """

    planner_usd: float = 0.0
    executor_usd: float = 0.0
    last_turn_usd: float = 0.0
    last_turn_phase: str = DEFAULT_PHASE

    @property
    def total_usd(self) -> float:
        """Sum of planner and executor totals (computed, not stored)."""
        return self.planner_usd + self.executor_usd

    def to_dict(self) -> dict:
        return {
            "planner_usd": round(self.planner_usd, 6),
            "executor_usd": round(self.executor_usd, 6),
            "total_usd": round(self.total_usd, 6),
            "last_turn_usd": round(self.last_turn_usd, 6),
            "last_turn_phase": self.last_turn_phase,
        }


@dataclass
class UsageAccumulator:
    """Track per-session cost split by phase (planner vs executor).

    Construct with an open transcript path. Call :meth:`observe` for
    each ``usage`` or ``phase`` record you read from the NDJSON
    stream. Call :meth:`summary` to get the 5-field shape.

    The accumulator is intentionally simple: it scans the file once
    and stores cumulative + last-turn values. It does NOT subscribe
    to live notifications (that's Path B; deferred).

    Example:
        acc = UsageAccumulator(transcript_path=Path("/tmp/s-1.jsonl"))
        with acc.transcript_path.open() as f:
            for line in f:
                rec = json.loads(line)
                acc.observe(rec)
        breakdown = acc.summary()
        # {"planner_usd": 0.012, "executor_usd": 0.045,
        #  "total_usd": 0.057, "last_turn_usd": 0.008,
        #  "last_turn_phase": "executor"}
    """

    transcript_path: Optional[Path] = None

    # Internal state (exposed for tests, not for callers).
    _planner_total: float = field(default=0.0, init=False)
    _executor_total: float = field(default=0.0, init=False)
    _last_turn_usd: float = field(default=0.0, init=False)
    _last_turn_phase: str = field(default=DEFAULT_PHASE, init=False)
    _current_phase: str = field(default=DEFAULT_PHASE, init=False)
    _turn_started_seen: bool = field(default=False, init=False)

    # ---- public API ----

    def observe(self, record: dict) -> None:
        """Fold one NDJSON record into the accumulator.

        Recognized record shapes (all carrying ``kind`` from
        :class:`LogCapture`):

        - ``{"kind": "phase", "text": "planner · planning", ...}`` —
          updates the current phase; usage records observed next will
          be attributed to this phase.
        - ``{"kind": "phase", "text": "executor · executing", ...}`` —
          same, for executor.
        - ``{"kind": "turn_started", ...}`` — resets the last-turn
          accumulator so the next usage record starts a new turn.
        - ``{"kind": "usage", "usage": {...wireUsage...}, ...}`` —
          adds the per-turn cost to the current phase's total and
          overwrites the last-turn snapshot.
        - Any other kind — ignored (forward-compatible with new
          notification kinds the upstream may add).
        """
        kind = record.get("kind")
        if kind == "phase":
            self._update_phase(record)
        elif kind == "turn_started":
            self._reset_last_turn()
        elif kind == "usage":
            self._record_usage(record)
        # else: ignore (kind may be "message", "tool_call", etc.)

    def summary(self) -> CostBreakdown:
        """Return the 5-field cost shape for this session."""
        return CostBreakdown(
            planner_usd=self._planner_total,
            executor_usd=self._executor_total,
            last_turn_usd=self._last_turn_usd,
            last_turn_phase=self._last_turn_phase,
        )

    def load_from_transcript(self) -> CostBreakdown:
        """Scan the transcript NDJSON end-to-end and return the summary.

        Convenience method: opens the file, calls :meth:`observe` for
        each line, and returns the resulting breakdown. Silently skips
        malformed lines (log them at WARNING).

        Returns:
            :class:`CostBreakdown` with the session's cost shape. If
            the transcript doesn't exist or is empty, returns a zero
            breakdown.
        """
        if self.transcript_path is None or not self.transcript_path.exists():
            log.warning("transcript path missing: %s", self.transcript_path)
            return CostBreakdown()

        with self.transcript_path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError as e:
                    log.warning(
                        "skipping malformed transcript line %d: %s",
                        lineno, e,
                    )
                    continue
                self.observe(rec)

        return self.summary()

    # ---- internals ----

    def _update_phase(self, record: dict) -> None:
        """Switch the current phase based on a ``phase`` notification.

        The text format follows Reasonix upstream's coordinator.go:
        ``"planner · planning"`` or ``"executor · executing"``. We
        match the leading word case-insensitively so we tolerate
        future format tweaks (e.g. ``"Planner · planning"``).
        """
        text = (record.get("text") or "").strip().lower()
        if text.startswith("planner"):
            self._current_phase = "planner"
        elif text.startswith("executor"):
            self._current_phase = "executor"
        else:
            # Unknown phase label — keep the current phase unchanged.
            log.debug("unknown phase label, keeping %r: %r",
                      self._current_phase, text)

    def _reset_last_turn(self) -> None:
        """Zero the last-turn snapshot for the next usage record."""
        self._last_turn_usd = 0.0
        self._last_turn_phase = self._current_phase
        self._turn_started_seen = True

    def _record_usage(self, record: dict) -> None:
        """Add this turn's cost to the current phase's total."""
        usage = record.get("usage") or {}
        # Prefer `costUsd` (clearer unit), fall back to `cost`.
        # The wire emits both when Pricing is attached; `costUsd` is
        # kept for older status consumers (see wire.go:84-86).
        cost = float(usage.get("costUsd") or usage.get("cost") or 0.0)
        if cost < 0:
            log.warning("negative cost in usage record, ignoring: %r", cost)
            return

        if self._current_phase == "planner":
            self._planner_total += cost
        else:
            self._executor_total += cost
        self._last_turn_usd = cost
        self._last_turn_phase = self._current_phase
