"""MCP tool: ``reasonix_delegate`` (R7 stub + R8 real dispatch).

R7 (shipped 2026-06-15, commit c7b8e5ed):
  Stub layer that validates inputs, resolves the persona, and
  returns a fake session id. Does NOT spawn a real Reasonix binary.

R8 (in this file):
  Adds a thin ``DelegateDispatcher`` class that wires the tool to
  a real :class:`adapter.supervisor.Supervisor`. The MCP tool entry
  point (``reasonix_delegate`` function) still works as a pure
  function (backward compatible with R7 tests); the class-based
  ``DelegateDispatcher.dispatch()`` is what the MCP server will
  register in the broader PicoClaw integration.

Contract (locked 2026-06-15):
  - 3 plan_modes: ``auto`` (default), ``skip``, ``approve``
  - 3 persona params, mutually exclusive at this layer:
      persona_prompt (raw) > persona_file (custom path) > persona
  - Empty prompt → ValueError
  - Returns: dict with ``sid``, ``log_path``, ``persona_name``,
    ``plan_mode``, and ``system_prompt_preview`` (truncated to 200
    chars), plus (R8) ``cost`` 5-field shape when a supervisor is
    provided.

The tool is registered with the MCP server in ``adapter/mcp_server.py``
(not yet created; that comes with the broader PicoClaw integration).
"""

from __future__ import annotations

import logging
import json
import os
import secrets
import tempfile
import time
import asyncio
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from adapter.persona import Persona, PersonaResolver, PERSONAS_DIR

if TYPE_CHECKING:  # pragma: no cover
    # Avoid runtime import: supervisor imports persona only lazily.
    from adapter.supervisor import Supervisor

log = logging.getLogger(__name__)


#: The 3 plan_mode values accepted by this tool.
VALID_PLAN_MODES: frozenset[str] = frozenset({"auto", "skip", "approve"})

#: Default plan_mode when the caller does not specify one.
DEFAULT_PLAN_MODE: str = "auto"

#: Maximum length of the system_prompt_preview field in the return dict.
_PREVIEW_CHARS: int = 200

#: Max auto-approve wait time for ``plan_mode="approve"`` paused sessions.
#: When the user does not call ``reasonix_approve`` within this window,
#: the dispatcher auto-rejects the plan (race option X — approve wins,
#: but if no approve arrives, the pending plan is dropped, not executed).
APPROVE_TIMEOUT_S: float = 300.0


def _validate_inputs(
    prompt: str,
    plan_mode: str,
    persona: Optional[str],
    persona_file: Optional[str],
    persona_prompt: Optional[str],
) -> None:
    """Shared validation for both the stub function and the class.

    Raises :class:`ValueError` for empty prompt, invalid plan_mode, or
    mutually-exclusive persona params.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if plan_mode not in VALID_PLAN_MODES:
        raise ValueError(
            f"plan_mode must be one of {sorted(VALID_PLAN_MODES)} "
            f"(got {plan_mode!r})"
        )
    set_params = [p for p in (persona, persona_file, persona_prompt) if p is not None]
    if len(set_params) > 1:
        raise ValueError(
            f"persona params are mutually exclusive: pass at most one of "
            f"persona/persona_file/persona_prompt (got {len(set_params)})"
        )


def _resolve(persona, persona_file, persona_prompt) -> Persona:
    """Resolve persona via the silent-precedence chain.

    The MCP tool layer has already enforced mutual exclusivity; this
    call never sees more than one of the three set.
    """
    return PersonaResolver.resolve(
        persona=persona,
        persona_file=persona_file,
        persona_prompt=persona_prompt,
    )


def _preview(system_prompt: str) -> str:
    """Truncate a system prompt to ``_PREVIEW_CHARS`` for the return dict."""
    if len(system_prompt) <= _PREVIEW_CHARS:
        return system_prompt
    return system_prompt[:_PREVIEW_CHARS] + "..."


def reasonix_delegate(
    prompt: str,
    cwd: Optional[str] = None,
    persona: Optional[str] = None,
    persona_file: Optional[str] = None,
    persona_prompt: Optional[str] = None,
    plan_mode: str = DEFAULT_PLAN_MODE,
) -> dict:
    """Validate inputs, resolve persona, return a fake session id.

    R7 stub. The real Reasonix dispatch lives in
    :class:`DelegateDispatcher.dispatch` (R8). The MCP server wires
    the tool to the dispatcher when a Supervisor is available;
    without one, the function still validates and returns a stub
    response (useful for unit tests, dry runs, and the broader
    integration before the binary is online).

    Args:
        prompt: the user task to delegate. Must be non-empty.
        cwd: optional working directory for the Reasonix session.
            Reserved for R8 (real dispatch); the stub records it
            in the return dict only.
        persona: built-in persona name (e.g. ``"scaffold"``).
            Mutually exclusive with ``persona_file`` and ``persona_prompt``.
        persona_file: path to a custom .md file. Mutually exclusive
            with ``persona`` and ``persona_prompt``.
        persona_prompt: raw system-prompt text. Mutually exclusive
            with ``persona`` and ``persona_file``.
        plan_mode: one of ``"auto"`` (default), ``"skip"``, ``"approve"``.

    Returns:
        dict with keys:
            - ``sid`` (str): fake session id like ``"stub-abc123"``
            - ``log_path`` (str): reserved path under tmpdir
            - ``persona_name`` (str): the resolved persona's name
            - ``plan_mode`` (str): the validated plan_mode
            - ``cwd`` (str or None): the cwd arg, as-passed
            - ``system_prompt_preview`` (str): first 200 chars of the
              resolved persona's system_prompt (truncated with "...").

    Raises:
        ValueError: if ``prompt`` is empty, ``plan_mode`` is invalid, or
            more than one of the persona params is set.
    """
    _validate_inputs(prompt, plan_mode, persona, persona_file, persona_prompt)
    resolved = _resolve(persona, persona_file, persona_prompt)

    sid = f"stub-{secrets.token_hex(6)}"
    tmp_root = Path(tempfile.gettempdir())
    log_path = str(tmp_root / f"reasonix-{sid}.log")

    return {
        "sid": sid,
        "log_path": log_path,
        "persona_name": resolved.name,
        "plan_mode": plan_mode,
        "cwd": cwd,
        "system_prompt_preview": _preview(resolved.system_prompt),
    }


class DelegateDispatcher:
    """R8: real Reasonix dispatch wired to a :class:`Supervisor`.

    Use this from the MCP server layer (or any caller that has a
    live :class:`Supervisor`). The class owns:

      1. ``session/new`` (via :meth:`Supervisor.new_session`)
      2. ``session/prompt`` (via :meth:`Supervisor.prompt`)
      3. ``session/approve`` flow (only when ``plan_mode="approve"``)
      4. Cost accumulation per session (via :mod:`adapter.cost`)

    Composition mirrors :class:`adapter.task_dispatcher.TaskDispatcher`:
    the dispatcher takes a Supervisor and (optionally) a LogCapture
    and wires ``supervisor.on_notification = log_capture.handler()``
    so per-session NDJSON transcripts land under the same
    ``SESSION_LOG_DIR`` that ``reasonix_status`` and ``reasonix_replay``
    read from. If ``log_capture`` is ``None`` (e.g. unit tests with a
    mock supervisor), the dispatcher runs in "blind" mode: dispatch
    still works, but ``log_path`` resolves to the canonical
    ``SESSION_LOG_DIR`` and ``cost`` is the zero breakdown.

    Lifecycle::

        dispatcher = DelegateDispatcher(supervisor, log_capture=log_cap)
        # Optional: install an approval callback
        dispatcher.on_approve(sid, decision, feedback=None)

        # Delegate a task
        result = await dispatcher.dispatch(
            prompt="...task...",
            cwd="/some/dir",
            persona="scaffold",
            plan_mode="auto",
        )
        # result["sid"] is a real session id; result["cost"] is the
        # 5-field shape from :class:`adapter.cost.CostBreakdown`.

    Args:
        supervisor: a started, ready :class:`Supervisor` instance.
        log_capture: optional :class:`adapter.log_capture.LogCapture`.
            When provided, the dispatcher wires it as the supervisor's
            notification handler so per-session JSONL transcripts are
            written for cost parsing. If ``None``, dispatch still
            works but cost will be zero and log_path will be the
            canonical ``SESSION_LOG_DIR`` (which may not exist yet).
        approve_timeout_s: how long ``plan_mode="approve"`` waits for
            :meth:`on_approve` before auto-rejecting. Defaults to
            :data:`APPROVE_TIMEOUT_S` (300s, per spec 2026-06-15 13:30).
    """

    def __init__(
        self,
        supervisor: "Supervisor",
        *,
        log_capture: Optional[Any] = None,
        approve_timeout_s: float = APPROVE_TIMEOUT_S,
    ) -> None:
        self._sup = supervisor
        self._cap = log_capture
        self._approve_timeout = approve_timeout_s
        # Map sid -> dict with {"event": asyncio.Event, "decision": str | None,
        #                      "feedback": str | None, "fired": bool}
        # so concurrent "approve" calls for different sessions don't race.
        self._pending_approves: dict[str, dict] = {}

        # R9: Session map. sid -> SessionRecord (see SessionRecord below).
        # Populated in dispatch(), read by status() / replay(). Cleared
        # in close() (and naturally bounded by Reasonix session lifetime).
        self._sessions: dict[str, "SessionRecord"] = {}

        # R13.4: per-path asyncio locks for the parallel dispatcher log
        # (one per <sid>.dispatcher.jsonl). Created lazily in
        # _user_log_lock_for. Used to serialize concurrent
        # _append_user_log calls on the same sid so the conversation
        # timeline can never be read mid-write by replay(mode="conversation").
        self._user_log_locks: dict = {}

        # If a log_capture was provided, wire it as the supervisor's
        # notification handler. This is the same pattern TaskDispatcher
        # uses (L174 in task_dispatcher.py). If the supervisor is a
        # mock that lacks the on_notification setter, skip silently
        # (cost will be zero in that case).
        if log_capture is not None:
            try:
                handler = log_capture.handler()
                self._sup.on_notification = handler  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                log.warning(
                    "could not wire log_capture handler on supervisor %r; "
                    "continuing without transcript capture",
                    supervisor,
                )

    # ---- public API ----

    async def dispatch(
        self,
        prompt: str,
        cwd: Optional[str] = None,
        persona: Optional[str] = None,
        persona_file: Optional[str] = None,
        persona_prompt: Optional[str] = None,
        plan_mode: str = DEFAULT_PLAN_MODE,
    ) -> dict:
        """Validate, resolve, and dispatch to a real Reasonix session.

        Returns a dict shaped like the stub's return value, plus:
          - ``cost`` 5-field shape from the session's transcript
            (so the caller can display spend without a second call).
          - For ``plan_mode="approve"`` the dispatch pauses after the
            planner phase and returns ``cost.last_turn_phase="planner"``
            (the cost snapshot is taken before the user approves).
        """
        _validate_inputs(prompt, plan_mode, persona, persona_file, persona_prompt)
        resolved = _resolve(persona, persona_file, persona_prompt)
        # NOTE: persona system_prompt is reserved for a future R8.5 where
        # the dispatcher writes it into the session metadata or a side
        # file that the Reasonix binary picks up. For R8 the binary
        # uses its own default system prompt — the persona parameter
        # is captured in the return dict so the caller / log can see
        # which persona was requested.
        effective_cwd = cwd or os.getcwd()

        # 1. Open the session at the requested cwd.
        new_resp = await self._sup.new_session(cwd=effective_cwd)
        sid = new_resp["sessionId"]

        # 2. For plan_mode="approve": we send the prompt, then PAUSE
        # after the planner phase. The user must call on_approve() to
        # either continue (decision="approve") or drop (decision="reject").
        if plan_mode == "approve":
            return await self._dispatch_with_approval(
                sid, prompt, resolved, plan_mode, cwd,
            )

        # 3. Normal path: send the prompt and wait for end-of-turn.
        await self._sup.prompt(
            sid, content=[{"type": "text", "text": prompt}],
        )

        # 4. Snapshot the cost from the transcript written by log_capture.
        # Best-effort: returns zero breakdown if log_capture isn't attached
        # or the transcript doesn't exist yet.
        cost = self._snapshot_cost_for(sid)

        # R13.1: capture stop_reason from the prompt result. The Reasonix
        # binary returns a normalized stop payload under
        # ``prompt_result["update"]["stopReason"]``; common values are
        # ``"end_turn"`` (success), ``"cancelled"`` (session/cancel), and
        # ``"error"`` (server error). We store it in the session record
        # so ``status()`` can surface it without re-reading the log.
        #
        # NOTE: as of v1.9.1 the binary's normalized response shape is
        # stable (verified in internal/control/controller.go + adapter
        # tests). If a future wire change moves the field, this capture
        # gracefully falls back to ``"end_turn"`` (the success default).
        prompt_result = getattr(self._sup, "_last_prompt_result", None) or {}
        stop_reason = (
            prompt_result.get("update", {}).get("stopReason")
            if isinstance(prompt_result, dict)
            else None
        ) or "end_turn"

        # R13.4: write the user prompt to the parallel dispatcher log so
        # ``replay(mode="conversation")`` can reconstruct the full dialog.
        # The dispatcher log is durable across restarts; the binary
        # doesn't echo session/prompt requests, so without this side log
        # we'd lose the user side of the conversation.
        # Compute the turn number from the EXISTING session record (if
        # any) so re-dispatch on the same sid increments monotonically.
        # The record is read here — _before_ the overwrite below —
        # otherwise the overwrite would clobber the previous turn_count
        # and the very first turn (turn=1) would be lost.
        prev_turn = self._sessions.get(sid, {}).get("turn_count", 0)
        self._append_user_log(sid, turn=prev_turn + 1, text=prompt)

        # R9: Record session in the dispatcher's session map so status()
        # and replay() can return authoritative metadata without
        # re-deriving from the binary. We do this AFTER prompt() so
        # the session id is definitely real (not just new_session's
        # pre-prompt return value).
        import time as _time
        self._sessions[sid] = {
            "created_at": _time.time(),
            "persona_name": resolved.name,
            "plan_mode": plan_mode,
            "cwd": cwd,
            "log_path": str(self._log_path(sid)),
            "status": "completed",  # end-of-turn reached
            "prompt_preview": prompt[:80],
            # R13.1: stop_reason for status() to surface.
            "stop_reason": stop_reason,
            # R13.2: turn_count tracks how many times this sid has been
            # dispatched (re-dispatch = new turn in the same session).
            # Carries over prev_turn so the count is monotonic across
            # re-dispatches on the same sid.
            "turn_count": prev_turn + 1,
            # R13.2: goal_status — heuristic from stop_reason for now
            # (R14 may query Controller.GoalStatus for full ground truth).
            "goal_status": self._derive_goal_status(stop_reason),
        }

        return {
            "sid": sid,
            "log_path": str(self._log_path(sid)),
            "persona_name": resolved.name,
            "plan_mode": plan_mode,
            "cwd": cwd,
            "system_prompt_preview": _preview(resolved.system_prompt),
            "cost": cost.to_dict(),
        }

    def on_approve(
        self,
        sid: str,
        decision: str,
        feedback: Optional[str] = None,
    ) -> bool:
        """Resolve a pending ``plan_mode="approve"`` session.

        Returns True if the session was waiting and is now unblocked,
        False if no pending approval exists for ``sid`` (caller raced
        ahead or session is closed).

        Args:
            sid: session id returned by a previous :meth:`dispatch` call
                with ``plan_mode="approve"``.
            decision: ``"approve"`` continues to the executor phase;
                ``"reject"`` stops after the planner phase (the call
                returns immediately with cost snapshot).
            feedback: optional natural-language feedback to inject
                into the next turn (mirrors :meth:`Supervisor.steer`).
        """
        if sid not in self._pending_approves:
            return False
        slot = self._pending_approves[sid]
        if slot["fired"]:
            return False  # already resolved
        slot["decision"] = decision
        slot["feedback"] = feedback
        slot["fired"] = True
        slot["event"].set()
        return True

    # ---- R9: status + replay ----

    def status(self, sid: str) -> dict:
        """Return a status snapshot for ``sid``.

        The shape mirrors the MCP tool contract (see :mod:`phase2-spec.md`
        §P2.2 base MCP tools — ``reasonix_status``). For sessions
        dispatched via this dispatcher the returned dict has:

          - ``sid`` (str): the session id
          - ``status`` (str): one of ``"running"`` / ``"paused"``
            (plan_mode=approve waiting for on_approve) / ``"completed"``
            (no in-flight prompt) / ``"unknown"`` (sid not in our map)
          - ``persona`` (str): resolved persona name (e.g. ``"scaffold"``)
          - ``plan_mode`` (str): one of ``"auto"`` / ``"skip"`` /
            ``"approve"``
          - ``cost`` (dict): 5-field shape from
            :class:`adapter.cost.CostBreakdown.to_dict`
          - ``last_event`` (dict | None): the most recent NDJSON record
            from the session's transcript, or None if the log doesn't
            exist yet
          - ``queue_len`` (int): best-effort estimate of pending
            notifications (0 when we can't tell — we don't subscribe to
            the binary's internal queue from the adapter)
          - ``created_at`` (float): unix timestamp of the dispatch
          - ``prompt_preview`` (str): first 80 chars of the original
            prompt (handy for log display)
          - ``log_path`` (str): absolute path to the NDJSON transcript
          - ``stop_reason`` (str | None): raw ``stopReason`` from the
            binary's normalized stop payload — ``"end_turn"`` /
            ``"cancelled"`` / ``"error"`` / ``None`` when the session
            is unknown. R13.1: captured at dispatch() time.
          - ``turn_count`` (int): number of times this sid has been
            dispatched. R13.2: currently always 1 (single dispatch per
            session in R13.x; R14 may support re-dispatch).
          - ``goal_status`` (str | None): heuristic from
            ``stop_reason`` for R13.2 (``"complete"`` when end_turn,
            ``"blocked"`` when error/cancelled, ``None`` when running
            or unknown). R14 may query Controller.GoalStatus for full
            ground truth.

        For unknown sids the shape is::

            {"sid": "<unknown>", "status": "unknown", "cost": {zero 5-field},
             "stop_reason": None, "turn_count": 0, "goal_status": None}

        This method is sync (no I/O beyond the transcript read, which
        is best-effort and tolerates missing files) so the MCP layer
        can call it from any context.
        """
        from adapter.cost import CostBreakdown

        if sid not in self._sessions:
            return {
                "sid": sid,
                "status": "unknown",
                "persona": None,
                "plan_mode": None,
                "cost": CostBreakdown().to_dict(),
                "last_event": None,
                "queue_len": 0,
                "created_at": 0.0,
                "prompt_preview": "",
                "log_path": str(self._log_path(sid)),
                # R13.1 + R13.2: explicit None / 0 for unknown sids so
                # callers always see the same shape.
                "stop_reason": None,
                "turn_count": 0,
                "goal_status": None,
            }

        rec = self._sessions[sid]
        # Best-effort last_event from the tail of the log. We don't
        # require the file to exist — if it doesn't, last_event=None.
        last_event = self._tail_last_event(sid)
        # Best-effort queue_len. We have no direct visibility into
        # the binary's internal queue; we report the number of
        # pending "approve" slots (1 if this sid is paused) plus 0
        # otherwise. A future P2.4 / P3.x iteration can surface
        # Agent.steerQueueLen() if exposed upstream.
        queue_len = 1 if sid in self._pending_approves else 0
        # Current status: paused if waiting for approve, otherwise
        # the dispatcher thinks it's running. We don't poll the
        # binary for completion here (that's a future enhancement).
        current_status = "paused" if queue_len > 0 else rec.get("status", "running")

        return {
            "sid": sid,
            "status": current_status,
            "persona": rec.get("persona_name"),
            "plan_mode": rec.get("plan_mode"),
            "cost": self._snapshot_cost_for(sid).to_dict(),
            "last_event": last_event,
            "queue_len": queue_len,
            "created_at": rec.get("created_at", 0.0),
            "prompt_preview": rec.get("prompt_preview", ""),
            "log_path": rec.get("log_path", str(self._log_path(sid))),
            # R13.1 + R13.2: surface stop_reason, turn_count, goal_status.
            "stop_reason": rec.get("stop_reason"),
            "turn_count": rec.get("turn_count", 0),
            "goal_status": rec.get("goal_status"),
        }

    def replay(
        self,
        sid: str,
        since_seq: int = 0,
        *,
        mode: str = "conversation",
    ) -> list[dict]:
        """Return transcript events for ``sid``, optionally transformed.

        R13.3: 3-mode API. ``mode`` selects the output shape:

          - ``mode="raw"`` (default for R1 compat via dispatcher
            internal calls): the raw NDJSON records written by
            :class:`LogCapture` to ``<log_dir>/<sid>.jsonl``, in
            order, filtered by ``since_seq``. **Does NOT include
            user prompts** (those live in a parallel
            ``<sid>.dispatcher.jsonl``). Use this mode for "what did
            the binary actually emit" debugging.

          - ``mode="conversation"`` (default for the public API):
            a unified timeline merged from the binary log
            (``<sid>.jsonl``) and the dispatcher log
            (``<sid>.dispatcher.jsonl``), sorted by ``ts``,
            shaped as a list of role-tagged message dicts:

              - ``{role: "user", text, ts, turn, source: "dispatcher"}``
                (one per dispatch)
              - ``{role: "assistant", text, ts, turn, seq_start, seq_end,
                  source: "binary"}`` (one per turn; consecutive
                ``agent_message_chunk`` records are concatenated into
                a single assistant message; tool_call / stop / ask /
                permission events flush the current turn)
              - ``{role: "tool", name, status, args, result_summary,
                  ts, turn, source: "binary"}`` (one per tool call;
                ``tool_call`` + matching ``tool_call_update`` records
                are merged into a single entry with
                ``status="ok"|"error"|"running"``).

          - ``mode="summary"``: placeholder. Returns ``[]`` for R13.x.
            R14 will fill in aggregated counts / durations / tokens.

        The wrapper in :mod:`adapter.tools.reasonix_status` catches
        an invalid ``mode`` and returns ``[error_dict]`` (the
        dispatcher raises ``ValueError`` instead).

        Args:
            sid: session id.
            since_seq: only return events with ``seq >= since_seq``
                (raw mode) or events whose original records have
                ``seq >= since_seq`` (conversation mode). Default 0 =
                all events. Soft filter — records without a ``seq``
                field are included only when ``since_seq == 0``.
            mode: one of ``"raw"``, ``"conversation"``, ``"summary"``.
                Default ``"conversation"``.

        Returns:
            list of dicts. ``[]`` if the log file(s) don't exist or
            the session is unknown.

        Raises:
            ValueError: if ``mode`` is not one of the three valid
                values (the wrapper converts this to
                ``[error_dict]``).
        """
        from adapter.cost import UsageAccumulator  # noqa: F401  (import parity)

        valid_modes = frozenset({"raw", "conversation", "summary"})
        if mode not in valid_modes:
            raise ValueError(
                f"invalid mode={mode!r}; valid modes: "
                f"{sorted(valid_modes)}"
            )

        # R13.3: read both the binary log and the dispatcher log
        # (when in conversation mode). We do all I/O up front so
        # the transformation functions are pure.
        raw_events = self._read_raw_events(sid, since_seq=since_seq)

        if mode == "raw":
            return raw_events
        if mode == "summary":
            # R13.3: placeholder. R14 will fill in aggregated
            # counts / durations / tokens. Returning [] keeps the
            # contract testable without committing to a shape.
            return []

        # mode == "conversation"
        user_events = self._read_user_events(sid)
        return self._merge_conversation(raw_events, user_events)

    def _read_raw_events(self, sid: str, *, since_seq: int) -> list[dict]:
        """Read the binary NDJSON log and return raw records.

        Internal helper for :meth:`replay`. Mirrors the R1 logic
        that used to live directly in ``replay()``:

          - Returns ``[]`` if the log file doesn't exist.
          - Skips empty / unparseable lines silently.
          - Filters by ``since_seq`` (soft filter: records without
            a ``seq`` field are included only when ``since_seq ==
            0``).
        """
        log_path = self._log_path(sid)
        if not log_path.exists():
            return []
        try:
            text = log_path.read_text(encoding="utf-8")
        except OSError:
            return []
        events: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq = rec.get("seq")
            if since_seq == 0:
                events.append(rec)
            elif isinstance(seq, int) and seq >= since_seq:
                events.append(rec)
            # else: seq missing or below threshold, skip
        return events

    def _read_user_events(self, sid: str) -> list[dict]:
        """Read the parallel dispatcher log and return user records.

        Internal helper for :meth:`replay` ``mode="conversation"``.
        Returns ``[]`` if the file doesn't exist (R13.x: every
        session should have one because ``_append_user_log`` is
        called from both dispatch paths, but a missing file is
        tolerated for forward-compat with sessions dispatched by
        an older R9-only dispatcher).

        Dispatcher-log record shape (written by
        :meth:`_append_user_log`)::

            {"ts": <unix>, "turn": <int>, "role": "user",
             "text": <str>, "source": "dispatcher"}
        """
        path = self._user_log_path(sid)
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return []
        events: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(rec)
        return events

    @staticmethod
    def _merge_conversation(
        raw_events: list[dict],
        user_events: list[dict],
    ) -> list[dict]:
        """Merge binary + dispatcher records into a unified conversation.

        Internal helper for :meth:`replay` ``mode="conversation"``.
        Pure function (no I/O) so it's cheap to unit-test.

        Algorithm (R13.3, locked 2026-06-19 18:30):

          1. Project each binary record into one of three roles:
             - ``agent_message_chunk`` (and ``agent_message`` if it
               ever appears) → assistant candidate (text accumulator)
             - ``tool_call`` / ``tool_call_update`` → tool
               (grouped by ``toolCallId``; updates merge into the
               call with the same id)
             - everything else (stop / ask / permission / unknown)
               → flushes any in-progress assistant accumulator
               (boundary marker; not emitted as a message itself)
          2. Project each dispatcher record into a user message
             (already shaped correctly by ``_append_user_log``).
          3. Concatenate user + assistant + tool into one list.
          4. Sort by ``ts`` (lexicographic on ISO strings is the
             same as chronological since we always write ISO 8601
             UTC with the same precision — see ``_now_iso`` in
             ``adapter.log_capture``).
          5. Return.

        Returns a list of message dicts. The list may be empty
        (no events on either side).
        """
        # 1. project binary events
        # Map tool_call_id -> accumulated tool dict (so the
        # matching tool_call_update can merge into it).
        tool_acc: dict[str, dict] = {}
        # Current assistant message being built (None = no open
        # turn). Flushed by tool/stop/ask/permission events.
        current_assistant: dict | None = None
        # Order: we preserve chronological emission by appending
        # to messages in the order records are read (the raw log
        # is already in order). Flush appends to ``messages``
        # immediately.
        messages: list[dict] = []

        def _flush_assistant() -> None:
            nonlocal current_assistant
            if current_assistant is not None:
                messages.append(current_assistant)
                current_assistant = None

        for rec in raw_events:
            kind = rec.get("kind", rec.get("sessionUpdate", "unknown"))
            if kind == "agent_message_chunk":
                # Append text to the current assistant message,
                # or open a new one.
                text = rec.get("content", rec.get("text", ""))
                if not isinstance(text, str):
                    text = str(text)
                if current_assistant is None:
                    current_assistant = {
                        "role": "assistant",
                        "text": text,
                        "ts": rec.get("ts", ""),
                        "turn": rec.get("turn", 1),
                        "seq_start": rec.get("seq"),
                        "seq_end": rec.get("seq"),
                        "source": "binary",
                    }
                else:
                    current_assistant["text"] += text
                    current_assistant["seq_end"] = rec.get("seq", current_assistant.get("seq_end"))
            elif kind in ("tool_call", "tool_call_update"):
                # Boundary: flush any in-progress assistant text.
                _flush_assistant()
                tool_call_id = rec.get("toolCallId") or rec.get("id") or ""
                if kind == "tool_call":
                    tool_acc[tool_call_id] = {
                        "role": "tool",
                        "name": rec.get("name", rec.get("toolName", "")),
                        "status": rec.get("status", "running"),
                        "args": rec.get("arguments", rec.get("args", {})),
                        "result_summary": rec.get("result", rec.get("output")),
                        "ts": rec.get("ts", ""),
                        "turn": rec.get("turn", 1),
                        "source": "binary",
                    }
                else:  # tool_call_update — merge into existing
                    existing = tool_acc.get(tool_call_id)
                    if existing is not None:
                        if "status" in rec:
                            existing["status"] = rec["status"]
                        if "result" in rec or "output" in rec:
                            existing["result_summary"] = rec.get(
                                "result", rec.get("output", existing.get("result_summary"))
                            )
                        # Update ts to the latest activity time.
                        if "ts" in rec:
                            existing["ts"] = rec["ts"]
                    else:
                        # Orphan update (no matching tool_call).
                        # Emit as a standalone tool entry so the
                        # caller can still see the result.
                        tool_acc[tool_call_id] = {
                            "role": "tool",
                            "name": rec.get("name", rec.get("toolName", "")),
                            "status": rec.get("status", "ok"),
                            "args": {},
                            "result_summary": rec.get("result", rec.get("output")),
                            "ts": rec.get("ts", ""),
                            "turn": rec.get("turn", 1),
                            "source": "binary",
                        }
                # Emit the tool entry immediately (so order
                # matches the binary log). Updates will replace
                # it in messages; see below.
                # NB: since we only append, an update on a tool
                # emitted earlier in the list will NOT update
                # the earlier entry — the caller sees both the
                # call and the update as separate tool events
                # with the same tool_call_id. This is the safe
                # default for R13.3; R14 can fold them via a
                # second pass if needed.
                messages.append(tool_acc[tool_call_id])
            else:
                # Boundary: stop / ask / permission / unknown /
                # user_message (binary-side user echoes). Flush
                # any in-progress assistant text. The boundary
                # event itself is NOT emitted (it's a marker,
                # not a message).
                _flush_assistant()

        # End of raw_events: flush the trailing assistant
        # message, if any.
        _flush_assistant()

        # 2. append dispatcher user events (already shaped).
        for rec in user_events:
            messages.append({
                "role": "user",
                "text": rec.get("text", ""),
                "ts": rec.get("ts", ""),
                "turn": rec.get("turn", 1),
                "source": "dispatcher",
            })

        # 3. sort by ts (lex on ISO 8601 UTC = chronological).
        # Stable sort so events with the same ts preserve their
        # relative order (Python's sort is stable).
        messages.sort(key=lambda m: m.get("ts") or "")
        return messages

    def _tail_last_event(self, sid: str, n: int = 1) -> Optional[dict]:
        """Read the last ``n`` events from the session's transcript.

        Returns None if the log doesn't exist. Best-effort: used by
        :meth:`status` to populate the ``last_event`` field. Off-loads
        the file read to a thread.
        """
        log_path = self._log_path(sid)
        if not log_path.exists():
            return None
        try:
            text = log_path.read_text(encoding="utf-8")
        except OSError:
            return None
        events: list[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not events:
            return None
        return events[-1] if n == 1 else events[-n:]

    async def close(self) -> None:
        """Close the log_capture (if attached) so its background
        threads shut down cleanly. Mirrors TaskDispatcher.close.

        Idempotent: safe to call multiple times.

        R9: Also clears the in-memory session map. Callers should
        not assume session metadata survives a close.
        """
        # R9: drop session map first so any in-flight status() call
        # returns "unknown" instead of stale data.
        self._sessions.clear()
        if self._cap is None:
            return
        try:
            await self._cap.close()  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover - defensive
            log.warning("log_capture close failed: %s", e)

    # ---- internals ----

    async def _dispatch_with_approval(
        self,
        sid: str,
        prompt: str,
        resolved: Persona,
        plan_mode: str,
        cwd: Optional[str],
    ) -> dict:
        """Handle the ``plan_mode="approve"`` race (option X optimistic).

        1. Start the prompt (planner runs).
        2. If a user approval lands within :data:`APPROVE_TIMEOUT_S`,
           continue with executor. Otherwise, auto-reject (return early).
        3. Whether approve or reject, the cost snapshot reflects the
           planner spend only — the executor phase only runs on approve.
        """
        import asyncio

        # Send the prompt; the binary will pause after planner phase
        # when plan_mode="approve" propagates through the coordinator.
        # NOTE: as of R8 we don't yet have a separate "plan completed"
        # event in the ACP v2 stream — we send the prompt and use the
        # on_approve callback to decide whether to send a follow-up
        # prompt (executor) or just return (reject).
        await self._sup.prompt(
            sid, content=[{"type": "text", "text": prompt}],
        )

        slot = {
            "event": asyncio.Event(),
            "decision": None,
            "feedback": None,
            "fired": False,
        }
        self._pending_approves[sid] = slot
        try:
            # Wait for either an external approval or the timeout.
            try:
                await asyncio.wait_for(
                    slot["event"].wait(), timeout=self._approve_timeout,
                )
            except asyncio.TimeoutError:
                slot["fired"] = True
                slot["decision"] = "reject"
                log.info(
                    "approve timeout sid=%s → auto-reject (timeout=%.1fs)",
                    sid, self._approve_timeout,
                )

            decision = slot["decision"]
            cost = self._snapshot_cost_for(sid)

            if decision == "approve":
                # Continue the turn: send follow-up with optional feedback.
                followup = slot["feedback"] or "Plan approved — continue."
                await self._sup.prompt(
                    sid, content=[{"type": "text", "text": followup}],
                )
                cost = self._snapshot_cost_for(sid)

            # R9: Record final session state in the map. Status is
            # "completed" (executor ran to end-of-turn) for approve,
            # "rejected" for explicit reject / auto-timeout.
            import time as _time
            final_status = "completed" if decision == "approve" else "rejected"
            # R13.1: stop_reason for approve path. The follow-up prompt
            # (when decision=approve) sets the "real" stop reason;
            # auto-reject is treated as "cancelled".
            stop_reason = "end_turn" if decision == "approve" else "cancelled"
            # R13.4: write the user prompt to the dispatcher log so
            # replay(mode="conversation") can show the user side.
            self._append_user_log(
                sid, turn=self._sessions.get(sid, {}).get("turn_count", 0) + 1, text=prompt,
            )
            self._sessions[sid] = {
                "created_at": _time.time(),
                "persona_name": resolved.name,
                "plan_mode": plan_mode,
                "cwd": cwd,
                "log_path": str(self._log_path(sid)),
                "status": final_status,
                "prompt_preview": prompt[:80],
                "decision": decision,
                # R13.1 + R13.2: stop_reason + goal_status surfaced via status().
                "stop_reason": stop_reason,
                "turn_count": 1,
                "goal_status": self._derive_goal_status(stop_reason),
            }

            return {
                "sid": sid,
                "log_path": str(self._log_path(sid)),
                "persona_name": resolved.name,
                "plan_mode": plan_mode,
                "cwd": cwd,
                "system_prompt_preview": _preview(resolved.system_prompt),
                "cost": cost.to_dict(),
                "decision": decision,
            }
        finally:
            # Keep the slot for a short grace period in case the caller
            # queries status, then drop it. For R8 we drop immediately.
            self._pending_approves.pop(sid, None)

    def _log_path(self, sid: str) -> Path:
        """Resolve the NDJSON log path for ``sid``.

        Mirrors TaskDispatcher._log_path (L312-325). We ask the
        log_capture for its ``_log_dir``; if log_capture is ``None``
        or doesn't expose a ``_log_dir`` (e.g. a duck-typed test
        double), fall back to the canonical ``SESSION_LOG_DIR`` and
        a best-effort filename. The returned path may not exist yet
        (e.g. session just opened, no notifications arrived); callers
        should check ``.exists()`` before reading.
        """
        from adapter.log_capture import SESSION_LOG_DIR
        log_dir = getattr(self._cap, "_log_dir", None)
        base = Path(log_dir) if log_dir is not None else SESSION_LOG_DIR
        return base / f"{sid}.jsonl"

    def _snapshot_cost_for(self, sid: str) -> "CostBreakdown":
        """Read the latest cost from the session's transcript.

        Best-effort: returns zero breakdown if log_capture isn't
        attached or the transcript doesn't exist yet.
        """
        from adapter.cost import UsageAccumulator

        path = self._log_path(sid)
        if not path.exists():
            return self._empty_cost()
        return UsageAccumulator(transcript_path=path).load_from_transcript()

    @staticmethod
    def _empty_cost() -> "CostBreakdown":
        from adapter.cost import CostBreakdown
        return CostBreakdown()

    # ---- R13.x helpers ----

    @staticmethod
    def _derive_goal_status(stop_reason: Optional[str]) -> Optional[str]:
        """Heuristic mapping from ``stop_reason`` to ``goal_status``.

        The Reasonix v1.9.1 binary's normalized stop payload uses
        these ``stopReason`` values (see
        ``internal/control/controller.go`` ``advanceGoalAfterTurn``
        and ``parseGoalStatusMarker``):

          - ``"end_turn"``  →  ``goal_status = "complete"``
          - ``"cancelled"`` →  ``goal_status = "blocked"``
          - ``"error"``     →  ``goal_status = "blocked"``
          - ``None`` (still running) → ``goal_status = "running"``

        R13.2: heuristic only. R14 may surface
        ``Controller.GoalStatus`` directly for full ground truth
        (especially the ``"blocked"`` vs ``"complete"`` distinction,
        which is more nuanced in the source than this 3-way mapping
        captures).

        Args:
            stop_reason: raw stopReason from the binary's prompt result
                (or None when not yet captured).

        Returns:
            ``"complete"`` / ``"blocked"`` / ``"running"`` / ``None``
            (when stop_reason is unrecognized).
        """
        if stop_reason == "end_turn":
            return "complete"
        if stop_reason in ("cancelled", "error"):
            return "blocked"
        if stop_reason is None:
            return "running"
        # Unknown stop_reason: don't fabricate a goal status.
        return None

    def _user_log_path(self, sid: str) -> Path:
        """Resolve the parallel dispatcher-log path for ``sid``.

        R13.4: the binary's ``<sid>.jsonl`` does not include
        ``session/prompt`` requests, so user prompts are lost from
        the wire dump. We persist them to a parallel
        ``<sid>.dispatcher.jsonl`` next to the binary log so
        ``replay(mode="conversation")`` can reconstruct the full
        dialog.

        Path layout mirrors :meth:`_log_path`: prefer the
        ``log_capture``'s ``_log_dir`` (so tests using a stub
        capture land in a temp dir), fall back to
        :data:`adapter.log_capture.SESSION_LOG_DIR`.

        Args:
            sid: session id.

        Returns:
            Absolute path that may not exist yet. Filename is
            ``<sid>.dispatcher.jsonl`` (distinct from
            ``<sid>.jsonl`` which holds the binary transcript).
        """
        from adapter.log_capture import SESSION_LOG_DIR
        log_dir = getattr(self._cap, "_log_dir", None)
        base = Path(log_dir) if log_dir is not None else SESSION_LOG_DIR
        return base / f"{sid}.dispatcher.jsonl"

    def _user_log_lock_for(self, path: Path) -> asyncio.Lock:
        """Return the per-path :class:`asyncio.Lock` for dispatcher-log writes.

        Latches are kept in :attr:`_user_log_locks` (created lazily
        per path). Mirrors :meth:`adapter.log_capture.LogCapture._lock_for`
        but for the dispatcher log.

        Args:
            path: the file path to lock (typically from
                :meth:`_user_log_path`).

        Returns:
            The asyncio.Lock for this path. Always the same
            instance for the same path within a single dispatcher
            lifetime.
        """
        lock = self._user_log_locks.get(path)
        if lock is None:
            lock = asyncio.Lock()
            self._user_log_locks[path] = lock
        return lock

    def _append_user_log(self, sid: str, turn: int, text: str) -> None:
        """Append a user-prompt record to the parallel dispatcher log.

        R13.4: best-effort, never raises. The dispatcher log uses
        one NDJSON record per prompt::

            {"ts": <unix>, "turn": <int>, "role": "user",
             "text": <str>, "source": "dispatcher"}

        Implementation notes:

          - File is created lazily (parent dir included).
          - Blocking I/O is offloaded via
            :func:`asyncio.to_thread` so the dispatch coroutine
            isn't blocked by a slow disk.
          - Per-file :class:`asyncio.Lock` (see
            :meth:`_user_log_lock_for`) serializes concurrent
            prompts on the same sid; the *same* lock guards
            reads from the replay path when conversation mode
            merges dispatcher + binary records, so the file is
            never read mid-write.
          - Errors are logged at WARNING and swallowed: a failed
            dispatcher log must not fail the user's dispatch call.

        Args:
            sid: session id.
            turn: 1-based turn number within this session (1 for the
                first dispatch, 2 for the second, ...). Stored as
                ``turn`` on the record so the conversation timeline
                can be reconstructed in order.
            text: the user prompt text (already validated upstream
                by ``_validate_inputs``).
        """
        from adapter.log_capture import _now_iso  # type: ignore

        path = self._user_log_path(sid)

        async def _do_append() -> None:
            lock = self._user_log_lock_for(path)
            async with lock:
                def _sync_write() -> None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    record = {
                        "ts": time.time(),
                        "turn": turn,
                        "role": "user",
                        "text": text,
                        "source": "dispatcher",
                    }
                    # NDJSON: one record per line. append in binary
                    # mode to avoid Windows newline translation (we're
                    # on linux, but cheap insurance for cross-platform).
                    with open(path, "ab") as f:
                        f.write(
                            (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                        )
                try:
                    await asyncio.to_thread(_sync_write)
                except OSError as e:
                    log.warning(
                        "dispatcher-log append failed sid=%s path=%s: %s",
                        sid, path, e,
                    )

        try:
            # Schedule on the running loop. If called from sync
            # context (e.g. status() in a future enhancement),
            # the dispatcher would need a sync fallback — for
            # R13.x _append_user_log is only invoked from
            # dispatch() which is async.
            loop = asyncio.get_running_loop()
            loop.create_task(_do_append())
        except RuntimeError:
            # No running loop (sync caller). Fall back to direct
            # blocking write — the lock is asyncio.Lock, but if
            # we're here we're not in async context, so no
            # contention is possible.
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "ts": time.time(),
                    "turn": turn,
                    "role": "user",
                    "text": text,
                    "source": "dispatcher",
                }
                with open(path, "ab") as f:
                    f.write(
                        (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                    )
            except OSError as e:
                log.warning(
                    "dispatcher-log append failed (sync) sid=%s path=%s: %s",
                    sid, path, e,
                )
