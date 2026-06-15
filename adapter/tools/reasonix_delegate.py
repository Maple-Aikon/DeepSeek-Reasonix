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
import os
import secrets
import tempfile
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

    async def close(self) -> None:
        """Close the log_capture (if attached) so its background
        threads shut down cleanly. Mirrors TaskDispatcher.close.

        Idempotent: safe to call multiple times.
        """
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
