"""MCP tool: ``reasonix_delegate`` (R7, P2.2 stub).

This is the **stub** implementation. It validates inputs, resolves the
persona, and returns a fake session id. It does NOT spawn a real Reasonix
binary; the real dispatch (wire to the binary's ``session/delegate``
handler) lands in R8.

Contract (locked 2026-06-15):
  - 3 plan_modes: ``auto`` (default), ``skip``, ``approve``
  - 3 persona params, mutually exclusive at this layer:
      persona_prompt (raw) > persona_file (custom path) > persona (builtin)
  - Empty prompt → ValueError
  - Returns: dict with ``sid``, ``log_path``, ``persona_name``,
    ``plan_mode``, and ``system_prompt_preview`` (truncated to 200 chars)

The tool is registered with the MCP server in ``adapter/mcp_server.py``
(not yet created; that comes with the broader PicoClaw integration).
"""

from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path
from typing import Optional

from adapter.persona import PersonaResolver, PERSONAS_DIR


#: The 3 plan_mode values accepted by this tool.
VALID_PLAN_MODES: frozenset[str] = frozenset({"auto", "skip", "approve"})

#: Default plan_mode when the caller does not specify one.
DEFAULT_PLAN_MODE: str = "auto"

#: Maximum length of the system_prompt_preview field in the return dict.
_PREVIEW_CHARS: int = 200


def reasonix_delegate(
    prompt: str,
    cwd: Optional[str] = None,
    persona: Optional[str] = None,
    persona_file: Optional[str] = None,
    persona_prompt: Optional[str] = None,
    plan_mode: str = DEFAULT_PLAN_MODE,
) -> dict:
    """Validate inputs, resolve persona, and return a fake session id.

    Args:
        prompt: the user task to delegate. Must be non-empty.
        cwd: optional working directory for the Reasonix session. Reserved
            for R8 (real dispatch); the stub records it in the return dict
            only.
        persona: built-in persona name (e.g. ``"scaffold"``). Mutually
            exclusive with ``persona_file`` and ``persona_prompt``.
        persona_file: path to a custom .md file. Mutually exclusive with
            ``persona`` and ``persona_prompt``.
        persona_prompt: raw system-prompt text. Mutually exclusive with
            ``persona`` and ``persona_file``.
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
    # ---- 1. prompt non-empty ----
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    # ---- 2. plan_mode allowed ----
    if plan_mode not in VALID_PLAN_MODES:
        raise ValueError(
            f"plan_mode must be one of {sorted(VALID_PLAN_MODES)} "
            f"(got {plan_mode!r})"
        )

    # ---- 3. persona params mutually exclusive (tool layer strict) ----
    set_params = [p for p in (persona, persona_file, persona_prompt) if p is not None]
    if len(set_params) > 1:
        raise ValueError(
            f"persona params are mutually exclusive: pass at most one of "
            f"persona/persona_file/persona_prompt (got {len(set_params)})"
        )

    # ---- 4. resolve persona (uses silent-precedence PersonaResolver) ----
    resolved = PersonaResolver.resolve(
        persona=persona,
        persona_file=persona_file,
        persona_prompt=persona_prompt,
    )

    # ---- 5. generate fake sid + log path ----
    sid = f"stub-{secrets.token_hex(6)}"
    tmp_root = Path(tempfile.gettempdir())
    log_path = str(tmp_root / f"reasonix-{sid}.log")

    # ---- 6. system_prompt preview (truncated) ----
    sp = resolved.system_prompt
    preview = sp if len(sp) <= _PREVIEW_CHARS else sp[:_PREVIEW_CHARS] + "..."

    return {
        "sid": sid,
        "log_path": log_path,
        "persona_name": resolved.name,
        "plan_mode": plan_mode,
        "cwd": cwd,
        "system_prompt_preview": preview,
    }
