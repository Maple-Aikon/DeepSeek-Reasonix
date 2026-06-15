"""Persona resolution for Reasonix delegations (R7, P2.2 stub).

A "persona" is a small system-prompt fragment prepended to the user prompt
so the model behaves like a specialist (Scaffolder, Test Engineer, etc.).

Resolution chain (4 tiers, highest to lowest priority):

    1. ``persona_prompt`` — raw string passed inline (caller fully specifies
       the prefix; no file is read).
    2. ``persona_file``   — path to a custom ``.md`` file (caller-supplied).
    3. ``persona``        — name of a built-in persona; loads
       ``PERSONAS_DIR/{name}.md``.
    4. default            — loads ``PERSONAS_DIR/default.md``.

When **multiple** of the three are set, the resolver uses the highest
priority tier (prompt > file > builtin) and ignores the others. It does
not raise. The MCP tool layer (``reasonix_delegate``) enforces mutual
exclusivity at the tool boundary, but this resolver stays permissive so
it can be called directly from non-MCP code paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


#: Directory holding built-in persona .md files (relative to this module).
PERSONAS_DIR: Path = Path(__file__).parent / "personas"

#: Built-in persona name → filename (without .md extension).
BUILTINS: dict[str, str] = {
    "default":   "default.md",
    "scaffold":  "scaffold.md",
    "test":      "test.md",
    "doc":       "doc.md",
    "refactor":  "refactor.md",
    "review":    "review.md",
}

#: Sentinel name used for inline ``persona_prompt`` (no file lookup).
_INLINE_NAME = "inline"


@dataclass(frozen=True)
class Persona:
    """A resolved persona — name plus the system-prompt fragment.

    ``prefix`` and ``system_prompt`` are kept as separate fields for
    forward-compat: the wire layer may need the raw markdown for debug
    while the LLM only consumes the plain text.
    """
    name: str
    prefix: str
    system_prompt: str


class PersonaResolver:
    """Stateless facade — caches loaded personas in a class-level dict."""

    _cache: dict[str, Persona] = {}

    # ---- public API --------------------------------------------------------

    @classmethod
    def resolve(
        cls,
        persona: Optional[str] = None,
        persona_file: Optional[str] = None,
        persona_prompt: Optional[str] = None,
    ) -> Persona:
        """Resolve a persona from the (up to) 3 caller-provided params.

        Raises:
            ValueError: if more than one of the 3 params is set, or if a
                built-in / file path does not exist.
        """
        # Silent precedence: prompt > file > builtin > default.
        # We do NOT raise on multiple-set; the chain picks the highest tier
        # automatically. The reasonix_delegate MCP tool layer enforces the
        # "mutually exclusive" contract at its boundary; this resolver is
        # permissive so it can be used from non-MCP callers too.

        # Tier 1: inline prompt (highest priority)
        if persona_prompt is not None:
            return Persona(
                name=_INLINE_NAME,
                prefix=persona_prompt,
                system_prompt=persona_prompt,
            )

        # Tier 2: custom file path
        if persona_file is not None:
            return cls._load_file(Path(persona_file))

        # Tier 3: built-in name
        if persona is not None:
            return cls._load_builtin(persona)

        # Tier 4: default (no params)
        return cls._load_builtin("default")

    # ---- internals ---------------------------------------------------------

    @classmethod
    def _load_builtin(cls, name: str) -> Persona:
        """Load ``PERSONAS_DIR/{name}.md`` (cached after first read)."""
        if name in cls._cache:
            return cls._cache[name]

        if name not in BUILTINS:
            raise ValueError(
                f"unknown persona '{name}'; built-ins: {sorted(BUILTINS)}"
            )

        path = PERSONAS_DIR / BUILTINS[name]
        if not path.is_file():
            raise FileNotFoundError(
                f"built-in persona file missing: {path} "
                f"(expected under PERSONAS_DIR={PERSONAS_DIR})"
            )

        text = path.read_text(encoding="utf-8").strip()
        persona = Persona(name=name, prefix=text, system_prompt=text)
        cls._cache[name] = persona
        return persona

    @classmethod
    def _load_file(cls, path: Path) -> Persona:
        """Load a caller-supplied .md file. Not cached (caller may edit it)."""
        if not path.is_file():
            raise FileNotFoundError(f"persona_file not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        # Derive a display name from the filename (stem only).
        name = path.stem
        return Persona(name=name, prefix=text, system_prompt=text)

    @classmethod
    def clear_cache(cls) -> None:
        """Reset the built-in cache (used by tests)."""
        cls._cache.clear()
