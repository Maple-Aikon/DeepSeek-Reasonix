"""Tests for adapter.persona (R7, P2.2 stub).

PersonaResolver implements a 4-tier resolution chain:
  1. persona_prompt (raw string)  ← highest priority
  2. persona_file   (custom .md path)
  3. persona        (built-in shortcut, loads adapter/personas/{name}.md)
  4. default        (loads adapter/personas/default.md)

All 3 persona params are optional. If none provided → default. If multiple
provided → ValueError (ambiguous).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from adapter.persona import Persona, PersonaResolver, BUILTINS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_persona_file(tmp_path: Path, name: str, content: str) -> Path:
    """Write a persona .md file and return its path."""
    p = tmp_path / f"{name}.md"
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# T1: default resolution (no params)
# ---------------------------------------------------------------------------

def test_resolve_no_params_returns_default(tmp_path):
    """No params → loads default persona."""
    # Patch PERSONAS_DIR to a tmp dir with our own default.md
    with patch("adapter.persona.PERSONAS_DIR", tmp_path):
        _write_persona_file(tmp_path, "default", "I am the default.\n")
        p = PersonaResolver.resolve()
    assert p.name == "default"
    assert "default" in p.prefix.lower() or "default" in p.system_prompt.lower()


# ---------------------------------------------------------------------------
# T2: builtin resolution (persona="scaffold")
# ---------------------------------------------------------------------------

def test_resolve_builtin_scaffold(tmp_path):
    """persona='scaffold' → loads adapter/personas/scaffold.md."""
    with patch("adapter.persona.PERSONAS_DIR", tmp_path):
        _write_persona_file(tmp_path, "scaffold", "I am the Scaffolder.\n")
        p = PersonaResolver.resolve(persona="scaffold")
    assert p.name == "scaffold"
    assert "Scaffolder" in p.system_prompt


# ---------------------------------------------------------------------------
# T3: file resolution (persona_file="<custom path>")
# ---------------------------------------------------------------------------

def test_resolve_file_path(tmp_path):
    """persona_file=path → loads that exact file."""
    custom = _write_persona_file(tmp_path, "custom", "I am custom-loaded.\n")
    p = PersonaResolver.resolve(persona_file=str(custom))
    assert p.name == "custom"
    assert "custom-loaded" in p.system_prompt


# ---------------------------------------------------------------------------
# T4: prompt resolution (raw persona_prompt)
# ---------------------------------------------------------------------------

def test_resolve_raw_prompt():
    """persona_prompt='raw text' → uses verbatim, no file load."""
    p = PersonaResolver.resolve(persona_prompt="Be concise and direct.")
    assert p.name == "inline"
    assert p.system_prompt == "Be concise and direct."


# ---------------------------------------------------------------------------
# T5: precedence (prompt > file > builtin > default)
# ---------------------------------------------------------------------------

def test_resolve_precedence_prompt_wins(tmp_path):
    """All 3 set → prompt wins (highest priority)."""
    custom = _write_persona_file(tmp_path, "x", "from-file\n")
    p = PersonaResolver.resolve(
        persona="x",  # would be file x.md (won't load, persona_file wins)
        persona_file=str(custom),  # would be custom (prompt wins)
        persona_prompt="RAW PROMPT WINS",  # ← this wins
    )
    assert p.name == "inline"
    assert p.system_prompt == "RAW PROMPT WINS"


def test_resolve_precedence_file_over_builtin(tmp_path):
    """persona + persona_file set → file wins (builtin ignored)."""
    builtin_path = tmp_path / "scaffold.md"
    builtin_path.write_text("BUILTIN content\n")
    custom = _write_persona_file(tmp_path, "my_custom", "FILE content\n")
    with patch("adapter.persona.PERSONAS_DIR", tmp_path):
        p = PersonaResolver.resolve(
            persona="scaffold",
            persona_file=str(custom),
        )
    assert p.name == "my_custom"
    assert "FILE content" in p.system_prompt
    assert "BUILTIN content" not in p.system_prompt


# ---------------------------------------------------------------------------
# T6: validation — multiple params (ambiguous)
# ---------------------------------------------------------------------------

def test_resolve_silent_file_over_builtin(tmp_path):
    """persona + persona_file set (no prompt) → file wins silently."""
    builtin_path = tmp_path / "scaffold.md"
    builtin_path.write_text("BUILTIN content\n")
    custom = _write_persona_file(tmp_path, "x", "FILE content\n")
    with patch("adapter.persona.PERSONAS_DIR", tmp_path):
        p = PersonaResolver.resolve(persona="scaffold", persona_file=str(custom))
    assert p.name == "x"  # filename stem
    assert "FILE content" in p.system_prompt
    assert "BUILTIN content" not in p.system_prompt


def test_resolve_silent_prompt_over_builtin():
    """persona + persona_prompt set → prompt wins silently (no raise)."""
    p = PersonaResolver.resolve(persona="scaffold", persona_prompt="RAW WINS")
    assert p.name == "inline"
    assert p.system_prompt == "RAW WINS"


def test_resolve_silent_prompt_over_file(tmp_path):
    """persona_file + persona_prompt set → prompt wins silently (no raise)."""
    custom = _write_persona_file(tmp_path, "x", "FILE content\n")
    p = PersonaResolver.resolve(persona_file=str(custom), persona_prompt="RAW WINS")
    assert p.name == "inline"
    assert p.system_prompt == "RAW WINS"
    assert "FILE content" not in p.system_prompt


# ---------------------------------------------------------------------------
# T7: BUILTINS catalog sanity
# ---------------------------------------------------------------------------

def test_builtins_catalog_has_6_names():
    """BUILTINS dict has the 6 expected persona names."""
    assert set(BUILTINS.keys()) == {
        "default", "scaffold", "test", "doc", "refactor", "review",
    }


# ---------------------------------------------------------------------------
# T8: caching — second resolve returns same object
# ---------------------------------------------------------------------------

def test_resolve_caches_builtin(tmp_path):
    """Built-in persona is loaded once, cached."""
    with patch("adapter.persona.PERSONAS_DIR", tmp_path):
        _write_persona_file(tmp_path, "scaffold", "x\n")
        p1 = PersonaResolver.resolve(persona="scaffold")
        p2 = PersonaResolver.resolve(persona="scaffold")
    assert p1 is p2  # same cached object
