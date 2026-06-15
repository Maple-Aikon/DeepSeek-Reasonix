"""Tests for the ``reasonix_delegate`` MCP tool stub (R7, P2.2).

The stub validates input, resolves the persona, and returns a fake
session id. It does NOT spawn a real Reasonix binary — that comes in R8.

Contract verified here:
  - plan_mode ∈ {"auto", "skip", "approve"} (default "auto")
  - Persona params follow chain: persona_prompt > persona_file > persona
  - The MCP tool layer enforces mutual-exclusivity of persona params
    (PersonaResolver itself is permissive; the tool layer is strict)
  - Returns: {sid, log_path, persona_name, plan_mode, system_prompt_preview}
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the tool function under test
from adapter.tools.reasonix_delegate import (
    reasonix_delegate,
    VALID_PLAN_MODES,
    DEFAULT_PLAN_MODE,
)
from adapter.persona import PersonaResolver


@pytest.fixture(autouse=True)
def _clear_persona_cache():
    """Reset PersonaResolver cache between tests so PERSONAS_DIR patches
    take effect (built-in lookups are cached on first read)."""
    PersonaResolver.clear_cache()
    yield
    PersonaResolver.clear_cache()


# ---------------------------------------------------------------------------
# T1: default params
# ---------------------------------------------------------------------------

def test_delegate_default_params(tmp_path):
    """No persona, no plan_mode → auto + default persona, fake sid returned."""
    with patch("adapter.persona.PERSONAS_DIR", tmp_path):
        (tmp_path / "default.md").write_text("Default persona.\n")
        out = reasonix_delegate(prompt="summarize this file")
    assert out["plan_mode"] == "auto"
    assert out["persona_name"] == "default"
    assert out["sid"].startswith("stub-")
    assert "Default persona" in out["system_prompt_preview"]
    # log_path under tmp (the stub uses tempfile)
    assert "reasonix-" in out["log_path"]


# ---------------------------------------------------------------------------
# T2: plan_mode="skip" accepted
# ---------------------------------------------------------------------------

def test_delegate_plan_mode_skip(tmp_path):
    with patch("adapter.persona.PERSONAS_DIR", tmp_path):
        (tmp_path / "default.md").write_text("x\n")
        out = reasonix_delegate(prompt="echo", plan_mode="skip")
    assert out["plan_mode"] == "skip"


# ---------------------------------------------------------------------------
# T3: plan_mode="approve" accepted
# ---------------------------------------------------------------------------

def test_delegate_plan_mode_approve(tmp_path):
    with patch("adapter.persona.PERSONAS_DIR", tmp_path):
        (tmp_path / "default.md").write_text("x\n")
        out = reasonix_delegate(prompt="do work", plan_mode="approve")
    assert out["plan_mode"] == "approve"


# ---------------------------------------------------------------------------
# T4: invalid plan_mode → ValueError
# ---------------------------------------------------------------------------

def test_delegate_invalid_plan_mode(tmp_path):
    with patch("adapter.persona.PERSONAS_DIR", tmp_path):
        (tmp_path / "default.md").write_text("x\n")
        with pytest.raises(ValueError, match="plan_mode"):
            reasonix_delegate(prompt="x", plan_mode="garbage")


# ---------------------------------------------------------------------------
# T5: persona + persona_file both set → ValueError (tool layer strict)
# ---------------------------------------------------------------------------

def test_delegate_rejects_persona_and_file(tmp_path):
    custom = tmp_path / "x.md"
    custom.write_text("x\n")
    with pytest.raises(ValueError, match="exclusive|ambiguous"):
        reasonix_delegate(
            prompt="x", persona="default", persona_file=str(custom)
        )


# ---------------------------------------------------------------------------
# T6: persona + persona_prompt both set → ValueError
# ---------------------------------------------------------------------------

def test_delegate_rejects_persona_and_prompt(tmp_path):
    with pytest.raises(ValueError, match="exclusive|ambiguous"):
        reasonix_delegate(
            prompt="x", persona="default", persona_prompt="RAW"
        )


# ---------------------------------------------------------------------------
# T7: persona_file + persona_prompt both set → ValueError
# ---------------------------------------------------------------------------

def test_delegate_rejects_file_and_prompt(tmp_path):
    custom = tmp_path / "x.md"
    custom.write_text("x\n")
    with pytest.raises(ValueError, match="exclusive|ambiguous"):
        reasonix_delegate(
            prompt="x", persona_file=str(custom), persona_prompt="RAW"
        )


# ---------------------------------------------------------------------------
# T8: persona="scaffold" → resolves to scaffold.md
# ---------------------------------------------------------------------------

def test_delegate_builtin_scaffold(tmp_path):
    with patch("adapter.persona.PERSONAS_DIR", tmp_path):
        (tmp_path / "default.md").write_text("d\n")
        (tmp_path / "scaffold.md").write_text("I scaffold.\n")
        out = reasonix_delegate(prompt="x", persona="scaffold")
    assert out["persona_name"] == "scaffold"
    assert "scaffold" in out["system_prompt_preview"]


# ---------------------------------------------------------------------------
# T9: persona_prompt (raw) → uses verbatim
# ---------------------------------------------------------------------------

def test_delegate_raw_prompt():
    out = reasonix_delegate(
        prompt="x", persona_prompt="BE CONCISE"
    )
    assert out["persona_name"] == "inline"
    assert "BE CONCISE" in out["system_prompt_preview"]


# ---------------------------------------------------------------------------
# T10: empty prompt → ValueError
# ---------------------------------------------------------------------------

def test_delegate_empty_prompt_rejected():
    with pytest.raises(ValueError, match="prompt"):
        reasonix_delegate(prompt="")


# ---------------------------------------------------------------------------
# T11: valid plan_modes exported
# ---------------------------------------------------------------------------

def test_valid_plan_modes_constant():
    assert VALID_PLAN_MODES == {"auto", "skip", "approve"}
    assert DEFAULT_PLAN_MODE == "auto"


# ---------------------------------------------------------------------------
# T12: sid is unique per call
# ---------------------------------------------------------------------------

def test_delegate_unique_sid(tmp_path):
    with patch("adapter.persona.PERSONAS_DIR", tmp_path):
        (tmp_path / "default.md").write_text("x\n")
        a = reasonix_delegate(prompt="a")
        b = reasonix_delegate(prompt="b")
    assert a["sid"] != b["sid"]
