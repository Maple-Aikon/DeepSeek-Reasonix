"""Shared pytest fixtures for the adapter test suite.

This file lives at ``adapter/tests/conftest.py`` so pytest picks it up
automatically for every test under ``tests/`` — no per-file
``@pytest.fixture`` duplication needed.

Why a shared fixture exists
---------------------------

Two test files (``test_supervisor.py`` and ``test_task_dispatcher.py``)
both spawn the **real** ``bin/reasonix`` binary to smoke-test the wire
format. Each originally had its own copy of the ``real_bin`` fixture
that just returned the binary path.

In v1.9.1 the binary started resolving the default model
(``deepseek-flash``) at ``session/new`` time and rejecting requests
with ``-32603 "model 'deepseek-flash' is not configured"`` when the
provider was not configured. Integration tests that only needed
``session/new`` to succeed were failing for an unrelated reason.

The fix is to give the binary a minimal ``reasonix.toml`` with the
``deepseek-flash`` provider declared. The binary reads it from
``$HOME/reasonix.toml`` on startup, so the fixture:

  1. creates a tmp dir
  2. copies the template from ``tests/fixtures/reasonix.toml`` into it
  3. returns the binary path AND the tmp dir (caller wires it as
     ``env={"HOME": tmpdir}`` on the ``Supervisor``)

Both pieces are returned in a small ``RealBin`` namedtuple so the
test signature is self-documenting::

    async def test_foo(real_bin, tmp_path):
        sup = Supervisor(
            binary=real_bin.path,
            cwd=tmp_path,
            env=real_bin.env,   # {"HOME": "<tmpdir with reasonix.toml>"}
        )
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import NamedTuple

import pytest


# Where the bundled reasonix binary lives. Path is repo-root-relative,
# which is what ``Path(__file__).resolve().parent.parent.parent``
# resolves to: adapter/tests/conftest.py → adapter/tests → adapter → repo.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_BIN = REPO_ROOT / "bin" / "reasonix"

# Template toml copied into the per-test HOME so the binary finds a
# valid provider on startup. See fixture docstring for the why.
CONFIG_TEMPLATE = Path(__file__).parent / "fixtures" / "reasonix.toml"


class RealBin(NamedTuple):
    """Pair of (binary path, env dict) for spawning the real reasonix.

    The env dict must be passed to ``Supervisor(env=...)`` so the
    subprocess sees a ``HOME`` pointing at a directory that contains
    a valid ``reasonix.toml``.
    """
    path: Path
    env: dict


@pytest.fixture
def real_bin(tmp_path):
    """Spawn the real reasonix binary, with a writable HOME containing
    a minimal ``reasonix.toml``.

    Skips the test if the binary is missing (lets ``-m 'not integration'``
    work as advertised in CI). The tmp_path is pytest-managed, so the
    config file is cleaned up automatically between tests.
    """
    if not REAL_BIN.exists():
        pytest.skip(f"real reasonix binary not found at {REAL_BIN}")
    if not CONFIG_TEMPLATE.exists():
        pytest.skip(f"reasonix.toml fixture template missing: {CONFIG_TEMPLATE}")

    # Copy template into a fresh tmp dir (tmp_path may be shared with
    # the test, so use a sibling). Write to BOTH locations the binary
    # might consult, because resolution order varies across versions:
    # - documented: flag > ./reasonix.toml > ~/.reasonix/config.toml
    # - the binary also reads <HOME>/reasonix.toml on some versions
    home_dir = tmp_path / "fake_home"
    home_dir.mkdir(exist_ok=True)
    shutil.copy(CONFIG_TEMPLATE, home_dir / "reasonix.toml")
    reasonix_dir = home_dir / ".reasonix"
    reasonix_dir.mkdir(exist_ok=True)
    shutil.copy(CONFIG_TEMPLATE, reasonix_dir / "config.toml")

    # Only override HOME — the binary also reads XDG paths, but HOME
    # is the highest-precedence location on Linux and macOS, and on
    # CI we want a deterministic, isolated config.
    env = {"HOME": str(home_dir)}
    return RealBin(path=REAL_BIN, env=env)
