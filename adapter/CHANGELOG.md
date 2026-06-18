# Changelog — Reasonix Python Adapter

All notable changes to `adapter/` are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- **2026-06-18 — Rebased onto upstream Reasonix v1.9.1** (`f944dfb7`).
  - Base: `b225dd7d` (desktop-v1.8.0) → `f944dfb7` (v1.9.1). +280 upstream commits.
  - Picked up new ACP method `session/steer` from upstream `e28a971c` ("Fix ACP Zed integration gaps"). Replaces local cancel+re-prompt workaround in `adapter/supervisor.py`.
  - Cherry-pick conflicts: 2 (`.gitignore` + `adapter/pyproject.toml`) — resolved manually.
  - Dropped: 3 codegraph-strip commits (`6417e415`, `72b76479`, `0df35eee`), 1 v1.7.0 merge commit (`d3e574f8`).
  - 188/188 tests pass (176 unit + 5 integration + 7 marker-skipped in 35.62s).
  - See: `memory/plan/reasonix-update-v191-20260618.md` for full plan + gate results.

## [0.1.0] — 2026-06-12

### Added
- **R1-R6** (Phase 1 close, commit `b2a98d17` on `main-v2`):
  - `adapter/reasonix_client.py` — ACP-over-stdio client (skeleton).
  - `adapter/supervisor.py` — supervisor state machine (idle → running → completed/failed).
  - `adapter/dispatcher.py` — task dispatch + result aggregation.
  - `adapter/persona.py` — persona resolver (R7 prep).
- **Test infrastructure** (commit `b2a98d17`): `pyproject.toml` `[tool.pytest.ini_options]` with `asyncio_mode = "auto"`, `integration` marker for slow tests that spawn the real reasonix binary. 7 `PytestUnknownMarkWarning` cleared.
- 54/54 tests pass (47 unit/functional + 7 integration), 0 warnings.

[Unreleased]: https://github.com/Maple-Aikon/DeepSeek-Reasonix/compare/v1.9.1...main-v2
[0.1.0]: https://github.com/Maple-Aikon/DeepSeek-Reasonix/compare/desktop-v1.8.0...b2a98d17
