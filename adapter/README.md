# Reasonix Subagent Adapter (Phase 1)

Python adapter that drives a [`reasonix`](https://github.com/maple/DeepSeek-Reasonix)
subprocess over JSON-RPC 2.0 / NDJSON (the **Agent Client Protocol** / ACP).

The adapter is the Phase 1 bridge between PicoClaw's tools layer and the
Reasonix Go binary. It owns one subprocess per `Supervisor` instance, exposes
a small async API for session lifecycle (`start` → `prompt` → `cancel` →
`close`), and streams every ACP `session/update` notification to an NDJSON
log file for later inspection (`replay` / `status`).

> **Status:** Phase 1 complete. `acp_client` / `supervisor` / `log_capture` /
> `task_dispatcher` are all wired; 47/47 unit tests pass. Live ACP integration
> is verified end-to-end via the `__main__` subcommands.

## Repository layout

```
sources/DeepSeek-Reasonix/
├── adapter/                ← this package
│   ├── acp_client.py       JSON-RPC 2.0 / NDJSON transport (no Reasonix types)
│   ├── supervisor.py       One subprocess; start / prompt / cancel / close
│   ├── log_capture.py      Subscribe to session/update → NDJSON log file
│   ├── task_dispatcher.py  High-level API: dispatch / cancel / steer / replay
│   ├── __main__.py         CLI: run / start / cancel / steer / status / replay
│   └── tests/              pytest suite (47 tests, 0 live-network required)
└── bin/reasonix            Go binary built from this repo (aarch64, statically linked)
```

`cli-bin/reasonix` (in the PicoClaw workspace) is a **symlink** to
`bin/reasonix`, so a `make build` here is picked up without an extra deploy.

## Quickstart

### 1. Run a one-shot task and stream the trace

```bash
cd sources/DeepSeek-Reasonix/adapter
python -m adapter run "say hi and stop" --no-approval
```

`--no-approval` auto-approves any server-initiated `session/request_permission`
(only useful if the LLM wants to call a tool — for a "hello world" task it has
no effect).

### 2. Start a session, capture `{sid, pid, log_path}`, then re-prompt

```bash
# start + return JSON
python -m adapter start "implement feature X" --no-approval
# → {"sid":"...","pid":...,"log_path":"/home/maple/.picoclaw/logs/reasonix-subagent/<sid>.jsonl"}

# inspect the last 20 events
python -m adapter status <sid> --n 20

# stream the full NDJSON log to stdout (NDJSON, one event per line)
python -m adapter replay <sid> | jq
```

### 3. Cancel / steer a running session

```bash
python -m adapter cancel <sid>           # best-effort cancel
python -m adapter steer <sid> "new task"  # cancel + re-prompt
```

### 4. As a library (used by PicoClaw tools layer)

```python
from adapter.task_dispatcher import TaskDispatcher

async def run_reasonix_task(task: str) -> dict:
    dispatcher = TaskDispatcher(
        cwd="/path/to/workspace",
        model="litellm-proxy/deepseek-flash-ai",
        auto_approve=True,  # trust the LLM to call tools
    )
    try:
        # dispatch = start session + prompt + await end-of-turn
        result = await dispatcher.dispatch(task)
        return result  # {"sid":..., "stopReason":..., "events":N}
    finally:
        await dispatcher.close()
```

## Architecture in one paragraph

`acp_client.Conn` owns the bytes — it reads NDJSON lines from the
subprocess's stdout, writes framed JSON-RPC 2.0 requests to its stdin, and
correlates responses by integer id using an in-memory `pending` map. It also
dispatches three kinds of inbound frames: **responses** (correlate to a
pending request), **notifications** (fire-and-forget `session/update`),
and **server-initiated requests** (rare; carry an `id` + `method` and
require the client to reply — handled via the `on_server_request` callback
in Phase 1's debug #5 wiring).

`supervisor.Supervisor` owns one `Conn` and one subprocess. It implements
the session lifecycle: `start()` (spawn + `initialize`), `new_session()`
(open a session at a workspace root), `prompt()` (send user content, await
`stop` notification), `cancel()` / `close()` (tear down). It does **not**
auto-recover a crashed process — the caller decides.

`log_capture.LogCapture` subscribes to the supervisor's `on_notification`
channel and writes every `session/update` as one NDJSON record to
`~/.picoclaw/logs/reasonix-subagent/<sid>.jsonl`. Rotation at 50 MB per
active file; unlimited rotated files (Phase 2 cleanup concern).

`task_dispatcher.TaskDispatcher` is the public surface PicoClaw's tools
layer talks to. It composes a Supervisor + LogCapture and exposes five
operations: `dispatch(task)`, `cancel(sid)`, `steer(sid, new_task)`,
`status(sid, n=20)`, `replay(sid)`. It also serializes a plain string task
into ACP `ContentBlock` form and reads log files for status / replay, so
callers never have to reach into the log directory.

## Subcommand reference

| Subcommand  | Purpose                                          | Returns                                |
|-------------|--------------------------------------------------|----------------------------------------|
| `run`       | One-shot task; print live trace to stderr        | exit 0 + final stop reason             |
| `start`     | Start + prompt + return `{sid, pid, log_path}`   | JSON on stdout                         |
| `cancel`    | Best-effort cancel of an in-flight session       | exit 0 (idempotent)                    |
| `steer`     | Cancel + re-prompt with a new task               | JSON `{sid, stopReason}`               |
| `status`    | Print last N events from the NDJSON log          | pretty-printed event list              |
| `replay`    | Stream all events from the NDJSON log to stdout  | raw NDJSON (one event per line)        |

### Shared flags

```
--cwd DIR         working directory for the session (default: $PWD)
--binary PATH     override the reasonix binary path
--log-dir PATH    override the per-session log directory
--pid-dir PATH    override the per-session PID directory
--model NAME      provider/model name (forwarded to `reasonix acp --model`)
--prompt-timeout  max seconds to wait for a single prompt (default 120)
--pretty          print pretty stdout for log_capture (off by default)
--no-approval     auto-approve server-initiated permission requests
```

### Exit codes

| Code | Meaning                                                |
|------|--------------------------------------------------------|
| 0    | success (or, for `cancel`, the session was idle)       |
| 1    | dispatcher error (prompt failed, log file unreadable)  |
| 2    | invalid arguments (argparse)                           |
| 3    | runtime setup error (binary missing, log dir not creatable) |

## Defaults & paths

| Resource       | Default location                                       |
|----------------|--------------------------------------------------------|
| Binary         | `sources/DeepSeek-Reasonix/bin/reasonix` (aarch64)     |
| Session log    | `~/.picoclaw/logs/reasonix-subagent/<sid>.jsonl`       |
| PID file       | `~/.picoclaw/state/reasonix/<sid>.pid`                 |

All three can be overridden via the matching `--*` flag or by setting
`REASONIX_BIN` in the environment.

## Testing

```bash
cd sources/DeepSeek-Reasonix/adapter
.venv/bin/pytest -v --tb=short
```

The suite has 47 tests; 41 are pure unit (mocked) and 6 are integration
(of which 3 hit the real `bin/reasonix` binary and may take 20-40s each
to settle — the LLM-driven tests run with `litellm-proxy/deepseek-flash-ai`
by default, which loops indefinitely without a `--max-steps` cap). Use `-k`
to filter:

```bash
# Fast: 41 unit tests, no subprocess.
.venv/bin/pytest -k 'not real and not integration' -q

# New Step-4 tests only.
.venv/bin/pytest -k 'start_returns_sid or cancel_long_task or steer_replaces_prompt or replay_prints_log' -v
```

## Debugging tips

- **Live NDJSON tail:** `tail -f ~/.picoclaw/logs/reasonix-subagent/<sid>.jsonl | jq`
- **Stderr from subprocess:** set `stderr_log=` when constructing a
  `Supervisor`; the file captures one line per stderr write.
- **Wire dump:** set `ACP_DEBUG=1` in the environment to log every
  inbound/outbound frame to `adapter.acp_client` at DEBUG level.
- **Standalone binary check:** `reasonix acp </dev/null` — should print
  one JSON-RPC error then exit (the binary is alive if you see
  `METHOD_NOT_FOUND` for the empty request).

## Known limitations (Phase 1)

- **One Supervisor = one session.** No pooling. Phase 2 adds a pool if/when
  PicoClaw needs concurrent Reasonix tasks.
- **Auto-approve is a binary knob.** Phase 3 (Telegram bridge) will replace
  it with a callback that surfaces `session/request_permission` to the
  user as an inline keyboard.
- **No transcript persistence.** The NDJSON log is the only ground truth;
  Reasonix's own session files are not read by the adapter (yet — see
  ACP `transcriptPath` discussion in the plan).
- **No reconnect.** A crashed process is a dead supervisor; callers must
  instantiate a fresh one.

## Related docs

- `memory/plan/reasonix-subagent-20260610.md` — Phase 1 plan + R4 close log
- `memory/202606/20260611.md` — daily notes (R4 closing entry)
- `sources/DeepSeek-Reasonix/internal/acp/protocol.go` — authoritative ACP
  wire format reference (Reasonix side)
