"""Command-line entry for the Reasonix subagent adapter.

R4 implementation: each subcommand wires a :class:`adapter.task_dispatcher.TaskDispatcher`
to a real :class:`adapter.supervisor.Supervisor` and
:class:`adapter.log_capture.LogCapture`. The wiring is identical for every
subcommand; what differs is which dispatcher method gets called and how
the result is printed.

Subcommands
-----------
``run``       — start a session, prompt, print live trace, return final stop reason.
``start``     — start a session, prompt, print ``{sid, pid, log_path}`` JSON, close.
``cancel``    — cancel an in-flight session (best-effort).
``steer``     — cancel + re-prompt with a new task.
``status``    — print the last N events from the session's NDJSON log.
``replay``    — stream all events from the session's NDJSON log to stdout.

All subcommands share the same flag set:
  ``--cwd DIR``         working directory for the session (default: $PWD)
  ``--binary PATH``     override the reasonix binary path
  ``--log-dir PATH``    override the per-session log directory
  ``--pid-dir PATH``    override the per-session PID directory
  ``--model NAME``      provider/model name (forwarded to ``reasonix acp --model``)
  ``--prompt-timeout``  max seconds to wait for a single prompt (default 120)
  ``--pretty``          print pretty stdout for log_capture (off by default)
  ``--no-approval``     auto-approve server-initiated permission requests (off by default)

Exit codes
----------
0 — success (or, for cancel, the session was already idle)
1 — dispatcher error (e.g. prompt failed, log file unreadable)
2 — invalid arguments (argparse)
3 — runtime setup error (binary missing, log dir not creatable)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Sequence

from adapter.log_capture import (
    default_log_dir,
    default_pid_dir,
    LogCapture,
)
from adapter.supervisor import DEFAULT_BINARY, Supervisor, SupervisorError
from adapter.task_dispatcher import TaskDispatcher, TaskDispatcherError


# A constant in ``adapter.__init__`` carries the version; we import it lazily
# so this module stays importable even if the package metadata is unavailable
# (e.g. running it as a flat script during early development).
_VERSION = "unknown"
try:  # pragma: no cover
    from adapter import __version__ as _VERSION  # type: ignore
except ImportError:  # pragma: no cover
    pass


# ---------------- shared helpers ----------------


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    """Flags every subcommand accepts. Keep this list small — anything
    specific to one subcommand belongs on its own subparser."""
    parser.add_argument(
        "--cwd", default=os.getcwd(),
        help="Working directory for the new session (default: $PWD).",
    )
    parser.add_argument(
        "--binary", default=None,
        help=f"Path to the reasonix binary (default: {DEFAULT_BINARY}).",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help=f"Per-session NDJSON directory (default: {default_log_dir()}).",
    )
    parser.add_argument(
        "--pid-dir", default=None,
        help=f"Per-session PID directory (default: {default_pid_dir()}).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Provider/model name forwarded to ``reasonix acp --model``.",
    )
    parser.add_argument(
        "--prompt-timeout", type=float, default=120.0,
        help="Max seconds to wait for a single prompt (default: 120).",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Enable pretty stdout for log_capture (off by default).",
    )
    parser.add_argument(
        "--no-approval", action="store_true",
        help="Auto-approve any ``session/request_permission`` the server "
             "sends. Off by default — when off, the server sees METHOD_NOT_FOUND "
             "and Reasonix will surface the request as an error.",
    )


def _resolve_binary(arg: Optional[str]) -> Path:
    """Resolve the reasonix binary path.

    Priority: ``--binary`` flag → ``$REASONIX_BIN`` env → DEFAULT_BINARY.
    """
    if arg:
        return Path(arg)
    env = os.environ.get("REASONIX_BIN")
    if env:
        return Path(env)
    return DEFAULT_BINARY


def _build_dispatcher(args: argparse.Namespace) -> TaskDispatcher:
    """Construct a Supervisor + LogCapture + TaskDispatcher from the
    shared CLI flags. Caller is responsible for ``close()`` (in a
    finally block) and the async lifecycle.

    Raises
    ------
    SystemExit
        If the binary doesn't exist on disk (exit code 3).
    """
    binary = _resolve_binary(args.binary)
    if not binary.exists():
        print(
            f"error: reasonix binary not found: {binary}\n"
            f"hint:  run ``make build`` in sources/DeepSeek-Reasonix, "
            f"or set $REASONIX_BIN, or pass --binary PATH.",
            file=sys.stderr,
        )
        sys.exit(3)

    log_dir = Path(args.log_dir) if args.log_dir else None
    pid_dir = Path(args.pid_dir) if args.pid_dir else None
    cwd = Path(args.cwd)

    supervisor = Supervisor(
        binary=binary,
        cwd=cwd,
        model=args.model,
        prompt_timeout=args.prompt_timeout,
        auto_approve=bool(args.no_approval),
    )
    log_capture = LogCapture(
        supervisor,
        log_dir=log_dir,
        pid_dir=pid_dir,
        pretty_stdout=args.pretty,
    )
    return TaskDispatcher(
        supervisor=supervisor,
        log_capture=log_capture,
        cwd=cwd,
        log_dir=log_dir,
        pid_dir=pid_dir,
        auto_approve=bool(args.no_approval),
    )


# ---------------- subcommand handlers ----------------
# Each handler is ``async`` so we can share the same async runner. They
# take a parsed ``Namespace`` and return an int exit code.


async def _cmd_run(args: argparse.Namespace) -> int:
    """``run <task>`` — start a session, prompt, print stop reason."""
    td = _build_dispatcher(args)
    try:
        result = await td.dispatch(args.task)
    except TaskDispatcherError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except SupervisorError as e:  # pragma: no cover - exercised via integration
        print(f"error: supervisor: {e}", file=sys.stderr)
        return 1
    finally:
        await td.close()
    stop_reason = result.get("stopReason", "unknown")
    print(json.dumps({"stopReason": stop_reason, "payload": result}, indent=2))
    return 0 if stop_reason in ("end_turn", "stop") else 1


async def _cmd_start(args: argparse.Namespace) -> int:
    """``start <task>`` — like ``run`` but prints the session metadata
    so a caller can later ``cancel`` / ``steer`` / ``replay`` it."""
    td = _build_dispatcher(args)
    try:
        result = await td.dispatch(args.task)
    except (TaskDispatcherError, SupervisorError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    # Find the sid the supervisor is currently on. TaskDispatcher doesn't
    # expose it directly post-dispatch, so we read it from the supervisor.
    sid = td._sup.current_sid if hasattr(td, "_sup") else None  # type: ignore[attr-defined]
    pid = td._sup.pid if hasattr(td, "_sup") else None  # type: ignore[attr-defined]
    log_path = td._log_path(sid) if sid else None
    print(json.dumps({
        "sid": sid,
        "pid": pid,
        "log_path": str(log_path) if log_path else None,
        "stopReason": result.get("stopReason"),
    }, indent=2))
    await td.close()
    return 0 if result.get("stopReason") in ("end_turn", "stop") else 1


async def _cmd_cancel(args: argparse.Namespace) -> int:
    """``cancel <sid>`` — best-effort cancel; never raises."""
    td = _build_dispatcher(args)
    try:
        await td.cancel(args.sid)
    except Exception as e:  # pragma: no cover - defensive
        print(f"warning: cancel failed: {e}", file=sys.stderr)
    await td.close()
    return 0


async def _cmd_steer(args: argparse.Namespace) -> int:
    """``steer <sid> <new_task>`` — cancel + re-prompt."""
    td = _build_dispatcher(args)
    try:
        result = await td.steer(args.sid, args.new_task)
    except TaskDispatcherError as e:
        print(f"error: {e}", file=sys.stderr)
        await td.close()
        return 1
    await td.close()
    stop_reason = result.get("stopReason", "unknown")
    print(json.dumps({"stopReason": stop_reason, "payload": result}, indent=2))
    return 0 if stop_reason in ("end_turn", "stop") else 1


async def _cmd_status(args: argparse.Namespace) -> int:
    """``status <sid>`` — print the last N events as NDJSON."""
    td = _build_dispatcher(args)
    try:
        events = await td.status(args.sid, n=args.tail)
    except TaskDispatcherError as e:
        print(f"error: {e}", file=sys.stderr)
        await td.close()
        return 1
    await td.close()
    for evt in events:
        print(json.dumps(evt))
    if not events:
        print(f"(no events found for sid={args.sid!r})", file=sys.stderr)
        return 0
    return 0


async def _cmd_replay(args: argparse.Namespace) -> int:
    """``replay <sid>`` — stream all events as NDJSON."""
    td = _build_dispatcher(args)
    try:
        count = 0
        async for evt in td.replay(args.sid):
            print(json.dumps(evt))
            count += 1
    except TaskDispatcherError as e:
        print(f"error: {e}", file=sys.stderr)
        await td.close()
        return 1
    await td.close()
    if count == 0:
        print(f"(no events found for sid={args.sid!r})", file=sys.stderr)
    return 0


# ---------------- argparse wiring ----------------


# Map subcommand name → (handler, takes_sid)
# ``takes_sid`` is True for cancel/status/replay (sid is a positional),
# and False for run/start/steer (which take <task> as positional).
_HANDLERS: dict[str, Callable[[argparse.Namespace], Awaitable[int]]] = {
    "run": _cmd_run,
    "start": _cmd_start,
    "cancel": _cmd_cancel,
    "steer": _cmd_steer,
    "status": _cmd_status,
    "replay": _cmd_replay,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adapter",
        description=(
            "Reasonix subagent adapter — Python supervisor over JSON-RPC/ACP. "
            "Run one task, manage live sessions, or replay past ones."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"adapter {_VERSION}",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # ---- run <task> ----
    p_run = sub.add_parser(
        "run", help="Start a session, prompt with <task>, and print the final stop reason.",
    )
    p_run.add_argument("task", help="The task string to send as the user prompt.")
    _add_shared_flags(p_run)

    # ---- start <task> ----
    p_start = sub.add_parser(
        "start", help="Like ``run`` but prints {sid, pid, log_path} JSON for later control.",
    )
    p_start.add_argument("task", help="The task string to send as the user prompt.")
    _add_shared_flags(p_start)

    # ---- cancel <sid> ----
    p_cancel = sub.add_parser(
        "cancel", help="Cancel an in-flight session by sid (best-effort, idempotent).",
    )
    p_cancel.add_argument("sid", help="The session id to cancel.")
    _add_shared_flags(p_cancel)

    # ---- steer <sid> <new_task> ----
    p_steer = sub.add_parser(
        "steer", help="Cancel <sid> and re-prompt with <new_task>.",
    )
    p_steer.add_argument("sid", help="The session id to steer.")
    p_steer.add_argument("new_task", help="The new task string to send after the cancel.")
    _add_shared_flags(p_steer)

    # ---- status <sid> ----
    p_status = sub.add_parser(
        "status", help="Print the last N events from the session's NDJSON log.",
    )
    p_status.add_argument("sid", help="The session id to inspect.")
    p_status.add_argument(
        "--tail", type=int, default=20,
        help="Number of trailing events to print (default: 20).",
    )
    _add_shared_flags(p_status)

    # ---- replay <sid> ----
    p_replay = sub.add_parser(
        "replay", help="Stream all events from the session's NDJSON log.",
    )
    p_replay.add_argument("sid", help="The session id to replay.")
    _add_shared_flags(p_replay)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = _HANDLERS[args.command]
    try:
        return asyncio.run(handler(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130  # 128 + SIGINT(2), per shell convention
    except SupervisorError as e:
        # Raised during _build_dispatcher's binary pre-check OR from
        # the supervisor itself if it dies before we get a chance to
        # wrap the call. The handlers above should catch most cases,
        # so this is the catch-all.
        print(f"error: {e}", file=sys.stderr)
        return 3


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    sys.exit(main())
