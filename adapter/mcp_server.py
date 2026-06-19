"""P2.0 MCP server: exposes 6 Reasonix tools over Model Context Protocol.

The MCP server is a thin glue layer. It does NOT contain any
business logic — every tool is a one-line wrapper that calls
into the existing :mod:`adapter.tools.*` modules (which in turn
talk to the live dispatcher — a :class:`DelegateDispatcher` or
:class:`TaskDispatcher` instance).

Tools exposed (6 total — P2.0 + 1 added in this PR):
  - ``reasonix_delegate(prompt, persona=...)``     — start a session
  - ``reasonix_status(sid)``                        — 5-field cost shape
  - ``reasonix_replay(sid, since_seq=0)``           — NDJSON events
  - ``reasonix_approve(sid, decision, feedback=?)`` — unblock approval
  - ``reasonix_steer(sid, text)``                   — queue mid-turn guidance
  - ``reasonix_cancel(sid)``                        — stop in-flight turn

Wire protocol: stdio (default — ``mcp dev`` / ``mcp run`` /
``mcp install`` all use stdio for desktop integration). HTTP/SSE
transports are available but not used here; Claude Desktop
expects stdio, and PicoClaw's tool layer can also be wired to
stdio via the MCP client.

Boot flow (see :func:`main`):
  1. Build a :class:`DelegateDispatcher` wrapping a started
     :class:`Supervisor` (and optional :class:`LogCapture`).
  2. Register that dispatcher in every tool module's slot via
     :func:`set_all_dispatchers`. The same instance is shared —
     the tool layer doesn't care whether the dispatcher is a
     ``DelegateDispatcher`` or ``TaskDispatcher``; both expose
     the same ``status() / replay() / on_approve() / steer() /
     cancel()`` surface.
  3. Start the FastMCP server — it registers the 6 tools and
     blocks on stdin.
  4. On SIGTERM / SIGINT: clear all dispatchers + close the
     supervisor cleanly via :func:`teardown_dispatcher`.

Test strategy (see ``tests/test_mcp_server.py``):
  - We don't spawn the full MCP wire (that requires a real client).
  - We DO verify:
      a) ``build_server()`` returns a FastMCP instance.
      b) 6 tools are registered with the canonical names.
      c) ``set_all_dispatchers / clear_all_dispatchers`` propagate
         to all 4 modules with module-level dispatcher slots
         (status / approve / steer / cancel).
      d) The registered tools are bound to the canonical functions
         in the tool modules (no lambda shadowing).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from adapter.tools import (
    reasonix_approve as approve_tool,
    reasonix_cancel as cancel_tool,
    reasonix_delegate as delegate_tool,
    reasonix_status as status_tool,
    reasonix_steer as steer_tool,
)

log = logging.getLogger(__name__)


# Modules that own a module-level ``_dispatcher`` slot (set via
# ``set_dispatcher`` / cleared via ``clear_dispatcher``). The
# FastMCP server wires them at boot via :func:`set_all_dispatchers`
# and unwires them at shutdown via :func:`clear_all_dispatchers`.
#
# NOTE: ``reasonix_delegate`` is NOT in this list. Its function
# ``reasonix_delegate(prompt, ...)`` takes ALL arguments explicitly
# (no module-level dispatcher slot) — the dispatcher pattern is
# the caller's responsibility for that tool. The other 4 tools
# are thin wrappers that read the shared dispatcher slot.
_TOOL_MODULES_WITH_SLOT = (
    status_tool,
    approve_tool,
    steer_tool,
    cancel_tool,
)


# FastMCP server factory. Split out from build_server so tests can
# inspect the registered tool list without spawning the transport.
def build_server(name: str = "reasonix") -> FastMCP:
    """Construct the FastMCP server with all 6 tools registered.

    The server is intentionally a thin glue layer — every tool is
    a one-line lambda that calls into the existing
    :mod:`adapter.tools.*` modules. We don't redefine contracts
    here; the tool modules own them.

    Returns:
        :class:`mcp.server.fastmcp.FastMCP` instance with 6 tools
        registered. Call :meth:`run` to start the stdio loop, or
        :meth:`_tool_manager.list_tools` to inspect the registered
        tools (used by tests).
    """
    mcp = FastMCP(
        name=name,
        instructions=(
            "Reasonix MCP server: delegates tasks to a local Reasonix "
            "binary via the ACP wire protocol (session/new, "
            "session/prompt, session/status, session/steer, "
            "session/cancel, session/request_permission). "
            "Use reasonix_delegate to start a session, "
            "reasonix_status to check progress, "
            "reasonix_replay to fetch the NDJSON transcript, "
            "reasonix_steer to guide mid-turn, "
            "reasonix_cancel to hard-stop, and "
            "reasonix_approve to unblock an approval-gated session."
        ),
    )

    # Register the 6 tools. Each is a 1-line lambda that delegates
    # to the tool module. Lambda avoids accidentally shadowing
    # function names if a tool needs to be inspected by name.
    mcp.tool()(delegate_tool.reasonix_delegate)
    mcp.tool()(status_tool.reasonix_status)
    mcp.tool()(status_tool.reasonix_replay)
    mcp.tool()(approve_tool.reasonix_approve)
    mcp.tool()(steer_tool.reasonix_steer)
    mcp.tool()(cancel_tool.reasonix_cancel)

    return mcp


def set_all_dispatchers(dispatcher: Any) -> None:
    """Register ``dispatcher`` in every tool module's slot.

    Idempotent — safe to call multiple times. Each module's
    ``set_dispatcher`` is independently thread-safe (per-module
    lock).
    """
    for mod in _TOOL_MODULES_WITH_SLOT:
        if hasattr(mod, "set_dispatcher"):
            mod.set_dispatcher(dispatcher)
        else:
            log.warning("tool module %s has no set_dispatcher() — skipped", mod.__name__)


def clear_all_dispatchers() -> None:
    """Clear the dispatcher slot in every tool module.

    Idempotent. Called on server shutdown so a restart picks up
    a fresh registration.
    """
    for mod in _TOOL_MODULES_WITH_SLOT:
        if hasattr(mod, "clear_dispatcher"):
            mod.clear_dispatcher()


def _build_real_dispatcher() -> Any:
    """Build a real :class:`TaskDispatcher` for production stdio use.

    This is what the ``__main__`` block calls at boot. Tests should
    NOT call this — they pass a MagicMock via :func:`set_all_dispatchers`
    directly.

    Reads optional env vars for tuning:
      - ``REASONIX_BINARY`` (default: ``bin/reasonix`` — the
        built binary checked into the repo at the v1.9.1 rebase).
      - ``REASONIX_CWD`` (default: current working directory).

    Returns:
        A :class:`adapter.task_dispatcher.TaskDispatcher` instance
        wired to a live :class:`adapter.supervisor.Supervisor` and
        :class:`adapter.log_capture.LogCapture`. The caller is
        responsible for ``await sup.start()`` if they want sessions
        to be created immediately.
    """
    # Local imports keep the test path light (no real binary
    # needed for the build_server / set_all_dispatchers tests).
    from adapter.log_capture import LogCapture
    from adapter.supervisor import Supervisor
    from adapter.task_dispatcher import TaskDispatcher

    binary = os.environ.get("REASONIX_BINARY", "bin/reasonix")
    cwd = os.environ.get("REASONIX_CWD", os.getcwd())
    log.info("boot: binary=%s cwd=%s", binary, cwd)

    sup = Supervisor(binary=binary, cwd=cwd, auto_approve=True)
    cap = LogCapture(sup, log_dir=os.path.join(cwd, "logs"), pid_dir=os.path.join(cwd, "pids"))
    td = TaskDispatcher(supervisor=sup, log_capture=cap, cwd=cwd)
    return td


def main() -> None:
    """Boot the MCP server over stdio with a real TaskDispatcher.

    Production entry point. Called by ``mcp dev`` / ``mcp run`` via
    the ``[project.scripts]`` entry in pyproject.toml, or directly
    via ``python -m adapter.mcp_server``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    dispatcher = _build_real_dispatcher()
    set_all_dispatchers(dispatcher)

    mcp = build_server(name="reasonix")

    try:
        # Blocking stdio loop. Returns when stdin closes.
        mcp.run(transport="stdio")
    finally:
        log.info("shutdown: clearing dispatchers")
        clear_all_dispatchers()


if __name__ == "__main__":
    main()
