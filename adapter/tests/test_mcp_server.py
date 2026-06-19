"""R4 tests: ``adapter.mcp_server`` — the FastMCP glue layer.

The MCP server is intentionally thin: it just registers the 6
:mod:`adapter.tools.*` modules as MCP tools and wires/clears the
dispatcher slot. These tests pin that contract:

  - ``build_server()`` returns a :class:`mcp.server.fastmcp.FastMCP`
    instance (not a coroutine, not None, not a different class).
  - The returned server exposes exactly 6 tools, with the canonical
    names: ``reasonix_delegate``, ``reasonix_status``,
    ``reasonix_replay``, ``reasonix_approve``, ``reasonix_steer``,
    ``reasonix_cancel``.
  - ``set_all_dispatchers(dispatcher)`` propagates the dispatcher
    to all 4 tool modules with module-level slots
    (status/approve/steer/cancel). ``delegate`` is NOT in this set
    because it uses class-based ``DelegateDispatcher`` and does not
    own a module-level dispatcher slot.
  - ``clear_all_dispatchers()`` clears the slot in all 4 modules.
  - The set + clear round-trip is symmetric (no leftover state).
  - ``set_all_dispatchers`` is idempotent (second call replaces,
    no error).
  - The server's ``instructions`` field mentions the 6 tool names
    (so the LLM can self-discover via ``tools/list``).

We don't spawn the stdio wire (that requires a real client) —
we only verify the glue. Real wire behavior is covered by MCP's
own test suite; we trust ``FastMCP.run(transport='stdio')`` to
do the right thing once the tools are registered.

All tests are sync and run in <100ms. No real binary required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adapter import mcp_server
from adapter.tools import (
    reasonix_approve as approve_tool,
    reasonix_cancel as cancel_tool,
    reasonix_delegate as delegate_tool,
    reasonix_status as status_tool,
    reasonix_steer as steer_tool,
)


# The 6 MCP tool names exposed by the server. If you add a tool,
# update this set AND the ``build_server()`` body in mcp_server.py
# (no new tuple needed — the server registers tools inline, not
# via a list of names).
EXPECTED_TOOL_NAMES = {
    "reasonix_delegate",
    "reasonix_status",
    "reasonix_replay",
    "reasonix_approve",
    "reasonix_steer",
    "reasonix_cancel",
}


# All 4 modules that hold a module-level dispatcher slot. The
# server's set_all_dispatchers / clear_all_dispatchers iterate
# over ``_TOOL_MODULES_WITH_SLOT``. We pin the set here so a
# missing module is caught. ``delegate`` is NOT in this set
# (class-based dispatcher pattern).
EXPECTED_TOOL_MODULES_WITH_SLOT = (
    status_tool,
    approve_tool,
    steer_tool,
    cancel_tool,
)


@pytest.fixture(autouse=True)
def _isolate_dispatcher_slots():
    """Save/restore all 4 module dispatcher slots around each test.

    set_all_dispatchers / clear_all_dispatchers touch every module
    in the suite, so we must snapshot+restore to keep tests
    independent (in case prior tests left a dispatcher registered).
    """
    saved = {mod: mod._dispatcher for mod in EXPECTED_TOOL_MODULES_WITH_SLOT}
    for mod in EXPECTED_TOOL_MODULES_WITH_SLOT:
        mod.clear_dispatcher()
    yield
    for mod, prev in saved.items():
        mod.clear_dispatcher()
        if prev is not None:
            mod.set_dispatcher(prev)


# ---------------------------------------------------------------------------
# build_server()
# ---------------------------------------------------------------------------


def test_build_server_returns_fastmcp_instance():
    """``build_server()`` returns a :class:`FastMCP` (not None, not a
    coroutine, not a coroutine wrapper)."""
    from mcp.server.fastmcp import FastMCP

    server = mcp_server.build_server()
    assert isinstance(server, FastMCP), f"got {type(server).__name__}"


def test_build_server_default_name():
    """Default name is ``"reasonix"`` (the MCP server's identity
    on the wire)."""
    server = mcp_server.build_server()
    assert server.name == "reasonix"


def test_build_server_custom_name():
    """``build_server(name="my-fork")`` uses the custom name — useful
    for parallel test instances on the same machine."""
    server = mcp_server.build_server(name="test-instance-42")
    assert server.name == "test-instance-42"


def test_build_server_registers_six_tools():
    """The server exposes exactly 6 tools with the canonical names."""
    server = mcp_server.build_server()

    # FastMCP stores tools in a ToolManager; we use the public-ish
    # ``_tool_manager.list_tools()`` API to enumerate them. This
    # is the same API the MCP server uses internally for
    # ``tools/list`` responses, so it's stable enough for tests.
    tools = server._tool_manager.list_tools()
    tool_names = {t.name for t in tools}

    assert tool_names == EXPECTED_TOOL_NAMES, (
        f"expected exactly {EXPECTED_TOOL_NAMES}, got {tool_names}"
    )


def test_build_server_instructions_mention_all_tools():
    """The server's ``instructions`` field (sent on ``initialize``)
    mentions all 6 tool names so an LLM client can self-discover
    them without a separate ``tools/list`` round trip."""
    server = mcp_server.build_server()
    instructions = server.instructions or ""
    for name in EXPECTED_TOOL_NAMES:
        assert name in instructions, (
            f"tool {name!r} missing from instructions; LLM cannot self-discover"
        )


# ---------------------------------------------------------------------------
# set_all_dispatchers / clear_all_dispatchers
# ---------------------------------------------------------------------------


def test_set_all_dispatchers_propagates_to_all_modules():
    """``set_all_dispatchers(fake)`` puts ``fake`` in all 4 tool
    module slots that have one."""
    fake = MagicMock(name="fake_dispatcher")
    mcp_server.set_all_dispatchers(fake)

    for mod in EXPECTED_TOOL_MODULES_WITH_SLOT:
        assert mod._dispatcher is fake, (
            f"{mod.__name__}._dispatcher is {mod._dispatcher!r}, expected {fake!r}"
        )


def test_clear_all_dispatchers_clears_all_modules():
    """``clear_all_dispatchers()`` sets all 4 slots to None."""
    fake = MagicMock(name="fake_dispatcher")
    mcp_server.set_all_dispatchers(fake)
    mcp_server.clear_all_dispatchers()

    for mod in EXPECTED_TOOL_MODULES_WITH_SLOT:
        assert mod._dispatcher is None, (
            f"{mod.__name__}._dispatcher is {mod._dispatcher!r}, expected None"
        )


def test_set_all_dispatchers_replaces_existing():
    """Second ``set_all_dispatchers`` call replaces the first
    (no error, useful for hot-reload of the supervisor)."""
    fake1 = MagicMock(name="fake1")
    fake2 = MagicMock(name="fake2")
    mcp_server.set_all_dispatchers(fake1)
    mcp_server.set_all_dispatchers(fake2)

    for mod in EXPECTED_TOOL_MODULES_WITH_SLOT:
        assert mod._dispatcher is fake2, (
            f"{mod.__name__}._dispatcher is {mod._dispatcher!r}, expected fake2"
        )


def test_set_all_dispatchers_with_none_clears():
    """``set_all_dispatchers(None)`` is treated as clear (defensive —
    some supervisors pass None on shutdown).

    Note: this works only because the modules' set_dispatcher()
    accept None and store it. If a module were to reject None
    (raise), this test would fail and the design would have to
    change. The current contract is "None is a valid value".
    """
    fake = MagicMock(name="fake_dispatcher")
    mcp_server.set_all_dispatchers(fake)
    mcp_server.set_all_dispatchers(None)

    for mod in EXPECTED_TOOL_MODULES_WITH_SLOT:
        assert mod._dispatcher is not fake, (
            f"{mod.__name__}._dispatcher is still {fake!r} after set_all_dispatchers(None)"
        )


def test_clear_all_dispatchers_is_idempotent():
    """Calling clear twice does not raise."""
    mcp_server.clear_all_dispatchers()
    mcp_server.clear_all_dispatchers()  # second call: should be a no-op


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_tool_modules_with_slot_has_all_four():
    """The internal ``_TOOL_MODULES_WITH_SLOT`` tuple lists all 4
    modules that own a module-level dispatcher slot. Pinning this
    prevents accidental additions or removals."""
    actual = set(mcp_server._TOOL_MODULES_WITH_SLOT)
    expected = set(EXPECTED_TOOL_MODULES_WITH_SLOT)
    assert actual == expected, (
        f"_TOOL_MODULES_WITH_SLOT = {actual}, expected {expected}"
    )


def test_delegate_module_not_in_slot_set():
    """Architectural invariant: ``reasonix_delegate`` uses class-based
    ``DelegateDispatcher`` and is NOT in ``_TOOL_MODULES_WITH_SLOT``.

    If this test fails, someone has either (a) added a module-level
    ``_dispatcher`` slot to ``reasonix_delegate`` (refactor!), or
    (b) accidentally added ``delegate_tool`` to the slot set.
    Either change is fine — but should be a deliberate one with
    the architectural note updated in the module docstrings.
    """
    assert delegate_tool not in mcp_server._TOOL_MODULES_WITH_SLOT, (
        "delegate_tool unexpectedly in _TOOL_MODULES_WITH_SLOT; "
        "the class-based DelegateDispatcher pattern has been broken"
    )


# ---------------------------------------------------------------------------
# End-to-end smoke (no real wire)
# ---------------------------------------------------------------------------


def test_registered_tools_call_into_correct_modules():
    """Verify the registered MCP tools are bound to the canonical
    functions in the tool modules (not copies, not lambdas that
    would mask the contract).

    This catches a refactor that accidentally wraps each tool in a
    lambda that drops the function name (which would break MCP
    debuggers and tool discovery).
    """
    server = mcp_server.build_server()
    tools = {t.name: t for t in server._tool_manager.list_tools()}

    # The FastMCP ``Tool`` wraps the underlying callable. We can
    # inspect the function it wraps via ``tool.fn``.
    assert tools["reasonix_delegate"].fn is delegate_tool.reasonix_delegate
    assert tools["reasonix_status"].fn is status_tool.reasonix_status
    assert tools["reasonix_replay"].fn is status_tool.reasonix_replay
    assert tools["reasonix_approve"].fn is approve_tool.reasonix_approve
    assert tools["reasonix_steer"].fn is steer_tool.reasonix_steer
    assert tools["reasonix_cancel"].fn is cancel_tool.reasonix_cancel
