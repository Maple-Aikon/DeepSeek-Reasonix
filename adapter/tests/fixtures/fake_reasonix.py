#!/usr/bin/env python3
"""Fake reasonix binary for supervisor tests.

Reads JSON-RPC 2.0 NDJSON from stdin, emits scripted responses to stdout,
optional diagnostics to stderr. Behavior is controlled by env vars:

  FAKE_REASONIX_MODE=happy|crash|hang|malformed|slow_init
  FAKE_REASONIX_INIT_DELAY=<seconds>   (for slow_init mode)
  FAKE_REASONIX_SESSION_DELAY=<seconds> (per-session/new delay)
  FAKE_REASONIX_PROMPT_DELAY=<seconds>   (per-prompt delay; pauses before emitting events)
  FAKE_REASONIX_VERBOSE=1               (log requests to stderr)

Modes:
  happy     - acknowledge initialize, ack session/new, emit one agent_message_chunk + stop per prompt
  crash     - exit(1) immediately (after emitting nothing)
  hang      - never respond to any request (test cancel/timeout)
  malformed - emit a non-JSON line then a valid response
  slow_init - wait FAKE_REASONIX_INIT_DELAY seconds before responding to initialize
"""
from __future__ import annotations
import json
import os
import sys
import time


def main() -> int:
    mode = os.environ.get("FAKE_REASONIX_MODE", "happy")
    init_delay = float(os.environ.get("FAKE_REASONIX_INIT_DELAY", "0"))
    sess_delay = float(os.environ.get("FAKE_REASONIX_SESSION_DELAY", "0"))
    verbose = os.environ.get("FAKE_REASONIX_VERBOSE") == "1"

    def log(msg: str) -> None:
        if verbose:
            print(f"[fake-reasonix] {msg}", file=sys.stderr, flush=True)

    if mode == "crash":
        print("crashing as instructed", file=sys.stderr, flush=True)
        return 1

    if mode == "hang":
        # Wait forever; let parent kill us
        log("hang mode: sleeping until killed")
        try:
            time.sleep(3600)
        except KeyboardInterrupt:
            pass
        return 0

    if mode == "malformed":
        # Emit a garbage line, then a valid initialize response
        print("this is not json {{{", flush=True)
        time.sleep(0.05)

    # Read loop: parse one line at a time
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"malformed request: {e}")
            continue
        method = frame.get("method", "")
        req_id = frame.get("id")
        log(f"recv {method} (id={req_id})")

        if method == "initialize":
            if init_delay > 0:
                log(f"slow_init: sleeping {init_delay}s")
                time.sleep(init_delay)
            result = {
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": True,
                    "sessionCapabilities": {
                        "list": {}, "resume": {}, "close": {}, "delete": {}
                    },
                    "promptCapabilities": {
                        "image": False, "audio": False, "embeddedContext": True
                    },
                    "mcpCapabilities": {"http": False, "sse": False}
                },
                "agentInfo": {"name": "fake-reasonix", "version": "test-fixture-0.0.1"},
                "authMethods": []
            }
            send({"jsonrpc": "2.0", "id": req_id, "result": result})
        elif method == "session/new":
            if sess_delay > 0:
                time.sleep(sess_delay)
            sid = f"fake-sid-{int(time.time()*1000)}"
            send({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": sid}})
        elif method == "session/prompt":
            # Read sessionId from params (we do not enforce it; just emit events)
            params = frame.get("params") or {}
            sid = params.get("sessionId", "fake-sid-0")
            prompt_delay = float(os.environ.get("FAKE_REASONIX_PROMPT_DELAY", "0"))
            if prompt_delay > 0:
                time.sleep(prompt_delay)
            # Emit a single agent_message_chunk and a stop update.
            # Wire format matches internal/acp/protocol.go SessionUpdateParams
            # (nested ``update`` wrapper, NOT flat ``sessionUpdate``).
            send({
                "jsonrpc": "2.0", "method": "session/update",
                "params": {
                    "sessionId": sid,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "hello from fake-reasonix"}
                    }
                }
            })
            send({
                "jsonrpc": "2.0", "method": "session/update",
                "params": {
                    "sessionId": sid,
                    "update": {
                        "sessionUpdate": "stop",
                        "stopReason": "end_turn"
                    }
                }
            })
            # The actual session/prompt response (no payload required)
            send({"jsonrpc": "2.0", "id": req_id, "result": {"stopReason": "end_turn"}})
        elif method == "session/cancel":
            # Acknowledge and stop
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        else:
            send({
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            })
    return 0


def send(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
