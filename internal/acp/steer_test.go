package acp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// TestSessionSteer_InvalidParams covers the request-validation branches the
// handler hits BEFORE it touches the session registry: bad JSON, missing
// sessionId, unsafe sessionId (path traversal etc.), and empty/whitespace
// text. All four must surface as ErrInvalidParams (-32602) and the handler
// must NOT touch the session map.
func TestSessionSteer_InvalidParams(t *testing.T) {
	svc := &service{sessions: make(map[string]*acpSession)}

	cases := []struct {
		name string
		raw  json.RawMessage
	}{
		{"malformed JSON", json.RawMessage(`{not json`)},
		{"missing sessionId", json.RawMessage(`{"text":"hello"}`)},
		{"empty sessionId", json.RawMessage(`{"sessionId":"","text":"hello"}`)},
		{"unsafe sessionId (path traversal)", json.RawMessage(`{"sessionId":"../etc","text":"hello"}`)},
		{"empty text", json.RawMessage(`{"sessionId":"abc","text":""}`)},
		{"whitespace-only text", json.RawMessage(`{"sessionId":"abc","text":"   \n\t  "}`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := svc.sessionSteer(context.Background(), tc.raw)
			if err == nil {
				t.Fatalf("expected error, got result %#v", res)
			}
			rpcErr, ok := err.(*RPCError)
			if !ok {
				t.Fatalf("expected *RPCError, got %T: %v", err, err)
			}
			if rpcErr.Code != ErrInvalidParams {
				t.Errorf("Code = %d, want %d (ErrInvalidParams)", rpcErr.Code, ErrInvalidParams)
			}
			if !strings.Contains(rpcErr.Message, "session/steer") {
				t.Errorf("Message = %q, want it to mention session/steer", rpcErr.Message)
			}
		})
	}
}

// TestSessionSteer_UnknownSession covers the "valid params but session id is
// not in the registry" branch. The handler must report invalid params rather
// than silently creating a session — callers should not see a Queued=true
// response for a non-existent session. This also implicitly guards the
// `s.session(...) == nil` path against future regressions where the handler
// might be tempted to call `s.sessions[pid] = ...` for unknown ids.
func TestSessionSteer_UnknownSession(t *testing.T) {
	svc := &service{sessions: make(map[string]*acpSession)}
	raw := json.RawMessage(`{"sessionId":"ghost","text":"hello"}`)
	_, err := svc.sessionSteer(context.Background(), raw)
	if err == nil {
		t.Fatal("expected error for unknown session id, got nil")
	}
	rpcErr, ok := err.(*RPCError)
	if !ok {
		t.Fatalf("expected *RPCError, got %T", err)
	}
	if rpcErr.Code != ErrInvalidParams {
		t.Errorf("Code = %d, want %d", rpcErr.Code, ErrInvalidParams)
	}
	if !strings.Contains(rpcErr.Message, "unknown session id") {
		t.Errorf("Message = %q, want it to mention unknown session id", rpcErr.Message)
	}
}

// TestSessionSteer_DoesNotMutateSessionsOnError guards a subtle invariant: if
// validation fails, the handler must not insert a new acpSession into the
// registry. This catches future regressions where someone might be tempted
// to "auto-create" a session as a convenience — that would violate the
// session/close contract and confuse clients that poll session/list.
func TestSessionSteer_DoesNotMutateSessionsOnError(t *testing.T) {
	svc := &service{sessions: make(map[string]*acpSession)}
	before := len(svc.sessions)

	// Try several error paths; registry size must stay at 0.
	rawBadJSON := json.RawMessage(`{not json`)
	if _, err := svc.sessionSteer(context.Background(), rawBadJSON); err == nil {
		t.Fatal("expected error for malformed JSON")
	}
	rawEmptySID := json.RawMessage(`{"sessionId":"","text":"x"}`)
	if _, err := svc.sessionSteer(context.Background(), rawEmptySID); err == nil {
		t.Fatal("expected error for empty sessionId")
	}
	rawUnknown := json.RawMessage(`{"sessionId":"nope","text":"x"}`)
	if _, err := svc.sessionSteer(context.Background(), rawUnknown); err == nil {
		t.Fatal("expected error for unknown session id")
	}

	if got := len(svc.sessions); got != before {
		t.Errorf("session registry mutated on error: before=%d after=%d", before, got)
	}
}

// Note: the success path of sessionSteer (valid params + valid session) is
// covered by the Python integration test in P2.1 T3, which drives the full
// stack (acp.Serve → control.Controller → agent.Agent) with a real Reasonix
// binary and asserts the response shape. Adding a unit test for it would
// require duplicating the e2eFactory wiring from e2e_test.go, which is
// significant overhead for marginal coverage gain.
