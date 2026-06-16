package control

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestCodegraphIndexRoot exercises the env-var resolver used to pin the
// codegraph MCP server to an explicit directory. It must NEVER fall back to a
// default — a forgotten env var should fail loudly so a misconfigured deploy
// cannot index the entire workspace by accident.
func TestCodegraphIndexRoot(t *testing.T) {
	// Capture and restore the env var so subtests don't leak into each other
	// or into the rest of the test binary.
	prev, hadPrev := os.LookupEnv("REASONIX_CODEGRAPH_ROOT")
	t.Cleanup(func() {
		if hadPrev {
			_ = os.Setenv("REASONIX_CODEGRAPH_ROOT", prev)
		} else {
			_ = os.Unsetenv("REASONIX_CODEGRAPH_ROOT")
		}
	})

	cases := []struct {
		name      string
		setEnv    bool
		envValue  string
		wantErr   bool
		errSubstr string
		wantPath    string // exact absolute path expected; only checked when !wantErr
		wantAbsOnly bool   // if true, assert result is absolute (no exact match)
	}{
		{
			name:     "absolute path passes through",
			setEnv:   true,
			envValue: "/tmp/codegraph-test",
			// On macOS /tmp is /tmp; on Linux it's usually /tmp. We resolve
			// the expected via filepath.Abs so the test is portable.
			wantPath: func() string { p, _ := filepath.Abs("/tmp/codegraph-test"); return p }(),
		},
		{
			name:        "relative path is resolved to absolute",
			setEnv:      true,
			envValue:    "rel/path",
			wantAbsOnly: true, // see post-check switch
		},
		{
			// REASONIX_FORK: post-v1.8.0 merge, codegraphIndexRoot is permissive
			// when the env is unset/empty/whitespace (returns "", nil). PicoClaw's
			// wrapper always sets the env, so the wrapper path keeps its strict
			// pinning; the legacy connectCodegraphMCPServer(cfg) path falls back
			// to cwd when the env is missing (v1.8.0 test contract).
			name:     "unset env falls back to empty path (no error)",
			setEnv:   false,
			wantPath: "",
		},
		{
			name:     "empty env treated as unset (no error)",
			setEnv:   true,
			envValue: "",
			wantPath: "",
		},
		{
			name:     "whitespace-only env treated as unset (no error)",
			setEnv:   true,
			envValue: "   \t  ",
			wantPath: "",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.setEnv {
				t.Setenv("REASONIX_CODEGRAPH_ROOT", tc.envValue)
			} else {
				_ = os.Unsetenv("REASONIX_CODEGRAPH_ROOT")
			}

			got, err := codegraphIndexRoot()

			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil (path=%q)", got)
				}
				if !strings.Contains(err.Error(), tc.errSubstr) {
					t.Fatalf("error %q does not contain %q", err.Error(), tc.errSubstr)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			// Three post-check branches:
			//   - tc.wantPath == "" (sentinel): helper must return "" (no path)
			//   - tc.wantPath is absolute: exact match
			//   - tc.wantPath == "" in the struct but with the abs-check sentinel:
			//     handled separately via wantAbsOnly flag
			switch {
			case tc.wantPath == "" && !tc.wantAbsOnly:
				if got != "" {
					t.Fatalf("expected empty path, got %q", got)
				}
			case tc.wantAbsOnly:
				if !filepath.IsAbs(got) {
					t.Fatalf("relative env did not resolve to absolute: got %q", got)
				}
			default:
				if got != tc.wantPath {
					t.Fatalf("path mismatch: got %q want %q", got, tc.wantPath)
				}
			}
		})
	}
}
