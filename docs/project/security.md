# Security — detail

Moved out of [PROMETHEUS.md](../../PROMETHEUS.md) on 2026-10-03 so that file fits the
12,000-character cap on a project instruction file. The text below is the original
section, word for word. Its rules are also kept in PROMETHEUS.md, which is
authoritative: where this copy disagrees, this copy is stale.

## Security Philosophy

Prometheus is designed for sovereign single-operator deployment on
dedicated hardware. The security model protects the operator from
autonomous agent actions, not from co-tenants. User-initiated commands
via Telegram have full trust. Background and self-improvement tasks run
under restricted trust with scanner verification.

### Trust Model
- User says it in Telegram → full trust, no blocks
- Background tasks (SENTINEL, AutoDream, cron) → SecurityGate applies
- External code from SYMBIOTE harvest → DangerousCodeScanner applies
- Self-improvement output (GEPA, SkillRefiner) → scanner applies
- Credentials loaded from local config files → always allowed
- Network commands (pip, curl) initiated by user → always allowed

This mirrors Hermes Agent's single-tenant assumption but without Docker
isolation, as Prometheus runs on dedicated hardware where the machine
itself is the security boundary.

### Origin classification
The trust origin is derived from `LoopContext.session_id`:
- `telegram:<chat_id>`, `slack:<channel>`, `cli`, `web` → **user**
- `system`, `None`, SYMBIOTE/GEPA/SENTINEL UUIDs → **system**

Helper: `prometheus.permissions.checker.origin_from_session_id()`.
Default for unrecognized values is `system` (the safer classification).

## Security
Shared security utilities live in `src/prometheus/security/`.

- `SecurityGate` (`permissions/checker.py`) — Trust-level evaluator wired
  into `AgentLoop` as `permission_checker`. Takes an `origin` parameter:
  `user` skips ExfiltrationDetector and the network/install
  approve-patterns; `system` applies the full restriction set. Always-
  blocked patterns (`rm -rf /`, `mkfs`, fork bomb), `denied_commands`,
  `denied_paths`, and the write_file workspace gate fire in BOTH origins.
- `ExfiltrationDetector` (`permissions/exfiltration.py`) — bash-command
  pattern detector. Flags only when an actual sensitive *file* on disk
  shows up in a network command (cat ~/.ssh/, < ~/.aws/, $(cat ~/...),
  pipes/redirects from sensitive paths, base64+sensitive_path+network).
  Bare `$VAR`-style env-var references are no longer flagged — that
  pattern is too coarse to distinguish exfil from legitimate auth.
- **`DangerousCodeScanner`** (`security/code_scanner.py`) — AST-based
  static-analysis pass on Python source. Flags `exec/eval/compile/__import__`
  and `os.system/popen/exec*/pty.spawn/ctypes.CDLL` at any scope, plus
  `subprocess/socket/httpx/requests/urllib` at module scope (suspicious,
  not blocking). Returns `ScanResult(verdict: clean|suspicious|dangerous)`.
  First introduced in GRAFT-SYMBIOTE for harvest-time gating; promoted to
  shared so any subsystem (hooks, audit pipelines, future eval gates) can
  reuse it. The old import path
  `prometheus.symbiote.code_scanner` still works via a re-export shim.
  - `scan_markdown_content()` — extracts Python from `​```python` /
    `​```py` fenced code blocks in markdown and runs the AST scan on each.
    Used by GEPA's promotion gate (`learning/gepa.py::_promote_winner`)
    and SkillRefiner (`learning/skill_refiner.py::maybe_refine`) before
    AI-generated skill variants are written to disk. A `dangerous`
    verdict refuses the write silently and continues.
- **`assert_path_under_roots`** (`security/path_guard.py`) — write-boundary
  helper. Resolves a candidate path BEFORE checking against an allow-list
  of roots, so `../` traversals that escape the allow-list are rejected
  even when the literal input starts with an allowed prefix. Used by
  `MemoryExtractor`'s `ObsidianWriter` to confine its write surface to
  `~/.prometheus/` (covering `MEMORY.md` and `wiki/`).

## Security Conventions

### Path Traversal Defense
Always resolve paths before prefix-checking. Never check prefix on the raw input string.

WRONG:

    if not str(user_path).startswith(str(allowed_root)):  # bypassable with ../

RIGHT:

    resolved = Path(allowed_root / user_path).resolve()
    if not str(resolved).startswith(str(allowed_root.resolve())):
        raise SecurityError(f"Path traversal attempt: {user_path}")

First caught in GRAFT-SYMBIOTE Step 5 (`GraftEngine._resolve_target`).
Test: `tests/test_graft.py::TestAllowedRoots::test_rejects_traversal`.
