# WEAVE and WEAVE-PRESS — detail

Moved out of [PROMETHEUS.md](../../PROMETHEUS.md) on 2026-10-03 so that file fits the
12,000-character cap on a project instruction file. The text below is the original
section, word for word. Its rules are also kept in PROMETHEUS.md, which is
authoritative: where this copy disagrees, this copy is stale.

## WEAVE — Core Web Tools + Capability Audit

Closed loop wired during the WEAVE sprint (commit on 2026-04-26).
Additive on top of the pre-existing `web_fetch` and `web_search` tools.

### New tools (`src/prometheus/tools/builtin/`)
- **`youtube_transcript`** (`youtube_transcript.py`) — wraps `yt-dlp`
  to fetch subtitle tracks without downloading video. Accepts full
  YouTube URLs, `youtu.be` short URLs, embed/shorts URLs, and bare
  11-character video IDs. Stdlib-only VTT and SRV3 parsers (no
  beautifulsoup). Read-only unless `save_to` is set; all errors
  return `ToolResult(is_error=True)` — never raises.
- **`download_file`** (`download_file.py`) — streams a URL to disk
  with `httpx.AsyncClient.stream`. Default destination
  `~/.prometheus/downloads/`. SSRF protection mirrors `web_fetch`.
  Path-traversal guard rejects `/etc`, `/sys`, `/boot`, `/proc`,
  `/dev`, `~/.ssh` by resolving the path before checking the prefix
  (matches the GraftEngine pattern). Size cap 100MB by default —
  trips on either the server's `Content-Length` header or
  bytes-written during streaming.

### Wiring
- Both tools are exported from `prometheus.tools.builtin` and
  registered in `__main__.create_tool_registry()` in the
  "Web + messaging" group, alongside `WebSearchTool` and
  `WebFetchTool`. The pre-existing tools were not modified.
- `config/prometheus.yaml.default` adds a `web_tools:` section with
  `fetch_timeout_seconds`, `fetch_max_chars`, `search_max_results`,
  `download_dir`, `download_max_mb`, and
  `youtube_transcript_language`. The corresponding `web_tools`
  block in `config/prometheus.yaml` is gitignored per project policy.

### `scripts/web_capability_audit.py`
Subclasses `SmokeTestRunner` from `scripts/smoke_test_tool_calling.py`
into `WebAuditRunner`. Adds:
- `expect_in_output_any` (any-of-list matching) and
  `expect_no_circuit_breaker` assertions.
- Failure classification: `circuit_breaker`, `wrong_tool`,
  `execution_failure`, `wrong_answer`, `timeout`.
- Hard time-budget gating; tasks past the cap are skipped.
- ~36 audit tasks across 7 categories (search, fetch, youtube,
  download, research, graceful, railway).
- JSON + Markdown reports written to `~/.prometheus/audits/`.

### Telegram surface — `/audit`
Three forms, all additive:
| Form | Behaviour |
|---|---|
| `/audit` | Show summary of the most recent audit JSON |
| `/audit run` | Spawn the full audit as a background subprocess |
| `/audit <category>` | Single-category run (search, fetch, etc.) |

The implementation lives at the bottom of `gateway/telegram.py`
(`_cmd_audit`, `_audit_show_last`, `_audit_kick_off`). It does not
touch any existing `/symbiote`, `/gepa`, `/sentinel`, or core
commands.

### Pytest marker
A new `network` marker is registered in `pyproject.toml` so
network-dependent tests can be skipped with
`pytest -m 'not network'`. The existing `integration` marker is
unchanged.

### Tests
- `tests/test_web_tools.py` (52 unit tests) — URL normalization,
  VTT/SRV3 parsing, error classification, path-traversal guard,
  SSRF guard, file-size formatting, network smoke tests behind the
  `network` marker.
- `tests/test_web_audit.py` (18 tests) — category aggregation,
  failure breakdown, circuit-breaker detection, time-budget gating,
  JSON/Markdown report generation, `/audit` Telegram wiring.
- `tests/test_wiring.py::TestWeaveWebToolsWiring` (6) — verifies
  both new tools are registered and the pre-existing `web_fetch` and
  `web_search` are still present.

## WEAVE-PRESS — Printing Press CLI auto-discovery

When Prometheus needs a CLI it doesn't have, it consults a local clone of
[printing-press-library](https://github.com/mvanhorn/printing-press-library)
(~70 Go-based service CLIs: Slack, Cal.com, Airbnb, Sentry, etc.).
On user approval, it `go install`s the binary, copies the bundled
`SKILL.md` into `~/.prometheus/skills/`, and hot-reloads the skill
registry — the new tool is usable in the same conversation.

### `prometheus.tools.printing_press`
- `PrintingPressRegistry` — discovers a library clone in this order:
  explicit `library_path` → `~/printing-press-library/` →
  `/tmp/printing-press-library/` → `~/go/pkg/mod/.../printing-press-library`.
  Returns empty if no clone is found (no GitHub fallback in this sprint).
- `list_available()` — enumerates `cli-skills/<name>/SKILL.md`, parses
  YAML frontmatter, builds `CLIRecord(name, skill_name, category,
  description, install_module, bin_name, skill_path, installed)`.
  Only Go installs are surfaced; npm/other kinds are skipped.
- `search(query)` — fuzzy-matches by name → skill_name → bin_name →
  description with descending weights.
- `install(cli_name)` — runs `go install <module>@latest`, verifies the
  binary lands on PATH or in `~/go/bin/`, copies `SKILL.md`, fires the
  reload callback. Fail-safe: returns `InstallResult(success=False,
  error=...)` for missing Go, missing CLI, subprocess failure, or
  binary-not-found-after-install.
- `update_library()` — `git pull --ff-only` in the clone. Silent-fail
  if not a git checkout.
- `set_reload_callback(fn)` — daemon wires this to
  `SkillRegistry.reload_user_skills` so a fresh install is picked up
  without a daemon restart.

### Bash command-not-found hook
`prometheus.engine.agent_loop._maybe_suggest_printing_press` —
when a bash tool returns `is_error=True` and the output contains
`command not found: <name>` (any of the three shell forms), and the
session is **user-initiated** (origin classifier from the
TRUST-CONTEXT commit), the hook searches the registry and appends a
suggestion to the tool result content:

> 💡 Printing Press has a CLI for this: **cal-com** (productivity).
> To install, the user can run `/press install cal-com` — installation
> requires their explicit approval and will not happen automatically.

The model relays this to the user. Background/automated sessions
(SENTINEL, GEPA, AutoDream, smoke-tests, cron) get **no** suggestions.
Already-installed CLIs that just errored aren't re-suggested.

### Telegram `/press` surface
| Subcommand | Trust | Behaviour |
|---|---|---|
| `/press list [category]` | 2 | Enumerate available CLIs, optionally filtered |
| `/press search <query>` | 2 | Fuzzy match by name / description |
| `/press install <name>` | 1 | Approval-gated `go install` + skill copy |
| `/press installed` | 2 | Show CLIs whose binary is on PATH or in `~/go/bin` |
| `/press update` | 2 | `git pull --ff-only` the library clone |

Install routes through the same `ApprovalQueue` used by `/gepa run`
and `/symbiote graft`. The user sees a `Permission requested:` message
with `/approve <id>` — same UX as every other Trust Level 1 action.

### Skill hot-reload
`SkillRegistry.reload_user_skills()` re-scans
`~/.prometheus/skills/` and merges new or updated entries (purely
additive — never removes existing). `ToolSearchTool.get_skill_registry()`
exposes the live registry so the daemon can wire the reload callback.

### Daemon wiring (`scripts/daemon.py`)
After `ApprovalQueue` is wired:
1. If `printing_press.enabled` is true, build `PrintingPressRegistry`
   with the configured `library_path`.
2. Wire the reload callback to the running `SkillRegistry` via
   `tool_search.get_skill_registry().reload_user_skills`.
3. If `auto_update_library` is true, spawn a non-blocking
   `git pull --ff-only` task on startup.
4. Attach to `agent_loop._tool_metadata["printing_press"]` so the bash
   hook can find it, and to `telegram._printing_press` for `/press`.

### Config (`prometheus.yaml.default`)
```yaml
printing_press:
  enabled: false                  # opt-in
  library_path: null              # auto-detect
  auto_suggest: true              # bash command-not-found hook
  auto_update_library: false      # git pull on startup
```

### Tests (46 new)
- `tests/test_printing_press.py` (40) — frontmatter parsing,
  command-not-found detection across three shell forms, registry
  discovery, fuzzy search, `is_installed` (PATH + ~/go/bin fallback),
  install (subprocess mocked: success, no-Go, unknown CLI, subprocess
  failure), reload callback firing, skill hot-reload (additive +
  idempotent), suggestion hook (positive case, clean output, library
  unavailable, already-installed, suffix stripping), `update_library`
  edge cases.
- `tests/test_wiring.py::TestWeavePressWiring` (6) — Telegram
  `_cmd_press` + helpers exist, registry importable, suggestion
  helper is async, `SkillRegistry.reload_user_skills` exists,
  `ToolSearchTool.get_skill_registry` round-trips, `AgentLoop`
  accepts `tool_metadata` kwarg.

Network-dependent paths (real `go install`, real `git pull`) are
mocked — no live network calls in the test suite.
