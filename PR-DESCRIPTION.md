# Onboarding Overhaul — PHASE 0: fix the floor

**Branch:** `feat/onboarding-phase0` off `main` (`dcae6d6`)
**Spec:** `docs/sprints/SPRINT-ONBOARDING-OVERHAUL.md` §1 (audit) + §2 PHASE 0 (contract)
**Status:** PR-ready — not merged, not pushed to main.

A brand-new user previously faced two competing wizards, a web dashboard
that was off by default with no token story, a README that promised a
systemd unit that didn't exist, a `Doctor` class with no CLI, and setup
dead ends that wrote known-broken configs. This PR is all six Phase 0
items.

## What shipped

### 1. ONE canonical wizard: `prometheus setup`
- New subcommand (`src/prometheus/cli/setup.py`) routing to the **rich**
  wizard by default (identity generation + gateway + smoke test —
  `setup_wizard.py`) and to the **fast** probe→yaml→env path with
  `--fast` / `--noninteractive` (`cli/init.py`).
- Thin forwarding aliases, no breakage: `prometheus --setup`,
  `prometheus --setup-gateway-only`, and the `prometheus-init` console
  script all forward to the canonical command (with a one-line notice).
- Wizard writes to the user config dir (`~/.prometheus/prometheus.yaml`)
  on pip installs; checkout installs keep the historical repo-local
  `config/prometheus.yaml` target.

### 2. Web on + token minted
- Every config the wizard writes has `web.enabled: true` (rich path:
  `_apply_wizard_fields`; fast path already had it; the reference
  `config/prometheus.yaml.default` flipped to `true` with a comment).
- **Env-file convention formalized** (`src/prometheus/config/env_file.py`):
  the canonical secrets file is `~/.config/prometheus/env` — exactly what
  the live systemd unit already loads via `EnvironmentFile=`. New:
  `prometheus daemon` now loads it itself at startup (setdefault — a
  systemd-populated environment always wins), so bare and systemd runs
  see the same secrets. `PROMETHEUS_ENV_FILE` overrides the path (tests
  use this exclusively).
- **Token bootstrap** (`src/prometheus/config/api_token.py`): on daemon
  start with web enabled and no token configured anywhere, mint
  `secrets.token_urlsafe(32)`, persist to the env file (0600), export to
  the process env (the web launcher/WS bridge read it), and print it
  ONCE in a loud banner. Every startup logs either
  `web auth: ENABLED (token from …)` or `web auth: OPEN — no token set`.
  An **explicit empty** `PROMETHEUS_API_TOKEN=` (env file or environment)
  is respected as deliberately-open — matching the existing REST/WS
  "empty token = auth off" convention.
- New `prometheus token show` / `prometheus token rotate` subcommands.

### 3. Systemd
- `packaging/prometheus.service` — user unit: `After=network.target`,
  `Restart=on-failure`, `ExecStart` resolving the installed `prometheus
  daemon`, optional `EnvironmentFile=-%h/.config/prometheus/env`.
- `prometheus install-service` (`src/prometheus/cli/service.py`): writes
  the unit to `~/.config/systemd/user/` (resolving the absolute
  `prometheus` binary into ExecStart), runs `daemon-reload`, enables it.
  **Idempotent** (byte-identical unit → up-to-date no-op) and **refuses
  to overwrite a differing existing unit without `--force`** (backs the
  old one up when forced). `--now` opt-in for immediate start.
  `--systemd-dir` / `PROMETHEUS_SYSTEMD_USER_DIR` override the target;
  the systemctl runner is injectable — tests never touch systemd.

### 4. `prometheus doctor`
- CLI entry (`src/prometheus/cli/doctor.py`) exposing the existing
  `infra/doctor.py` Doctor class (anatomy-scan checks, deduped, and with
  its repo-relative Config check superseded) **plus** the extended
  onboarding checks: config found+parses (reporting the search order),
  inference server reachable, model detected, web port free / served by
  Prometheus / squatted by something else, token set (ENABLED vs OPEN),
  `~/.prometheus` dirs writable, whisper available when voice enabled.
- Human-readable ✓/✗/! output with a one-line fix per failure; exit
  code 1 when any check errors. `--no-scan` skips the deep anatomy pass.

### 5. Dead ends killed — "no path may write a known-broken config"
- **Rich wizard**: "I don't have one running yet" and the
  failed-connection path no longer save a config pointing at a dead URL
  (the old behavior literally printed "Saving config with this URL").
  Both now enter a recovery menu: (a) remote base URL (probed before
  accept), (b) cloud provider — a pasted API key is persisted to the
  **env file**, not the yaml and not just "add this to your shell
  profile", (c) install instructions + clean exit with an explicit
  "nothing was written" message.
- **Fast path**: same contract. `--noninteractive` with nothing detected
  prints Ollama/llama.cpp install instructions and exits cleanly (rc 2,
  nothing written) instead of writing a default config aimed at a server
  that isn't there. Interactive gets the same remote/cloud/exit menu; the
  cloud fast path refuses to write when no usable key exists.
- **Detection hardened**: a candidate only counts as a server when its
  models endpoint returns JSON in a known shape (Ollama `models` /
  OpenAI-compat `data`). Found live on this dev box: a web UI answering
  200 HTML on `:8080/v1/models` used to be "detected" as llama.cpp and
  noninteractive setup would have written a broken config pointing at it.

### 6. Papercuts
- **Whisper/voice**: new `voice` optional extra (`faster-whisper`);
  `_detect_whisper_engine` also detects the python package (with a new
  python-API transcription path); a missing engine now produces a loud,
  actionable error (`pip install 'oara-prometheus[voice]' …`) at the
  tool, in the CLI voice loop (which previously swallowed it as an
  empty-recording lookalike), and in `prometheus doctor`. Piper misconfig
  in voice output warns once with the exact config key.
- **Config search order documented**: README, the
  `prometheus.yaml.default` header, and `load_config`'s docstring all
  state the same three-step order (`--config` → repo
  `config/prometheus.yaml` → `$PROMETHEUS_CONFIG_DIR/prometheus.yaml`).
- **Migration offer**: verified `_offer_migration` already returns
  silently when `detect_sources()` finds nothing — now pinned by a test.
- **README rewritten** around the one canonical path:
  `pip install 'oara-prometheus[full]'` → `prometheus setup` →
  `prometheus` / `prometheus daemon`; token section; `install-service`
  makes the long-promised systemd line true; doctor section.

## Tests

Full suite: **3065 passed, 1 failed, 4 skipped** (`uv run pytest`).
The 1 failure is the known pre-existing one on main —
`tests/test_bootstrap.py::TestMemoryInPrompt::test_empty_memory_files_no_section`
— not touched by this branch.

New/updated test files (~90 tests around this PR):
- `tests/test_env_file.py` — env-file parse/update/load (setdefault
  semantics, comment preservation, 0600), token mint/persist/rotate,
  deliberate-blank stays open, idempotent re-mint, banner prints the
  token exactly once, `token show|rotate` CLI.
- `tests/test_setup_command.py` — `setup` arg routing (rich default,
  `--fast`, `--noninteractive` implies fast, `--gateway-only`), plus all
  three aliases (`--setup`, `--setup-gateway-only`, `prometheus-init`).
- `tests/test_install_service.py` — refuses without `--force` (zero
  systemctl calls made), force+backup, idempotent rerun, `--now`,
  packaging-file-vs-template drift guard, env-dir override.
- `tests/test_doctor_cli.py` — mock server up/down/empty (real ephemeral
  HTTP servers), cloud key present/absent, web port free/foreign
  listener, token set/unset (never leaks the value), dirs writable,
  whisper gate, end-to-end exit codes.
- `tests/test_setup_deadends.py` — every no-server path writes a valid
  config or writes NOTHING; the cloud key lands in the env file and
  never in the yaml; wizard web-enable; migration-offer skip.
- `tests/test_cli_init.py` — updated: a write now requires a detected
  server; new HTML-impostor detection test.

All tests are confined to tmp dirs via `PROMETHEUS_CONFIG_DIR`,
`PROMETHEUS_ENV_FILE`, and `PROMETHEUS_SYSTEMD_USER_DIR` — no test
touches `~/.prometheus`, `~/.config/prometheus`, or the live systemd
unit, and the live daemon was never restarted.

## Acceptance (spec §2 Phase 0)

Fresh box simulated in a tmp `PROMETHEUS_CONFIG_DIR` + `PROMETHEUS_ENV_FILE`,
against the live local Ollama (read-only probing):

```
$ export PROMETHEUS_CONFIG_DIR=$SCRATCH/dot-prometheus PROMETHEUS_ENV_FILE=$SCRATCH/env
$ prometheus setup --noninteractive
┌─ Prometheus setup (fast) ─────────────────────────────────────┐
  Config will be written to …/dot-prometheus/prometheus.yaml
└───────────────────────────────────────────────────────────────┘
Probing for local inference servers …
Local inference: 1 server(s) detected:
  • Ollama     @ http://localhost:11434       1ms (4 models, first: qwen2.5:14b-instruct)
Env template written to …/env
Setup complete. Next steps:
  1. Chat now:       prometheus
  2. Always-on:      prometheus daemon   (Beacon dashboard on http://localhost:8005; …)
  Health check anytime:  prometheus doctor
--- exit: 0
```

(Note: a non-inference web UI live on this box's `:8080` answered 200
HTML — correctly rejected by the hardened probe; Ollama was picked.)

Written config (excerpt) — **`web.enabled: true` confirmed**:

```yaml
model:
  provider: ollama
  base_url: http://localhost:11434
  model: qwen2.5:14b-instruct
web:
  enabled: true
  api_port: 8005
  ws_port: 8010
```

```
$ prometheus doctor --config $PROMETHEUS_CONFIG_DIR/prometheus.yaml
prometheus doctor

Platform:
  ✓ Config: loaded …/dot-prometheus/prometheus.yaml
  ✓ Data dirs: writable (…/dot-prometheus)
  ✓ Python: Python 3.11.15
  ✓ uv: installed
  ✓ Data Dir: …/dot-prometheus
  ! Bootstrap: Missing: SOUL.md, AGENTS.md
      fix: Run `prometheus setup` (the rich wizard) to generate identity files.
  ✓ Dependencies: all required packages installed

Connectivity:
  ✓ Inference: ollama reachable at http://localhost:11434
  ✓ Web: Prometheus web API serving on :8005 (auth required)
  ! API token: web auth OPEN — no PROMETHEUS_API_TOKEN set
      fix: Run `prometheus token rotate` (the daemon also mints one automatically …)
  ! Telegram: bot token not configured
      fix: Get a token from @BotFather and set PROMETHEUS_TELEGRAM_TOKEN env var.

Model:
  ✓ Model: detected: qwen2.5:14b-instruct (+3 more)
  · Vision: Qwen 2.5 does not support vision
  ✓ Function Calling: Qwen 2.5 supports tool calling

Resources:
  · Whisper STT: voice disabled — check skipped
  ✓ GPU: NVIDIA GeForce RTX 3090 Ti — 11.7 GB VRAM free
  ✓ Disk: 66.4 GB free

RESULT: OK with 3 warning(s)
--- exit: 0
```

(":8005 auth required" is this dev box's live daemon — on a true fresh
box the check reports "port 8005 free (daemon not running)". The three
warnings are the honest state of a fast noninteractive setup: no
identity files yet, no token minted yet — first daemon start mints it —
and no Telegram.)

Token + install-service also exercised live (against tmp targets only):

```
$ prometheus token rotate       # → 43-char token, saved to $PROMETHEUS_ENV_FILE
$ prometheus token show         # → prints it back, exit 0
$ prometheus install-service    # with PROMETHEUS_SYSTEMD_USER_DIR pointing at a
                                #   dir seeded with a hand-rolled unit:
REFUSING to overwrite existing unit: …/systemd-user/prometheus.service
It differs from what install-service would write.
Inspect it, then re-run with --force to replace it.
--- exit: 1   (existing unit byte-identical afterwards; no systemctl invoked)
```

The daemon-mint path is covered by unit tests (`ensure_api_token`:
mints/persists/exports, idempotent on second start, deliberate-blank
respected); the daemon itself was not started on this box (a live
prometheus.service owns :8005/:8010 — hard rule).

## Deviations / notes

- **Spec's `enable --now`** for install-service: the default is
  `daemon-reload` + `enable` (per the task instruction), with `--now`
  opt-in — safer default on boxes with an existing setup.
- **`prometheus init` alias**: the spec names `prometheus init` as an
  alias; the shipped `prometheus-init` console script forwards (that's
  what exists today as an entry point). A bare `prometheus init`
  subcommand was NOT added — `setup --fast` is the one canonical
  spelling; easy to add later if wanted.
- **Fast-path cloud menu** offers anthropic/openai (per the task's item
  5b); the rich wizard's menu covers all four cloud providers.
- **`config/prometheus.yaml.default` now has `web.enabled: true`** —
  affects anyone who copies the reference default, not just wizard
  output. Deliberate (spec §2.2), flag if unwanted.
- **Detection hardening** (JSON-shape requirement) is a behavior change
  to `detect_local_servers` beyond the letter of the spec; without it,
  noninteractive setup on this very dev box writes a config pointing at
  a web UI. Test-pinned.
- The `voice` extra is not folded into `[full]` (faster-whisper pulls
  ctranslate2 — heavy); doctor/error messages point at
  `oara-prometheus[voice]` explicitly.
- This file replaces the stale root `PR-DESCRIPTION.md` from the merged
  force-search sprint (#77/#78), per the per-branch convention.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
