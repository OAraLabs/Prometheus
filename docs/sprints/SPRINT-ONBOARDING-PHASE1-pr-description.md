<!--
RESCUED WRITE-UP — provenance, because this file was nearly lost.

Source:   PR-DESCRIPTION.md, UNCOMMITTED working copy of the worktree on
          branch `feat/onboarding-phase1` (last commit 2026-07-05; the branch
          was 124 commits behind origin/main when this was recovered).
Rescued:  2026-08-15, during a prune of 27 stale worktrees.
Author:   the session that built Onboarding Phase 1. Not me — I moved it.

WHY IT WAS AT RISK. `PR-DESCRIPTION.md` is a single tracked file at the repo
root that EVERY branch overwrites with its own PR text. `main` currently
carries `feat/api-provider-keys`'s version; this branch's committed version
is Phase 0's; only the uncommitted working copy held Phase 1. So the sole
copy of this write-up existed as an uncommitted diff on a long-stale branch,
one `git worktree remove --force` away from gone.

STATUS OF THE WORK IT DESCRIBES: NOT MERGED. `feat/onboarding-phase1` still
holds the code. This document is the description, preserved verbatim below —
it has NOT been re-verified against current main, and the line numbers,
commit SHAs and API shapes it cites are as of 2026-07-05.

The content below is unmodified.
-->

# feat(onboarding): PHASE 1 — pairing handshake (setup-mode daemon + wizard "Connect a client" end screen)

**Branch:** `feat/onboarding-phase1` off `main` (`6e55075`)
**Spec:** `docs/sprints/SPRINT-ONBOARDING-OVERHAUL.md` §2 PHASE 1 (items 1+2; item 3 is the
companion beacon-desktop PR `feat/pairing-firstrun`)
**Status:** PR-ready — not merged, not pushed to main. Builds directly on Phase 0 (#86):
`config/api_token.py` ensure/rotate machinery, `config/env_file.py`, the one canonical
`prometheus setup`.

## What changed

### Item 1 — Wizard "Connect a client (Beacon)" end screen
- `config/api_token.py`: new `format_connect_client_block(config, api_port)` — boxed block
  with the address (`socket.gethostname()` + `web.api_port`, **never a hardcoded host**, plus
  a "or this machine's Tailscale / LAN address" hint) and the API token: the actual value when
  one exists (config/env/env-file, via `resolve_api_token`), otherwise
  "minted on first daemon start — re-print with `prometheus token show`". Print-only — it
  never writes the token anywhere.
- Wired at the END of **both** paths: `setup_wizard._print_summary` (rich) and
  `cli/init.run_init` (fast/`--noninteractive`).

### Item 2 — Daemon setup mode + pairing API
- New `src/prometheus/web/setup_server.py` — a **dedicated minimal FastAPI app**, not a flag
  on `web/server.py`: the full route surface is not mounted, so it is unreachable by
  construction.
  - `GET /api/setup/status` → `{setup_mode: true, configured: false, pairing:
    "available"|"locked", version}`
  - `POST /api/setup/pair` `{code}` → `200 {token, api_base_port, ws_port}` on success;
    `401 {error: invalid_code, attempts_remaining}` on a wrong code;
    `403 {error: pairing_locked|pairing_expired|pairing_used}` otherwise
  - **everything else** (any method, any path) → `403 {error: setup_mode, detail: …}`
- Pairing code: 6 digits (`secrets.randbelow`, zero-padded), printed ONCE in a banner
  style-matched to `format_minted_banner`. One-time use; **15-min TTL**; **max 5 failed
  attempts** → `"locked"`; compared via `hmac.compare_digest`; neither code nor token logged
  after the banner. **Documented decision:** an expired/locked/used code is NOT re-minted in
  place — restart `prometheus daemon` for a new one (keeps "printed once" true).
- On success: token minted/persisted via the existing `ensure_api_token` → env file
  (`set_env_value`). A pre-existing env-file token is **reused, not replaced**, and
  `resolve_api_token` later finds the same token — the paired client keeps working after
  `prometheus setup` + real daemon start (test-pinned + live-proven below).
- `daemon.py main()`: no config found (explicit `--config` → repo-local
  `config/prometheus.yaml` → `$PROMETHEUS_CONFIG_DIR`/`~/.prometheus`) → boots setup mode
  instead of half-starting on defaults. Detection via a new no-mkdir `find_config_file`
  (`get_config_dir()` would create `~/.prometheus`); the gate runs BEFORE the file logger so
  setup mode creates **no** `~/.prometheus` state (test-pinned). Logs loudly that it is in
  setup mode; SIGTERM/SIGINT shut down cleanly. Port: 8005 default, `PROMETHEUS_WEB_API_PORT`
  override (there is no config to read a port from in setup mode).
- `prometheus` (interactive CLI) with no config keeps today's behavior (error + pointer to
  `prometheus setup`) — verified unchanged.

## Tests

- Suite: **3100 passed, 1 failed (pre-existing, known:
  `tests/test_bootstrap.py::TestMemoryInPrompt::test_empty_memory_files_no_section`),
  4 skipped** — verified the same single failure exists on clean main `6e55075`.
- New `tests/test_setup_server.py` (24): pair happy path; env-file persist; **token
  continuity into `resolve_api_token`**; pre-existing token reused; wrong code ×5 counts
  down 4→0 then locks (right code post-lock also rejected); TTL expiry rejected+locked
  (within-TTL passes); one-time reuse rejected; malformed body = 400 and burns no attempt;
  8 parametrized non-setup routes → 403 (incl. with a *valid* bearer); banner style/content;
  `PROMETHEUS_WEB_API_PORT`; `find_config_file` search order + no-mkdir pin.
- New `tests/test_connect_block.py` (11): hostname-not-hardcoded; port from config;
  env-file token printed; mint-note fallback; never writes the env file; block present at
  the end of the fast path (real `run_init` against a fake llama.cpp HTTP server) and the
  rich path (`_print_summary`, block is the last thing on screen).
- Test-hygiene note baked into the fixture: `ensure_api_token` exports the minted token into
  `os.environ`; `monkeypatch.delenv` on an absent var records nothing, so the fixture pops it
  explicitly (without this, 5 unrelated web-API tests 401'd later in the suite).

## Live acceptance (isolated: tmp `PROMETHEUS_CONFIG_DIR` + `PROMETHEUS_ENV_FILE`, high ports; live daemon untouched)

**(1) Setup mode boots + prints the code** (`PROMETHEUS_WEB_API_PORT=18205`):

```
WARNING No prometheus.yaml found — starting in SETUP MODE (pairing-only API on :18205; …)
====================================================================
  PROMETHEUS IS IN SETUP MODE — no configuration found

  Pairing code (printed once — valid 15 min, one client, 5 tries):

    836075
  …
====================================================================
INFO:     Uvicorn running on http://0.0.0.0:18205
```

**(2) Pair flow via curl:**

```
GET  /api/setup/status → {"setup_mode":true,"configured":false,"pairing":"available","version":"0.1.0"}
POST /api/setup/pair {"code":"000000"} → 401 {"error":"invalid_code","attempts_remaining":4,…}
POST /api/setup/pair {"code":"836075"} → 200 {"token":"gveEICC…","api_base_port":18205,"ws_port":8010}
POST /api/setup/pair {"code":"836075"} → 403 {"error":"pairing_used",…}
```

env file after pair: `PROMETHEUS_API_TOKEN=gveEICC…` · tmp config dir: **empty** (no state created).

**(3) Any other route → 403:**

```
GET  /api/status     → 403 {"error":"setup_mode","detail":"…only /api/setup/* is available…"}
POST /api/chat/send  → 403 (same body)
```

SIGTERM → `Application shutdown complete … Setup-mode server stopped.` — clean exit.

**(4) Token continuity end-to-end:** `prometheus setup --noninteractive` into the same tmp
config (detected the local Ollama) — its end screen printed the SAME token pairing minted:

```
====================================================================
  CONNECT A CLIENT (Beacon)

    Address:  <hostname>:8005
              (or this machine's Tailscale / LAN address, port 8005)
    Token:    gveEICC…   (stored in <tmp env file>)
====================================================================
```

Then the REAL daemon on the tmp config (ports edited to 18205/18210):
`GET /api/status` without token → **401**; with the paired token → **200**
(`{"state":"idle","model":"qwen2.5:14b-instruct",…}`).

**(5)** Beacon transport check against this daemon: see the companion PR (5/5 live checks).

## Deviations / notes

- Expiry policy: restart-to-re-mint (allowed by spec, documented above and in the banner).
- `ws_port` in the pair answer is the default 8010 (env-overridable) — setup mode has no
  config to read a real value from; Beacon treats it as advisory with the same defaults.
- The setup-mode port override is a setup-mode-only env var (`PROMETHEUS_WEB_API_PORT`),
  not a new global config override — the general `ENV_OVERRIDES` map was left untouched.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
