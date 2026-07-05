# Onboarding Overhaul — PHASE 2 (Part A): setup mode becomes remotely drivable

**Branch:** `feat/onboarding-phase2` off `main` (`4d89ad7`)
**Spec:** `docs/sprints/SPRINT-ONBOARDING-OVERHAUL.md` §2 PHASE 2 (daemon side)
**Pairs with:** beacon-desktop `feat/setup-wizard` (the stepped wizard that drives this API)
**Status:** PR-ready — not merged, branch pushed, main untouched.

## What this is

Phase 1 gave the no-config daemon a pairing-only setup server (code → token). Phase 2 lets the
paired client finish the ENTIRE setup remotely and flips the same process into the real daemon:

```
prometheus daemon            (no config → setup mode, prints 6-digit code)
  POST /api/setup/pair       {code}                → {token, ports}          (Phase 1, unchanged)
  GET  /api/setup/status     → {setup_mode, configured, pairing, version}    (open; discovery probe)
  GET  /api/setup/detect     (authed) → {backends:[{name, provider, base_url, models, latency_ms}]}
                             ?base_url= probes ONE remote/custom URL instead
  POST /api/setup/configure  (authed) {provider, base_url, model, agent_name?, persona?,
                             telegram_token?} → validates by RE-PROBING the backend, writes
                             prometheus.yaml + identity + env-file telegram token; idempotent
  POST /api/setup/complete   (authed) → verifies config parses, replies, then exits the serve
                             loop with a restart sentinel → daemon.main() RE-CHECKS for config
                             and falls through into the REAL daemon boot IN THE SAME PROCESS
```

No systemd dependency: the in-process fallthrough works standalone (under systemd a plain
exit + `Restart=` also works, but is not required).

## Design decisions

- **Auth:** every `/api/setup/*` mutation requires `Authorization: Bearer` with the token
  `resolve_api_token` sees — i.e. the one `/api/setup/pair` minted into the env file. No token
  anywhere → `401 not_paired` (pair first). `GET /api/setup/status` stays open: it is the
  client's discovery probe. Comparisons via `hmac.compare_digest`; tokens never logged.
- **Shared code, not forks:**
  - Detection = `prometheus.cli.init.detect_local_servers` (the Phase 0 JSON-models-shape
    hardening applies over the wire too — an HTML dashboard is not a backend). New shared
    helpers `remote_server_candidates()` (also used by the interactive remote prompt) and
    `probe_backend()` (configure's validation re-probe).
  - Config writer = `prometheus.cli.init._default_config` + `write_config` — the SAME writer
    `prometheus setup --fast` uses. Tested by byte-comparing the model/web yaml sections
    against a `run_init` output.
  - Identity = the SetupWizard's `generate_identity_files`, extended (not forked) with
    `agent_name` + `persona`: `templates/SOUL.md.template` now has `{{AGENT_NAME}}` slots
    (default "Prometheus" keeps the historical rendering byte-identical); a non-empty persona
    appends a `## Persona` section.
- **Ports follow the pairing:** configure writes `web.api_port`/`ws_port` as the ports setup
  mode is serving on (defaults 8005/8010) so the paired client's address keeps working after
  the flip.
- **Dead-end rule holds:** configure re-probes the chosen provider+URL and refuses
  (`400 backend_unreachable`) rather than write a known-broken config.
- **No-state rule holds:** booting setup mode creates nothing; only pair (env-file token) and
  an explicit configure (config dir + identity) write. Pairing lockout/TTL semantics unchanged.
- **Resume:** `/api/setup/status` gains a real `configured: true|false` so a client can resume
  a half-done wizard (configure ran, complete didn't → jump to the wake step).
- **Idempotent configure:** re-POST overwrites cleanly (`backup_existing=False` — the only
  thing it can overwrite is its own earlier output; no backup litter).
- **Telegram token** goes to the env file (`PROMETHEUS_TELEGRAM_TOKEN`, the env-override path
  the daemon already reads), never into the yaml; `gateway.telegram_enabled: true` set.
  Response bodies never contain tokens.

## Files

- `src/prometheus/web/setup_server.py` — detect/configure/complete + auth + `SetupModeState` +
  `SETUP_COMPLETE` sentinel; `run_setup_mode()` returns it after a verified complete.
- `src/prometheus/daemon.py` — `main()` re-checks for config after setup mode exits and falls
  through into the normal boot in the same process (`logging.basicConfig(force=True)` so the
  file handler attaches after the fallthrough).
- `src/prometheus/cli/init.py` — shared `remote_server_candidates()` + `probe_backend()`.
- `src/prometheus/cli/generate_identity.py` + `templates/SOUL.md.template` — `agent_name` /
  `persona` (defaults keep old output byte-identical).
- `tests/test_setup_api_phase2.py` — NEW (26 tests, see below).

## Tests

`PYTHONPATH=$PWD/src uv run pytest` — **3126 passed, 1 failed** = the KNOWN pre-existing
`tests/test_bootstrap.py::TestMemoryInPrompt::test_empty_memory_files_no_section` (not this
branch; fails identically on main). Setup-related files:
`test_setup_server.py + test_setup_api_phase2.py + test_cli_init.py + test_setup_deadends.py +
test_setup_command.py + test_setup_wizard.py + test_connect_block.py + test_config_env.py`
= **125 passed**; identity consumers `test_clean_slate.py + test_wiring.py` = 339 passed.

New coverage (26 tests):
- authed-vs-unauthed: all three mutations 401 pre-pair (`not_paired`) and with a wrong token
  (`unauthorized`); status stays open; no state created by refused calls
- detect: fake JSON backend via `?base_url=` (models + latency), HTML-dashboard rejection,
  unreachable → empty list, non-http scheme → 400
- configure: **byte-compare of model/web yaml sections vs the CLI wizard's output**; summary
  shape with zero token leakage (telegram token → env file, not yaml, not response); identity
  generated with agent name + persona (SOUL.md/AGENTS.md; none without agent_name); unreachable
  + HTML backends refused with nothing written; missing fields 400; idempotent re-configure
  (overwrites cleanly, no backups)
- status.configured false → true transition; complete 409 before configure / on garbage yaml;
  complete 200 + `restart_requested` + stop callback runs AFTER the response
- **subprocess**: no-config daemon → banner code → pair → unauthed-configure 401 → detect →
  configure(TestAgent) → complete → the SAME PID serves the real daemon (`/api/status` 200
  with the paired token), config + SOUL.md verified on disk

## Acceptance (live E2E on this box — isolated tmp dirs, high port 18731; live Ollama probed read-only)

```
== (1) no-config daemon → setup mode:
{"setup_mode":true,"configured":false,"pairing":"available","version":"0.1.0"}
== no state created yet: OK (confdir absent)
== (5) unauthed configure:
{"error":"not_paired","detail":"no client is paired yet — POST /api/setup/pair with the pairing code first, ..."} [HTTP 401]
== (2) pair:
{"token": "<redacted>","api_base_port":18731,"ws_port":18732}
== (3) detect (server-side probe; live local Ollama):
{"backends": [{"name": "Ollama", "provider": "ollama", "base_url": "http://localhost:11434",
  "models": ["qwen2.5:14b-instruct", "llama3.1:8b", "qwen2.5:7b-instruct", "qwen3:32b"], "latency_ms": 1.0}]}
== configure (agent_name TestAgent, Ollama backend):
{"configured": true, "config_path": ".../confdir/prometheus.yaml", "provider": "ollama",
 "model": "qwen2.5:14b-instruct", "agent_name": "TestAgent",
 "identity": {"SOUL.md": "created", "AGENTS.md": "created", ...}, "telegram_token_saved": false,
 "web": {"enabled": true, "api_port": 18731, "ws_port": 18732}}
== status now reports configured:
{"setup_mode":true,"configured":true,"pairing":"locked","version":"0.1.0"}
== complete:
{"restarting":true,"detail":"setup verified — the real daemon is starting in this process. ..."}
== (4) polling /api/status with the paired token (same PID 4091269):
REAL DAEMON UP — /api/status 200 (same process; PID 4091269 still alive: yes)
{'state': 'idle', 'model': 'qwen2.5:14b-instruct', 'provider': 'ollama', 'profile': 'full', ...}
== unauthed /api/status on the REAL daemon: [HTTP 401]
-- SOUL.md exists: yes → "# TestAgent — Core Identity" + "## Persona / calm, precise, briefly witty"
-- config web section: {'enabled': True, 'api_port': 18731, 'ws_port': 18732}
-- config model section: {'provider': 'ollama', 'base_url': 'http://localhost:11434', 'model': 'qwen2.5:14b-instruct', ...}
system: {'name': 'TestAgent', 'version': '0.1.0'}
```

Live state untouched: no `~/.prometheus` / `~/.config/prometheus` writes, `prometheus.service`
never restarted, :8005/:8010 never bound (live daemon verified active + unharmed throughout).

## Deviations / notes

- `agent_name` also lands in `system.name` (summary + future display); owner name is not in the
  API body (spec's fixed shape), so SOUL.md renders the wizard's "User" default for the owner.
- Configure runs blocking work via `asyncio.to_thread`; detect is a sync (threadpool) endpoint.
- The macOS/UI walk of the Beacon wizard that drives this API was NOT done (standing eyes-on debt).
