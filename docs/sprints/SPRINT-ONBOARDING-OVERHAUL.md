# SPRINT: Onboarding Overhaul — from clone-and-pray to a guided setup

> **Drafted:** 2026-07-05 · **Status:** PHASES 0+1+2 ALL SHIPPED 2026-07-05.
> Phase 0 = Prom #86 (6e55075). Phase 1 = Prom #87 (4d89ad7) + beacon #25 (4268f04).
> Phase 2 = Prom #88 (main b354e52: setup mode remotely drivable — authed
> detect/configure/complete + same-process flip into the real daemon) + beacon #26
> (main ccff866: 8-step first-run wizard, quick-connect escape hatch, first-flight
> checklist on Mission home, empty-state copy on Files/Board/Status). Full remote
> journey E2E-verified (pair→detect→configure→complete→same-PID real daemon w/ custom
> SOUL.md); live daemon restarted + smoked each phase.
> PHASE 3 SHIPPED 2026-07-05 — Prom #89 (CI tracked at last + release workflow; the
> first-ever CI run immediately caught an env-dependent test, fixed via #90, main
> b4ce937) + beacon #27 (linux AppImage/deb targets, v0.1.0, release workflow; main
> e318da4). **Draft release "Beacon v0.1.0" LIVE with CI-built arm64 dmg + AppImage +
> deb.** TRACK COMPLETE. Remaining debt: macOS eyes-on walk (pairing UI, wizard,
> first-flight card), publish the draft release when happy, oara.ai install page
> (manual deploy; workflow scope now unblocked).
> **Repos:** Prometheus (daemon-side floor + setup API) and beacon-desktop (UI walkthrough).
> **Audit basis:** two full-repo sweeps 2026-07-05 (Prometheus install/first-run surface;
> Beacon first-launch surface). Findings summarized in §1 with file:line cites.

## North star

A brand-new user goes from "found Prometheus" to **a working chat in under 10 minutes**,
with one obvious path, no dead ends, and a polished UI walkthrough as the flagship route.
Two personas, two paths, one shared floor:

- **Terminal path** (developer): `pip install oara-prometheus[full]` → `prometheus setup`
  → chatting in the CLI. One canonical wizard.
- **UI path** (the flagship): install daemon → daemon prints a pairing line → Beacon's
  setup wizard walks connect → model → identity → gateways → smoke test → guided first chat.

---

## §1 What a new user actually faces TODAY (audit findings)

### Prometheus daemon

**What exists (good bones):**
- A REAL rich wizard: `prometheus --setup` (`src/prometheus/setup_wizard.py`) — detects
  llama.cpp/:8080, Ollama/:11434, LM Studio/:1234, vLLM/:8000; picks model; picks gateway;
  generates identity (SOUL.md/AGENTS.md); runs a smoke test.
- A fast path: `prometheus-init` (`src/prometheus/cli/init.py`) — probe → minimal yaml →
  env template. Has `--noninteractive`.
- A `Doctor` diagnostics class (`src/prometheus/infra/doctor.py`) — model/capability/context
  mismatch detection. **No CLI entry point.**
- Sane auto-creation of `~/.prometheus/*` dirs and DBs on first use.

**The sharp edges (ranked):**
1. **`web.enabled: false` by default** — the Beacon dashboard/API, the marquee feature,
   is OFF. A new user who then downloads Beacon has nothing to connect to. Neither wizard
   enables it or mentions Beacon at all.
2. **No API token story** — `PROMETHEUS_API_TOKEN` is never generated or prompted; no
   feedback at startup that auth is on/off; no `prometheus generate-token`.
3. **README inconsistent** — front section says `pip install oara-prometheus[full]` +
   `prometheus init`; Quick Start says `pip install -e .` + `python3 -m prometheus --setup`;
   references a systemd unit (`systemctl --user enable --now prometheus`) that is NOT
   shipped anywhere in the repo.
4. **Two competing init paths** (`--setup` vs `prometheus-init`) with different outputs
   (identity vs none) and no guidance on which to use.
5. **Wizard dead-end**: no local server detected → "continue with cloud provider" → no
   cloud provider selection exists → user ends with a config that fails on first run.
6. **`prometheus doctor` doesn't exist** as a command despite the class existing.
7. Whisper used for voice but not declared as a dependency (silent failure); config search
   order (repo `config/prometheus.yaml` vs `~/.prometheus/`) undocumented.

### Beacon desktop

**What exists (good bones):**
- A real first-run screen (`FirstRunSetup.tsx`): address + optional token, read-only Test
  probe (`GET /api/status`), keychain storage, Enter-to-connect, honest 401 labels.
- A full Connection Settings modal (⌘,) with separate REST+WS test, fail-loud
  NotAuthorized states across all panels, command palette (⌘K) + shortcut sheet (?).

**The sharp edges (ranked):**
1. **Post-connect silence** — user lands on Mission home (decorative orrery + HUD +
   "A light for the voyage") with no orientation, no next step, no explanation of the 12
   views. The single biggest UX gap.
2. **No distribution** — no prebuilt releases; user must `npm install && npm run dist`
   themselves; macOS-only target; unsigned (Gatekeeper right-click dance).
3. First-run form doesn't help diagnose (401 vs wrong address vs no-auth-needed); port
   stripping surprises ("I typed :9000, why :8005?"); no connecting spinner (~3–5 s freeze).
4. No empty-state guidance anywhere outside the sessions sidebar.

### The combined killer

The two halves don't know about each other. The daemon wizard never mentions Beacon,
never enables the web API, never mints a token. Beacon's first-run asks for an address +
token the daemon-side flow never produced. **A new user following both READMEs perfectly
still ends up with Beacon showing "Unreachable."**

---

## §2 Strategy — three phases, floor first

Principle: the polished UI walkthrough (Phase 2) is only as good as the daemon floor under
it. Phase 0 is where most of the pain dies; Phase 1 is the handshake that makes the UI
path possible; Phase 2 is the flagship experience.

### PHASE 0 — Fix the floor (Prometheus, one sprint)

1. **ONE canonical wizard: `prometheus setup`.** Merge `--setup` (rich) and
   `prometheus-init` (fast) into a single subcommand; `--fast` / `--noninteractive` flags
   keep the quick path. `prometheus init` and `--setup` become aliases that forward (no
   breakage). README rewritten around the one true path.
2. **Web on + token minted by default.** `web.enabled: true` in the default config the
   wizard writes. First daemon start with auth unset → generate a token, persist to
   `~/.config/prometheus/env`, print it ONCE loudly, and log "web auth: ENABLED" at every
   startup (or "OPEN — localhost only" if deliberately blank). Add
   `prometheus token show|rotate`.
3. **Ship the systemd unit.** `packaging/prometheus.service` + `prometheus install-service`
   (writes user unit, daemon-reload, enable --now). README's existing systemd line becomes
   true.
4. **`prometheus doctor`.** Expose the existing Doctor class as a CLI command; extend to
   check: config exists/parses, inference server reachable, model detected, web port free,
   token set, dirs writable, whisper present if voice enabled. This is both the wizard's
   final verify step and the eternal support answer ("run prometheus doctor").
5. **Kill the dead ends.** No server detected → offer: (a) point at a remote URL,
   (b) cloud provider key entry (anthropic/openai already supported as providers),
   (c) print copy-paste install instructions for Ollama and exit cleanly.
6. **Papercuts:** declare/gate whisper; document config search order; skip the
   Hermes/OpenClaw migration offer when nothing to migrate.

**Acceptance:** fresh VM/container → `pip install oara-prometheus[full]` →
`prometheus setup` (accepting defaults against a running Ollama) → CLI chat works →
`prometheus daemon` → `curl :8005/api/status` with the printed token → 200. And
`prometheus doctor` green on that box.

### PHASE 1 — The pairing handshake (small, both repos)

The bridge that makes the UI path one step instead of a scavenger hunt:

1. **Setup summary block.** The wizard's last screen prints a boxed "Connect a client":
   `address <host>:8005 · token <value>` (+ note it's in the env file).
2. **Setup mode on the daemon (stretch).** First daemon start with NO config: instead of
   erroring, boot the web server with a setup-only surface (`/api/setup/*`, everything else
   403) and print a one-time 6-digit pairing code. Beacon (or a browser) can then drive
   the entire setup remotely. This unlocks "install one thing in the terminal, do
   everything else in the UI."
3. **Beacon "pair" affordance.** First-run screen accepts `host + code` as an alternative
   to `host + token`; exchanges the code for a real token via `/api/setup/pair`, stores it
   in the keychain.

### PHASE 2 — The polished UI walkthrough (Beacon, the flagship)

Replace the single first-run form with a **stepped wizard** (keep the current form as the
"I know what I'm doing" escape hatch):

1. **Welcome** — what Beacon + Prometheus are, one paragraph, one picture.
2. **Connect** — address (+ token or pairing code), inline Test with tri-state diagnosis
   (reachable/auth/version), visible progress spinner.
3. **Model** — list backends/models over the existing model REST (GET /api/models +
   session model routes shipped in Prom #71); pick provider + model; show what was
   auto-detected. (Needs small daemon additions only if setup-mode selection is wanted
   pre-config — otherwise reuse what ships.)
4. **Identity** — name your agent, pick a persona seed → daemon generates SOUL.md
   (needs one small POST /api/setup/identity or reuse of the wizard's generator).
5. **Gateways (optional, skippable)** — Telegram token paste with a "how to get one from
   @BotFather" link; Slack later.
6. **Smoke test** — live round-trip in the UI: send "hello" → watch the reply stream.
   Green check = setup done.
7. **Orientation handoff** — land on Mission home WITH a dismissible "first flight"
   checklist: ① send a message ② open the command palette (⌘K) ③ explore one panel
   (deep-link chips to Chat / Board / Loop Manager). Empty-state copy added to Files,
   Board, Status panels. This kills Beacon gap #1.

### PHASE 3 — Distribution polish (when ready to hand to strangers)

- GitHub Releases with prebuilt artifacts (`npm run dist` in CI); add linux target to
  electron-builder; sign/notarize when the Apple ID is available.
- Landing-page install instructions on oara.ai matching the ONE canonical path.

---

## §3 Sequencing recommendation

Phase 0 is the sprint to run first — it's all Prometheus, it fixes the terminal path AND
lays the rails (web-on, token, doctor) the UI walkthrough needs. Phase 1+2 together are
the "polished UI walk-thru" Will asked for and make a natural second sprint (Beacon-heavy,
small daemon additions). Phase 3 is opportunistic.

## Out of scope (this track)

- Windows/Linux Beacon builds beyond adding electron-builder targets.
- Cloud-hosted Prometheus / multi-user; this is single-user local-first onboarding.
- Redesigning Mission home itself (the checklist overlays it; the orrery stays).
