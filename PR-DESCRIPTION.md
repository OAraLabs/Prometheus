# feat(gateway): SPRINT G2 — Discord gateway

**Branch:** `feat/gateway-discord-g2` off main (`7867b36`)
**Status:** PR-ready — not merged, main untouched, live daemon untouched.
Prometheus repo only. Onboarding surfaces (G3) untouched — no wizard prompt
flows, no setup API; only the commented env-template line the wizard writes.

Third gateway surface: a `DiscordAdapter` on discord.py (new optional
extra), wired through the G1 `GatewaySubsystemRegistry` with **zero
per-subsystem daemon wiring**, all 43 command families registered as
Discord app commands (ZERO discord entries in the command allowlist), and
attachments routed through the **same** media services Telegram uses —
extracted to a shared module with Telegram behaviour pinned byte-identical.

---

## What shipped

| Piece | Where |
| --- | --- |
| DiscordAdapter — gateway WS, DMs + whitelisted guild channels, 2000-char chunking, long replies → threads, 👀→✅ reactions, skill/memory/curator signal notifications, last-channel persistence | `src/prometheus/gateway/discord.py` |
| All 43 families as app commands under ONE `/prometheus` command (4 section groups) | `discord.py::_register_app_commands` |
| Shared media services (vision / Whisper STT / doc context-budget) extracted from Telegram | `src/prometheus/gateway/media_services.py` |
| Telegram delegates to the shared services — byte-identical, pinned | `telegram.py` + `tests/test_gateway_media_pins.py` |
| Optional extra `discord = ["discord.py>=2.3.0"]`, `full` extra updated | `pyproject.toml` (+ `uv.lock`) |
| Daemon: optional-construction block, ONE `register_adapter(discord_adapter)` call, start/stop lifecycle alongside telegram/slack | `src/prometheus/daemon.py` |
| Config defaults `gateway.discord.{enabled,token,guild_ids,channel_ids,skill_event_notifications,long_reply_threshold}` | `config/prometheus.yaml.default` |
| `PROMETHEUS_DISCORD_TOKEN` — env template line (wizard-written template ONLY) + `ENV_OVERRIDES` mapping | `src/prometheus/cli/init.py`, `src/prometheus/config/env_override.py` |
| `Platform.DISCORD` + `discord_inbound_allowed` whitelist semantics | `src/prometheus/gateway/config.py` |
| Parity manifest: discord column, 43/43 registered | `tests/test_gateway_parity.py` |
| Adapter test suite — fakes only, no token, no network | `tests/test_discord.py` (61 tests) |
| Source-scan test extended: forbids `discord_adapter.<subsystem> =`, requires ≥3 `register_adapter` calls | `tests/test_gateway_g1.py` |

---

## 1. Parity manifest — the chart (green in CI; printed by `pytest -s tests/test_gateway_parity.py::TestParityReport`)

```
family        telegram       slack                     discord       shared
-----------------------------------------------------------------------------
start         start          — (gap: TG convention)    start         — (shared_gap: one fixed greeting)
clear         clear          — (gap: alias of reset)   clear         — (shared_gap: one-line clear alias)
reset         reset          prometheus-reset          reset         — (shared_gap: one-line clear + string)
help          help           prometheus-help           help          cmd_help
status        status         prometheus-status         status        cmd_status
model         model          prometheus-model          model         cmd_model
wiki          wiki           prometheus-wiki           wiki          cmd_wiki
note          note           prometheus-note           note          cmd_note
sentinel      sentinel       prometheus-sentinel       sentinel      cmd_sentinel
benchmark     benchmark      prometheus-benchmark      benchmark     — (shared_gap: handler IS the benchmark)
context       context        prometheus-context        context       cmd_context
skills        skills         prometheus-skills         skills        cmd_skills + 5 subcommand fns
memory        memory         prometheus-memory         memory        cmd_memory_show, cmd_memory_limits
curator       curator        prometheus-curator        curator       cmd_curator_show/_status/_run
notifications notifications  prometheus-notifications  notifications cmd_notifications
health        health         prometheus-health         health        cmd_health
events        events         prometheus-events         events        cmd_events
steer         steer          prometheus-steer          steer         cmd_steer
queue         queue          prometheus-queue          queue         cmd_queue
unqueue       unqueue        prometheus-unqueue        unqueue       cmd_unqueue
clearsteers   clearsteers    prometheus-clearsteers    clearsteers   cmd_clearsteers
anatomy       anatomy        prometheus-anatomy        anatomy       cmd_anatomy
doctor        doctor         prometheus-doctor         doctor        cmd_doctor
profile       profile        prometheus-profile        profile       cmd_profile
beacon        beacon         prometheus-beacon         beacon        cmd_beacon
tools         tools          prometheus-tools          tools         cmd_tools
pairs         pairs          prometheus-pairs          pairs         cmd_pairs
approve       approve        prometheus-approve        approve       cmd_approve
deny          deny           prometheus-deny           deny          cmd_deny
pending       pending        prometheus-pending        pending       cmd_pending
gepa          gepa           prometheus-gepa           gepa          cmd_gepa
symbiote      symbiote       prometheus-symbiote       symbiote      cmd_symbiote
audit         audit          prometheus-audit          audit         cmd_audit
press         press          prometheus-press          press         cmd_press
escalations   escalations    prometheus-escalations    escalations   cmd_escalations
voice         voice          prometheus-voice          voice         cmd_voice
claude        claude         prometheus-claude         claude        cmd_provider_override
gpt           gpt            prometheus-gpt            gpt           cmd_provider_override
gemini        gemini         prometheus-gemini         gemini        cmd_provider_override
xai           xai            prometheus-xai            xai           cmd_provider_override
grok          grok           prometheus-grok           grok          cmd_provider_override
local         local          prometheus-local          local         cmd_local_override
route         route          prometheus-route          route         cmd_route
```

**Discord command allowlist: EMPTY — 43/43 families registered**, including
`start`/`clear` which are deliberately gapped on Slack. The discord column
stores the family leaf name; the user-facing command is
`/prometheus <section> <leaf>` (flagged deviation below). The manifest's
registration regex scans `discord.py`'s `self._register(<group>, "<name>",
self.<handler>, …)` lines and `hasattr`-checks each handler, both directions
(unregistered manifest entry FAILS; unlisted registration FAILS).

Non-command capability allowlist (`NON_COMMAND_GAPS`, updated by G2):

| Gap | Reason |
| --- | --- |
| slack: media ingestion | the shared media pipeline now EXISTS (`gateway/media_services.py`, used by telegram + discord); Slack `file_shared` wiring is still open |
| slack + discord: TTS voice-note replies | piper→opus/ogg is bound to Telegram's voice-message API; `voice` on both surfaces replies with an explicit boundary. Discord voice-message **INPUT works** (Whisper) |
| slack + discord: inline dispatch on override commands | no message-dispatch context on slash payloads / interactions; handlers append an honest note instead of silently dropping text |
| telegram: emoji reaction ack | Slack/Discord-native affordance; Telegram uses the typing indicator |
| discord: sticker vision analysis | Discord stickers arrive as `message.stickers`, not attachments; image ATTACHMENTS get full vision analysis |
| approval prompt delivery | ApprovalQueue's outbound transport is still the Telegram adapter; approve/deny/pending work from every gateway |

## 2. Test counts

- Baseline, MEASURED on main `7867b36` in a clean worktree with the
  standard extras (`web anthropic slack` + dev): **3194 passed, 4 skipped,
  0 failed** in this environment. (The sprint brief's "3243" figure does
  not reproduce on this box with these extras — flagged rather than
  copied; the 4 skips are pre-existing optional-dep skips, e.g. `mcp`
  not installed.)
- This branch, same command + `--extra discord`: **3265 passed, 4 skipped,
  0 failed, 0 deselected** (`PYTHONPATH=$PWD/src uv run pytest`).
- Delta: exactly **+71** — 61 `tests/test_discord.py` +
  10 `tests/test_gateway_media_pins.py`; identical skips, nothing
  deselected, no xfails added.
- `uv sync --extra web --extra anthropic --extra slack --extra discord
  --group dev` resolves cleanly (discord.py 2.7.1, no conflicts);
  `uv sync --extra discord` alone also resolves (dry-run verified).
  `uv.lock` diff = the new extra's dependency set only.
- The 43-command tree was additionally built against the REAL discord.py
  2.7.1 (no network): one `/prometheus` root, sections core=24, session=4,
  ops=8, provider=7, every leaf carrying an optional `args` string option —
  under Discord's 25-options-per-command cap at every level.

## 3. Grep-proof: no discord-specific subsystem assignments in daemon.py

```
$ grep -nE '^\s*(telegram|slack_adapter|discord_adapter)\.(_\w+|cost_tracker|escalation_engine|signal_bus)\s*=' src/prometheus/daemon.py
(no output — exit 1)
$ grep -c "gateway_registry.register_adapter(" src/prometheus/daemon.py
3
```

The daemon's entire Discord wiring is: construct `DiscordAdapter` →
`gateway_registry.register_adapter(discord_adapter)` → `await start()`.
All 8 subsystems (cost tracker, approval queue, printing press, escalation
engine, signal bus, backup vault, morph engine, GEPA) arrive via the G1
registry replay — proven in
`test_discord.py::TestRegistryInheritance` (attach-before-register replay +
signal_bus setter subscription). The CI source-scan test
(`test_gateway_g1.py::TestDaemonUsesGenericWiring`) now also forbids
`discord_adapter.<subsystem> =` lines and requires ≥3 `register_adapter`
calls, so the constraint is enforced forward.

## 4. First live run — honest statement

**A live Discord connection was NOT tested.** No Discord bot token exists
on this box and none was created for this sprint (hard rule: no real
tokens, no daemon restart). Everything above is fakes + real-library tree
construction.

What the first live run needs from Will:

1. **Create the Discord application + bot**
   (https://discord.com/developers/applications → New Application → Bot →
   Reset Token → copy token).
2. **Enable the MESSAGE CONTENT intent** (Bot tab → Privileged Gateway
   Intents → *Message Content Intent* ON). The adapter requests
   `intents.message_content = True`; if the portal toggle is off the
   gateway rejects the connection outright (`PrivilegedIntentsRequired`) —
   loud, not silent.
3. **Invite URL with BOTH scopes** — `bot` **and** `applications.commands`
   (OAuth2 → URL Generator). Without `applications.commands`, `/prometheus`
   never appears. Suggested bot permissions: View Channels, Send Messages,
   Send Messages in Threads, Create Public Threads, Add Reactions,
   Attach Files, Read Message History.
4. **Config + install on the daemon host**:
   `PROMETHEUS_DISCORD_TOKEN=` in `~/.config/prometheus/env` (template line
   added), and in `prometheus.yaml`:
   ```yaml
   gateway:
     discord:
       enabled: true
       guild_ids: [<server id>]   # instant slash commands; [] = global (~1h)
   ```
   plus `uv sync --extra discord` (or `pip install 'oara-prometheus[discord]'`),
   then a daemon restart — Will's call, not done here.

What could surprise us:

- **Global command propagation**: with `guild_ids: []`, Discord can take up
  to ~1 hour to surface `/prometheus`. Guild-scoped sync is instant — set
  `guild_ids`.
- **DM slash commands under guild-scoped sync**: when `guild_ids` is set we
  deliberately skip the global sync (per spec), so the command picker only
  offers `/prometheus` inside those guilds. DM *chat* and DM media still
  work — plain messages need no registration.
- **Rate limits**: discord.py absorbs per-route 429s internally (sleep +
  retry), but a very long agent reply chunked into many 2000-char messages
  will visibly trickle (~5 msgs/5s per channel). Long-reply→thread reduces
  channel spam, not the rate limit.
- **Interaction deadlines**: app commands must be acked within 3s — slow
  handlers (benchmark, anatomy, doctor, curator run, gepa, symbiote, audit,
  press) `defer()` first, and multi-message flows post directly to the
  channel because followup webhooks expire after 15 minutes. Anything
  unexpectedly slow without a defer shows "The application did not respond".
- **Message-content intent at scale**: past 100 guilds the intent needs
  Discord's manual approval — irrelevant for a personal bot, noted for
  completeness.

## 5. Telegram media refactor — byte-identical pins (G1 pattern)

`_describe_image` / `_transcribe_audio` / `_truncate_for_context` moved
from `telegram.py` into shared `prometheus.gateway.media_services` so
Discord routes attachments through the SAME services (no duplication).
Commit-sequenced so the proof is in history:

- `cf39888` — 10 pins authored and run **green against the pre-refactor
  tree**: exact vision question string ("Describe this image in detail."),
  provider-metadata pass-through, error/empty→None fallbacks, exact
  truncation formula + suffix text, server-context cap, and an end-to-end
  `_handle_photo` pin asserting the exact `[Image: …]\ncaption` event text.
- refactor commit — telegram delegates; the same 10 pins pass unchanged
  (`10 passed` before AND after the extraction).

## Flagged deviations

1. **`/prometheus <section> <family>`, not `/prometheus <family>`.**
   Discord hard-caps a slash command at 25 options (and a group at 25
   subcommands); 43 families cannot sit flat under one group, so they sit
   one section deeper: `core` (24) / `session` (4) / `ops` (8) /
   `provider` (7) — still ONE top-level `/prometheus` in the picker.
   Sections were chosen so every cross-command reference the shared layer
   renders stays inside one section (approvals live with gepa/symbiote/
   press in `ops` because their flows render "watch for the …approve
   prompt"), so a single `prefix` string per section produces correct,
   typeable help text — verified in tests
   (`Usage: /prometheus ops approve {request_id}`,
   `/prometheus provider local`, …).
2. **Whitelist semantics.** The spec said "empty = DMs only, mirroring
   Telegram's allowed_chat_ids semantics — check and MATCH the existing
   semantics, cite them." Checked: Telegram's actual semantics
   (`PlatformConfig.chat_allowed`) are *empty list = allow every chat*.
   Carrying that to guilds would make the bot answer every message in any
   server anyone invites it to, so Discord deviates deliberately: DMs
   always allowed (matches Telegram's open-DM posture when unrestricted),
   guild channels require an explicit `channel_ids`/`guild_ids` hit, both
   empty = DMs only — exactly the spec's stated behaviour, documented with
   the Telegram citation at `PlatformConfig.discord_inbound_allowed`.
3. **`ENV_OVERRIDES` rider**: `PROMETHEUS_DISCORD_TOKEN` also got a mapping
   in `env_override.py` (nested path `gateway.discord.token`) so the env
   var behaves like the Telegram/Slack token vars everywhere (config
   loading), not only in the daemon's construction block. One line.

## Hard-rule compliance

- No real tokens; no live Discord (or any) connection attempted.
- Live daemon untouched: no restart, no second daemon, nothing bound to
  :8005/:8010 — all tests are fakes; the discord gateway task only exists
  inside `start()`, which no test calls with a real client.
- Telegram behaviour byte-identical (pinned); slack.py untouched.
