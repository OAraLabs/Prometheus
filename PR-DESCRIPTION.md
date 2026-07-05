# feat(gateway): SPRINT G1 — gateway parity foundation

**Branch:** `feat/gateway-parity-g1` off main (`cbfed06`)
**Status:** PR-ready — not merged, main untouched, live daemon untouched.
Prometheus repo only. Discord (G2) and onboarding (G3) untouched.

Slack goes from a 23-command subset to full slash-command parity with
Telegram (41 commands), the daemon's subsystem wiring becomes
gateway-generic (a new gateway inherits everything for free — the G2
Discord hook), and a parity-manifest CI test makes the chart impossible to
silently regress.

---

## Workstream 1 — Telegram-embedded command logic → shared layer

Every Telegram-only command family was mechanically extracted from
`telegram.py` handler bodies into `src/prometheus/gateway/commands.py` as
plain, adapter-free `cmd_*` functions (existing style: no `self`, no
platform APIs, injected subsystem objects). Telegram handlers are now thin
wrappers, exactly like Slack's pre-existing ones.

Extracted families (new shared functions):

| family | shared function(s) | notes |
|---|---|---|
| provider overrides `/claude /gpt /gemini /xai /grok` | `cmd_provider_override` | returns `(text, applied)` so the gateway decides on inline dispatch |
| `/local` | `cmd_local_override` | |
| `/route` | `cmd_route` | `prefix=` kwarg renders per-surface command names |
| `/approve /deny /pending` | `cmd_approve` `cmd_deny` `cmd_pending` | |
| `/escalations` | `cmd_escalations` | |
| `/gepa` | `cmd_gepa` + `gepa_run_with_approval` | multi-message flows take an injected `async send(text)` |
| `/symbiote` (all 11 subcommands) | `cmd_symbiote` + `symbiote_run_with_approval`, `symbiote_restore_with_approval`, `_symbiote_*` sub-handlers | ~570 lines moved verbatim |
| `/audit` | `cmd_audit` + `_audit_show_last` / `_audit_kick_off` | |
| `/press` | `cmd_press` + `_press_*` sub-handlers | |
| `/voice` | `cmd_voice` + `get_voice_mode` / `set_voice_mode` / `load_voice_modes` / `save_voice_modes` | per-chat persistence is platform-independent; Telegram's `_get_voice_mode` etc. now delegate |
| `/tools` `/pairs` | `cmd_tools` `cmd_pairs` | pure formatters; kills the /tools duplication that already existed between telegram.py and slack.py |

`PROVIDER_PRESET_DISPLAY_NAMES` moved to commands.py
(`telegram._PRESET_DISPLAY_NAMES` kept as an alias).

**`prefix` convention:** shared text that names sibling commands is built
with a `prefix` kwarg — Telegram passes the default `"/"` (byte-identical
output), Slack passes `"/prometheus-"` so its replies name commands the
user can actually type.

`telegram.py`: 4004 → 2663 lines. Telegram-visible reply text preserved
byte-for-byte — see the pins below.

### Acceptance 2 — before/after reply-text pins (byte-identical)

`tests/test_gateway_command_pins.py` hard-codes the exact reply strings of
**27** representative handler invocations across the refactored families
(`/route`, `/local`, `/claude` (3 failure modes), `/approve`, `/deny`,
`/pending`, `/escalations` (incl. full armed-stats block), `/gepa`,
`/symbiote`, `/audit`, `/press` (incl. full usage block), `/voice` (3
modes), `/tools`, `/pairs`).

Run against the PRE-refactor tree (main `cbfed06…`, only the pin file
added):

```
$ PYTHONPATH=$PWD/src uv run pytest tests/test_gateway_command_pins.py -q
...........................                                              [100%]
27 passed in 0.23s
(HEAD = cbfed06850cb2ce621e2957338987225be3aafbb)
```

Same file, unchanged, against the POST-refactor tree:

```
$ PYTHONPATH=$PWD/src uv run pytest tests/test_gateway_command_pins.py -q
...........................                                              [100%]
27 passed in 0.23s
```

## Workstream 2 — gateway-generic subsystem wiring in daemon.py

New `GatewaySubsystemRegistry` (`src/prometheus/gateway/platform_base.py`):

* `register_adapter(adapter)` — adds a gateway and **replays every
  subsystem attached so far** onto it (fixes the ordering problem: the
  approval queue and printing press are constructed before the Slack
  adapter exists).
* `attach(name, value)` — records a subsystem and `setattr`s it on every
  registered adapter, current and future. Property setters (e.g. the
  `signal_bus` subscribe-on-set contract) are invoked naturally; a failing
  setter is logged loudly and never blocks other adapters or startup.

`BasePlatformAdapter.__init__` now defaults all seven subsystem slots to
`None` (`cost_tracker`, `escalation_engine`, `_approval_queue`,
`_gepa_engine`, `_printing_press`, `_backup_vault`, `_morph_engine`), so
any adapter subclass has well-defined "not wired" state.

daemon.py changes are surgical: one registry constructed before the
Telegram block, `register_adapter()` after each adapter's construction,
and each former `telegram.<slot> = X` line became
`gateway_registry.attach("<slot>", X)` in place. No startup restructuring.

**Adding a gateway (Discord, G2) = construct the adapter +
`gateway_registry.register_adapter(discord)` — it inherits ALL subsystems.**

### Acceptance 4 — grep-level proof

```
$ grep -nE '^\s*(telegram|slack_adapter)\.(_\w+|cost_tracker|escalation_engine|signal_bus)\s*=' src/prometheus/daemon.py
(no output)

$ grep -n "gateway_registry" src/prometheus/daemon.py
459:    gateway_registry = GatewaySubsystemRegistry()
461:        gateway_registry.attach("cost_tracker", cost_tracker)
487:        gateway_registry.register_adapter(telegram)
507:            gateway_registry.attach("_approval_queue", approval_queue)
560:            gateway_registry.attach("_printing_press", press_registry)
614:            gateway_registry.register_adapter(slack_adapter)
818:        gateway_registry.attach("escalation_engine", escalation_engine)
953:            gateway_registry.attach("signal_bus", signal_bus)
1122:                gateway_registry.attach("_backup_vault", sym_backup_vault)
1150:                gateway_registry.attach("_morph_engine", sym_morph_engine)
1182:            gateway_registry.attach("_gepa_engine", gepa_engine)
```

This proof is codified as CI in
`tests/test_gateway_g1.py::TestDaemonUsesGenericWiring` (source-scan: no
by-name injection + every slot attached through the registry).

Remaining `telegram` references in daemon.py are the **reverse direction**
(the adapter handed to a subsystem as its delivery transport), deliberately
out of G1 scope: `ApprovalQueue(telegram_adapter=…)` (approval-prompt
delivery), `ActivityObserver(gateway=…)`, and
`TaskCompletionHandler(inject_turn=telegram.inject_turn)`.

## Workstream 3 — Slack's missing commands

**Slack slash handlers: 23 → 41** (`grep -c 'self._app.command('`;
before-count taken from `git show main:src/prometheus/gateway/slack.py`).

18 new `/prometheus-*` handlers, all thin wrappers over the shared layer,
registered in `start()` and listed in `/prometheus-help`: `note`, `pairs`,
`approve`, `deny`, `pending`, `escalations`, `gepa`, `symbiote`, `audit`,
`press`, `voice`, `claude`, `gpt`, `gemini`, `xai`, `grok`, `local`,
`route`.

Also: `_slash_tools` now delegates to the shared `cmd_tools` (its body was
a copy of Telegram's), and `_slash_status` passes the (newly attached)
`cost_tracker` so cloud-spend reporting reaches Slack too.

Multi-message flows (gepa run, symbiote, audit, press install) use a
`_channel_sender` that posts via `chat_postMessage` (durable) and falls
back to the slash `respond` URL — a `respond` URL alone dies after ~30
minutes / 5 messages, too tight for approval-gated background flows.

### Platform-honest exceptions table

| capability | Slack behavior | why |
|---|---|---|
| `/voice` TTS voice-note replies | `/prometheus-voice` is registered and replies: *"Voice replies are not supported on Slack yet — the TTS pipeline (piper → opus/ogg voice notes) is wired to Telegram's voice-message API only…"* | the synth/upload pipeline lives in the Telegram adapter; porting it is not a G1 goal |
| inline message dispatch (`/claude what is 2+2?`) | override applies; reply appends *"inline message dispatch isn't supported on Slack yet — send your question as a normal message"* | Telegram re-enters `_dispatch_to_agent` with a synthetic event; Slack slash payloads have no message/thread context wired for that |
| approval **prompt delivery** | `/prometheus-approve/deny/pending` fully work; the outbound "Permission requested… /approve <id>" prompt still lands on the queue's configured transport (Telegram) | `ApprovalQueue`'s transport is a constructor arg; multi-gateway prompt delivery is flagged for G2 |
| approval queue when Telegram is disabled | queue is only constructed inside the Telegram block, so a Slack-only deployment reports "Approval queue not active." | pre-existing structure; honest reply rather than half-working |
| `/start`, `/clear` | not on Slack (manifest gap entries with reasons) | Telegram-native onboarding ping / alias of `-reset` |
| media ingestion (photo/voice/document/sticker) | out of G1 scope | rides with G2's shared media work |
| turn-level teacher-escalation hook | `/prometheus-escalations` reports the engine; the per-turn escalation anchor itself still runs only in Telegram's `_run_agent_turn` | turn-pipeline unification is beyond command parity (G2+ candidate) |

## The drift-proof parity test (acceptance 1)

`tests/test_gateway_parity.py` — a single `MANIFEST` of 43 command
families, each declaring its shared `commands.py` function(s) and its
registered command name per platform (or `None` + mandatory `gap_reason`).
Asserted mechanically, both directions:

* every manifest command is registered in that adapter's source AND its
  handler method exists on the adapter class;
* **every registered command on every platform appears in the manifest** —
  adding a command to one gateway and forgetting another (or forgetting
  the chart) fails CI;
* platform gaps and shared-function gaps must carry documented reasons;
* `TestParityReport` prints the full chart + the deliberate-gap allowlist
  into CI logs.

G2 extensibility: add one `PlatformSpec` for discord to `PLATFORMS` and a
`"discord": …` key per family —
`test_every_family_covers_every_platform` fails until every family takes an
explicit stance on Discord.

```
$ PYTHONPATH=$PWD/src uv run pytest tests/test_gateway_parity.py -q
........                                                                 [100%]
8 passed in 0.20s
```

### Parity chart after G1 (printed by the test)

```
family        telegram                  slack                     shared
------------------------------------------------------------------------
start         start                     — (gap)                   — (one fixed greeting string)
clear         clear                     — (gap)                   — (alias of /reset)
reset         reset                     prometheus-reset          — (one-line session clear)
help          help                      prometheus-help           cmd_help
status        status                    prometheus-status         cmd_status
model         model                     prometheus-model          cmd_model
wiki          wiki                      prometheus-wiki           cmd_wiki
note          note                      prometheus-note           cmd_note
sentinel      sentinel                  prometheus-sentinel       cmd_sentinel
benchmark     benchmark                 prometheus-benchmark      — (handler IS the benchmark)
context       context                   prometheus-context        cmd_context
skills        skills                    prometheus-skills         cmd_skills + 5 subcommand fns
memory        memory                    prometheus-memory         cmd_memory_show, cmd_memory_limits
curator       curator                   prometheus-curator        cmd_curator_show/status/run
notifications notifications             prometheus-notifications  cmd_notifications
health        health                    prometheus-health         cmd_health
events        events                    prometheus-events         cmd_events
steer         steer                     prometheus-steer          cmd_steer
queue         queue                     prometheus-queue          cmd_queue
unqueue       unqueue                   prometheus-unqueue        cmd_unqueue
clearsteers   clearsteers               prometheus-clearsteers    cmd_clearsteers
anatomy       anatomy                   prometheus-anatomy        cmd_anatomy
doctor        doctor                    prometheus-doctor         cmd_doctor
profile       profile                   prometheus-profile        cmd_profile
beacon        beacon                    prometheus-beacon         cmd_beacon
tools         tools                     prometheus-tools          cmd_tools
pairs         pairs                     prometheus-pairs          cmd_pairs
approve       approve                   prometheus-approve        cmd_approve
deny          deny                      prometheus-deny           cmd_deny
pending       pending                   prometheus-pending        cmd_pending
gepa          gepa                      prometheus-gepa           cmd_gepa
symbiote      symbiote                  prometheus-symbiote       cmd_symbiote
audit         audit                     prometheus-audit          cmd_audit
press         press                     prometheus-press          cmd_press
escalations   escalations               prometheus-escalations    cmd_escalations
voice         voice                     prometheus-voice          cmd_voice
claude        claude                    prometheus-claude         cmd_provider_override
gpt           gpt                       prometheus-gpt            cmd_provider_override
gemini        gemini                    prometheus-gemini         cmd_provider_override
xai           xai                       prometheus-xai            cmd_provider_override
grok          grok                      prometheus-grok           cmd_provider_override
local         local                     prometheus-local          cmd_local_override
route         route                     prometheus-route          cmd_route

Deliberate non-command gaps (allowlist):
  * slack: media ingestion (photo/voice/document/sticker)
      rides with Sprint G2's shared media pipeline
  * slack: TTS voice-note replies
      piper→opus/ogg pipeline is bound to Telegram's voice-message API;
      /prometheus-voice replies with an explicit not-supported boundary
  * slack: inline message dispatch on override commands
      handler appends an explicit note instead of silently dropping text
  * telegram: emoji reaction ack (eyes → white_check_mark)
      Slack-native affordance; Telegram uses typing indicator instead
  * approval prompt delivery
      ApprovalQueue's outbound prompt transport is the Telegram adapter;
      /approve /deny /pending work from every gateway
```

## Acceptance 3 — full suite

```
$ PYTHONPATH=$PWD/src uv run pytest -q \
    --deselect "tests/test_bootstrap.py::TestMemoryInPrompt::test_empty_memory_files_no_section"
3193 passed, 4 skipped, 1 deselected, 4 warnings in 102.93s
```

(The deselected test is the known pre-existing failure; CI deselects the
same one. Warnings are pre-existing event-loop teardown noise.)

New tests: 27 pins + 8 parity + 30 registry/Slack-handler tests = 65.
Updated: 4 wiring assertions in `tests/test_wiring.py` /
`tests/test_web_audit.py` that pinned the old private-method locations now
point at the shared layer (same guarantees, new home).

## Safety

No adapters were started, no tokens read, no ports bound, no daemon
restarted, no second daemon run. All tests use fakes, `MagicMock`
transports, and tmp-dir `PROMETHEUS_CONFIG_DIR`. Registration is asserted
by source scan + `hasattr`, never by instantiating a live bot.

## Deviations / notes for review

* `cmd_provider_override` returns `(text, applied)` — the one shared
  function that isn't a plain `-> str`, because the gateway must know
  whether to dispatch an inline message. Documented in its docstring.
* `cmd_note`'s usage strings still say `/note …` on Slack (no `prefix`
  kwarg added — it's a pre-G1 shared function also used by the web slash
  router; threading `prefix` through it is a trivial follow-up).
* Slack `/prometheus-status` output gains the cost-tracker block when a
  cloud provider is active (parity improvement, not drift — Slack had
  simply never received the tracker).
* Two daemon/commands log lines changed wording ("Approval queue wired to
  gateway adapters"; override log lines no longer say "Phase 4"); no
  user-visible reply text changed.
* Slack handler count is 23 → 41 (the sprint brief's "18 → ~38" counted a
  slightly different baseline; both deltas are the same 18 missing
  commands).
* Telegram's `/status`, `/wiki`, `/sentinel`, `/context`, `/benchmark`
  bodies remain embedded duplicates of their shared functions (pre-G1
  state). Left untouched per the "bias toward leaving Telegram code paths
  alone" rule — the parity manifest covers them because the shared
  functions exist and both gateways register the commands.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
