# SPRINT — Polish & Platform

**Type:** Implementation sprint
**Branch:** `feat/polish-platform` (off main after Sprint 2 merges)
**Estimated time:** 4–7 focused days
**Prerequisites:** Sprints 1 and 2 merged
**Trust level:** Touches gateway, Beacon, packaging. Lower-risk than Sprint 2 but broad surface.

---

## Why this sprint exists

After Sprints 1–2, the agent has visible memory, visible skills, weekly curation, mid-turn steering, and silent-failure detection. What's still missing is the *frictionless arrival experience* that Hermes earned with its onboarding story. Three things specifically:

1. **The Slack gateway is half-built.** README claims 9 slash commands; most aren't wired. Either finish it to Telegram parity or unclaim. We're finishing.
2. **Beacon dashboard runs but doesn't show much.** Per verification, `start_web()` fires on every daemon boot. But the UI doesn't surface the live signals from Sprint 1 — skill events, memory events, Curator reports, agent activity. The hard work was already done; this sprint makes it visible.
3. **Installation is `git clone` + `pip install -e .` + edit config + start daemon.** Hermes is `pip install hermes-agent && hermes init`. The difference isn't capability, it's narrative — and the narrative drives adoption.

Plus one minor item from Sprint 1's natural extension: natural-language cron parsing. People love being able to say "every Monday at 9am" instead of writing `0 9 * * 1`.

---

## Read These Files First

### Slack gateway
- `src/prometheus/gateway/slack.py` — current scaffold (verification: registered, mostly unimplemented)
- `src/prometheus/gateway/telegram.py` — reference implementation to mirror
- README.md sections claiming Slack commands

### Beacon
- `src/prometheus/web/server.py` — FastAPI routes
- `src/prometheus/web/ws_server.py` — WebSocket bridge (Sprint 1 added signal broadcasts here)
- `src/prometheus/web/static/` or wherever the Beacon frontend lives
- `src/prometheus/web/templates/` if applicable

### Cron tools
- `src/prometheus/tools/builtin/cron_create.py` (or wherever the cron tools are)
- `src/prometheus/tools/builtin/cron_list.py`
- `src/prometheus/tools/builtin/cron_delete.py`

### Packaging
- `pyproject.toml`
- `setup.py` if present
- `scripts/daemon.py` startup entry points
- `src/prometheus/setup_wizard.py`
- `src/prometheus/__main__.py`

---

## Work Stream 1: Slack Gateway to Telegram Parity

### Goal

Every command and notification Telegram supports works in Slack. No claim in README is false.

### What to build

1. **Audit existing Slack handlers.** Run `grep -n "@slack_command\|@command\|app.command\|app.event" src/prometheus/gateway/slack.py`. List what's actually wired.

2. **Mirror Telegram command surface.** Every Telegram command should have a Slack equivalent, with the same signature where Slack's idioms allow:
   - `/status`, `/help`, `/profile`
   - `/memory show`, `/memory show user`, `/memory limits` (Sprint 1)
   - `/skills list`, `/skills show <name>`, `/skills pin <name>` (Sprint 1)
   - `/curator show`, `/curator run`, `/curator status` (Sprint 1)
   - `/steer <text>`, `/queue <text>`, `/unqueue` (Sprint 2)
   - `/notifications quiet|verbose|off`

3. **Notification routing.** When SignalBus emits SKILL_CREATED, MEMORY_UPDATED, CURATOR_REPORT (Sprint 1's signals), they route to *both* Telegram and Slack if both are enabled — not just whichever was the originating gateway. Use the same `skill_event_notifications` config knob for each.

4. **Threading.** Slack threads are an idiomatic way to handle long agent responses. Long replies (>800 chars) should reply in a thread, not flood the channel. Telegram doesn't have this; Slack expects it.

5. **Workspace-scoped permissions.** Slack workspace tokens are workspace-wide. Add an `allowed_channels` config list (mirror of `telegram_allowed_users`) so the agent only responds in specific channels. Default: empty list = respond in any channel the bot is added to.

### Config additions

```yaml
gateway:
  slack:
    enabled: false
    bot_token: ""              # or read from PROMETHEUS_SLACK_BOT_TOKEN env
    app_token: ""              # for socket mode
    allowed_channels: []       # empty = all channels
    skill_event_notifications: quiet
    long_reply_threshold: 800  # chars; replies longer than this go in thread
```

### Tests

- `test_slack_status_command` — `/status` returns same content shape as Telegram
- `test_slack_skill_event_routes` — SKILL_CREATED reaches Slack when Slack is enabled
- `test_slack_long_reply_threads` — reply >800 chars uses thread, not main channel
- `test_slack_allowed_channels_filter` — bot ignores messages from non-allowed channels

### If Slack isn't actually being used

Halt and ask. The work is real (M-effort, maybe 1.5 days), and if you don't run a Slack workspace, the alternative is to **delete the claims from README and downgrade Slack to "scaffolded, not maintained."** Don't build a feature nobody uses.

---

## Work Stream 2: Beacon Live Activity Dashboard

### Goal

Beacon (already running on every daemon boot) shows a live activity feed of what the agent is doing — tool calls streaming, skills being created, memory being updated, Curator reports landing.

### What to build

#### Backend (light additions)

In `src/prometheus/web/ws_server.py`, ensure these event types are broadcast:

- `tool_call_start` — fired when a tool is invoked
- `tool_call_complete` — fired when a tool returns
- `model_call_start` — fired when calling the model provider
- `model_call_complete` — fired with token counts
- `skill_created` / `skill_refined` / `memory_updated` / `curator_report` (from Sprint 1)
- `steer_received` / `prompt_queued` (from Sprint 2)
- `file_mutation_summary` (from Sprint 2's verifier)

Each event includes a timestamp, type, summary, and optional structured payload.

#### Frontend — Activity Feed

In Beacon's existing UI, add an Activity Feed panel showing the live stream. Doesn't need to be fancy; chronological list with:

- Icon by event type (🔧 tool, 🧠 model, 🎓 skill, 📝 memory, 📋 curator, 📍 steer, 📁 file)
- Timestamp (relative — "12s ago")
- One-line summary
- Click to expand for full payload

If Beacon's frontend is React, this is a single `<ActivityFeed>` component subscribed to the WebSocket. If it's vanilla HTML/JS, it's a `<ul>` with WebSocket subscription. Either way, scope is small — maybe 200 lines of frontend code total.

#### Frontend — Memory & Skills panels (lightweight)

Add two side panels:

- **Memory panel** — shows current MEMORY.md and USER.md content with character counts vs limits. Read-only.
- **Skills panel** — shows skill list with usage counts. Click a skill to see its markdown content. Pin/unpin buttons that hit a new `/api/skills/pin` endpoint.

#### Backend — small API additions

```python
@router.get("/api/activity/recent")        # last 100 events for feed initial load
@router.get("/api/memory/current")          # current MEMORY.md and USER.md
@router.get("/api/skills/list")             # skill list with usage counts
@router.get("/api/skills/{name}")           # specific skill content
@router.post("/api/skills/{name}/pin")      # pin a skill
@router.delete("/api/skills/{name}/pin")    # unpin
```

These mirror the Telegram command surface, which keeps the mental model consistent.

### Tests

- `test_beacon_serves_activity_feed` — `/api/activity/recent` returns last events
- `test_beacon_websocket_broadcasts_skill_event` — emit SKILL_CREATED, confirm Beacon WS receives it
- `test_beacon_memory_endpoint` — `/api/memory/current` returns USER.md and MEMORY.md content
- `test_beacon_pin_skill` — POST to `/api/skills/foo/pin` makes the skill survive a Curator pass

---

## Work Stream 3: Natural-Language Cron

### Goal

`cron_create` accepts either standard cron syntax OR natural language ("every Monday at 9am", "tomorrow at 3pm", "in 30 minutes", "every weekday at noon").

### What to build

In `src/prometheus/tools/builtin/cron_create.py` (or wherever the tool lives):

```python
class CronCreate(Tool):
    def execute(self, schedule: str, command: str, name: str = None):
        # Try parsing as cron first
        if self._is_cron_syntax(schedule):
            cron_spec = schedule
        else:
            # Fall back to NL parsing
            cron_spec = self._parse_natural_language(schedule)
            if cron_spec is None:
                return error("Couldn't parse schedule. Try cron syntax or 'every X at Y'.")
        # ... existing cron creation logic ...
```

#### NL parser

Use a small dependency-free parser (or pull in `recurrent` / `dateparser` / `parsedatetime` — pick one based on what's already in `pyproject.toml`).

Support these patterns:

- `every <day> at <time>` → standard cron
- `every <weekday|weekend|day-of-week>`
- `in <N> <units>` → one-shot at_time
- `tomorrow at <time>`
- `at <time>` → today, if future; tomorrow if past
- `every <N> <hours|minutes|days>`

If parsing fails, fall through to an LLM-assisted parse (use the local model with constrained decoding — there's already a constrained decoding path in `src/prometheus/evals/`):

```python
def _llm_parse_schedule(self, nl: str) -> str | None:
    """Last-resort: ask local model to convert NL to cron with strict JSON schema."""
    grammar = {"type": "object", "properties": {"cron": {"type": "string"}}, "required": ["cron"]}
    # Use existing constrained decoding infrastructure
```

### Tests

- `test_cron_create_standard_syntax` — `"0 9 * * 1"` works unchanged
- `test_cron_create_nl_weekly` — `"every Monday at 9am"` parses correctly
- `test_cron_create_nl_relative` — `"in 30 minutes"` creates a one-shot
- `test_cron_create_ambiguous_falls_back` — unparseable strings get the LLM fallback
- `test_cron_create_invalid` — gibberish returns a useful error

---

## Work Stream 4: Onboarding Polish

### Goal

`pip install oara-prometheus && prometheus init` should get a new user to a running agent in under 5 minutes.

### What to build

#### PyPI packaging

Audit `pyproject.toml`:
- Package name: `oara-prometheus` (or whatever you want claimed on PyPI — check availability first)
- Console scripts:
  ```toml
  [project.scripts]
  prometheus = "prometheus.__main__:main"
  prometheus-init = "prometheus.cli.init:main"
  prometheus-daemon = "prometheus.cli.daemon_cmd:main"
  ```
- Optional dependencies for the various providers/tools, so `pip install oara-prometheus[full]` gets everything

Don't publish yet — that's a release decision. The goal of this sprint is *being publishable*.

#### `prometheus init` command

Create `src/prometheus/cli/init.py`:

```python
def main():
    """First-run interactive setup.

    1. Detect environment:
       - Running local inference server? (probe llama.cpp:8080, Ollama:11434, LM Studio:1234)
       - Available GPUs? (use existing AnatomyScanner)
       - Existing config? (offer to migrate or back up)

    2. Ask the essentials:
       - Primary model: detected local / cloud API / both
       - Gateway: Telegram (prompt for BotFather link), Slack, CLI-only
       - Memory mode: solo / multi-user

    3. Write config to ~/.config/prometheus/config.yaml
    4. Write env file template to ~/.config/prometheus/env (with placeholders)
    5. Print: "Edit ~/.config/prometheus/env to add tokens, then run: prometheus daemon"
    """
```

Reuse the existing setup wizard logic where it makes sense; this is mostly a friendlier wrapper.

#### Auto-detection of local inference servers

In `src/prometheus/cli/init.py` and ideally exposed as a helper module:

```python
KNOWN_LOCAL_SERVERS = [
    ("llama.cpp", "http://localhost:8080/v1/models", "openai_compat"),
    ("Ollama", "http://localhost:11434/api/tags", "ollama"),
    ("LM Studio", "http://localhost:1234/v1/models", "openai_compat"),
    ("vLLM", "http://localhost:8000/v1/models", "openai_compat"),
]

def detect_local_servers() -> list[dict]:
    """Probe each known server; return list of those responding."""
```

If any are detected, `prometheus init` offers them as default model providers. **This is the feature Hermes GitHub issue #523 was asking for — and Prometheus is the right project to build it well because the Adapter Layer can offer matching strictness recommendations per model.**

#### README rewrite (front matter only)

The top of README.md should answer in 60 seconds:

1. What is Prometheus (one sentence)
2. The thesis (one sentence)
3. Install:
   ```
   pip install oara-prometheus
   prometheus init
   prometheus daemon
   ```
4. What it gives you (5-bullet feature summary)
5. Link to deeper docs

Move the architecture deep-dive to `docs/`. The README is marketing.

### Tests

- `test_detect_llama_cpp` — when llama.cpp is running, detection finds it (mock the HTTP call)
- `test_init_writes_config` — `prometheus init --noninteractive` produces a valid config file
- `test_init_preserves_existing` — running `init` on a system with existing config offers to back up, doesn't clobber

---

## Acceptance criteria for the whole sprint

A new user, on a fresh machine with Ollama already running:

1. Runs `pip install oara-prometheus` (or `pip install -e .` from a clone if not yet on PyPI)
2. Runs `prometheus init`
3. The wizard detects Ollama, asks two or three questions, writes config
4. Runs `prometheus daemon`
5. Opens Telegram, talks to their bot
6. Opens Beacon in a browser (`http://localhost:8765` or whatever the port is), sees live activity feed populate as they interact

For an existing Slack user, every command they use on Telegram also works in Slack with the same shape.

For cron: `"every Monday at 9am, run the weekly report skill"` creates the cron entry successfully.

---

## Constraints

- **Branch `feat/polish-platform`.** Off main after Sprint 2 merges.
- **No commits to main.** Will squash-merges.
- **Don't publish to PyPI in this sprint.** Build the packaging, verify it works locally (`pip install -e .` produces working console scripts), but the publish step is a separate decision.
- **No Beacon visual overhaul.** This sprint adds three small panels to existing Beacon, not a UI rewrite.
- **NL cron parsing falls back to LLM only as last resort.** Most patterns should work without an LLM call.
- **If Slack isn't actually used by Will, downgrade instead of finish.** Don't build a feature nobody uses.
- **README rewrite is FRONT MATTER ONLY.** Don't touch the architecture sections — those are for the architecture docs in `docs/`.

---

## Reporting back

1. Branch + commit SHAs across the four work streams
2. Slack: list of newly wired commands (or confirmation that Slack was downgraded instead)
3. Beacon: screenshot of activity feed populated during a real session
4. Cron: 3–5 example NL strings that parse, with the resulting cron spec
5. `prometheus init` transcript on a fresh-ish environment
6. PyPI: confirmation that `pip install -e .` produces a working `prometheus` console script
7. Drive-by findings — note, don't fix
