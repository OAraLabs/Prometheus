# SPRINT — Visible Memory & Skills

**Type:** Implementation sprint
**Branch:** `feat/visible-memory-skills` (off origin/main)
**Estimated time:** 3–5 focused days
**Trust level:** Internal code changes only. No new external dependencies. No commits to main.

---

## Why this sprint exists

Verification commit `b190f19` confirmed that SkillCreator, SkillRefiner, PeriodicNudge, MemoryExtractor, and GoldenTraceExporter are already wired into the daemon and post-task hooks. The components fire on every successful task. What's missing is **visibility**: users don't see skills being created, don't see memory being curated, don't have commands to inspect what's stored, and have no weekly summary of what the agent learned.

Hermes's reputation for "stays sharp on day 30" and "watch it build itself a skill" comes from making this machinery visible to the user, not from having more machinery. Prometheus already has more machinery than Hermes (LCM + Adapter Layer + AnatomyScanner + GRAFT-SYMBIOTE). This sprint surfaces it.

Three coordinated work streams. All small. All ship together because they share the Telegram/Beacon surface and the SignalBus.

---

## Read These Files First

Before writing any code, view these to understand the current wiring:

### Daemon and lifecycle
- `scripts/daemon.py` — particularly lines 288-302 (SkillCreator post-task hook) and 590-603 (PeriodicNudge / SkillRefiner)
- `src/prometheus/__main__.py` — tool registry construction, gateway startup

### Memory subsystems
- `src/prometheus/memory/extractor.py` — MemoryExtractor
- `src/prometheus/memory/lcm_engine.py` — LCM core
- USER.md handling — grep for `USER.md` across `src/prometheus/memory/`
- MEMORY.md handling — grep for `MEMORY.md` across `src/prometheus/memory/`

### Learning subsystems
- `src/prometheus/learning/skill_creator.py` — SkillCreator
- `src/prometheus/learning/skill_refiner.py` — SkillRefiner
- `src/prometheus/learning/nudge.py` — PeriodicNudge
- `src/prometheus/learning/gepa.py` — GEPAOptimizer
- `src/prometheus/sentinel/gepa_engine.py` — GEPAEngine

### Gateway and signaling
- `src/prometheus/gateway/telegram.py` — Telegram command dispatch + outbound message helpers
- `src/prometheus/signal_bus.py` or wherever SignalBus lives — internal event system
- `src/prometheus/web/server.py` — Beacon FastAPI server (verification confirmed `start_web()` runs on every daemon boot)
- `src/prometheus/web/ws_server.py` — WebSocket bridge

### Config
- `config/prometheus.yaml.default` — current memory/learning config keys

---

## Work Stream 1: Curator Weekly Pass

**Goal:** A scheduled weekly consolidation pass that grades, prunes, and consolidates the skill library, emitting a human-readable report to disk and a summary to Telegram.

### What to build

Create `src/prometheus/learning/curator.py`:

```python
class Curator:
    """Weekly skill library consolidation.

    Reads all skills in ~/.prometheus/skills/, grades each on:
    - Usage frequency (from telemetry.db)
    - Last successful execution (from telemetry.db)
    - Drift from current tool surface (skill references tools that no longer exist)
    - Redundancy (skill duplicates a more general skill)

    Uses the existing LLM judge at src/prometheus/evals/judge.py for grading.

    Outputs:
    - ~/.prometheus/curator/run-YYYY-MM-DD.json (structured report)
    - ~/.prometheus/curator/REPORT.md (human-readable summary)
    - Optional: Telegram message via SignalBus

    Actions taken:
    - PIN: skill is high-value, mark as protected
    - KEEP: skill is fine as-is
    - REFINE: hand to SkillRefiner for improvement
    - CONSOLIDATE: merge into another skill
    - PRUNE: move to ~/.prometheus/skills/archive/ (NEVER delete)

    Pinned skills are protected from CONSOLIDATE and PRUNE.
    """
```

### Wiring

Add a scheduler entry in `scripts/daemon.py`:

```python
# After existing PeriodicNudge wiring
if config["learning"].get("curator_enabled", True):
    curator_interval = config["learning"].get("curator_interval_seconds", 7 * 24 * 3600)
    # Use whichever scheduler the daemon already uses
    # (APScheduler, asyncio task, or existing periodic_nudge scheduler)
    schedule_periodic(curator_interval, lambda: Curator(...).run())
```

### Config additions

In `config/prometheus.yaml.default`, under the existing `learning:` section:

```yaml
learning:
  # ... existing keys ...
  curator_enabled: true
  curator_interval_seconds: 604800  # 7 days
  curator_telegram_summary: true    # send weekly summary to Telegram
  curator_prune_threshold_days: 90  # skills unused for 90+ days are PRUNE candidates
```

### Tests

Add to `tests/test_wiring.py` (or split into `tests/wiring/test_curator.py` if the wiring file is being broken up — see Sprint 3):

- `test_curator_runs_on_schedule` — confirm scheduler fires the Curator at the configured interval
- `test_curator_writes_report` — confirm REPORT.md and run-YYYY-MM-DD.json land in `~/.prometheus/curator/`
- `test_curator_respects_pins` — pinned skills are never pruned or consolidated
- `test_curator_archives_dont_delete` — pruned skills go to `archive/`, not deleted

---

## Work Stream 2: User-Visible Skill & Memory Events

**Goal:** When the agent creates a skill, refines one, updates memory, or runs Curator, the user sees it via Telegram and Beacon.

### What to build

In `src/prometheus/signal_bus.py` (or wherever SignalBus is defined), add four event types if they don't already exist:

```python
class SignalType(Enum):
    # ... existing ...
    SKILL_CREATED = "skill_created"          # SkillCreator fired
    SKILL_REFINED = "skill_refined"          # SkillRefiner updated a skill
    MEMORY_UPDATED = "memory_updated"        # MemoryExtractor wrote to MEMORY.md / USER.md
    CURATOR_REPORT = "curator_report"        # Weekly Curator pass completed
```

### Wire emission

In each of:
- `src/prometheus/learning/skill_creator.py` — emit `SKILL_CREATED` after writing the skill file
- `src/prometheus/learning/skill_refiner.py` — emit `SKILL_REFINED` after a refinement
- `src/prometheus/memory/extractor.py` — emit `MEMORY_UPDATED` after writes to MEMORY.md or USER.md
- `src/prometheus/learning/curator.py` (from Stream 1) — emit `CURATOR_REPORT` after a run

Each emission includes a structured payload:

```python
{
    "skill_name": str,           # for SKILL_*
    "skill_path": str,           # absolute path
    "trigger_task": str,         # what the agent was doing when it learned this
    "summary": str,              # one-line description
}
```

### Wire consumption — Telegram

In `src/prometheus/gateway/telegram.py`, subscribe to these signals and emit user-facing messages. **Default behavior:** quiet (just a 🎓 emoji + skill name), but `/notifications verbose` flips to full descriptions.

```
🎓 New skill: docker-network-debug
   (built while troubleshooting your container network issue)

📚 Updated skill: web-research-deep
   (added pagination handling)

🧠 Memory updated: 2 new facts about your project structure

📋 Weekly Curator report: 23 skills reviewed, 1 refined, 0 pruned
   /curator show for details
```

### Wire consumption — Beacon

In `src/prometheus/web/ws_server.py`, broadcast the same signals over WebSocket so Beacon's dashboard can show them as a live activity feed. Don't build a new dashboard panel in this sprint — just emit the events. Beacon UI polish is Sprint 3.

### Config additions

```yaml
gateway:
  telegram:
    # ... existing ...
    skill_event_notifications: quiet   # quiet | verbose | off
    memory_event_notifications: quiet
    curator_summary_notification: true
```

### Tests

- `test_skill_created_emits_signal` — SkillCreator post-task hook produces a SKILL_CREATED signal
- `test_signal_routes_to_telegram` — SKILL_CREATED arrives at Telegram outbound queue
- `test_signal_routes_to_beacon_ws` — same signal broadcast over WebSocket
- `test_notification_quiet_mode` — quiet mode produces short message, verbose produces full

---

## Work Stream 3: Memory & Skill Inspection Commands

**Goal:** User can inspect what the agent knows and has learned, directly from Telegram.

### Commands to add

In `src/prometheus/gateway/telegram.py`, add handlers for:

**Memory commands**
- `/memory show` — display current MEMORY.md content with character count vs limit
- `/memory show user` — display current USER.md content with character count vs limit
- `/memory limits` — show hard limits and current usage
- `/memory edit <key> <value>` — request agent to update a specific memory entry (goes through MemoryExtractor with explicit user override flag)

**Skills commands**
- `/skills list` — list all skills with usage count from telemetry
- `/skills show <name>` — display a specific skill's markdown content
- `/skills pin <name>` — mark a skill as pinned (Curator won't prune or consolidate)
- `/skills unpin <name>` — remove pin
- `/skills history <name>` — show refinement history (git log of the skill file)

**Curator commands**
- `/curator show` — display the most recent REPORT.md
- `/curator status` — show next scheduled run, last run, pinned skill count
- `/curator run` — trigger an immediate Curator pass (user-initiated, off-schedule)

### Implementation pattern

Each command:
1. Validates the requesting user is authorized (existing pattern in Telegram gateway)
2. Reads from `~/.prometheus/skills/` or `~/.prometheus/memory/` directly — no LLM call needed
3. Truncates to Telegram message limits (4096 chars), offers `/more` continuation if needed
4. Logs the inspection event for telemetry

### Hard character limits enforcement

In `src/prometheus/memory/extractor.py` (or wherever MEMORY.md / USER.md writes happen):

```python
MEMORY_MD_LIMIT = 2200    # match Hermes
USER_MD_LIMIT = 1375      # match Hermes

def write_memory(content: str, target: str):
    """target: 'MEMORY.md' or 'USER.md'"""
    limit = MEMORY_MD_LIMIT if target == 'MEMORY.md' else USER_MD_LIMIT
    if len(content) > limit:
        # Don't truncate. Request consolidation pass instead.
        # The agent self-curates to fit; this is the discipline that
        # forces sharp memory.
        raise MemoryOverflowError(
            f"{target} would be {len(content)} chars, limit is {limit}. "
            f"Run consolidation before writing."
        )
    Path(f"~/.prometheus/memory/{target}").expanduser().write_text(content)
```

When `MemoryOverflowError` is raised, the calling code (MemoryExtractor or agent loop) should:
1. Catch the error
2. Inject a system prompt addendum: "MEMORY.md is at capacity. Consolidate before adding new facts: review existing entries, merge duplicates, remove stale items, then write the consolidated version."
3. Retry the write

This is the Hermes pattern. It forces the agent to keep memory sharp instead of letting it bloat.

### Tests

- `test_memory_show_displays_content` — `/memory show` returns the expected content
- `test_skills_list_includes_usage` — `/skills list` includes usage counts from telemetry
- `test_skills_pin_protects_from_curator` — pinned skills survive a Curator pass that would otherwise prune them
- `test_memory_overflow_triggers_consolidation` — writing past the limit raises MemoryOverflowError and the agent loop catches it
- `test_curator_run_immediate_returns_report` — `/curator run` produces a report synchronously

---

## Acceptance criteria for the whole sprint

A user on Telegram should be able to:

1. Send a task to the agent that requires building a new skill, and **receive a Telegram notification** that the skill was created.
2. Run `/skills list` and see the new skill in the list with usage count `1`.
3. Run `/skills show <name>` and see the markdown content.
4. Run `/memory show` and see current MEMORY.md content with character count.
5. Run `/curator run` and receive a report.
6. Run `/skills pin <name>` and have that skill survive a subsequent Curator pass.
7. Watch the Beacon dashboard and see skill/memory events stream in via WebSocket (even without a dedicated UI panel — raw events in console are fine for this sprint).

If memory exceeds the hard limit, the next agent action should trigger consolidation, not a silent truncation or unbounded write.

---

## Constraints

- **Branch `feat/visible-memory-skills`.** Off current origin/main (commit `b190f19` or later).
- **No commits to main.** Will squash-merges.
- **No new external dependencies.** Scheduler should use whatever the daemon already uses (APScheduler, asyncio task, or the existing PeriodicNudge scheduler).
- **No changes to LCM, Adapter Layer, or GRAFT-SYMBIOTE.** This sprint is surface-only; don't touch the moats.
- **Pruned skills go to archive, never delete.** Curator archives to `~/.prometheus/skills/archive/`. The user can recover anything.
- **Notifications default to `quiet`, not `verbose`.** Users opt in to full descriptions.
- **All three work streams ship together.** Don't merge one without the others — the user-visible value comes from the combination.
- **Wiring tests are mandatory.** Every new signal type, command handler, and scheduler entry gets a wiring test in `tests/test_wiring.py` (or a split file).

---

## Reporting back

When done:

1. Branch + commit SHAs (likely 3–6 commits across the three work streams)
2. Full list of new Telegram commands wired
3. Confirmation that all four signal types route to both Telegram and Beacon WebSocket
4. A screenshot or transcript of a Telegram session showing: task → skill created notification → `/skills list` → `/skills show` → `/curator run` → report
5. The output of `~/.prometheus/curator/REPORT.md` after the first scheduled run (or after `/curator run` if you trigger it manually)
6. Any drive-by findings — bugs noticed while reading the code that weren't part of this sprint. Don't fix them; note them.
