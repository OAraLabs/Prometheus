# SPRINT ADDENDUM — Read Hermes Source First

**Applies to:** SPRINT-1 (Visible Memory & Skills), SPRINT-2 (Durability & Steering), SPRINT-3 (Polish & Platform)

**Purpose:** The three sprint specs reference Hermes patterns by name (`/steer`, `/queue`, Curator, file-mutation verifier, NL cron) but don't point Claude Code at the actual Hermes implementations. Both projects are Python and Hermes is MIT-licensed. Reading the upstream first prevents reinventing patterns that already work, and produces better-attributed code.

This addendum prepends to each sprint and modifies Step "Read These Files First" to include the Hermes source.

---

## Why this matters

Hermes is MIT-licensed Python. Their implementations of `/steer`, Curator, file-mutation verifier, and the rest have shipped, been used by 545+ contributors per release, and have hit edge cases we haven't thought of yet. Reading their source before writing ours produces:

1. **Better designs** — they've solved problems we haven't encountered yet (queue cancellation semantics, concurrency under load, edge cases in NL parsing)
2. **Proper attribution** — provenance headers on adapted code, same pattern Prometheus already uses for OpenHarness donors
3. **Honest "we built on prior art" framing** — not pretending we invented things we borrowed

This is the same pattern Prometheus already follows for OpenHarness extraction. Hermes joins the donor registry for these specific subsystems.

---

## Method — Per Sprint

Before writing code in each sprint, do this:

### 1. Read the relevant Hermes modules from GitHub

Use `web_fetch` (or `gh` CLI with raw content URLs) against the Hermes repo. Do NOT clone Hermes — read what's needed, paraphrase the patterns, attribute properly.

Base URL: `https://github.com/NousResearch/hermes-agent`
Raw content base: `https://raw.githubusercontent.com/NousResearch/hermes-agent/main/`

### 2. Identify the right modules

For each Prometheus feature being built, find the analog in Hermes. Suggested grep targets on the Hermes side:

**For SPRINT-1 (Visible Memory & Skills):**
- Curator implementation: search for `curator` in `hermes_agent/` — look for `curator.py`, scheduler wiring, grading logic
- Skill creation: search for `skill_creator`, `skill_extract`, autonomous skill writing
- Skill refinement: `skill_refiner`, `skill_update`
- Memory limits enforcement: search for `MEMORY.md`, `USER.md`, character limit constants
- Slash commands (`/skills list`, `/memory show`, `/curator`): search for command registration in their gateway code

**For SPRINT-2 (Durability & Steering):**
- `/steer` and `/queue`: search for `steer`, `queue`, session injection — likely in their ACP server or gateway code
- File-mutation verifier: search for `file_mutation`, `verifier`, `disk_changes`, per-turn footer logic — likely in their hook system
- Session state during agent loop: how they hold per-session queues without breaking concurrency

**For SPRINT-3 (Polish & Platform):**
- Slack gateway: their full Slack implementation as reference for command-set parity
- Natural-language cron: search for `cron`, `schedule_parse`, NL date parsing
- Setup wizard / init flow: `hermes init` implementation
- Local server auto-detection: per their GitHub issue #523, this is something they want but haven't built — we may not find a reference here, but check anyway

### 3. Paraphrase, don't copy verbatim

The MIT license allows copying with attribution. But for Prometheus to keep its architectural identity, adapt rather than copy:

- **Read** the Hermes implementation
- **Understand** the design decisions and edge cases
- **Adapt** to Prometheus's idioms (e.g., Prometheus's SignalBus, hook system, session state)
- **Attribute** with a provenance header on the new file:

```python
# Pattern adapted from Hermes Agent (NousResearch/hermes-agent)
# Original: hermes_agent/learning/curator.py
# License: MIT
# Adaptation notes:
# - Routes through Prometheus SignalBus instead of Hermes's internal events
# - Uses Prometheus LLM judge (constrained decoding) instead of Hermes's API-based grading
# - Outputs to ~/.prometheus/curator/ instead of ~/.hermes/curator/
```

### 4. Note where Prometheus deliberately diverges

For each pattern adapted, write a one-line comment explaining where Prometheus's implementation departs from Hermes and why. This is documentation for future maintainers (including us) about which decisions were deliberate.

Examples:
- "Hermes runs Curator on a 7-day cycle; Prometheus also defaults to 7 days but exposes interval via config because solo-operator skill libraries grow slower."
- "Hermes's `/steer` accepts multi-line text via a continuation prompt; Prometheus accepts single-message text only because Telegram doesn't have the same multi-input UX."
- "Hermes file-mutation verifier covers shell builtins via tracing; Prometheus uses regex parsing of bash command strings because we don't have the tracing infrastructure."

---

## Updated "Read These Files First" sections

Add this to the top of each sprint's "Read These Files First" section:

### SPRINT-1 addendum

```
### Hermes source (read for patterns; do NOT clone)

Use web_fetch against these Hermes paths to read the upstream patterns
before writing Prometheus equivalents:

- https://github.com/NousResearch/hermes-agent — explore the tree, find:
  - hermes_agent/learning/curator.py (or similar) — Curator implementation
  - hermes_agent/learning/skill_creator.py — autonomous skill creation
  - hermes_agent/learning/skill_refiner.py — skill self-improvement
  - hermes_agent/memory/ — MEMORY.md and USER.md handling, character limits
  - hermes_agent/gateway/ — /skills, /memory, /curator command handlers

If the file paths above don't match the current Hermes layout, fetch
the repo tree first via `web_fetch https://github.com/NousResearch/hermes-agent/tree/main/hermes_agent`
and locate the equivalents.

Attribute every adapted file with a provenance header (see addendum).
```

### SPRINT-2 addendum

```
### Hermes source (read for patterns; do NOT clone)

- hermes_agent/ — find their /steer and /queue implementations
  (likely in gateway or ACP server code)
- hermes_agent/hooks/ — file-mutation verifier (v0.14 release)
- Search for "queued_steers", "queued_prompts" or equivalent session-state
  patterns to see how they handle concurrency

Pay specific attention to:
- How they distinguish system-message injection from user-message turns
- How they handle queue draining when the agent has stopped tool-calling
- Their cancellation semantics (`/unqueue`, `/clear-steers`)
- Their file-mutation diff format and how the footer is structured

Attribute every adapted file with a provenance header.
```

### SPRINT-3 addendum

```
### Hermes source (read for patterns; do NOT clone)

- hermes_agent/gateway/slack/ — full Slack implementation as reference
  for command-set parity with our Telegram surface
- hermes_agent/scheduling/ or hermes_agent/cron/ — natural-language cron
  parsing
- hermes_agent/cli/init.py or hermes_agent/setup_wizard.py — the
  `hermes init` flow as design reference

For local server auto-detection: per Hermes GitHub issue #523, they
don't have this yet. We're building it before they do — check the issue
for design hints from the community discussion.

Attribute every adapted file with a provenance header.
```

---

## Constraints

- **MIT license preserved.** Hermes is MIT; provenance headers carry the license forward.
- **No verbatim copying.** Adapt to Prometheus idioms. The Hermes code is reference; the Prometheus code is ours.
- **One file per pattern.** If a Hermes module covers multiple concepts (e.g., Curator + scheduler + report writer in one file), split into Prometheus's idiomatic layout.
- **Read before designing.** Don't write a Prometheus design first and then check Hermes — read theirs first, then design. This is the discipline that prevents reinventing.
- **Note the diff.** Every adapted file ends with a one-paragraph "Differences from Hermes" comment in the module docstring.

---

## Reporting back additions

In each sprint's reporting-back section, add:

- List of Hermes files read (by URL or path)
- For each adapted pattern: original Hermes source path + Prometheus destination
- Explicit list of where the Prometheus implementation diverges deliberately, with reasoning

This makes the borrowing legible to future audits and gives you (Will) an honest "we built on Hermes for these specific subsystems" story when talking publicly about Prometheus.
