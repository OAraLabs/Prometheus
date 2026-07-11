# Memory & knowledge

Prometheus remembers in layers. Every message you exchange is persisted losslessly to SQLite, a pair of bounded memory files rides every system prompt, a background extractor distills conversations into structured facts, those facts are recalled automatically when relevant, and a compiled wiki turns them into a cross-linked knowledge base you can browse in Obsidian. An optional idle-time layer (SENTINEL) keeps the whole thing tidy while you're away. This page explains each layer, what it does by default, and where the data lives on disk.

[← README](../../README.md)

## The layers at a glance

| Layer | What it does | Default |
|---|---|---|
| LCM | Persists every message to SQLite, losslessly; compacts the live context when it grows | On |
| File memory | `MEMORY.md` + `USER.md` ride every system prompt; the agent edits them | On |
| Memory extractor | Mines conversations into structured facts every ~30 minutes | On |
| Passive recall | Injects relevant stored facts into each turn's system prompt | On |
| Wiki | Compiles facts into cross-linked entity pages, browsable in Obsidian | On (tools always available; automatic recompiles ride SENTINEL) |
| SENTINEL | Idle-time observer + "dreaming" maintenance phases | **Off** (opt-in) |

## Lossless Context Management (LCM)

**Default: on** (`compaction.enabled: true`)

Every message in a session — yours, the model's, and every tool call and result — is written to a SQLite database as it happens. Nothing is ever truly thrown away.

When the live context grows past its budget, compaction happens in two tiers:

- **Tier 1** strips the bodies of old tool results (the parts least likely to matter again). This is free — no model call needed.
- **Tier 2** batch-summarizes older stretches of conversation with an LLM, organizing the summaries into a DAG so each summary knows exactly which original messages it covers.

Because the originals are always still in the database, compaction is reversible: the agent can call `lcm_expand` to pull the full original text of any summarized region back into view, and `lcm_grep` runs full-text (FTS5) search across the entire durable history — including everything that has been compacted out of the live window.

The `/context` command shows the current context budget and how much of it is in use. (The Status panel in Beacon shows the same LCM token gauge — see `../assets/shots/panel-status.png`.)

## File memory

**Default: on**

Two plain markdown files ride every system prompt:

- **`MEMORY.md`** (bounded to 12K characters) — the agent's working notes: ongoing projects, decisions, things it has learned.
- **`USER.md`** (bounded to 8K characters) — what it knows about you: preferences, context, standing instructions.

The agent reads and edits these itself over time. You can inspect them at any point with the `/memory` command, or in Beacon under **Config → Memory**. Because they are size-bounded, the agent has to keep them curated rather than letting them grow without limit.

## Memory extractor

**Default: on**

Roughly every 30 minutes, a background pass mines recent conversations into structured facts, organized by entity (a person, a client, a project, a topic) and tagged with a confidence score. These facts land in `memory.db` and become the raw material for passive recall and the wiki. Machine-generated sessions (automated jobs, evals) are excluded so the fact store reflects real conversations, not the system talking to itself.

## Passive recall

**Default: on** (config: `memory.recall`)

At the start of every agent turn, your latest message is matched (FTS5, any-token) against the facts in `memory.db`, and the best few ride that turn's system prompt as a "# Recalled memory" section. Mention a client from three weeks ago and the relevant facts are simply there — no explicit lookup needed.

It is deliberately conservative:

- **Request-only** — recalled facts never enter durable history, so the extractor never re-ingests its own output.
- **Fails open** — a missing or broken `memory.db` never blocks a turn.
- **Chat surfaces only** — Telegram, Slack, Discord, web, and CLI recall; coding mode, the gym, and evals never do.
- **Capped** — at most 6 facts per turn (`max_facts`), 900 characters rendered (`max_chars`), and only facts at or above 0.6 confidence (`min_confidence`). All three are tunable under `memory.recall` in the config.

## Wiki knowledge system

**Default: on** (query/capture tools always available; automatic recompiles ride SENTINEL's dream cycle)

The WikiCompiler projects the facts in `memory.db` into cross-linked markdown entity pages — `people/`, `clients/`, `projects/`, `topics/` — under `~/.prometheus/wiki/`. Pages link to each other, so the store reads like a small personal wiki rather than a flat database.

Working with it:

- **`/wiki`** shows compiler stats.
- **`/note [@entity] <text>`** is quick capture — it writes a durable, maximum-trust fact straight to `memory.db`, and the compiler projects it onto a page. This is the *only* supported way to put your own notes into the wiki.
- The agent itself uses **`wiki_query`** (read the knowledge base) and **`wiki_lint`** (check page hygiene) as tools.

### Obsidian view

The wiki is Obsidian-compatible, and there is a supported read-only setup — full details in [OBSIDIAN-VIEW.md](../OBSIDIAN-VIEW.md). The short version:

- The wiki is **compiled, not authored**: the compiler wipes and rewrites the pages from `memory.db` on each cycle, so anything you hand-edit in Obsidian is gone on the next compile. Capture goes through `/note`, never the editor.
- `scripts/install_obsidian_view.sh` installs the repo's Obsidian config (from `config/obsidian/`) into the vault. It includes a graph color group that highlights **manually captured** facts — your `/note` entries render as distinct nodes against the auto-extracted mass. The config survives recompiles, so you install it once.
- From another machine, mount `~/.prometheus/wiki/` over Tailscale/SSHFS and open it as a vault. Mount the pages read-only, but give `.obsidian/` a writable path — Obsidian needs to write its own workspace state even when your notes stay untouchable.

## SENTINEL

**Default: OFF** (`sentinel.enabled: false`) — opt-in

SENTINEL is the idle-time layer: it runs only when you have been inactive (15 minutes by default) and does housekeeping while you're away. It has two halves:

- **Observer** — watches signal patterns and, when something looks worth your attention, **nudges you via Telegram. It never auto-executes anything**; the nudge is the entire action.
- **AutoDream** — periodic "dreaming" (every 30 minutes of idle, by default) in four phases:
  1. **Wiki lint** — hygiene checks on the compiled pages.
  2. **Memory consolidation** — deduplicates and tidies the fact store.
  3. **Telemetry digest** — rolls up usage data.
  4. **Knowledge synthesis** — the only phase that calls an LLM, and it is budget-capped at 2000 tokens per cycle.

The first three phases use zero LLM tokens, so an idle SENTINEL costs essentially nothing. Check its status with the `/sentinel` command, or in Beacon under **Config → Sentinel**. Enable it by setting `sentinel.enabled: true` in your config (idle threshold and dream interval are tunable alongside it).

## Where the data lives

Everything user-generated sits under `~/.prometheus/` (config and data are kept apart from the code):

- **`memory.db`** — the structured fact store (extractor output, `/note` captures) that recall and the wiki read from.
- **`data/lcm.db`** — the lossless conversation history.
- **`wiki/`** — the compiled entity pages (plus the Obsidian config, once installed).
- **`sentinel/`** — SENTINEL's persisted signals and state.
- **`telemetry.db`**, **`data/security/audit.db`** — usage telemetry and the security audit log.

Each concern gets its own database — conversations, facts, telemetry, and audit never share a file, so you can inspect or wipe one without touching the others.

`prometheus --reset-data` deletes all of it — `telemetry.db`, `memory.db`, `lcm.db`, the audit log, `eval_results/`, `wiki/`, `sentinel/`, and auto-generated skills (`skills/auto/`) — after listing exactly what it found and asking for confirmation. Your config files are preserved. (`--reset-telemetry` wipes only the telemetry database.)
