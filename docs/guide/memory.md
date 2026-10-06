# Memory & knowledge

Prometheus remembers in layers. Every message you exchange is persisted to SQLite, a pair of bounded memory files rides every system prompt, a background extractor distills conversations into structured facts, those facts are recalled automatically when relevant, and a compiled wiki turns them into a cross-linked knowledge base you can browse in Obsidian. A context compactor keeps the prompt inside the model's window, and an optional idle-time layer (SENTINEL) keeps the whole thing tidy while you're away. This page explains each layer, what it does by default, and where the data lives on disk.

[← README](../../README.md)

## The layers at a glance

| Layer | What it does | Default |
|---|---|---|
| LCM | Persists every message to SQLite and summarizes older stretches into a searchable DAG | On (no switch) |
| Context compactor | Summarizes the oldest turns of the request sent to the model when it nears the context window | On in the shipped config (`compaction.enabled: true`) |
| File memory | `MEMORY.md` + `USER.md` ride every system prompt, as read when the daemon starts; the agent edits them | On |
| Memory extractor | Mines conversations into structured facts every ~30 minutes | On |
| Passive recall | Injects relevant stored facts into each turn's system prompt | On |
| Wiki | Compiles facts into cross-linked entity pages, browsable in Obsidian | On (pages recompile after each extraction pass) |
| SENTINEL | Idle-time observer + "dreaming" maintenance phases | **Off** (opt-in) |

**Retrieval is keyword-only.** Passive recall and `lcm_grep` use SQLite full-text search (FTS5), and `wiki_query` matches words against the wiki's index. There are no embeddings and no vector store, so a fact is found by the words it shares with the question, not by meaning.

## Lossless Context Management (LCM)

**Default: on** (there is no switch)

Every message in a session — yours, the model's, and every tool call and result — is written to a SQLite database (`data/lcm.db`) as the conversation goes. A few things never reach it, or leave it later:

- **Secrets are redacted.** Token-shaped strings (API keys and the like) are masked before a row is written. The live conversation keeps what you sent, so a key you paste for the agent to use still works for the rest of that session; only the stored copy is masked. `oara scrub` redacts anything an older build already kept.
- **Ephemeral sessions aren't stored.** After `/ephemeral on`, a chat writes no LCM rows, so it never reaches the extractor, `memory.db` or the wiki either.
- **Forgotten sessions can be purged.** Forgetting a session hides it but keeps its rows. `oara retention --apply` deletes forgotten sessions once they pass their window (7 days for machine traffic, 90 for conversations) and haven't spoken since. Without `--apply` it only prints the plan.

In the background, LCM batch-summarizes each session's older messages with the model (everything except the most recent 32) and organizes the summaries into a DAG, so each summary knows exactly which original messages it covers. As leaf summaries pile up, they are summarized again one level higher.

**The summaries are for search, not for the prompt.** Nothing from the DAG goes into what the model is sent — keeping the prompt inside the window is the [context compactor](#context-compactor)'s job, and it works separately. The agent reaches the stored history through the LCM tools:

- **`lcm_grep`** — full-text (FTS5) search over every stored message and summary, in one session or across all of them, including everything that has long since left the live window. Results are 300-character snippets.
- **`lcm_describe`** — inspect a summary node.
- **`lcm_expand`** — open a summary. A depth-0 summary lists its source messages, each cut to its first 400 characters; a higher one lists its child summaries.
- **`lcm_expand_query`** — search the summaries for a question and expand the best matches, up to 500 characters per message.

None of these returns a long message whole. On local models they also aren't in the advertised tool list: the shipped `tools.deferred_loading` setting (`auto`) advertises only a short `always_loaded` set on local backends, and no `lcm_*` or `wiki_*` tool is in it, so the model has to find them with `tool_search` first. Cloud providers get the full catalog.

Beacon's Status panel shows an LCM token gauge (see `../assets/shots/panel-status.png`). It measures the DAG's own view of the session — its summaries plus the recent messages — not the request the model is sent. `/context` measures that.

## Context compactor

**Default: on** in the shipped config (`compaction.enabled: true`). If the key is missing from your config, the compactor is off and the daemon warns at boot.

LCM keeps the history; the compactor keeps the prompt inside the model's window. Before each model call it estimates the whole request — system prompt, conversation and tool schemas. Once that passes 75% of the window (after reserving 4,096 tokens for the reply), it summarizes the oldest stretch of turns with the same model and sends the summary in their place. The 8 most recent user turns are never summarized.

The swap happens only in the request. The session's own messages are untouched, and the compactor never writes to `lcm.db`. If a summary fails, it says so — an error in the log and a `context_compaction_failed` event — and sends the request unchanged.

The `/context` command shows the window, how much of it is in use, and the point where the compactor fires.

Separately, on local models the agent loop trims old tool results during a long run: results from more than three rounds back are cut to a short excerpt. It costs no model call, and it is skipped on cloud providers by default (`context.microcompact_on_cloud`), where rewriting history would throw away the provider's prompt cache.

## File memory

**Default: on**

Two plain markdown files ride every system prompt:

- **`MEMORY.md`** (bounded to 12K characters) — the agent's working notes: ongoing projects, decisions, things it has learned.
- **`USER.md`** (bounded to 8K characters) — what it knows about you: preferences, context, standing instructions.

The agent reads and edits these itself over time. You can inspect them at any point with the `/memory` command, or in Beacon under **Config → Memory**.

**The prompt carries a copy taken when the daemon starts.** The daemon builds its system prompt once, at startup, with both files read into it. Edits made after that — the agent's own `memory` tool calls, or yours through Beacon or the API — land on disk straight away and show in `/memory`, but the copy in the system prompt stays as it was until the daemon restarts. Within a conversation, the agent still has its own edits in front of it, as the `memory` calls it made.

**A full file drops its oldest entries, silently.** Each line is one entry. When the agent's add or replace would push a file past its limit, the oldest entries are removed until it fits, and nothing reports it. (A single entry bigger than the whole limit is refused instead.) Keeping the files curated is the only way to keep old notes from falling off the top.

### Editing memory remotely

You can also edit both files yourself, from anywhere the API reaches — `PUT /api/memory/current` replaces the content of `MEMORY.md` and/or `USER.md`, and Beacon's **Config → Memory** tab is the UI over it. The write path is deliberately careful:

- **Budgets are enforced.** Over-budget content is refused with a 400 and **nothing is written** — the same character limits the agent lives under apply to you.
- **Every edit is reversible.** The previous content is snapshotted to `~/.prometheus/memory-history/` before each write.
- **Optional optimistic concurrency.** Send `base_memory`/`base_user` (the content you loaded) alongside your edit; if the agent moved the file in the meantime, the write is refused with a **409 that returns the current truth**, so your editor can rebase the draft instead of silently clobbering the agent's changes. Omit the base fields and you get plain last-writer-wins.

Like the agent's edits, yours reach the system prompt at the next daemon restart.

## Memory extractor

**Default: on**

Roughly every 30 minutes, a background pass mines recent conversations into structured facts, organized by entity (a person, a client, a project, a topic) and tagged with a confidence score. These facts land in `memory.db` and become the raw material for passive recall and the wiki. Machine-generated sessions (automated jobs, evals) are excluded so the fact store reflects real conversations, not the system talking to itself. A fact that restates one already stored folds into it and raises its mention count.

- **At most 500 messages per pass.** Anything beyond that waits for the next pass.
- **Summarized messages are never mined.** Once LCM has summarized a message, the extractor skips it. LCM runs an extraction pass on a session just before summarizing it, but a message that pass doesn't reach — because it failed, or because of the 500-message cap — is skipped for good.

## Passive recall

**Default: on** (config: `memory.recall`)

At the start of every agent turn, your latest message is matched (FTS5, any-token) against the facts in `memory.db`, and the best few ride that turn's system prompt as a "# Recalled memory" section. Mention a client from three weeks ago and the relevant facts are simply there — no explicit lookup needed. Facts you captured with `/note` go first.

It is deliberately conservative:

- **Request-only** — recalled facts never enter durable history, so the extractor never re-ingests its own output.
- **Fails open** — a missing or broken `memory.db` never blocks a turn.
- **Chat surfaces only** — Telegram, Slack, Discord, web, and CLI recall; coding mode, the gym, and evals never do.
- **Capped** — at most 6 facts per turn (`max_facts`), 900 characters rendered (`max_chars`), only facts at or above 0.6 confidence (`min_confidence`), at most 2 facts from any one entity (`per_entity_cap`), and at most 12 search terms taken from your message (`max_keywords`). All five are tunable under `memory.recall` in the config.

## Wiki knowledge system

**Default: on** (pages recompile after each extraction pass; the `wiki_query` and `wiki_compile` tools are always registered)

The WikiCompiler projects the facts in `memory.db` into cross-linked markdown entity pages — `people/`, `clients/`, `projects/`, `topics/` — under `~/.prometheus/wiki/`. Pages link to each other, so the store reads like a small personal wiki rather than a flat database.

An entity gets a page once it has at least **2 mentions** in total (counted across all its facts, ignoring case), or as soon as it has a `/note`. Below that, its facts stay in `memory.db`, where recall can still find them, but there is no page.

**When pages change.** After each scheduled extraction pass that finds new facts, the extractor hands them to the compiler. It rebuilds, from `memory.db`, the page of each entity those facts are about, then rewrites `index.md`. Pages for entities the pass didn't touch are left as they are.

Working with it:

- **`/wiki`** shows compiler stats.
- **`/note [@entity] <text>`** is quick capture — it writes a durable, maximum-trust fact straight to `memory.db`. It doesn't write the wiki: the page appears (or updates) on a later compile that touches that entity, usually the next extraction pass that turns up a fact about it. Until then the note is in `memory.db`, and recall can already use it. This is the *only* supported way to put your own notes into the wiki.
- The agent uses **`wiki_query`** to read the knowledge base. It matches the words of the question against the page names and one-line summaries in `index.md` — plain word overlap, not full-text search — and reads up to five pages, `/note` pages first. A substantial multi-page answer is saved to `wiki/queries/` for next time.
- **`wiki_compile`** lets the agent compile facts saved since the last compile, on demand.
- **`wiki_lint`** checks page hygiene. It is registered only when SENTINEL is on.

### Obsidian view

The wiki is Obsidian-compatible, and there is a supported read-only setup — full details in [OBSIDIAN-VIEW.md](../OBSIDIAN-VIEW.md). The short version:

- The wiki is **compiled, not authored**: each compile rewrites, from `memory.db`, the page of every entity it touches, so anything you hand-edit in Obsidian is gone as soon as a compile touches that entity. Capture goes through `/note`, never the editor.
- `scripts/install_obsidian_view.sh` installs the repo's Obsidian config (from `config/obsidian/`) into the vault. It includes a graph color group that highlights **manually captured** facts — your `/note` entries render as distinct nodes against the auto-extracted mass. The config survives recompiles, so you install it once.
- From another machine, mount the wiki root (`wiki.root`, default
  `~/.prometheus/wiki/`) over Tailscale/SSHFS and open it as a vault. Mount the pages read-only, but give `.obsidian/` a writable path — Obsidian needs to write its own workspace state even when your notes stay untouchable.

## SENTINEL

**Default: OFF** (`sentinel.enabled: false`) — opt-in

SENTINEL is the idle-time layer: it runs only when you have been inactive (15 minutes by default) and does housekeeping while you're away. It has two halves:

- **Observer** — watches signal patterns and, when something looks worth your attention, **nudges you via Telegram. It never auto-executes anything**; the nudge is the entire action.
- **AutoDream** — periodic "dreaming" (every 30 minutes of idle, by default) in four phases:
  1. **Wiki lint** — hygiene checks on the compiled pages: orphans, broken links, stale pages, likely duplicates, missing cross-references, category balance. It only reports. `sentinel.auto_fix_wiki` is on by default but has nothing to apply: no automatic fixes are defined, because links and cross-references belong to the compiler.
  2. **Memory consolidation** — merges near-duplicate facts about the same entity (the lower-confidence copy is deleted and its mentions added to the one kept), decays the confidence of facts nobody has mentioned for 90 days (0.05 per 30 days, never below 0.1), and **permanently deletes** any fact below 0.1 confidence. A decayed fact drops out of passive recall once it falls under recall's 0.6 floor.
  3. **Telemetry digest** — rolls up usage data.
  4. **Knowledge synthesis** — the only phase that calls an LLM. It finds entities whose facts keep coming from the same messages, asks the model for a short insight about each cluster, and writes it to `wiki/queries/insight-<topic>.md`, skipping clusters that haven't changed since their page was written. It stops starting new calls once a cycle has spent 2000 tokens.

The first three phases use zero LLM tokens, so an idle SENTINEL costs essentially nothing. Check its status with the `/sentinel` command, or in Beacon under **Config → Sentinel**. Enable it by setting `sentinel.enabled: true` in your config (idle threshold and dream interval are tunable alongside it).

## Where the data lives

Everything user-generated sits under `~/.prometheus/` (config and data are kept apart from the code):

- **`memory.db`** — the structured fact store (extractor output, `/note` captures) that recall and the wiki read from.
- **`data/lcm.db`** — the conversation history and its summary DAG.
- **`wiki/`** — the compiled entity pages, plus `queries/` (saved `wiki_query` answers and SENTINEL's insight pages) and the Obsidian config, once installed.
- **`sentinel/`** — SENTINEL's persisted signals and state.
- **`telemetry.db`**, **`data/security/audit.db`** — usage telemetry and the security audit log.

Each concern gets its own database — conversations, facts, telemetry, and audit never share a file, so you can inspect or wipe one without touching the others.

The fact store is also **self-healing**: full-text search indexes are kept in sync by SQLite triggers (they can't drift from the tables they index), schema migrations are versioned and applied once per database, and every migration snapshots a `memory.db.backup-<timestamp>` copy before touching anything.

`oara --reset-data` deletes all of it — `telemetry.db`, `memory.db`, `lcm.db`, the audit log, `eval_results/`, `wiki/`, `sentinel/`, and auto-generated skills (`skills/auto/`) — after listing exactly what it found and asking for confirmation. Your config files are preserved. (`--reset-telemetry` wipes only the telemetry database.)
