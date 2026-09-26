# Skill usage: does the model use the skills Prometheus writes for it?

**Date:** 2026-09-26 · **Code investigated:** origin/main `a7080e6` (the mini's daemon runs v0.9.3
`b2603ad`; every file on the skill path is identical in both) · **Data:** the mini's nightly snapshot
`20260926T065501Z` · **Status:** report only. Nothing in `src/` changes; the scripts in
`docs/audits/skill-usage/` hold no data.

## Answer

**No.** Across every tool call in telemetry (16,414 calls, 2026-04-06 → 2026-09-24) the model called
`skill` **3 times**. All three were in July. All three loaded a skill from the user skills directory,
and **no auto-written skill has ever been loaded**. The last load was on 2026-07-21. In LCM, **1 of 865**
human turns loaded a skill. SkillCreator has written 57 skills (20 still served, 37 archived), and not
one has been read by the model.

The near-zero is not an artifact of GEPA's broken detector (§6): the underlying count is near zero
too. It has more than one cause. Ranked:

1. **The model is never shown what the skills are.** The prompt has never listed a non-core skill.
   Today it names the three builtins, then says "131 additional skills available on demand… use
   tool_search… for any task you're unsure how to approach". On the 151 runs that did have the
   `skill` tool in their tool list, it was called 0 times. None of the 329 recorded `tool_search`
   queries mentions skills.
2. **On most runs the `skill` tool isn't in the tool list.** Deferred loading advertises the 13-tool
   `always_loaded` set, which has `tool_search` but not `skill`. That was **822 of 973** main-registry
   runs (84%), and every local run since 2026-08-03. **167 of 287** Qwen 3.8 Max runs (58%) also
   went without it. Until #462 the advertisement was resolved before routing, so those runs were
   served with the local model's deferred set.
3. **Search rarely leads to a load.** All three loads came two seconds after a `tool_search` in July.
   Since the last one, 299 searches (80 outside the nightly evals) have produced no load. When a search
   does list skills, it is mostly by accident: 20 of the 25 user searches that listed one got there by
   edit distance alone. The listing then tells the model to call a tool that most runs weren't given.
4. **Few requests match a skill, and the auto skills are thin.** A skill clearly matched about
   **0.8–2.5%** of human turns: 7 turns verified by hand, about 22 estimated. None of those turns loaded a skill. Of the
   57 auto skills, 13 are reusable procedures. 28 duplicate something generic or another skill, and 16
   are one-off recipes.
5. **The learning loop can't see any of this.** It has no usage signal. SkillRefiner "refines" the
   newest auto skill after every task with 3+ tool calls (208 model calls, none on a reuse). The Curator prunes on file
   mtime, which it labels `last_used_days_ago`. GEPA filters on the wrong tool name, field and export
   shape. A skill body that *is* loaded on a local run gets microcompacted to about 500 chars three
   user messages later.

**Fix options (§9).** The plumbing (C) comes first under any option: **+90 tokens** per deferred
request, low risk. Relevance-picked skills (B) is the recommended next step, framed as an Instinct
pick-or-abstain decision. At T = 0.75 it fires on **1.7%** of turns, costs **~1.3 tokens per turn**
on average (76 when it fires), and 7 of its 15 picks were right. The full catalog in the prompt (A)
costs **7,193 tokens** per request, 10% of the local 72k window.

---

## Data, method, constraints

- **Sources.** Everything was read on the mini.
  - The snapshot's top-level `telemetry.db` and its `data/lcm.db`, both opened `mode=ro&immutable=1`.
    Neither live DB was opened.
  - The skill files under `~/.prometheus/skills` (including `auto/`, `auto/.archive/` and
    `auto/_state.json`), the Curator's `run.json` records, and the 26 golden-trace export files. All
    were read read-only.
  - The deployed config, read for selected keys only.
- **What left the mini.** Only aggregates: counts, shares, sizes, token totals and cosine quantiles.
  Calibrating the matcher and labelling its hits needed some request texts and skill names/descriptions.
  Those were printed to the operator's terminal only (with secret-shaped strings masked), never written
  to disk, and are not in this report. **This report contains no skill names, bodies or conversation
  text.**
- **Windows.**
  - Telemetry `tool_calls`: 2026-04-06 → 2026-09-24, 16,414 calls. The 15,019 `_loop_transition`
    pseudo-rows are excluded.
  - LCM: 2026-05-27 → 2026-09-24, with 865 human turns on user surfaces.
  - Tool-advertisement rows start 2026-07-31. `tool_calls.session_id` starts 2026-08-15.
  - No tool call or LCM message is newer than 2026-09-24 20:02 UTC: the daemon had no conversations
    between then and the snapshot.
- **Matcher.** The Instinct kNN baseline's encoder, reproduced exactly:
  - model `BAAI/bge-small-en-v1.5` at `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, its own ONNX export;
  - onnxruntime on CPU, 256-token truncation, CLS pooling, L2-normalised, no query instruction;
  - both files checked against Instinct's SHA-256 pins. They were copied from the Mac to the mini, so
    the model came to the data.
- **Token counts.** Measured with the Qwen3 tokenizer (`Qwen/Qwen3-8B` at `b968826d…`), next to
  Prometheus's own 4-chars-per-token estimator. Qwen 3.8's tokenizer is only on the
  4090, which this work did not touch. Treat the counts as close, not exact.
- **Rules kept.** The daemon was not restarted and the 4090 was not used. The only processes started
  were short foreground SSH scans, and no process was stopped. The mini's scratch directory held only
  public model files, the scripts and a public tool list. It has been removed.

---

## 1. Real usage

### Calls by provider and model (all telemetry tool calls)

| provider | model | calls | `tool_search` | `skill` |
|---|---|---:|---:|---:|
| qwen (Alibaba cloud) | qwen3.8-max | 6,772 | 8 | 0 |
| llama.cpp (4090) | Qwen3.8-27B-UD-Q4_K_XL.gguf | 3,126 | 68 | 0 |
| llama.cpp (4090) | gemma4-26b | 2,362 | 37 | 0 |
| llama.cpp (4090) | blank model name (served the Qwen3.8-27B GGUF) | 1,462 | 221 | 0 |
| llama.cpp (4090) | google_gemma-4-26B-A4B-it-Q4_K_M.gguf | 1,071 | 0 | 0 |
| llama.cpp (4090) | Qwen3.6-27B-UD-Q4_K_XL.gguf | 563 | 1 | **1** |
| llama.cpp (4090) | qwen3.8-27b | 434 | 0 | 0 |
| xai (cloud) | grok-4.5 | 322 | 3 | **2** |
| qwen (Alibaba cloud) | qwen3.8-flash | 196 | 0 | 0 |
| anthropic (cloud) | claude-haiku-4-5 / claude-sonnet-4-5 | 104 | 0 | 0 |
| xai (cloud) | grok-3 | 2 | 0 | 0 |
| **total** | | **16,414** | **338** | **3** |

By provider: llama.cpp 9,018 / 327 / 1 · qwen 6,968 / 8 / 0 · xai 324 / 3 / 2 · anthropic 104 / 0 / 0.

### Calls by surface

| surface (session prefix) | calls | `tool_search` | `skill` |
|---|---:|---:|---:|
| no session column yet (before 2026-08-15) | 6,237 | 52 | 3 |
| beacon | 5,218 | 11 | 0 |
| telegram | 2,240 | 2 | 0 |
| nightly evals (session `system`, a 7-tool registry without `skill`) | 1,520 | 219 | 0 |
| web (the pre-#458 namespace) | 516 | 0 | 0 |
| desktop | 516 | 47 | 0 |
| ios | 116 | 0 | 0 |
| coding runs / test harness / other | 51 | 7 | 0 |

### By ISO week

| week | calls | `tool_search` | `skill` | | week | calls | `tool_search` | `skill` |
|---|---:|---:|---:|---|---|---:|---:|---:|
| W15 | 256 | 9 | 0 | | W28 | 177 | 1 | **2** |
| W17 | 105 | 0 | 0 | | W29 | 25 | 0 | 0 |
| W18 | 108 | 0 | 0 | | W30 | 133 | 2 | **1** |
| W19 | 80 | 0 | 0 | | W31 | 267 | 1 | 0 |
| W20 | 82 | 0 | 0 | | W32 | 141 | 0 | 0 |
| W21 | 46 | 0 | 0 | | W33 | 3,945 | 11 | 0 |
| W22 | 129 | 0 | 0 | | W34 | 661 | 2 | 0 |
| W23 | 81 | 0 | 0 | | W35 | 1,059 | 1 | 0 |
| W24 | 989 | 28 | 0 | | W36 | 761 | 52 (47 desktop) | 0 |
| W25 | 269 | 0 | 0 | | W37 | 3,235 | 219 (evals) | 0 |
| W26 | 36 | 0 | 0 | | W38 | 2,043 | 1 | 0 |
| W27 | 148 | 0 | 0 | | W39 (to 09-24) | 1,638 | 11 | 0 |

### `tool_search`: what was asked, and did the answer contain a skill?

- **Inputs** (from `parsed_tool_call`):
  - 272 searches, plus 2 with an empty query;
  - 55 `select` calls, none naming a skill;
  - 9 with no recorded input.
  - **0 of the 329 recorded queries mention skills.**
  - Outcomes: 331 OK, 5 tool errors, 2 validation failures. The model uses `tool_search` to find
    deferred tools, not skills.
- **Recorded results (LCM, 27 calls).**
  - 12 searches: 8 listed at least one skill (15 skill entries in all). The other 4 listed none, but
    all 4 were cut short in storage.
  - 12 selects listed none, and 3 selects errored.
- **Replay of every recorded input** (`tool_search_replay.py`).
  - Method: each query goes through the daemon's own `_score_tool`/`_score_skill`, with the same stable
    sort and top-5 cut. It is ranked against the 55 registered tools and the skills that existed at
    the call's time.
  - Check: the replay agrees with **every one of the 14 whole-JSON recorded results**. Its only
    disagreements are the 4 results cut short.

  | calls | searches | … list ≥1 skill | … only by edit distance (no query word matched) | selects listing a skill |
  |---|---:|---:|---:|---:|
  | user surfaces | 28 | 25 | **20** | 0 of 32 |
  | before 2026-08-15 | 24 | 21 | 1 | 0 of 19 |
  | nightly evals | 219 | 187 | 82 | – |
  | test/coding | 3 | 3 | 2 | 0 of 4 |

  So a search usually does return a skill. On user surfaces, only 5 of 28 searches returned one
  because it matched the query's text.

  - Why ranking works this way: the scorer ranks on substring and Levenshtein distance, and equal
    scores keep insertion order, so tools win ties with skills. A query that matches no text anywhere
    ranks everything by edit distance to the *name*, and 134 skill names outnumber 55 tool names.
  - A listed skill is followed by the hint `Use skill(name=…)`. On deferred runs that tool is not in
    the model's tool list.
  - Search → load has worked, rarely. Each of the three loads came two seconds after a `tool_search`.
    The one LCM recorded (2026-07-21) loaded a skill that its search had listed by name. Since the last
    load there have been 299 searches (219 in evals) and no load.

### `skill` calls and the share of turns that load one

- **3 calls, 3 successes, 0 "Skill not found".** No other tool name contains "skill". No blank-name
  call's raw output mentions one.
  - Two were grok-4.5 on 2026-07-10, in the same second. Both are marked `is_golden`.
  - One was the local Qwen3.6 GGUF on 2026-07-21.
  - All three resolve to **user-directory skills**. None was an auto skill.
- **Turns.** 1 of 865 human turns on user surfaces loaded a skill (0.12%).
- **Runs.** 2 of 2,886 runs since 2026-06-12 contained a `skill` call (0.07%).
- None since 2026-07-21.

## 2. The catalog

| source | files | served | core | bodies (Qwen3 tokens): median / max / total |
|---|---:|---:|---:|---|
| builtin (`src/prometheus/skills/builtin/`) | 3 | 3 | **3** | 290 / 300 / 870 |
| user directory `~/.prometheus/skills/*.md` | 111 | 111 | 0 | 1,327 / 4,552 / 166,773 |
| auto `~/.prometheus/skills/auto/*.md` | 20 | 20 | 0 | 460 / 1,000 / 8,974 |
| auto, archived (`auto/.archive/`) | 39 (2 are refiner backups) | 0 | – | – |

- **Served.** 134 distinct names, with no collisions. All 134 have a frontmatter description (median
  207 chars, p90 335, max 941).
- **The user directory is mostly the repo's own skills.**
  - 89 of the 111 are Lane-1 skills from the repo's `skills/`. 40 of those differ from the copy in the
    deployed tree.
  - 22 exist only locally.
  - By file mtime: April 91, May 17, June 2, August 1.
- **Auto skills by creation month:**

  | | May | Jun | Jul | Aug | Sep |
  |---|---:|---:|---:|---:|---:|
  | live | 1 | 5 | 7 | 5 | 2 |
  | archived | 8 | 15 | 6 | 10 | – |
  | `skill_created` signals | 3 | 20 | 13 | 14 | 2 |

  - Sizes: live median 1,912 B, archived median 1,148 B.
  - 18 of the 20 live skills are pinned.
  - Of the 39 archived, the Curator pruned 22 (July 5, August 10, September 7). The other 17 have no
    Curator record.
- **Ever loaded:** 0 of 59 auto files. The 3 loads in §1 went to user-directory skills.

## 3. What the model sees

**The skills section, as the daemon builds it at boot** (rendered with the deployed
`build_runtime_system_prompt` + `skills_for_prompt`; the three core lines are redacted to placeholders
here):

```
# Available Skills

## Core skills (always available)
- **<builtin skill 1>**: <its description>
- **<builtin skill 2>**: <its description>
- **<builtin skill 3>**: <its description>
## 131 additional skills available on demand. Use tool_search to find skills for any task you're unsure how to approach, then use the skill tool to load the skill's instructions.
```

- That is 112 Qwen3 tokens. It is built once per adapter at daemon start, so skills written later
  don't change the count until a restart.
- Since the initial commit the prompt has **never named a non-core skill**. First it carried a
  generic hint; since #192 (2026-08-14) it carries the three builtins and a count.

**The two tool descriptions:**

- `skill`: *"Read a builtin or user-defined skill by name."* Its one parameter is `name`. 90 tokens
  as an OpenAI-format schema.
- `tool_search`: *"Search for available tools and skills by name or description. Use 'search' to find
  tools or skills matching a query, or 'select' to load a specific tool by exact name. Use the skill
  tool to load a skill's instructions."* 194 tokens.

**Which runs had them.**

- The mini's `tools.deferred_loading`: `enabled: auto`, and `always_loaded` is `bash, task_create,
  read_file, write_file, edit_file, grep, glob, tool_search, vault_search, vault_read, web_search,
  web_fetch, memory`. **`tool_search` is in that list; `skill` is not.**
- Deferral resolves ON for every local provider. The shipped default (`SHIPPED_ALWAYS_LOADED`) has
  no `skill` either.
- The rows cover 2026-07-31 → 2026-09-24 (`advertised.py`):

  | runs (main registry) | `skill` advertised | `tool_search` advertised |
  |---|---:|---:|
  | local llama.cpp / ollama, 833 | 11 (Qwen3.6, 2026-07-31 → 08-01, deferral off) | 808 (25 older/profile sets unknown) |
  | cloud, 140 | 140 | 140 |
  | **all 973** | **151 (15.5%)** | 948 |

  The 1,406 eval and coding runs used 6- and 7-tool registries with no `skill` tool.

**Is `tool_search` advertised on every Qwen 3.8 Max turn? Yes. `skill` is not.**

- Method: each of the 287 runs Qwen 3.8 Max served (round-0 `loop_round` rows, W33–W39) was paired
  with its own advertisement row.
- **`tool_search` was advertised on all 287.**
- **`skill` was advertised on 120 (42%).** The other 167 got the *local* model's deferred set (13 or
  15 tools). Until #462 (`cd1b798`, merged 2026-09-11) the advertisement was resolved before routing,
  so a session routed to the cloud model got the catalog of the local model it was routed away from.

| week | W33 | W35 | W36 | W37 | W38 | W39 |
|---|---:|---:|---:|---:|---:|---:|
| runs with `skill` / without | 0 / 114 | 47 / 7 | 0 / 5 | 19 / 33 | 39 / 8 | 15 / 0 |

On the 120 runs where Qwen 3.8 Max did have `skill`, it called it 0 times.

**Cost of listing every skill's name + description instead of a count.** Lines are in the prompt's
own `- **name**: description` format, under a one-line header.

| what is listed | skills | Qwen3 tokens | estimator | descriptions cut to 120 chars |
|---|---:|---:|---:|---:|
| everything served | 134 | **7,193** | 8,637 | 4,146 |
| builtin + auto only | 23 | 716 | 850 | 687 |
| user directory only | 111 | 6,492 | 7,806 | 3,474 |

One line costs a median of 49 tokens (p90 80, max 204). Against the local `effective_limit` of 72,000,
the full list is 10% of the window, and 13% of the headroom before compaction triggers at 0.75.

## 4. Missed opportunities: requests an existing skill clearly matched, where nothing was loaded

**Method.**

- **Turns.** Every human turn in LCM on a user surface: 865. That is all of them, not a sample.
- **Candidates.** Each turn is embedded and compared with every skill (name + description) that
  existed at that moment: born before the turn, and not yet archived (median 132 per turn).
- **Loaded.** A turn counts as having loaded a skill when a `skill` tool_use follows it in the same
  session before the next human turn.

**Calibration.** The clearest matches that exist are each auto skill against the request it was
generated from: the `trigger_task` in its `skill_created` signal. There are 52 such pairs.

| | p10 | p25 | median | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| cosine, skill vs its own request | 0.504 | 0.586 | 0.680 | 0.743 | 0.829 |
| cosine, best *other* skill for that request | 0.580 | 0.618 | 0.664 | 0.700 | 0.745 |

- The skill's own request ranks it first only 28 times of 52.
- The share of pairs that clear each threshold:

  | threshold | 0.70 | 0.72 | 0.75 | 0.78 | 0.80 |
  |---|---:|---:|---:|---:|---:|
  | own skill | 40% | 38% | 23% | 19% | 13% |
  | best other skill | 25% | 21% | 8% | 4% | 4% |

bge-small on raw request text is a weak separator here. A match threshold needs to be high to mean
anything.

**Threshold: cosine ≥ 0.75** (bge-small-en-v1.5, CLS, L2-normalised).

- Nearest-other-skill false matches drop to 8% there.
- Hand-labelling agreed. A first stratified sample found 0 clear matches among 6 in [0.70, 0.75).
  Labelling every turn at ≥ 0.75 found 5 clear of 12 in [0.75, 0.80), and 2 of 3 at ≥ 0.80.

**Result.**

| top-1 cosine | turns | clear match by hand | loaded a skill |
|---|---:|---:|---:|
| ≥ 0.80 | 3 | 2 | 0 |
| [0.75, 0.80) | 12 | 5 | 0 |
| **≥ 0.75, total** | **15 (1.7%)** | **7** (3 auto, 3 user-directory, 1 builtin core) | **0** |
| [0.70, 0.75) | 105 | 2 of 14 sampled, so about 15 (wide interval, about 4–42) | 0 |
| < 0.70 | 745 | not separable with this encoder | 0 |

- **Headline: about 7 verified and about 22 estimated turns (0.8–2.5%) had a skill that clearly fit.
  None of them loaded one.**
- What the other 8 of 15 looked like: topic-only. They were status questions or opinions about a
  subject that a procedure skill happens to share.
- Where those turns fall:
  - Surfaces: telegram 10, desktop 2, beacon 1, no-prefix 2.
  - Advertisement: 5 of the 15 ran *without* the `skill` tool advertised. 1 ran with it, and that
    one was a topic-only match. The other 9 have no advertisement row: 7 predate the telemetry and 2
    were not found.
  - The one clear match with a core skill (named in the prompt) was on a run without the tool.
- **The recall caveat.** On the calibration pairs only 23% of true matches reach 0.75. If the turns
  behave the same way, the real count could be up to 2–3× higher (about 6% of turns). Below 0.70 this
  encoder cannot tell a match from noise without hand-reading hundreds of turns.
- **What it does not count.** The matcher sees only the human message. A follow-up such as "yes, do
  that" carries its task in context, so those turns are undercounted.

## 5. Skill quality (all 57 auto skills ever written, from name + description only)

| class | all | live (20) | archived (37) |
|---|---:|---:|---:|
| reusable procedure (encodes something environment-specific or non-obvious) | 13 | 7 | 6 |
| one-off recipe (tied to one entity, job or incident) | 16 | 5 | 11 |
| duplicate: generic work the model already does unprompted | 21 | 7 | 14 |
| duplicate: of another skill in the catalog | 7 | 1 | 6 |

- Two archived skills have a malformed description: the literal text `name: …`, from a frontmatter
  parse.
- One name was written three times.
- SkillCreator made 119 generation calls: 68 full files, 45 SKIP-sized answers and 6 empty ones. They
  produced 52 `skill_created` events.

## 6. Knock-on effects: everything that depends on detecting skill use

Nothing in Prometheus counts skill use. Every consumer substitutes something else:

- **SkillRefiner** (`learning/skill_refiner.py`, `maybe_refine_recent`, a post-task hook since #16,
  2026-05-25).
  - What it does: after *every* task with ≥ 3 tool calls, it takes the **most recently modified** auto
    skill and asks the model to refine it against that task's trace. The prompt tells the model "A
    skill was used to guide a task." Nothing checks that.
  - How often: **208 calls** (206 OK, 2 timeouts) on the 4090's single slot, **29.7 min** of model
    time.
  - What came of them: 163 NO_CHANGE-sized answers, 5 empty, and 38 full rewrites. Only 2 writes
    left a backup, and there is 1 `skill_refined` event (2026-08-03). So most rewrites were stopped
    by its `---`-prefix or scanner check.
  - Since no auto skill was ever loaded, it has **never once refined on reuse**. The hooks run
    SkillCreator first, so after the 52 tasks that produced a new skill, the target was that new skill,
    checked against the trace it had just been written from. After every other task the target was
    whichever auto skill had changed last, unrelated to the task.
  - Its backups are written *into* `auto/` as `*.bak-<ts>.md`. The loader and the Curator both treat
    those as skills.
- **Curator** (`learning/curator.py`).
  - Staleness is file **mtime**, handed to its model as `last_used_days_ago`.
  - Over 24 runs (9 with an LLM review) it moved 35 skills from active to stale and archived **22**.
    It also proposed 15 consolidations, which are report-only.
  - 15 of the 22 pruning reasons cite staleness; the others cite low signal (12), duplication (10) or
    one-off (9).
  - An auto skill used daily but never edited would age out the same way. It "works" today only
    because use really is zero.
- **`skill_state.py`** says so itself: usage "stays 0 until a follow-up sprint wires a counter from
  the skill-loading tool." That was never wired. Its note that Hermes keeps usage in a module "NOT in
  the public Hermes tree" is out of date: Hermes now ships `tools/skill_usage.py` (§7).
- **GEPA** (`learning/gepa.py` `_find_candidate_skills`).
  - It filters `_meta.tool_name == "Skill"`, reads `input.skill`, and parses a `Reference parsed
    call:` marker. The tool is `skill`, the field is `name`, and the marker is in **0 of the 5,519**
    export lines.
  - Fixing all three would still find nothing. The only two golden `skill` calls (rowids 4,569–4,570,
    2026-07-10) predate every export file, whose earliest row is 2026-08-15 17:41 UTC. So the exports
    hold **0 `skill` lines** either way.
  - GEPA is off on the mini (`learning.gepa_enabled: false`). Switched on, it would report "no
    auto-skill matches in traces" every cycle.
- **UI.** The Telegram `/skills` list and Beacon's `/api/skills/list` show file mtime as "last used".
- **Not detection, but it undercuts any fix: microcompaction.**
  - On local tiers, `_microcompact_old_results` runs every round. It trims any tool result older than
    3 user-role messages to about 500 chars, and tool results count as user-role messages.
  - A skill body arrives as a tool result, so it would be cut to its first ~500 chars three tool
    rounds after loading. User-directory bodies have a median of 1,327 tokens.
  - The Agent Skills client guide says to exempt skill content from pruning.

## 7. How others surface skills

**Hermes Agent** (NousResearch/hermes-agent, MIT, read at `8afaab3`, 2026-09-26).

- **The prompt lists every skill.** `agent/prompt_builder.py` `build_skills_system_prompt` →
  `_render_skills_index` (L1273–1414) writes *every* visible skill's name + description into the
  system prompt, grouped by category, inside `<available_skills>`.
- **It tells the model to load them.** The directive asks the model to scan the list before replying
  and to load any skill that is even partly relevant ("you MUST load it with skill_view(name)"),
  erring toward loading, even for tasks it already knows how to do.
- **Nothing is dropped.** Out-of-context categories shrink to a names-only line. A code comment gives
  the reason: the model would not find agent-created skills again through `skills_list`. The block is
  cached (in-process LRU plus an on-disk snapshot) so the prompt stays byte-stable.
- **Loading is a tool, and it is counted.** `tools/skills_tool.py` (L684–752) exposes `skills_list`
  and `skill_view`. `skill_view`'s description explains what a skill is. Every successful view bumps
  `view_count` and `use_count` (`_skill_view_with_bump` → `tools/skill_usage.py` `bump_view`/`bump_use`,
  L492–510, in a `.usage.json` sidecar).
- **The curator ages on real activity.** `agent/curator.py` `apply_automatic_transitions`
  (L208–255) uses `last_used_at`/viewed/patched. A skill that was never used but is younger than the
  stale window is never archived, since zero use is not treated as evidence. Skills that cron jobs
  reference are protected.

**The Agent Skills pattern** (agentskills/agentskills, Apache-2.0, `69ef37e`, which anthropics/skills'
`spec/agent-skills-spec.md` now points to).

- **Progressive disclosure.** `docs/specification.mdx` defines three tiers:
  1. **metadata**: every skill's `name` + `description`, about 100 tokens each, loaded at startup;
  2. the `SKILL.md` body (under 5,000 tokens recommended), loaded when the skill is activated;
  3. bundled files, loaded as needed.
- **How clients implement it.** `docs/client-implementation/adding-skills-support.mdx` gives the
  mechanics:
  - a tier-1 catalog at 50–100 tokens per skill, in the system prompt or embedded in the activation
    tool's description;
  - a short behavioural instruction telling the model to call the activation tool when a task
    matches a skill's description;
  - **model-driven** activation rather than harness-side keyword triggers;
  - filtered skills hidden from the catalog entirely;
  - skill content exempted from context compaction, and repeat activations deduplicated.
- **Prometheus's current design inverts tier 1.** The model gets a count and a search tool instead of
  the catalog.

## 8. Causes, ranked

| # | cause | evidence |
|---|---|---|
| 1 | **The catalog is invisible.** The prompt carries a count and a trigger for when the model is "unsure how to approach" a task. A capable model is rarely unsure, so it never learns a skill exists. | Full catalog advertised on 151 runs, `skill` calls 0. `tool_search` queries mentioning skills: 0 of 329 recorded. §3: no non-core skill has ever been named in the prompt. |
| 2 | **The tool to load a skill is missing from most runs.** | `skill` is not in `always_loaded`. Advertised on 151 of 973 main-registry runs, on 11 of 833 local runs, and on 120 of 287 Qwen 3.8 Max runs (pre-#462 ordering). 5 of the 6 flagged missed turns whose run is known lacked it. |
| 3 | **Search → load barely works.** The scorer surfaces skills by accident and points at a tool that often isn't there. | All 3 loads followed a search (July); 299 searches since, 0 loads. 20 of 25 user searches that listed a skill did so by edit distance alone. Ties go to tools. |
| 4 | **There's little to match, and auto skills add little.** | A clear match on about 0.8–2.5% of turns (§4). 44 of 57 auto skills are generic, duplicates or one-offs; 7 live skills are reusable. |
| 5 | **No usage signal, so nothing self-corrects**, and a loaded body wouldn't survive local microcompaction. | §6: the refiner, Curator, GEPA and UI all run on mtime, a newest-file guess or broken filters. |

Causes 1 and 2 are what stop a load. Cause 4 sets the ceiling on what fixing them can win: with
today's catalog, even a perfect mechanism helps about 1 turn in 40–125.

## 9. Fix options

All token costs are Qwen3 tokens per request, measured on the mini's catalog and turns.

**C. Fix the plumbing (required under A and B too).** Recommended as step 1.

- Add `skill` to `always_loaded` and `SHIPPED_ALWAYS_LOADED`.
- Reword the prompt line from "unsure how to approach" to "check before starting a procedure".
- Stop the scorer listing edit-distance-only matches, and stop tools winning every tie.
- Count loads in `SkillTool.execute` into `skill_state`, and key the Curator, the UI's "last used" and
  GEPA on that count.
- Gate SkillRefiner on a `skill` call actually being in the trace. That alone ends the 208 wasted
  4090 calls.
- Exempt `skill` results from microcompaction, and reload the tool_search skill registry after
  SkillCreator writes.

*Cost:* **+90 tokens** on deferred runs (the schema); the prompt line stays about the same size.

*Risk:* low. On its own it probably won't change behaviour: runs that already had `skill` advertised
never used it. What it buys is the counter, and without the counter no fix can be measured.

**B. Relevance-picked skills per turn.** Recommended as step 2.

- What it does: embed each human turn and match it against skill name + description embeddings.
  Surface the top 1–2 only when the match is confident, and nothing otherwise, as a short block
  appended to the user turn. Appending it there keeps the system-prompt prefix, and its KV cache,
  stable.
- **This is a natural Instinct decision.** "Show a skill or abstain" is exactly Instinct's
  pick-or-abstain shape: a calibrated choice with an explicit abstain, trained and gated like its
  other decision points.

*Cost, measured on the 865 turns:*

| T | fires on | average tokens per turn | tokens when it fires |
|---:|---|---:|---:|
| 0.72 | 54 turns (6.2%) | 4.6 | 74 |
| **0.75** | **15 (1.7%)** | **1.3** | **76** |
| 0.80 | 3 (0.3%) | 0.2 | 47 |

On top of that come C's 90 tokens. Encoding adds p50 4.7 ms, p95 22 ms on the mini's CPU.

*Risk:* precision. At 0.75, 7 of 15 picks were right, and the 8 wrong ones would inject off-topic
instructions. Recall is also low with raw-text bge-small: 23% on known pairs. It needs calibration,
context-aware inputs (request plus recent turns plus the skill's "when to use" line), and a
precision gate (for example ≥ 80% on a labelled set) before it ships. The threshold also needs
upkeep as skills come and go.

**A. The full Agent Skills catalog in the prompt** (the Hermes / agentskills.io pattern).

- List every served skill's name + description with a "when a task matches, call `skill`"
  instruction.

*Cost:* **+7,193 tokens** per request for 134 skills, 10% of the local 72k window. Other variants:
4,146 with descriptions cut to 120 chars; 716 for builtin + auto only. The block is stable within a
daemon lifetime, so it caches.

*Risk:* context pressure on the local model, which moves compaction earlier. It puts about 130
distractors in front of a 27B model's tool choice. A Hermes-style "MUST load when even partially
relevant" would also fire on topic-only matches, which were 8 of 15 at 0.75, and each false load
costs a round plus about 1.3k body tokens. The builtin + auto variant is cheap, but 13 of those 20
auto skills are generic or one-offs.

**Recommendation.** Do C now: it is cheap, low-risk, and the only way anyone learns whether skills
help. Then run B as an Instinct pick-or-abstain experiment, gated on precision. Don't list 134 skills
in a 72k local window.

Separately, SkillCreator needs attention. It writes about 0.4 skills per eligible task, and 44 of 57
are generic, duplicates or one-offs. On today's numbers, the catalog, not the surfacing, is what limits
the upside.

## Reproduce

The scripts are in `docs/audits/skill-usage/`. They print aggregates only. `--chat-only` modes print
examples to the terminal and must never be committed. Run on the mini, with the deployed tree on the
path. `$M/bge` holds the pinned bge-small files (`onnx/model.onnx`, `tokenizer.json`) and `$M/qwen3`
the Qwen3 `tokenizer.json`:

```
S=~/.prometheus/db-snapshots/20260926T065501Z
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=~/prometheus-deploy/src PROMETHEUS_CONFIG_DIR=~/.prometheus
python3 catalog.py $S --tokenizer $M/qwen3/tokenizer.json   # §2
python3 usage.py $S                                          # §1
python3 advertised.py $S                                     # §3 (who had skill / tool_search)
python3 seen_by_model.py $S $M/qwen3/tokenizer.json          # §3 (rendered section, schemas, token costs)
python3 tool_search_replay.py $S tools.json                  # §1 (did a result list a skill?)
python3 missed.py $S $M/bge --threshold 0.75                 # §4 and option B's cost
```

`tools.json` is public tool metadata. It comes from `dump_tool_catalog.py`, run on a development
checkout with an empty config directory.
