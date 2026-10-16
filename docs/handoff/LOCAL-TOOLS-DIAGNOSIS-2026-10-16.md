# Local tool discovery after 0.9.12: diagnosis and handoff (2026-10-16)

Handoff to an audit session. Diagnosis is complete; the approved fixes 1–4 were
started and are **not committed**; the live config workaround **was applied**.
Line references are to `e3c1787` (v0.9.12) unless stated.

## 0. State at handoff

| What | State |
|---|---|
| Live daemon (`prometheus.service`) | **Changed.** Config workaround applied on Will's go, restarted 2026-10-16 12:27:34Z, verified clean (§0.1). Nothing else on live was touched. |
| Fix branch | `fix/local-tool-reach-0913`, worktree `.claude/worktrees/local-tool-reach`, cut from `origin/main` `e3c1787`. **No commits, not pushed, no PR.** Uncommitted work in 7 files (§0.2, full diff in Appendix B). |
| This document | Branch `docs/local-tools-diagnosis-2026-10-16` (this file only). |
| Scratch daemons and proxies | All stopped. Their worktrees and scratch HOMEs remain (§0.3). |
| Archaeology agent (Will's second message) | Started, then stopped on Will's word. No results were used. The questions are open (§8). |

### 0.1 The live change

- Backup: `~/prometheus-deploy/config/prometheus.yaml.bak-20261016T122703Z-pre-agent-cap4096` (mode 600, byte-identical to the pre-edit file, same directory and naming as the earlier backups).
- Edit, 4 lines, applied as text so the 650 comment lines stay in place (651 after, one new comment):
  - `context.model_overrides["<served path of the Qwen3.8-27B GGUF>"].max_output_tokens: 4096`, with a one-line comment.
  - `agent` appended to the pinned `tools.deferred_loading.always_loaded`.
- Checked before the restart:
  - A dict diff (`yaml.safe_load` of backup vs new) shows exactly those two changes.
  - The deployed resolvers give `OutputCaps.resolve(...) == (4096, "model_override")`, where it was `(8192, "local")`, and `resolve_always_loaded` now includes `agent`.
  - Only `context/compactor.py`, `context/budget.py` and `providers/output_caps.py` read `model_overrides`. The first two ignore an entry without `effective_limit` or a threshold key, so the window stays the detected 32768.
- Procedure: OAra maintenance window armed 12:27:28Z, then restart at 12:27:34Z (pid 1709901).
- Boot journal:
  - 0 ERROR lines, 0 tracebacks, and the same 9 WARNINGs as the 0.9.12 deploy boot.
  - Confinement ACTIVE.
  - `Tool advertisement baseline: 17 always_loaded of 54 registered` (was 16), and the divergence line lists `agent`.
  - Compactor `limit=32768 [detected] … threshold=26214`, unchanged.
  - No `LOOP TOOL-METADATA DIVERGENCE` line.
- After the restart: config_audit ticked clean at 12:30:47Z, the watcher printed `OK — 5 component(s) fresh`, and `/api/cron` matched `cron_jobs.json` 4/4. The window was cleared at 12:32:11Z with `findings_pending 0`.
- The dev copy `~/Prometheus/config/prometheus.yaml` was **not** updated. The repo-paths note says the two are kept in step; that call is open.
- **Revert** (arm the maintenance window first, per the standing procedure):
  `cp -p ~/prometheus-deploy/config/prometheus.yaml.bak-20261016T122703Z-pre-agent-cap4096 ~/prometheus-deploy/config/prometheus.yaml && systemctl --user restart prometheus.service`

### 0.2 The uncommitted fix branch

| Fix | Status on the branch |
|---|---|
| 1. Local output cap back to 4096, quarter-of-window ceiling kept | **Written.** `providers/output_caps.py` (constant and docstrings), the template comment, `tests/test_output_caps.py` (repointed so each test still discriminates, plus a new test pinning the 32k-window arithmetic). `tests/test_output_caps.py`: 37 passed. |
| 2. `agent` always loaded by default | **Written.** A new set appended to `ALWAYS_LOADED_DEFAULTS_SHIPPED`, the template list and comment, `docs/reference/config-keys.md` regenerated with `scripts/gen_reference.py`, history data in `tests/test_always_loaded_follows_the_default.py`, new `tests/test_agent_tool_is_advertised.py`. That file plus the always_loaded, absence, setup, skill, generated-reference and config-defaults suites: 74 passed, 1 skipped. |
| 3. Machine-written endings streamed to Beacon | **Designed, not written** (§7). |
| 4. tool_search near-miss resolution and a 5-name miss list | **Not started.** |
| Not run on the branch | ruff, the mypy gate, the full suite, parity, bench, the real-model check of the branch itself. |

### 0.3 Scratch artifacts still on disk

- Session scratchpad `…/scratchpad/diag/`:
  - Worktrees `wt-0910` (ac8b592) and `wt-0912` (e3c1787), each with its own `.venv`.
  - Scratch HOMEs `h0910` and `h0912`.
  - Configs `cfg-*.yaml` (these contain the real model base URL; never commit them) and their `.token` files.
  - Captures `capture-*.jsonl`, frame logs `frames-*.jsonl`, daemon and proxy logs.
  - The scratchpad is session-local and may disappear. The scripts are reproduced in Appendix A.
- `.claude/worktrees/diag-0910` and `diag-0912` are unused detached worktrees with only a venv; safe to remove.
- `.claude/worktrees/local-tool-reach` is the fix branch above.
- Unrelated: another Prometheus daemon was running from `~/film-hero-demo/prometheus.yaml` (up about 9 hours at 00:10Z). It was not started by this session and was left alone.

## 1. Summary

1. **"Never called `agent`" is not a 0.9.12 regression.**
   - v0.9.10 fails the same request the same way.
   - On the local stack (llama.cpp, Qwen3.8-27B, adapter tier `light`, native tool calling), a tool that isn't in the request's `tools` is never called.
   - `tool_search` select returns the schema as a tool result, but the run's tool list is frozen at run start.
   - Since 2026-08-01, local runs followed **2 of 52** selects with a call of the selected tool.
2. **0.9.12 has a separate regression that affects every local turn.**
   - `LOCAL_OUTPUT_CAP = 8192` (#806) feeds #773's projected-overflow rule.
   - On the 32768 window the compaction line falls from 26,214 to about 22.3k tokens.
   - With 18,634 tokens of fixed overhead, the **first** turn is summarised mid-flight after 2–3 tool results.
3. **The turn wasn't silent at the daemon, but it reached Beacon empty.**
   - The repeat guard ended it with a "Halted: no progress…" message, which was stored.
   - The WS bridge forwards only streamed deltas, and that ending is not streamed.
   - Nine other machine-written endings share this.

## 2. The incident (live, v0.9.12)

A Beacon session titled "Sub-agent file listing attempt", 2026-10-16 03:45:48Z–03:46:01Z, served by the local Qwen3.8-27B (journal `ModelRouter: … (primary)`). Request: "use a sub-agent to list the files in your workspace".

- **Journal:**
  - Three `tool_search` audits.
  - `ContextCompactor: 5 msgs → 1 summary, 2 tool result(s) trimmed … tokens 22731 → 20068 (ceiling 22556, target 15789, trigger projected)`.
  - `Microcompacted 1 old tool results at turn 4 (-7012 chars)`.
  - `Unproductive repeat: tool_search called 3× with identical arguments, identical result each time — halting turn.`
- **LCM** (`~/.prometheus/data/lcm.db`, read-only), 12 rows:
  - Turn 1: `tool_search {"query":"agent subagent spawn","action":"search"}`.
  - Turns 3, 7, 9: `select agent` (ok).
  - Turn 5: `select "agent explorer"`. Its error listed all **192** names (55 tools + 137 skills): 5,569 chars, **1,701 tokens**.
  - Turn 11: assistant `Halted: no progress. \`tool_search\` was called 3 times…`.
- **Telemetry** (`~/.prometheus/telemetry.db`, table `turns`): stop reason `unproductive_repeat`, outcome `forced_stop`, 5 tool calls, 101,992 input tokens over 5 rounds, 194 output tokens.
- Beacon's meter showed 21k of 33k after the one message: 18.6k fixed (§5) plus the tool_search results.
- The 0.9.12 deploy note lists "the `agent` tool from a Beacon chat" as a manual smoke still to try. This incident is that smoke.

## 3. Was it a 0.9.12 regression? A/B on scratch daemons

**Method:**
- Two scratch daemons, v0.9.10 `ac8b592` and v0.9.12 `e3c1787`, each from its own worktree **outside** `~/Prometheus`. Otherwise PROMETHEUS.md stacks twice in the project-instructions walk.
- One config for both, derived from the live one:
  - Prompt-shaping keys kept: tools, context, bootstrap, anatomy, adapter, MCP.
  - Web on loopback spare ports with a scratch token.
  - Gateways, cron, sentinel, mDNS, pairing and push off; no fallback.
  - Workspace and HOME inside the scratch dir.
- Copies of SOUL, AGENTS, ANATOMY, MEMORY and USER.md and `skills/`, plus sparse replicas (same names, sizes and mtimes) of the 41 saved files.
- The model was reached through a logging proxy that recorded every llama.cpp request and reply.
- The request was sent over the WebSocket the way Beacon sends it.

| | v0.9.10 | v0.9.12 |
|---|---|---|
| Tools in every request | 16, no `agent` | the same 16 |
| Round-0 prompt (server count) | 18,634 | 18,634 |
| `max_tokens` | 4096 | 8192 |
| Mid-turn compaction | none | 3 passes: `22723 → 20935 (ceiling 22285, projected)`, then `25379 → 22083 (ceiling 21981)` **did not converge**, then a trim `22761 → 19388` |
| `agent` called | no; select ×3, then the repeat guard | no; select, `bash ls` of the daemon cwd, search, select ×2, then the repeat guard |
| What the model saw | its own history | "[Compacted summary of turns 1–5] … [The user's current request, verbatim:] …", then trimmed results; it said "my previous search was interrupted" |
| Halt text in any WS frame | 0 frames | 0 frames |

v0.9.10's model wrote "I have the schema. The tool isn't in my directly callable function list, so I'll invoke it now (it was registered via tool_search):" and then called `tool_search` again.

Follow-up runs on v0.9.12:

- **E1, cap 4096** (`context.model_overrides.<gguf>.max_output_tokens: 4096`):
  - Prompts 18,634 → 21,771 → 22,160 → **23,496** → 20,781 → 20,897. 23,496 is past the 8192-cap line, and no ContextCompactor pass ran (only microcompaction).
  - Still no `agent`: the model ran `bash` itself, then told the user it had used "a fresh `explorer` subagent", which is false.
- **E2, cap 4096 and `agent` in always_loaded:**
  - `agent` was called on round 1 with no tool_search.
  - An `explorer` subagent (tools read_file, glob, grep) ran, and the reply streamed.
  - 106 s, about 94 s of it in the subagent.
- **Direct probes** through the proxy (thinking off, max_tokens 300). The system prompt gave the `agent` schema; the user said "Call the agent function now…":

| Probe | Tools offered | Result |
|---|---|---|
| A | `[tool_search]`, with "Do not call tool_search." | `tool_search {"query":"agent","action":"select"}` |
| B | `[tool_search, agent]` | `agent {"description":"list files","prompt":"List the files in the workspace"}` |
| C | `[memory]` | No call. Text: "…the `agent` tool mentioned in my system prompt doesn't appear in my actual function definitions either." |

- **Server:**
  - llama.cpp build `b1-9d57ce456`, n_ctx 32768, **total_slots 1** (shared with live).
  - `chat_template_caps.supports_tools: true`.
  - The template uses the Qwen3-Coder XML call format (`<tool_call><function=…><parameter=…>`).

## 4. Answers to the five questions

**Q1. After a select, is the tool added to the next request?**

No.
- Code:
  - [`dynamic_tools.py:241`](../../src/prometheus/context/dynamic_tools.py) `schemas_for_run` is computed once per run; [`:204`](../../src/prometheus/context/dynamic_tools.py) `resolve_deferred` documents the frozen set.
  - [`tool_search.py:159`](../../src/prometheus/tools/tool_search.py) `_handle_select` only returns `to_api_schema()` as text.
  - The NOTE at [`dynamic_tools.py:295`](../../src/prometheus/context/dynamic_tools.py) says deferred tools stay reachable because "the model either calls them by name anyway … the 'lucky guess' path … or discovers them via the tool_search TOOL RESULT".
  - [`tests/support/advertisement.py:32`](../../tests/support/advertisement.py) asserts "DEFERRED IS NOT THE SAME AS UNREACHABLE".
  - **Neither premise holds for this stack** (probes A and C; telemetry below).
- Captures: the same 16 tools in every round after a select, on both versions.
- Telemetry, live, 2026-08-01 → 2026-10-15, read-only:
  - 39,610 `tool_calls` rows, 11,385 of them local (`.gguf`).
  - **52 local selects; 2 followed by a call** of the selected tool in the same session, both `sessions_list`.
  - Selected: `sessions_list` 24, `cron_create` 9 (called 0 times), `lcm_grep` 5, `agent` 3, "select sessions_list" 2, `message` 2.
  - Once each: `sessions`, `mcp__context7__resolve_library_id`, "bash tool", `lcm`, `lsp`, "select cron_create", "agent explorer".
- Grammar or model? **Not settled.** Probe C shows the model itself declines an unlisted function. Whether llama.cpp's tool-call grammar would also block it was not tested (§8).

**Q2. Why did "agent explorer" fail, and should near-misses resolve?**
- Select is exact-name ([`tool_search.py:159`](../../src/prometheus/tools/tool_search.py)).
- The agent registry in the system prompt lists `explorer` as a `subagent_type`, and the model joined the two words.
- A miss returns every tool and skill name ([`tool_search.py:181-186`](../../src/prometheus/tools/tool_search.py)): 1,701 tokens on this box.
- Near-misses seen in telemetry: "agent explorer", "select sessions_list" ×2, "select cron_create", "bash tool". Each has one real tool name as a whole word, so they can resolve unambiguously.
- "sessions" and "lcm" are ambiguous. Return the 5 closest names, not 192.
- On its own this saves rounds and tokens, but it doesn't make a deferred tool callable.

**Q3. Did the turn end silently, and why?**
- The repeat guard ended it ([`agent_loop.py:3439`](../../src/prometheus/engine/agent_loop.py); stop `REPEAT_DETECTOR`). It wasn't the breaker or an empty reply.
- The text was stored (LCM turn 11) but never sent while the turn was live:
  - [`ws_server.py:1746`](../../src/prometheus/web/ws_server.py) only counts the round on `AssistantTurnComplete`.
  - Text reaches clients only as `chat_delta` from `AssistantTextDelta` ([`:1776`](../../src/prometheus/web/ws_server.py)).
  - Beacon re-keys the empty bubble on `chat_done.row_id` without a re-read (`beacon-desktop` `src/shared/message-order.ts` `finalizeRow`).
- The same applies to these endings in `agent_loop.py`, which yield only `AssistantTurnComplete`:
  - 2458 context pre-flight
  - 2628 context overflow
  - 2853, 3027, 3351, 3359 circuit breaker
  - 2891 empty response twice
  - 3088 boundary escape
  - 3108 iteration limit
  - 3439 repeat
  - 3542 divergence halt
- Only the degenerate-output ending (2697–2704) also streams its text, with the comment "Streamed too: the web bridge shows deltas, not the final message".
- Other surfaces:
  - **Telegram** is unaffected: `run_async` uses the final message text ([`agent_loop.py:6288`](../../src/prometheus/engine/agent_loop.py)).
  - **The OpenAI-compatible endpoint** already emits the unstreamed remainder ([`openai_api.py:227-233`](../../src/prometheus/web/openai_api.py)).
  - **The CLI** (`__main__.py:1090`, `:1358`; `cli/worker.py:62`) prints the final text only if nothing streamed in the whole response, so it misses a halt that follows earlier text.

**Q4. The fixed overhead:** §5.

**Q5. Which tools must be reachable without searching:** §6.

## 5. The 18,634-token round-0 request, by piece

Counted with the 4090's own `/tokenize`; the pieces sum to within 3 tokens of the server's `prompt_tokens`. System prompt 48,527 chars = 13,488 tokens.

| Piece | Tokens |
|---|---|
| Tool schemas, 16 offered tools (Context7 MCP pair alone: 1,131) | 4,762 |
| `# Memory`: MEMORY.md (10.3k chars) + USER.md + preamble | 3,232 |
| Project instructions: PROMETHEUS.md of the daemon cwd (`~/prometheus-deploy`), in every Beacon chat without a workspace | 2,828 |
| The tool list again as prose, plus "respond ONLY with a JSON object" (adapter `QwenFormatter`, [`formatter.py:137`](../../src/prometheus/adapter/formatter.py), applied on every tier at [`adapter/__init__.py:146`](../../src/prometheus/adapter/__init__.py)); contradicts the template's XML call format | 1,440 |
| Saved-files listing (41 files) | 1,194 |
| Base prompt (System, Doing tasks, Using your tools, Executing actions, Tone, Environment, Reasoning) | 1,137 |
| SOUL.md | 1,012 |
| ANATOMY.md (infrastructure) | 1,004 |
| Documents library | 806 |
| Agent registry (AGENTS.md) | 610 |
| Skills list | 222 |
| Chat-template framing | 373 |
| The user's message | 11 |

Per offered tool:

| Tool | Tokens |
|---|---|
| task_create | 676 |
| mcp__context7__resolve_library_id | 642 |
| mcp__context7__query_docs | 489 |
| grep | 347 |
| vault_read | 345 |
| memory | 308 |
| write_file | 261 |
| bash | 231 |
| edit_file | 218 |
| web_search | 198 |
| read_file | 188 |
| web_fetch | 187 |
| glob | 186 |
| tool_search | 184 |
| vault_search | 155 |
| skill | 132 |

`agent` would add about 220 tokens as a native schema, plus 27 in the prose block.

## 6. Which tools need to be reachable on local

Local calls since 2026-08-01, for the tools that are always loaded:

| Tool | Local calls |
|---|---|
| bash | 3,709 |
| read_file | 1,077 |
| write_file | 516 |
| edit_file | 378 |
| grep | 341 |
| glob | 317 |
| tool_search | 325 |
| web_fetch | 65 |
| task_create | 49 |
| web_search | 48 |
| vault_search | 36 |
| vault_read | 11 |
| memory | 9 |
| mcp__context7__resolve_library_id | 4 |
| skill | 1 |
| mcp__context7__query_docs | 0 |

- Cloud calls of tools that are deferred on local (top of the list): `task_output` 160, `lcm_grep` 129, `task_get` 30, `agent` 17, `lcm_expand_query` 17, `task_stop` 14, `lcm_describe` 12, `task_list` 11, `todo_write` 11.
- `task_create` is offered on local but `task_output`, `task_get` and `task_list` are not.
- Keep: bash, read/write/edit, grep, glob, web_search, web_fetch. Add `agent`.
- While "deferred" means "unreachable" (until fix 5), anything that matters must be always loaded.
- Low-use tools that cost tokens on every request (Will's config choices): the Context7 pair (1,131 tokens, 4 calls), memory, skill, vault_read.
- **Unexplained, for the audit:** a few local calls of tools that were deferred at the time.
  - Telemetry shows `lcm_grep` 28, `lcm_expand_query` 10 and `lcm_expand` 10 (mostly one Beacon session, 2026-08-26 → 09-14), plus `task_list`, `sessions_list` and `mcp_status` 2 each.
  - Possibly runs with deferral toggled off, or smoke runs. It may also be the "lucky guess" path having worked at the time. This is the most direct lead for §8.

## 7. The 0.9.12 compaction regression, mechanism

- [`output_caps.py:41`](../../src/prometheus/providers/output_caps.py) `LOCAL_OUTPUT_CAP = 8192` (#806), with [`:47`](../../src/prometheus/providers/output_caps.py) `LOCAL_ROOM_FRACTION = 0.25`.
- [`agent_loop.py:2303`](../../src/prometheus/engine/agent_loop.py) hands the compactor `_output_cap(...)` as `max_tokens`.
- [`compactor.py:1175-1180`](../../src/prometheus/context/compactor.py): a pass fires when `tokens_before + max_tokens + largest_recent_result > window`, with `ceiling = min(threshold, window − max_tokens − largest)`.
- On 32768 with a ~2k tool result:

| Cap | Ceiling | Conversation that fits over the 18.6k overhead |
|---|---|---|
| 8192 | about 22.3k (22,556 live, 22,285 scratch) | about 3.7k tokens |
| 4096 | 26,214 (the 80% line) | about 7.6k tokens, what #805 intended |

- #805's calibration puts the estimate in server tokens. That's correct behavior, but it removed the slack that had hidden this.

## 8. Will's archaeology questions (open; not answered here)

Will believes the April tool-calling sprint gave deferred loading two safety nets that worked for months:
- (a) select returns the full schema, so the tool is callable that run;
- (b) a `lucky_guess` dispatch path executes any registered tool even if it wasn't offered, logged as `lucky_guess`.

His questions:
1. When did each net stop working? Is the frozen-at-run-start catalog the cause, and which commit or PR?
2. Does `lucky_guess` still exist? Does llama.cpp's grammar let an unoffered call through?
3. Estimate fix 5 as a restore scoped to local providers: lines changed, risk, and the re-prefill cost when the tool list changes mid-run.

Known so far, unverified against history:
- [`dynamic_tools.py:295`](../../src/prometheus/context/dynamic_tools.py) says `on_demand()` was deleted in `feat/deferred-tools-tier-aware` with "zero production call sites", and cites "the #120 prefix-mutation bug class".
- Whether dispatch still runs an unoffered registered tool was not re-checked.
- §6 lists the local deferred-tool calls to explain.

Hypothesis to test, not established: the nets may have depended on **text-mode** tool calling (adapter tier `full`).
- There, Prometheus renders tools into the prompt and parses calls from text, and the boot GBNF grammar (`model.grammar_enforcement: true` on live) may cover every registered tool.
- On tier `light` with `--jinja`, tools travel natively, llama.cpp renders them, and the provider drops its own grammar ([`llama_cpp.py:649`](../../src/prometheus/providers/llama_cpp.py), [`:680`](../../src/prometheus/providers/llama_cpp.py)).
- When the 27B moved to tier `light` is a history question: `config/model_registry.yaml`, `adapter/__init__.py`.

Planned tests that were not run:
- **Grammar vs model:** render the probe-A chat with llama.cpp `/apply-template`, send that raw prompt to `/completion` with no grammar, and compare with `/v1/chat/completions`.
- **Re-prefill cost:** replay a captured round twice (same tools, then with one tool appended) and read llama.cpp `timings.cache_n`, `prompt_n` and `prompt_ms`.
  - Check in the template whether `# Tools` follows the system content. If it does, an appended tool keeps the ~13.5k-token system prefix cached.
  - Reference: the cold 18,634-token round took 7.77 s, and a cached next round 1.5–2.4 s.

## 9. Proposed fixes, ranked

| # | Fix | Impact | Risk | Status |
|---|---|---|---|---|
| 1 | Local output cap back to 4096; keep the quarter-of-window ceiling | High: every local turn | Low: 0.9.10's local behavior; a reply cut at 4096 is told to split | Approved; written, uncommitted |
| 2 | `agent` always loaded by default (≈250 tokens) | High for subagent requests | Low; E2 proved it end to end | Approved; written, uncommitted. Will's pinned list needs `agent` too (done live). The one-shot CLI registers `agent` with no spawner (only `daemon.py:1428-1429` attaches one), as cloud CLI runs already do. |
| 3 | Stream machine-written endings to Beacon | Medium: no silent turns | Low | Approved; not written. Design below. |
| 4 | tool_search: resolve unambiguous near-misses; on a miss list the 5 closest names, not 192 | Low–medium | Low | Approved; not started |
| 5 | select makes the tool callable for the rest of the run or session, local providers only | High: every deferred tool | Medium: breaks the frozen-catalog rule; a partial prompt-cache miss per select; parity likely re-recorded | Deferred to after launch; issue to open. Estimate pending (§8). |
| 6 | Drop the QwenFormatter prose tool list and JSON-format line on tier `light` (−1,440 tokens, removes the JSON-vs-XML contradiction) | Medium | Medium: changes every local prompt; needs a bench | Deferred to after launch; issue to open |

**Fix 3 design**, chosen over a loop-level change:
- In `ws_server._run_agent_locked`, on `AssistantTurnComplete`, compute the round's streamed text `accumulated[round_start:]`.
- If the final message's text starts with it and is longer, broadcast the remainder as `chat_delta` and append it to `accumulated`, before moving `round_start`.
- This mirrors `openai_api.py:227-233` and covers all ten endings plus any future one.
- It shouldn't change parity goldens, which drive the loop directly.
- Tests to write:
  - A halt-only round yields a `chat_delta` before `chat_done`.
  - A streamed round is not doubled.
  - A rewritten final (not a prefix) emits nothing.
  - Far side: the real repeat guard through the WS bridge.

**Other observations:**
- "Your workspace" in Beacon resolves to the daemon cwd (`~/prometheus-deploy` live), not `security.workspace_root`.
- The single 4090 slot means every scratch probe evicts live's cached prefix.
- §8's planned tests run on the 4090's single slot, shared with live: keep them short.

## Appendix A — the scratch kit

Commands (`DIAG` = a scratch dir outside `~/Prometheus`):
- Worktrees: `git worktree add --detach $DIAG/wt-0910 ac8b592` (and `wt-0912 e3c1787`), then the house `uv sync` in each.
- Config: `wt-0912/.venv/bin/python mkcfg.py $DIAG/h0912 18605 18610 18690 $DIAG/cfg-0912.yaml`.
- Start: `run.sh 0912` starts the proxy and the daemon. `rund.sh 0912 <cfg> <log>` restarts only the daemon.
- Drive: `drive.py 18610 cfg-0912.yaml.token "beacon:<uuid>" "<message>" frames.jsonl`.
- Read the capture: `summarize.py capture-0912.jsonl`, `breakdown.py 18690 capture-0912.jsonl <seq>`.
- Stop by PID, never `pkill`.
- The configs this produces contain the real model base URL: keep them out of git.

Replace `<DIAG>` with your scratch dir in each file.

### proxy.py

```python
"""Logging reverse proxy between a scratch daemon and the 4090 llama.cpp.

Forwards every request unchanged and streams the response back. For
POST */chat/completions it writes one JSONL record per call: the request body
as sent, and a digest of the streamed reply (finish_reason, usage, tool calls,
content). The upstream address is read in-process from the live config and
is never printed or logged.

Usage: python proxy.py <listen_port> <log_path>
"""

from __future__ import annotations

import json
import sys
import time
from urllib.parse import urlparse

import httpx
import uvicorn
import yaml
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

_live = yaml.safe_load(open("/home/will/prometheus-deploy/config/prometheus.yaml"))
_u = urlparse(_live["model"]["base_url"])
UPSTREAM = f"{_u.scheme}://{_u.netloc}"
del _live

PORT = int(sys.argv[1])
LOG = sys.argv[2]
_seq = [0]
client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))


def _write(rec: dict) -> None:
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _digest_sse(raw: bytes) -> dict:
    """Fold an OpenAI-shape SSE stream into one summary."""
    content, reasoning = [], []
    calls: dict[int, dict] = {}
    finish = None
    usage = None
    timings = None
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        if chunk.get("timings"):
            timings = chunk["timings"]
        for ch in chunk.get("choices") or []:
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            d = ch.get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
    return {
        "finish_reason": finish,
        "usage": usage,
        "timings": timings,
        "content": "".join(content),
        "reasoning_chars": sum(len(r) for r in reasoning),
        "tool_calls": [calls[i] for i in sorted(calls)],
    }


async def proxy(request: Request) -> Response:
    path = request.url.path
    query = request.url.query
    url = UPSTREAM + path + (("?" + query) if query else "")
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    is_chat = request.method == "POST" and path.endswith("/chat/completions")
    if not is_chat:
        r = await client.request(request.method, url, content=body, headers=headers)
        if request.method == "POST" or path.endswith("/props"):
            _write({"t": time.time(), "kind": "other", "method": request.method, "path": path,
                    "status": r.status_code, "resp_len": len(r.content)})
        return Response(r.content, status_code=r.status_code,
                        headers={k: v for k, v in r.headers.items()
                                 if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")})
    _seq[0] += 1
    seq = _seq[0]
    try:
        req_json = json.loads(body)
    except Exception:
        req_json = {"_unparsed": body[:2000].decode("utf-8", "replace")}
    started = time.time()
    upstream = client.build_request("POST", url, content=body, headers=headers)
    resp = await client.send(upstream, stream=True)
    buf = bytearray()

    async def gen():
        try:
            async for chunk in resp.aiter_raw():
                buf.extend(chunk)
                yield chunk
        finally:
            await resp.aclose()
            digest = _digest_sse(bytes(buf)) if req_json.get("stream") else {"raw": bytes(buf)[:4000].decode("utf-8", "replace")}
            _write({"t": started, "dt": round(time.time() - started, 2), "kind": "chat", "seq": seq,
                    "status": resp.status_code, "request": req_json, "response": digest})

    return StreamingResponse(gen(), status_code=resp.status_code,
                             headers={k: v for k, v in resp.headers.items()
                                      if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")})


app = Starlette(routes=[Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
```

### mkcfg.py

```python
"""Build a scratch daemon config from the live one, isolated.

Usage: python mkcfg.py <scratch_home> <api_port> <ws_port> <proxy_port> <out_yaml> [key=value ...]

Keeps every prompt-shaping setting of the live config (tools, context,
bootstrap, anatomy, learning, adapter, MCP servers), and isolates the rest:
loopback-only web on spare ports with a scratch token, no gateways, no cron,
no sentinel, no mDNS, no pairing, no push, workspace inside the scratch home,
the model reached through the logging proxy, no fallback. Nothing printed but
the output path. Extra key=value pairs (dotted keys, YAML values) override.
"""

from __future__ import annotations

import copy
import random
import string
import sys
from urllib.parse import urlparse

import yaml

home, api, ws, pport, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
extra = sys.argv[6:]

live = yaml.safe_load(open("/home/will/prometheus-deploy/config/prometheus.yaml"))
cfg = copy.deepcopy(live)
del live

proxy = f"http://127.0.0.1:{pport}" + urlparse(cfg["model"]["base_url"]).path
token = "".join(random.choice(string.ascii_letters) for _ in range(40))

cfg["web"] = {
    "enabled": True, "api_port": api, "ws_port": ws,
    "tls_api_port": api + 1, "tls_ws_port": ws + 1,
    "bind": "127.0.0.1", "trusted_proxies": [], "api_token": token,
    "dashboard_port": api + 7000,
}
gw = cfg["gateway"]
gw["telegram_enabled"] = False
gw["cron_enabled"] = False
for k in ("telegram_token", "allowed_chat_ids", "briefing_chat_id", "proxy_url"):
    gw.pop(k, None)
gw["slack_enabled"] = False
cfg["sentinel"]["enabled"] = False
cfg["heartbeat"] = {"maintenance_db": ""}
cfg["pairing"] = {"requests_enabled": False}
cfg["discovery"] = {"mdns": False}
cfg["network"] = {"home_network": False, "allow_plaintext_lan": False}
cfg["push"] = {"enabled": False}
cfg["anatomy"]["scan_on_startup"] = False
cfg["anatomy"].pop("ssh_key", None)
cfg["learning"]["auto_skill_creation"] = False
cfg["learning"]["skill_refinement_enabled"] = False
cfg["trajectory_export"]["enabled"] = False
cfg["training"]["capture_enabled"] = False
cfg["training"]["cloud_golden_capture"] = False
cfg["printing_press"]["auto_update_library"] = False
cfg["evals"].pop("judge_base_url", None)
cfg["doctor"]["startup_check"] = False

sec = cfg["security"]
ws_dir = f"{home}/workspace"
sec["workspace_root"] = [ws_dir]
sec["bash_write_allow"] = [ws_dir]
sec.pop("grants", None)
sec.pop("allowed_commands", None)

cfg["model"]["base_url"] = proxy
cfg["model"].pop("fallback_url", None)
cfg["model"].pop("fallback_provider", None)
cfg["router"] = {"fallback": []}
cfg["backends"] = {"4090": {"provider": "llama_cpp", "base_url": proxy}}
cfg.pop("image_generation", None)
cfg.pop("whisper", None)

for item in extra:
    key, _, raw = item.partition("=")
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)

with open(out, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
with open(out + ".token", "w") as fh:
    fh.write(token)
print(out)
```

### run.sh

```bash
#!/usr/bin/env bash
# run.sh <0910|0912> : start the logging proxy and the scratch daemon for one version
set -u
D=<DIAG>
v=$1
case $v in 0910) P=18590;; 0912) P=18690;; esac
H=$D/h$v
WT=$D/wt-$v
nohup $WT/.venv/bin/python $D/proxy.py $P $D/capture-$v.jsonl > $D/proxy-$v.log 2>&1 &
echo $! > $D/proxy-$v.pid
cd $WT
nohup env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$H PROMETHEUS_CONFIG_DIR=$H/.prometheus XDG_CONFIG_HOME=$H/.config \
  LANG=C.UTF-8 PYTHONPATH=$WT/src PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 \
  $WT/.venv/bin/python scripts/daemon.py --config $D/cfg-$v.yaml > $D/daemon-$v.log 2>&1 &
echo $! > $D/daemon-$v.pid
```

### rund.sh

```bash
#!/usr/bin/env bash
# rund.sh <0910|0912> <config> <logname> : start only the scratch daemon (proxy already running)
set -u
D=<DIAG>
v=$1; CFG=$2; LOGN=$3
H=$D/h$v; WT=$D/wt-$v
cd $WT
nohup env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$H PROMETHEUS_CONFIG_DIR=$H/.prometheus XDG_CONFIG_HOME=$H/.config \
  LANG=C.UTF-8 PYTHONPATH=$WT/src PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 \
  $WT/.venv/bin/python scripts/daemon.py --config $CFG > $D/$LOGN 2>&1 &
echo $! > $D/daemon-$v.pid
for i in $(seq 1 40); do
  if grep -q "Uvicorn running" $D/$LOGN 2>/dev/null; then echo "daemon $v up (pid $(cat $D/daemon-$v.pid))"; exit 0; fi
  kill -0 $(cat $D/daemon-$v.pid) 2>/dev/null || { echo "daemon $v DIED"; tail -20 $D/$LOGN; exit 1; }
  sleep 3
done
echo "timeout"; exit 1
```

### drive.py

```python
"""Drive one chat turn over the scratch daemon's WebSocket, the way Beacon does.

Usage: python drive.py <ws_port> <token_file> <session_id> <message> <frames_out.jsonl> [timeout_s]

Records every frame to frames_out, then prints what a Beacon chat bubble
would have shown: tool calls, streamed text (chat_delta), and the end frames.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid

import websockets

port, token_file, sid, message, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
timeout = float(sys.argv[6]) if len(sys.argv) > 6 else 600.0
token = open(token_file).read().strip()


async def main() -> None:
    frames = []
    deltas = []
    t0 = time.time()
    async with websockets.connect(f"ws://127.0.0.1:{port}", max_size=None) as ws:
        await ws.send(json.dumps({"type": "auth", "token": token}))
        await ws.send(json.dumps({"type": "subscribe", "payload": {"sessions": [sid]}}))
        await ws.send(json.dumps({"type": "send_message", "payload": {
            "session_id": sid, "content": message, "client_msg_id": str(uuid.uuid4()), "mode": "agent"}}))
        done = False
        while time.time() - t0 < timeout:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                if done:
                    break
                continue
            fr = json.loads(raw)
            fr["_t"] = round(time.time() - t0, 2)
            frames.append(fr)
            typ = fr.get("type")
            p = fr.get("payload") or {}
            if typ == "chat_delta" and p.get("session_id") == sid:
                deltas.append(p.get("content", ""))
            elif typ == "tool_call_start" and p.get("session_id") == sid:
                print(f"[{fr['_t']:7.2f}] TOOL START {p.get('tool_name')} {json.dumps(p.get('inputs'))[:200]}")
            elif typ == "tool_call_end" and p.get("session_id") == sid:
                print(f"[{fr['_t']:7.2f}] TOOL END   {p.get('tool_name')} success={p.get('success')} result={str(p.get('result'))[:160]!r}")
            elif typ in ("chat_done", "error", "chat_discard", "provider_degraded", "turn_completed", "compaction"):
                print(f"[{fr['_t']:7.2f}] {typ} {json.dumps(p)[:300]}")
                if typ == "chat_done" and p.get("session_id") == sid:
                    done = True
            elif typ == "agent_state" and done and p.get("state") == "idle":
                break
    with open(out, "w") as fh:
        for fr in frames:
            fh.write(json.dumps(fr) + "\n")
    text = "".join(deltas)
    print(f"--- streamed chat_delta text ({len(text)} chars): {text[:1500]!r}")
    print(f"--- frame types: {sorted({f.get('type') for f in frames})}")
    print(f"--- elapsed {time.time() - t0:.1f}s")


asyncio.run(main())
```

### summarize.py

```python
import json, sys
path, start = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
rows = [json.loads(l) for l in open(path)][start:]
for r in rows:
    if r.get("kind") != "chat": continue
    req, resp = r["request"], r["response"]
    if not req.get("stream"): continue
    tools = [t["function"]["name"] for t in req.get("tools") or []]
    msgs = req["messages"]
    sys_len = sum(len(m.get("content") or "") for m in msgs if m["role"] == "system")
    u = resp.get("usage") or {}
    print(f"seq={r['seq']} dt={r['dt']}s max_tokens={req.get('max_tokens')} tool_choice={req.get('tool_choice')} n_tools={len(tools)} n_msgs={len(msgs)} sys_chars={sys_len} "
          f"prompt_tokens={u.get('prompt_tokens')} completion={u.get('completion_tokens')} finish={resp.get('finish_reason')}")
    print("   tools:", tools)
    print("   calls:", [(c['name'], c['arguments'][:120]) for c in resp.get('tool_calls') or []], "content:", repr(resp.get('content'))[:300])
```

### breakdown.py

```python
"""Token breakdown of a captured round-0 request, in the server's own tokenizer."""
import json, re, sys, httpx
port, cap_path, seq = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
r = next(json.loads(l) for l in open(cap_path) if '"kind": "chat"' in l and json.loads(l).get("seq") == seq)
req = r["request"]; usage = r["response"]["usage"]
def ntok(text):
    resp = httpx.post(f"http://127.0.0.1:{port}/tokenize", json={"content": text}, timeout=60)
    return len(resp.json()["tokens"])
sysmsg = next(m["content"] for m in req["messages"] if m["role"] == "system")
user = [m["content"] for m in req["messages"] if m["role"] == "user"]
# split the system prompt at top-level headings (lines starting with "# ") and the dynamic boundary
parts, cur, title = [], [], "(preamble)"
for line in sysmsg.split("\n"):
    if line.startswith("# ") or line.startswith("<!-- Bootstrap:") or line.strip() == "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__":
        if cur: parts.append((title, "\n".join(cur)))
        title, cur = line.strip()[:70], [line]
    else:
        cur.append(line)
if cur: parts.append((title, "\n".join(cur)))
total_sys = ntok(sysmsg)
rows = [(t, len(txt), ntok(txt)) for t, txt in parts]
print(f"server prompt_tokens for the whole request: {usage['prompt_tokens']}")
print(f"system prompt: {len(sysmsg)} chars, {total_sys} tokens (sections tokenized separately sum to {sum(x[2] for x in rows)})")
for t, c, n in sorted(rows, key=lambda x: -x[2]):
    print(f"   {n:6d} tok  {c:6d} ch  {t}")
tools = req.get("tools") or []
tool_rows = [(t["function"]["name"], ntok(json.dumps(t["function"], ensure_ascii=False))) for t in tools]
tools_total = ntok("\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools))
print(f"tool catalog ({len(tools)} tools, as JSON lines the template renders): {tools_total} tokens")
for name, n in sorted(tool_rows, key=lambda x: -x[1]):
    print(f"   {n:6d} tok  {name}")
u = sum(ntok(x) for x in user)
print(f"user message(s): {u} tokens")
print(f"template/framing remainder: {usage['prompt_tokens'] - total_sys - tools_total - u} tokens")
```

### probe.py

```python
"""Small direct probes of llama.cpp tool calling through the logging proxy."""
import json, sys, httpx
port = int(sys.argv[1]); which = sys.argv[2]
cap = [json.loads(l) for l in open("<DIAG>/capture-0910.jsonl") if '"kind": "chat"' in l]
tools_by_name = {t["function"]["name"]: t for r in cap for t in (r["request"].get("tools") or [])}
model = cap[-1]["request"]["model"]
agent_schema = json.loads(cap[-2]["request"]["messages"][-1]["content"]) if False else None
# agent schema as tool_search returned it (Anthropic shape) -> OpenAI function
sel = None
for r in cap:
    for m in r["request"]["messages"]:
        if m["role"] == "tool" and '"name": "agent"' in (m.get("content") or ""):
            try:
                cand = json.loads(m["content"])
            except Exception:
                continue
            if isinstance(cand, dict) and cand.get("name") == "agent":
                sel = cand; break
    if sel: break
agent_tool = {"type": "function", "function": {"name": "agent", "description": sel["description"], "parameters": sel["input_schema"]}}
ts = tools_by_name["tool_search"]
msgs = [
  {"role": "system", "content": "You are a helpful agent. Besides the listed functions, a function named `agent` exists that spawns a sub-agent; its schema: " + json.dumps(sel)},
  {"role": "user", "content": "Call the agent function now with description='list files' and prompt='List the files in the workspace'. Do not call any other function."},
]
tools = {"A": [ts], "B": [ts, agent_tool], "C": [tools_by_name["memory"]]}[which]
body = {"model": model, "messages": msgs, "tools": tools, "tool_choice": "auto", "max_tokens": 300, "stream": True,
        "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
calls, content, finish = {}, [], None
with httpx.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=120) as r:
    for line in r.iter_lines():
        if not line.startswith("data:") or line.strip() == "data: [DONE]": continue
        ch = json.loads(line[5:])
        for c in ch.get("choices") or []:
            finish = c.get("finish_reason") or finish
            d = c.get("delta") or {}
            if d.get("content"): content.append(d["content"])
            for tc in d.get("tool_calls") or []:
                s = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                f = tc.get("function") or {}
                s["name"] += f.get("name") or ""; s["arguments"] += f.get("arguments") or ""
print(f"probe {which}: tools={[t['function']['name'] for t in tools]} finish={finish}")
print("  calls:", list(calls.values()))
print("  content:", repr("".join(content))[:400])
```

## Appendix B — uncommitted diff on `fix/local-tool-reach-0913` (fixes 1 and 2)

As of the handoff; the worktree `.claude/worktrees/local-tool-reach` still holds it.

```diff
diff --git a/config/prometheus.yaml.default b/config/prometheus.yaml.default
index cc36798c..aa491caf 100644
--- a/config/prometheus.yaml.default
+++ b/config/prometheus.yaml.default
@@ -223,7 +223,7 @@ context:
   #
   # And how many tokens one reply may spend (the request's max_tokens). Each
   # model already gets its own cap: a table for the cloud models we know,
-  # 8192 for llama.cpp, 4096 for anything else, and never more than a quarter
+  # 4096 for anything else (llama.cpp included), and never more than a quarter
   # of a window a local server measured (providers/output_caps.py). Set this
   # to override that for one model; it is clamped to the vendor's maximum.
   #
@@ -260,12 +260,16 @@ tools:
     # skill promoted 2026-09-26 (docs/audits/SKILL-USAGE.md, option C2): the
     # system prompt tells every run to "use the skill tool", and on 822 of 973
     # runs it was not in the list. ~150 tokens.
+    # agent promoted 2026-10-16: a local model never calls a tool its request
+    # does not list, and tool_search select does not add one, so "use a
+    # sub-agent" searched until the repeat guard stopped the turn. Local runs
+    # followed 2 of 52 selects with a call (2026-08-01 to 10-15). ~250 tokens.
     # Leave this key out to follow the shipped default across upgrades. A list
     # equal to one Prometheus shipped as the default (this one, or an earlier
     # one that setup wrote into your config) follows the default too, with one
     # INFO line at boot. Any other list is yours and is used exactly as
     # written (`[]` advertises nothing); `oara doctor` says which applies.
-    always_loaded: [bash, task_create, read_file, write_file, edit_file, grep, glob, tool_search, skill, web_search, web_fetch, memory]
+    always_loaded: [bash, task_create, read_file, write_file, edit_file, grep, glob, tool_search, skill, web_search, web_fetch, memory, agent]
     # MCP tools follow the servers you configured: a configured server IS
     # the advertise decision, so its tools sit in the catalog like any
     # always_loaded tool, and a server added or removed over REST changes
diff --git a/docs/reference/config-keys.md b/docs/reference/config-keys.md
index 477afbac..9b9e4e69 100644
--- a/docs/reference/config-keys.md
+++ b/docs/reference/config-keys.md
@@ -71,7 +71,7 @@ none either — the key is documented so its absence is visible.
 | `tools` | *(section)* |
 | `tools.deferred_loading` | *(section)* |
 | `tools.deferred_loading.enabled` | `auto` |
-| `tools.deferred_loading.always_loaded` | `['bash', 'task_create', 'read_file', 'write_file', 'edit_file', 'grep', 'glob', 'tool_search', 'skill', 'web_search', 'web_fetch', 'memory']` |
+| `tools.deferred_loading.always_loaded` | `['bash', 'task_create', 'read_file', 'write_file', 'edit_file', 'grep', 'glob', 'tool_search', 'skill', 'web_search', 'web_fetch', 'memory', 'agent']` |
 | `tools.deferred_loading.mcp_always_deferred` | `False` |
 | `tools.deferred_loading.search_mcp` | `True` |
 | `adapter` | *(section)* |
diff --git a/src/prometheus/config/shipped_defaults.py b/src/prometheus/config/shipped_defaults.py
index 6fe261f6..5e18bcec 100644
--- a/src/prometheus/config/shipped_defaults.py
+++ b/src/prometheus/config/shipped_defaults.py
@@ -53,6 +53,12 @@ ALWAYS_LOADED_DEFAULTS_SHIPPED: tuple[tuple[str, ...], ...] = (
     # + skill (#593, option C2, 2026-09-26).
     ("bash", "task_create", "read_file", "write_file", "edit_file", "grep", "glob",
      "tool_search", "skill", "web_search", "web_fetch", "memory"),
+    # + agent (2026-10-16): a local model never calls a tool its request does not
+    # list. "use a sub-agent" ran tool_search select agent until the repeat guard
+    # halted the turn, and since 2026-08-01 local runs followed 2 of 52 selects
+    # with a call. About 250 tokens a request.
+    ("bash", "task_create", "read_file", "write_file", "edit_file", "grep", "glob",
+     "tool_search", "skill", "web_search", "web_fetch", "memory", "agent"),
 )
 
 # The tool set a fresh install advertises: the newest shipped default. Absent
diff --git a/src/prometheus/providers/output_caps.py b/src/prometheus/providers/output_caps.py
index e143fd6f..e66c411e 100644
--- a/src/prometheus/providers/output_caps.py
+++ b/src/prometheus/providers/output_caps.py
@@ -22,7 +22,7 @@ Whatever wins is clamped to the model's documented maximum when one is known,
 so no setting can ask a vendor for more than it serves. Last, on a window the
 server MEASURED (a local box reports n_ctx), the cap never takes more than
 :data:`LOCAL_ROOM_FRACTION` of it, so a small local window keeps most of its
-room for the conversation. A 32768 window allows 8192; a 16384 one, 4096.
+room for the conversation. A 16384 window allows 4096; an 8192 one, 2048.
 """
 
 from __future__ import annotations
@@ -34,11 +34,17 @@ from typing import Any
 
 log = logging.getLogger(__name__)
 
-#: The cap for a llama.cpp backend, whatever GGUF it serves: a quarter of the
-#: 4090's 32768 window. Ollama is not included: its probe reports the model's
-#: TRAINED context length, not the runtime ``num_ctx``, so the room a reply has
-#: there is not known and it keeps the default.
-LOCAL_OUTPUT_CAP = 8192
+#: The cap for a llama.cpp backend, whatever GGUF it serves: 4096, what every
+#: local request sent before per-model caps. 0.9.12 sent 8192, a quarter of the
+#: 4090's 32768 window, and the compactor's projected-overflow rule (#773)
+#: counts that as room the next round needs: with ~18.6k tokens of system
+#: prompt and tool schemas, the line fell from 26,214 to ~22.3k and a FIRST
+#: turn was summarised after two or three tool results (2026-10-16). A reply
+#: cut at 4096 is told to send its work in smaller pieces. Ollama is not
+#: included: its probe reports the model's TRAINED context length, not the
+#: runtime ``num_ctx``, so the room a reply has there is not known and it keeps
+#: the default.
+LOCAL_OUTPUT_CAP = 4096
 
 #: The most of a MEASURED window one reply may ask for. The compactor's
 #: projected-overflow rule (#773) counts the request's ``max_tokens`` as room
diff --git a/tests/test_agent_tool_is_advertised.py b/tests/test_agent_tool_is_advertised.py
new file mode 100644
index 00000000..6e183e02
--- /dev/null
+++ b/tests/test_agent_tool_is_advertised.py
@@ -0,0 +1,64 @@
+"""The ``agent`` tool is advertised by default (2026-10-16).
+
+Asked from Beacon to "use a sub-agent to list the files in your workspace", the
+local Qwen3.8-27B called ``tool_search`` five times and never ``agent``, until
+the repeat guard halted the turn. ``agent`` was deferred, and on llama.cpp with
+native tool calling a deferred tool cannot be called: ``tool_search`` select
+returns the schema as a tool result, the run's tool list stays as it was, and
+the model will not call a function its request does not list. Since
+2026-08-01 local runs followed 2 of 52 selects with a call of the selected
+tool. Offered ``agent``, the same model called it on its first round.
+
+This changes the advertised tool list, so it is trace-changing by design.
+"""
+
+from __future__ import annotations
+
+import json
+
+from prometheus.config.shipped_defaults import (
+    ALWAYS_LOADED_DEFAULTS_SHIPPED,
+    SHIPPED_ALWAYS_LOADED,
+    resolve_always_loaded,
+)
+from prometheus.context.dynamic_tools import DynamicToolLoader
+from prometheus.tools.base import ToolRegistry
+from prometheus.tools.builtin.agent import AgentTool
+from prometheus.tools.builtin.cron_list import CronListTool
+from tests.support.advertisement import template_always_loaded
+
+
+def test_agent_is_in_the_shipped_always_loaded():
+    assert "agent" in SHIPPED_ALWAYS_LOADED
+
+
+def test_the_template_ships_it_too():
+    assert "agent" in template_always_loaded()
+
+
+def test_a_deferred_run_is_offered_the_agent_tool():
+    registry = ToolRegistry()
+    registry.register(AgentTool())
+    registry.register(CronListTool())
+    loader = DynamicToolLoader(registry, None)  # no config: the shipped set
+    names = [s["name"] for s in loader.schemas_for_run(True)]
+    assert "agent" in names
+    assert "cron_list" not in names  # still deferred: the set is not "everything"
+
+
+def test_a_config_holding_the_0_9_12_default_gets_agent_too():
+    """Setup wrote the default into the configs it made, so a config holding the
+    previous shipped set exactly did not choose it: it follows the newest."""
+    previous = ALWAYS_LOADED_DEFAULTS_SHIPPED[-2]
+    assert "agent" not in previous
+    assert "agent" in resolve_always_loaded({"always_loaded": list(previous)})
+
+
+def test_an_operators_own_list_is_still_used_as_written():
+    mine = ["bash", "read_file", "vault_search"]
+    assert resolve_always_loaded({"always_loaded": mine}) == mine
+
+
+def test_its_schema_stays_small_enough_to_ride_every_request():
+    # About 220 tokens as the chat template renders it (measured 2026-10-16).
+    assert len(json.dumps(AgentTool().to_api_schema())) <= 1200
diff --git a/tests/test_always_loaded_follows_the_default.py b/tests/test_always_loaded_follows_the_default.py
index 86fc8fa4..5b170b44 100644
--- a/tests/test_always_loaded_follows_the_default.py
+++ b/tests/test_always_loaded_follows_the_default.py
@@ -44,6 +44,9 @@ FROM_GIT = (
      "tool_search", "web_search", "web_fetch", "memory"),
     ("bash", "task_create", "read_file", "write_file", "edit_file", "grep", "glob",
      "tool_search", "skill", "web_search", "web_fetch", "memory"),
+    # + agent, 2026-10-16 (after the git read above).
+    ("bash", "task_create", "read_file", "write_file", "edit_file", "grep", "glob",
+     "tool_search", "skill", "web_search", "web_fetch", "memory", "agent"),
 )
 V094 = FROM_GIT[2]  # what 0.9.4's setup wrote
 
diff --git a/tests/test_output_caps.py b/tests/test_output_caps.py
index 097ca28b..1dd0d7b1 100644
--- a/tests/test_output_caps.py
+++ b/tests/test_output_caps.py
@@ -74,9 +74,25 @@ def test_a_known_model_gets_its_cap(model, cap):
 
 def test_a_llama_cpp_backend_gets_the_local_cap_whatever_its_model_is_called():
     _table, local = _limits()
-    assert local == 8192
+    assert local == 4096
     for name in ("", "qwen3.8-27b", "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"):
-        assert _caps().resolve(name, provider=LLAMA, default=4096) == (8192, "local")
+        assert _caps().resolve(name, provider=LLAMA, default=4096) == (4096, "local")
+
+
+def test_on_a_32k_local_window_the_cap_leaves_the_80_percent_line_in_charge():
+    """2026-10-16: with 8192 sent on the 4090's 32768 window, the projected-overflow
+    rule (#773) lowered the compaction line from 26,214 to ~22.3k, and with ~18.6k
+    tokens of system prompt and tool schemas a FIRST turn was summarised after two
+    or three tool results. The rule's line is window - max_tokens - the largest
+    recent tool result; a 2k result is ordinary (a tool_search page)."""
+    from prometheus.context.compactor import compaction_threshold
+
+    window, ordinary_result = 32768, 2000
+    cap, source = _caps().resolve("local-27b", provider=LLAMA, default=4096,
+                                  window=window, window_measured=True)
+    assert (cap, source) == (4096, "local")
+    assert window - cap - ordinary_result >= compaction_threshold(window) == 26214
+    assert window - 8192 - ordinary_result < compaction_threshold(window), "premise: 8192 moved the line"
 
 
 def test_an_unknown_model_keeps_the_default_on_ollama_and_in_the_cloud():
@@ -190,12 +206,13 @@ def _anthropic_sse(text: str) -> str:
     return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
 
 
-def _sent_max_tokens(provider, model: str, output_caps, *, tool=None) -> LoopContext:  # noqa: ANN001
+def _sent_max_tokens(provider, model: str, output_caps, *, tool=None,  # noqa: ANN001
+                     max_tokens: int = 4096) -> LoopContext:
     registry = ToolRegistry()
     if tool is not None:
         registry.register(tool)
     ctx = LoopContext(provider=provider, model=model, system_prompt="You are a test.",
-                      max_tokens=4096, tool_registry=registry, output_caps=output_caps)
+                      max_tokens=max_tokens, tool_registry=registry, output_caps=output_caps)
 
     async def drain() -> None:
         async for _ in run_loop(ctx, [ConversationMessage.from_user_text("hi")]):
@@ -223,8 +240,10 @@ def test_an_opus_5_turn_asks_for_room_for_its_thinking(monkeypatch):
 
 def test_a_llama_cpp_turn_asks_for_the_local_cap(monkeypatch):
     seen = _serve(monkeypatch, [_openai_sse(text="ok")])
-    _sent_max_tokens(LlamaCppProvider(base_url="http://unit.test"), "local-27b", _caps())
-    assert [s["max_tokens"] for s in seen] == [8192]
+    # The loop's own max_tokens differs, so 4096 here is the local cap and not the default.
+    _sent_max_tokens(LlamaCppProvider(base_url="http://unit.test"), "local-27b", _caps(),
+                     max_tokens=2000)
+    assert [s["max_tokens"] for s in seen] == [4096]
 
 
 def test_a_route_max_tokens_reaches_the_request(monkeypatch):
@@ -252,9 +271,9 @@ def test_a_fallback_request_asks_for_the_fallback_models_cap():
     fallback = SimpleNamespace(model="qwen3.8-max",
                                provider=SimpleNamespace(provider_name="qwen",
                                                         route_max_tokens=None))
-    ctx = SimpleNamespace(output_caps=_caps(), max_tokens=4096, model="local-27b",
+    ctx = SimpleNamespace(output_caps=_caps(), max_tokens=2000, model="local-27b",
                           provider=primary, fallback=fallback, compactor=None)
-    assert _output_cap(ctx, "local-27b") == 8192
+    assert _output_cap(ctx, "local-27b") == 4096
     assert _output_cap(ctx, "qwen3.8-max") == 16384
 
 
@@ -302,8 +321,10 @@ def test_a_cut_call_is_told_the_cap_that_was_actually_sent(tmp_path, monkeypatch
 
 # ── a measured local window keeps most of its room ───────────────────────────
 
-@pytest.mark.parametrize("window,cap", [(32768, 8192), (16384, 4096), (8192, 2048), (2048, 1024)])
+@pytest.mark.parametrize("window,cap", [(65536, 4096), (32768, 4096), (16384, 4096), (8192, 2048),
+                                        (2048, 1024)])
 def test_a_measured_window_caps_the_reply_at_a_quarter_of_it(window, cap):
+    """The quarter is a ceiling under the local cap: it bites below a 16384 window."""
     assert _caps().resolve("local-27b", provider=LLAMA, default=4096,
                            window=window, window_measured=True)[0] == cap
 
@@ -317,7 +338,7 @@ def test_the_room_rule_applies_to_every_source_but_only_on_a_measured_window():
                         window=16384, window_measured=False) == (20000, "model_override")
     # A window nobody knows (no compactor): no clamp.
     assert _caps().resolve("local-27b", provider=LLAMA, default=4096,
-                           window=0, window_measured=True) == (8192, "local")
+                           window=0, window_measured=True) == (4096, "local")
 
 
 class _Compactor:
@@ -345,17 +366,17 @@ def test_a_small_local_window_sends_and_plans_around_a_quarter_of_it(monkeypatch
     from prometheus.engine.agent_loop import _output_cap
 
     ctx = SimpleNamespace(output_caps=_caps(), max_tokens=4096, model="local-27b",
-                          provider=LLAMA, fallback=None, backend=None, compactor=_Compactor(16384))
-    assert _output_cap(ctx, "local-27b") == 4096
+                          provider=LLAMA, fallback=None, backend=None, compactor=_Compactor(8192))
+    assert _output_cap(ctx, "local-27b") == 2048
     ctx.compactor = _Compactor(65536)
-    assert _output_cap(ctx, "local-27b") == 8192, "a large window still gets the local cap, no more"
+    assert _output_cap(ctx, "local-27b") == 4096, "a large window still gets the local cap, no more"
 
 
 def test_the_compactor_plans_around_the_cap_the_request_sends(monkeypatch):
     """#773's projected-overflow rule counts the request's max_tokens as room the
     next round needs: it must be the cap actually sent, not a flat 4096."""
     seen = _serve(monkeypatch, [_openai_sse(text="ok")])
-    compactor = _Compactor(16384)
+    compactor = _Compactor(8192)
     ctx = LoopContext(provider=LlamaCppProvider(base_url="http://unit.test"), model="local-27b",
                       system_prompt="You are a test.", max_tokens=4096,
                       tool_registry=ToolRegistry(), output_caps=_caps(), compactor=compactor)
@@ -365,8 +386,8 @@ def test_the_compactor_plans_around_the_cap_the_request_sends(monkeypatch):
             pass
 
     asyncio.run(drain())
-    assert [s["max_tokens"] for s in seen] == [4096], "a quarter of the 16384 window"
-    assert compactor.planned == [4096]
+    assert [s["max_tokens"] for s in seen] == [2048], "a quarter of the 8192 window"
+    assert compactor.planned == [2048]
 
 
 # ── every entry point passes the caps in ─────────────────────────────────────
```
