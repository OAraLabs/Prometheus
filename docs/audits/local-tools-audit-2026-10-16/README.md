# Local tool calling and context: audit, regression gate, fix plan (2026-10-16)

Fresh-eyes audit of how a local Qwen3.8-27B turn (llama.cpp on the 4090, 32k
window, adapter tier `light`) reaches tools and spends its context, in
Prometheus 0.9.12 (`e3c1787`). Read-only on live: the live daemon was not
changed or restarted. Live data came from copies of the telemetry and
conversation databases. Real-model checks ran on scratch daemons through a
logging proxy, in short turns.

No fix is written. The plan in §9 waits for approval.

## Summary

1. **Deferred tools have never been reachable from the local model.** This is
   not a 0.9.12 or 0.9.10 regression. It started when deferral was switched on
   for local providers: #121 on 31 Jul for Telegram and the CLI, #126 on 1 Aug
   for Beacon, and live from 3 Aug. Two things block it, both measured on the
   4090:
   - **The model** will not call a function that is missing from its tool
     list, even right after `tool_search` returned the schema. It selects the
     tool again. With no grammar at all it does the same (§4).
   - **The server.** llama.cpp builds its tool-call grammar from the request's
     `tools`, so a call can only name an offered tool. In the probe the model
     tried to call `lcm_grep`; the grammar turned it into `tool_search` or
     `memory`.

   Select never adds the tool to the request; the run's tool list was never
   grown at any point in the repo's history. The two "safety nets" of the
   April design (docs/PROMETHEUS-FULL.md, Sprint 25) were written for a setup
   where every tool was advertised by default. They were never needed until
   #121, and they don't work on native tool calling.

2. **"Lucky guess" still exists in dispatch, but it can't fire on the native
   path.** There are 28 lucky-guess rows (26 Aug → 19 Sep). Of these, 25 were
   mislabels: the run's tools had been chosen for a cloud model, so all 52–55
   were offered. That was the advertise-before-routing bug (#462, fixed by
   #477), and later a cloud→local fallback. The other 3 were genuine, and all
   went through the text path. The model wrote a bare JSON call (the format
   the QwenFormatter prose tells it to use), which the grammar doesn't catch
   and the text extractor runs. Two of the three were smoke tests.

3. **What 0.9.12 did break is compaction timing.** The 8,192 local output cap
   (#806) feeds #773's projected-overflow rule. On a 32k window the line drops
   from 26,214 to 24,576 minus the largest recent tool result. On scratch with
   the live overhead:
   - one 0.9.12 cron turn compacted 27 times, 8 of them under the old line;
   - v0.9.10 had none early.

   The live incident logged ceiling 22,556 (projected).

4. **Silent turns.** Eleven endings that the loop writes are saved but never
   streamed to Beacon (repeat guard, iteration cap, empty response,
   pre-flight, context overflow, circuit breaker ×4, boundary escape,
   divergence). Only the degenerate-output and round-cap endings stream (since
   0.9.10). Live, 3 of about 51 local Beacon turns since 3 Oct ended this way.

5. **Overhead.** Round 0 is 18,634 server tokens on the live config, 57% of
   the window. Prometheus itself adds about 6.5–7.1k of that (§8). Two pieces
   carry no information on this setup:
   - the QwenFormatter repeats every tool description in prose, with an
     instruction ("respond ONLY with a JSON object") that contradicts the
     template's XML call format: 1,440 tokens;
   - the deploy checkout's PROMETHEUS.md, a developer doc for this repo, rides
     every Beacon chat that has no workspace: 2,828 tokens.

6. **Known good.**
   - **Reach:** v0.1.0 (27 Jul) is the last release before #121, and it
     advertised everything. In the gate, 49 of its 51 tools were reachable;
     the two misses are MCP tools, which didn't exist yet.
     With every tool offered (0.9.12 with deferral off), the real 27B picked
     the right tool every time: `agent`, `cron_create`, `wiki_query`,
     `lcm_grep`, and the reachability tier in §6.
   - **Compaction:** v0.9.10 / v0.9.11.
   - **Endings reaching Beacon:** no version.

7. **Context windows (§7).**
   - **llama.cpp:** read correctly, from the served per-slot `n_ctx`.
   - **Ollama:** budgeted at the model's trained length (`/api/show`), and
     Prometheus sends no `num_ctx`. An over-window prompt is cut from the
     front with no error (measured on the 7b). The 7b/14b "32k" is the
     trained length. It happens to match what the 7b was loaded with on this
     box, but the 4090's qwen3.5:9b would be budgeted at 262,144.
   - **Overrides:** they match the exact model id, so a renamed GGUF silently
     drops its override.
   - **Reply room:** three different numbers describe it.
   - **Cloud:** 23 reachable cloud models run on the 1M default. grok-4.5
     (500k) and kimi-k2.6 (262k) are overestimated 2–4×.

8. **The gate (§6)** has two parts:
   - three integration suites that boot the real daemon with a scripted
     llama.cpp/Ollama stand-in;
   - a scripted real-model suite for a scratch daemon.

   On 0.9.12 the integration suites give 32 passed, 258 xfailed. Parity is
   green (182/182).

## 1. One local turn, end to end (e3c1787)

| Step | Where | What happens on a local turn |
|---|---|---|
| Ingress | `web/server.py:1471` `/api/chat/send` → `web/ws_server.py:1716` `_run_agent_locked` → `run_loop(..., surface="beacon")` | Beacon posts the message; the WS bridge runs the loop |
| Tool directive | `engine/agent_loop.py:1683-1708` | `auto` unless the turn forces a tool |
| Routing | `agent_loop.py:1713` → `engine/stages/routing.py` | Primary route keeps the local adapter (tier light) |
| **Tool catalog, frozen** | `agent_loop.py:1738-1753`; `context/dynamic_tools.py:204-239` (`auto` → deferred for every tier but `off`), `:241-252`, `:306-312` | Only `always_loaded` schemas, computed once per run, never grown |
| Profile filter, telemetry row | `agent_loop.py:1760-1839` | `tool_advertisement` row: `deferred_active`, `advertised` |
| System prompt | boot: `context/prompt_assembler.py` (SOUL 272, AGENTS 279, ANATOMY 287, skills 347, project instructions 147, memory 82, saved files 390), `context/system_prompt.py:34-121` (base; names `lcm_grep`/`lcm_expand`/`lcm_describe` at 56-59 and 102); per run: workspace swap 1861, memory refresh 1884, `adapter.format_request` 1912 → `adapter/formatter.py:123-139` (prose tool list + JSON instruction, every tier), recall 1924 | Byte-stable within a run |
| Round loop | `agent_loop.py:2124` | One model call per round |
| Microcompact | `:2127` → `:4414` | From round 3, old results cut to ~500 chars |
| Request tools | `:2223-2252` | tier light: native `tools`; tier full or a forced round: none (then the provider sends Prometheus's own GBNF grammar, generated from the **full** registry) |
| Compaction | `:2269-2308` → `context/compactor.py:1168-1181` | Every round, mid-turn included; projected rule uses `max_tokens = _output_cap()` (`agent_loop.py:3857`, `providers/output_caps.py:150-171`, `LOCAL_OUTPUT_CAP = 8192` at :41) |
| Request | `:2312-2330`; `providers/llama_cpp.py:624-690` | Native `tools`, `tool_choice: "auto"`, grammar dropped when tools are present (`:676-680`) |
| Pre-flight / overflow | `:2427-2472`; `:2540-2634` | Refuses an over-window estimate; one compact-and-retry on a 400 (#805) |
| Server (llama.cpp b1-9d57ce456) | `common/chat.cpp`, Qwen3-Coder handler | Lazy grammar triggered by `<tool_call>`; function names limited to the request's `tools` |
| Text path | `agent_loop.py:2719-2740` | If no native call came back, calls are parsed from text and checked against the **full** registry |
| Dispatch | `_execute_tool_call` `:5115`; validator `adapter/validator.py:309-320` (full registry); unknown tool `:5328`; **lucky guess** `:5350-5385` | Any registered tool runs; a marker is written if it was not advertised (recomputed, so it can mislabel) |
| tool_search | `tools/tool_search.py:159-166` select (schema as text), `:181-186` miss (lists all 192 tool and skill names, 1,701 tokens), `:189-238` search (top 5) | Never changes the request |
| Endings | repeat guard `agent_loop.py:3402-3444`, others listed in §5 | Appended to history, `AssistantTurnComplete` only |
| Stream to Beacon | `ws_server.py:1746-1749` (ATC only counts rounds), `:1771-1783` (`chat_delta` from text deltas), `:1851-1858` (`chat_done` without a reason) | An ending that was not streamed never shows live |

## 2. Archaeology

The full table of 191 merged PRs since 1 Jul is in
[pr-table.md](pr-table.md). These are the ones that decide the outcome:

| PR | Date | What it changed for a local turn |
|---|---|---|
| (initial commit) | 20 Apr | Native `tools` on tier light; Prometheus's grammar dropped when tools are present. Deferral off by default, all tools advertised. tool_search + lucky guess designed |
| #76, #77 | 2 Jul | Forced rounds withhold native tools and send a grammar from the full registry. Multi-tool grammars were invalid until #77 |
| **#121** | 31 Jul | `deferred_loading.enabled: auto` → deferral **on** for every local tier. Catalog frozen per run (it never grew before either). `on_demand()` deleted (never called) |
| #126 | 1 Aug | Beacon/web turns get the tool loader, so deferral reaches Beacon |
| #159, #593 | 12 Aug, 27 Sep | Shipped always_loaded 8 → 11 → 12. The live config pins its own list, so neither reached it |
| #263, #323 | 20 Aug, 29 Aug | Repeat guard and self-halt: endings appended, not streamed |
| #462 → **#477** | 8 → 12 Sep | Advertisement decided before routing gave local turns all 55 tools (the "lucky guess" mislabels); fixed |
| #583 | 26 Sep | Tier decided from registry, then the served template; the Qwen stays light |
| #605 | 27 Sep | Lucky-guess marker moved to `subsystem_runs` (telemetry only) |
| #772/#777, #774/#775 | 14 Oct | Degenerate-output and round-cap endings now stream; the other eleven do not |
| #778 (#773) | 14 Oct | Compaction 80% + projected-overflow rule |
| #784 | 14 Oct | `context_compaction` WS frames (summary passes only; trim-only passes send none) |
| **#805** | 16 Oct | Counts in server tokens (calibration): removes the slack that hid the next one |
| **#806** | 16 Oct | Local output cap 4096 → 8192; feeds the projected rule; early compaction |

Seams you asked about. None of these changed the tools or prompt a local model
receives, except where noted:
- **Telemetry v2** (#634, #635, #638, #640): observe only. One flaw: its
  `tool_sets` hash records the advertised set, not what was sent, so on tier
  full or forced rounds it names 12 tools while none were sent.
- **Routing-stage move** (#570): a pure move.
- **Parity harness** (#568): tests only. The goldens pin full request bodies.
  Adding a tool mid-run, or dropping the Qwen prose list, breaks 11 of the 12
  local scenarios and needs a live re-record.
- **hooks/1.0 contract** (#566): docs only. Its §6.3 says `{tool:X}` must name
  an advertised tool; the code accepts any registered one.
- **Hooks:** none touch the system prompt or `tools`. The file-mutation
  verifier and LSP diagnostics add to messages and tool results.
- **Changes that did alter what local turns receive:**
  - H2 (tier full withholds tools);
  - #76/#77 forced rounds;
  - the circuit breaker's light→full bump (shared across sessions until
    #422);
  - #477;
  - #580 (identity line).
- **`wp35-instinct-shadow`:** local only, with no remote branch and no PR.
  2 commits (13 Oct), not merged, not in the deploy checkout. Its
  `pillars.shadow.enabled` defaults to false, and the live config has no
  pillars keys. Nothing Instinct-related is on main.

## 3. Telemetry by date (copy of the live DB, read-only)

- **Local deferral on live:**
  - **3 Aug → 1 Sep:** 13 tools advertised;
  - **1 → 24 Sep:** 15 (Context7 added);
  - **30 Sep → 16 Oct:** 16 (`skill` added);
  - **16 Oct, after the workaround:** 17 (`agent`).

  Deferral was decided per run, tier light, for every local Beacon turn.
- **Before 3 Aug** local runs advertised all tools: 50 on Qwen3.6, "registry
  direct" / "explicitly disabled".
- **Reach, 1 Aug → 16 Oct:**
  - 52 local `tool_search` selects; 2 were followed by a call of the selected
    tool (the 31 Aug test campaign: one session in about twenty).
  - Other than the 28 lucky-guess rows (§Summary 2), no deferred tool was
    called by a local turn, apart from coding-mode runs with their own
    6-tool registry.
- **Compaction:** the one live local turn on 0.9.12 before the workaround
  logged `ceiling 22556, trigger projected`. The deploy-to-workaround window
  saw almost no local use (6 local rounds), so the scratch runs carry the
  0.9.12 evidence.
- **Endings:** the `turns` table (from 3 Oct) has 3 `unproductive_repeat`
  local Beacon turns, all saved and none streamed, and 1
  `max_turns_exhausted`, which streams.

## 4. Grammar or model? Probes on the 4090

Thinking off, temperature 0, prompts of 1–5k tokens, each prompt sent twice:
- **raw:** `/apply-template`, then `/completion` with no grammar;
- **chat:** `/v1/chat/completions` with native tools.

| Probe | Offered | Raw (model alone) | Chat (with the server's grammar) |
|---|---|---|---|
| Base prompt names `lcm_grep` | bash, read_file, tool_search | `tool_search` "lcm_grep" | same |
| After `select agent` returned the schema | same | `tool_search select agent` again | same |
| After `select cron_create` | same | `tool_search select cron_create` | same |
| Told outright to call `lcm_expand` | same / no tool_search | `select` / `bash echo "not available"` | same |
| History shows an earlier `lcm_grep` call | same | **`<function=lcm_grep>`** | **`tool_search select lcm_grep`** (forced) |
| Same, live 16-tool set | 16 tools | **`lcm_grep`** | **`memory`** (forced) |
| Same, thinking on | 16 tools | **`lcm_grep`** | **`tool_search`** (forced) |
| Same, 27 Aug build's tool schemas | 12 tools | **`lcm_grep`** | **`tool_search`** (forced) |
| Same, streamed like Prometheus | | | **`tool_search`** (forced) |
| `agent` offered | +agent | `agent` | `agent` |

The 4090's server is unchanged since 15 Aug (llama.cpp built from
9d57ce456, model file from 14 Aug, unit file from 15 Aug; process restarted
3 Oct). So the August "lucky guesses" were not a server difference: they were
the mislabels in §Summary 2.

## 5. Endings that never reach Beacon

`ws_server.py:1746-1749` ignores the content of `AssistantTurnComplete`, and
`chat_done` carries no reason. The halt text is saved (LCM), so it appears
after a reload as an ordinary, trusted reply. The endings that yield only
`AssistantTurnComplete` are:

- context pre-flight
- context overflow
- circuit breaker ×4
- empty response twice
- iteration cap
- boundary escape (the warning part)
- repeat guard
- divergence halt

Degenerate output and the round cap stream (#777, #775). Provider errors are
the reverse: they show live as an `error` row and are never saved, because
the turn is rolled back. Telegram, Slack and Discord get every ending, since
they send the final text.

## 6. The regression gate

### 6.1 How it works

- **Integration suites** (`tests/test_local_gate_*.py`,
  `tests/test_local_tool_reachability.py`):
  - **What runs:** the real daemon boots from config as a child process, from
    the checkout under test (`LOCAL_GATE_CHECKOUT`), with the release's own
    shipped template.
  - **The stand-in** (`tests/support/scripted_llama.py`) replaces only the
    model. It keeps the 4090's constraints: 32k window and its 400, calls
    only to offered tools, `tool_calls` reported on a cut-off call, one
    prompt cache. `tests/support/fake_ollama.py` keeps the measured Ollama
    behaviour.
  - **Fixtures:** operator files are synthetic, at live sizes, so round 0 is
    18,595 tokens against 18,634 live. Outbound HTTP goes to a dead proxy,
    and local GPU services are pointed at a dead port.
  - **Grading:** every check reads what the WebSocket client received, or
    what the model server received.
  - **Running the old release:** v0.1.0 binds every interface, so its
    baseline ran inside a private network namespace.
- **Real-model suite** (`scripts/local_gate/`):
  - **Setup:** a scratch daemon from a given checkout, with an isolated config
    built from the operator's (no secrets, loopback, no gateways), and a
    logging proxy to the 4090.
  - **Driving:** each turn is driven like Beacon and waits for live to be
    idle.
  - **Reports:** pass/fail per gate item, plus found / called / correct for
    the release reachability tier.

### 6.2 Results

Integration suites, run with `--runxfail` to show raw outcomes:

| Item | 0.9.12 | 0.9.10 | 0.1.0 (pre-#121) |
|---|---|---|---|
| a sub-agent spawn | fail | fail | fail (no `agent` on the web path then) |
| b deferred tool same turn (cron, wiki, memory, MCP) | 0/4 | 0/4 | 3/4 (MCP did not exist) |
| c no early compaction / finishes after compacting / never over 32k | **fail** / pass / pass | pass / n/a (never needed to compact) / pass | unmeasured (no compaction frames then) |
| d endings reach the client (repeat, empty ×2, pre-flight) | 0/3 | 0/3 | 0/3 (endings partly absent) |
| e measured per component / breakdown visible / within target | pass / fail / fail (6,494 > 5,000) | pass / fail / fail | pass / fail / fail |
| f cloud: every tool offered, prefix stable | pass | pass | fail (old override path) |
| Access, 53 tools | 12 / 53 (the loaded ones) | 12 / 53 | 49 / 51 (the 2 misses: MCP, absent then) |
| Awareness, 53 tools + count | 12 / 53, count fail | 12 / 53 | 49 / 51 |
| Choice: find_me declared / top 3 / near-miss select | 0 / 52 each | 0 / 52 each | 0 / 50 each |
| Context window (11 checks) | 3 pass | 2 pass | — |

On 0.9.12, run as a gate: 32 passed, 258 xfailed. Parity: 182 passed.

Real model, Qwen3.8-27B on the 4090, scratch daemon with the live
(pre-workaround) config:

| Scenario | 0.9.12 | 0.9.10 | 0.9.12, deferral off |
|---|---|---|---|
| a agent | not called (tool_search ×2, then bash/glob) | not called | **called** |
| b cron | not called; 45 rounds, 283 s, bash thrash, ending silent | not called; 51 rounds | **called** |
| b wiki | not called | not called | **called** |
| b memory (lcm_grep) | not called | not called | called (ending not streamed) |
| b MCP (deferred) | not called; 149 rounds | not called | — |
| d iteration cap | ending not streamed | not streamed | — |
| early compaction passes (any scenario) | 8 of 27 (cron), 1 (agent), 1 (MCP) | 0 | 0 |
| c 32k coding session (9 files, unbound chat) | 6 passes, **all 6 early** (ceiling ≈22.3k, so it kept less each time); task finished | 2 passes, 0 early (line 26,214); task finished | — |

Release reachability tier (plain requests, none naming a tool; 12-round cap):
| Request (none names a tool) | Right tool | 0.9.12 as shipped: found / called / correct | 0.9.12, deferral off |
|---|---|---|---|
| "Every weekday at 9am, remind me to check the build." | cron_create | yes / no / no (bash thrash) | yes / yes / yes |
| "Get a helper to look through my workspace…" | agent | yes / no / no; ending not streamed | yes / yes / yes |
| "What did we talk about earlier regarding the release checklist?…" | lcm_* | yes / no / no (bash) | yes / yes / yes |
| "Check my Prometheus wiki for the deploy checklist." | wiki_query | no (never searched; grep/read_file/vault_search) | yes / yes / yes |
| "What background tasks do I have running right now?" | task_list | yes / no / no; tool_search ×5, ending not streamed | yes / yes / yes |
| "Show me my recent chat sessions." | sessions_list | no / no / no (bash) | yes / yes / yes |
| "Keep a to-do list for this conversation…" | todo_write | no tool at all (answered in text) | yes / yes / yes |
| **Total** | | **found 4/7, called 0/7** | **7/7** |

## 7. Context-window resolution

What Prometheus reads:

- **llama.cpp:** `GET /props` → `default_generation_settings.n_ctx`
  (`providers/llama_cpp.py:295-310`, and the backend probe). This is the
  served per-slot window, never `n_ctx_train`. The gate checks both, with
  `n_ctx_train` 262,144 present and a per-slot 16,384 (pass).
- **Ollama:**
  - **The probe** (`providers/backends.py:615-619`) reads `POST /api/show` →
    `model_info.*.context_length`. That is the trained length.
  - **The provider** (`providers/ollama.py:295-330`) uses
    `/v1/chat/completions`, which has no `options`, so no `num_ctx` is ever
    sent.
  - **What gets served** is set when the model loads: from the loader's
    `num_ctx`, otherwise `OLLAMA_CONTEXT_LENGTH`, otherwise a VRAM tier
    ([docs](https://docs.ollama.com/context-length): 4k under 24 GiB, 32k for
    24–48 GiB, 256k above).
  - **Neither Ollama sets `OLLAMA_CONTEXT_LENGTH`.**
  - **This box (Ollama 0.23.0):** `/api/ps` showed the 7b loaded at 32,768,
    equal to its trained length.
  - **The 4090's Ollama (0.17.4):** qwen3.5:9b reports a trained 262,144, so
    Prometheus would budget 262k for a model Ollama serves at its default.
- **Truncation, measured on the 7b at 32k:**
  - an 18.4k-token system prompt kept both its tools and its top line;
  - a 36.8k-token one came back as exactly 32,768 prompt tokens, with no
    error. Its tools survived (the template renders them after the system
    text), but the top of the system prompt was gone.

  So an over-window Ollama prompt silently loses Prometheus's identity and
  rules, and the overflow handling never fires.
- **Where each live model's window comes from:**
  - **27B:** detected (32,768).
  - **mini and gpu_ollama models:** "detected" from the trained length, which
    is the wrong field. 32,768 for qwen2.5 and 262,144 for qwen3.5:9b.
  - **4 Claude models:** override.
  - **23 other reachable cloud models:** fallback (`cloud_default_limit`
    1,000,000).

Real windows for the cloud models that fall back. Most figures are
third-party listings, not vendor docs:

| Model | Real window | Note |
|---|---|---|
| qwen3.8-max | ≈1M (991,808 in + 131,072 out) | [ref](https://www.getmaxim.ai/bifrost/model-library/compare/dashscope/qwen3.8-max.md) |
| qwen3.8-flash | 1M | [Model Studio](https://help.aliyun.com/en/model-studio/qwen3-8-flash) |
| gpt-5.6-luna | 1,050,000 | [ref](https://pricepertoken.com/pricing-page/model/openai-gpt-5.6-luna) |
| gemini-2.5-pro | 1,048,576 | |
| deepseek-v4-pro / -flash | 1M | [ref](https://cometapi.com/models/deepseek/deepseek-v4-flash/) |
| glm-5.3 | ≈1,048,576 | [ref](https://haimaker.ai/models/z-ai/glm-5.3) |
| mimo-v2.6-pro | ≈1,048,576 | [ref](https://www.requesty.ai/models/xiaomi/mimo-v2.6-pro) |
| **grok-4.5** | **500,000** | [ref](https://haimaker.ai/models/xai/grok-4.5): 1M fallback is 2× too big |
| **kimi-k2.6** | **262,144** | [ref](https://www.requesty.ai/models-new/moonshot/kimi-k2.6): 1M fallback is 4× too big |
| **claude-sonnet-4-6** | 1M only with the beta flag, otherwise 200k | [ref](https://www.neowin.net/news/claude-sonnet-46-is-here-with-a-1-million-token-context-window-but-theres-a-catch/) |

Not looked up: qwen3.7-plus, gpt-5-nano, gpt-5.6-sol/terra, gpt-6-astra,
gemini-2.5-flash, kimi-k2.7-code(-highspeed), kimi-k3, glm-5.3-flash(x),
mimo-v2.6-flash. qwen3.8-max's real window is right; its #773 incident
(breaking down near 241k) is a quality limit, not the window.

Overrides and reply room:

- **`context.model_overrides` matches the model id exactly.**
  - The 27B's id is its full GGUF path on the 4090. Renamed or moved, the
    override stops matching with no message. In the gate, `max_tokens` went
    back to 8,192.
  - The live config also has entries no backend serves (Gemma, Qwen 3.5 32B,
    gpt-4o), with no warning.
- **Reply room, three numbers:**
  - The request's `max_tokens` is the per-model output cap (override, then
    route, then table, then 8,192 local, then 4,096). That is what the
    compactor's projected rule counts (`compactor.py:1173`).
  - `context.reserved_output` (2,000 live) is read only by the `/context`
    budget (`context/budget.py:274-320`) and media sizing
    (`gateway/media_services.py:158`).
  - `compaction.reserve_tokens` (4,096) is read only when no cap is passed.
  - The #356 pre-flight leaves no reply room at all.

  In the gate, `/context` (which works on Beacon) showed "Reserved output:
  2,000" while the request asked for 3,000.

## 8. Overhead per component

Round 0 of a local Beacon turn on the live config, in server tokens (the
prior diagnosis, `/tokenize`): 18,634. The pieces:
- tools 4,762, of which Context7 is 1,131 and the vault pair about 500 (both
  operator choices);
- memory 3,232;
- PROMETHEUS.md of the deploy checkout 2,828;
- QwenFormatter prose and JSON line 1,440;
- saved-files list 1,194;
- base 1,137;
- SOUL 1,012;
- ANATOMY 1,004;
- documents 806;
- AGENTS 610;
- skills 222;
- framing 373.

Prometheus-owned (everything but the operator's files and choices): about
7.1k, or about 9.9k counting PROMETHEUS.md. The gate measures the shipped
config at 6,494 Prometheus-owned (12 tools at 3,192).

**Proposed target:**
- Prometheus-owned round-0 overhead ≤ **5,000** tokens, including any
  deferred-tool listing.
- Round 0 on the live config ≤ **12,000** (37% of 32k). That leaves ≥ 16.6k
  for the conversation after a 4,096 reply room.

## 9. Fix plan, in order (nothing written; awaiting approval)

Each item gives the gate items it moves, what it could break, and an estimate
of lines changed. All are local-only unless stated, and none changes a cloud
request (gate f, parity).

1. **Stream every ending to Beacon.** On `AssistantTurnComplete`, the WS
   bridge sends any part of the final text that was not streamed as a
   `chat_delta`, the way `openai_api.py:227-233` already does. `chat_done`
   gains `stop_reason`.
   - **Moves:** d (3/3), and the silent-ending part of every b failure.
   - **Could break:** doubled text if a final message rewrites what streamed
     (it sends only a true suffix); Beacon ignores the new field.
   - **Size:** about 25 lines in `ws_server.py`, plus tests.
2. **Local output cap back to 4,096**, keeping the 25%-of-window clamp.
   Already written, uncommitted, on `fix/local-tool-reach-0913`; re-check
   it against this gate.
   - **Moves:** c, the early-compaction check.
   - **Could break:** a long single reply is cut at 4,096. The #806 refusal
     tells the model to split a cut tool call.
   - **Size:** about 5 lines, plus tests.
3. **Make every registered tool callable on a local turn, without a mid-run
   catalog change.**
   - **The change:** advertise deferred tools as *stubs*: name, one-line
     description and a permissive `{"type": "object"}` schema, built once at
     boot. The server's grammar then allows every name and the model sees
     every name. A stub call is validated against the real schema at
     dispatch. Today a bad argument on tier light is answered with "Invalid
     input for X: …" and no schema (`agent_loop.py:5519`); for a stub it must
     carry the real schema, in the shape the adapter's retry prompt already
     uses (`adapter/retry.py:117-122`).
   - **Why this shape:** the catalog stays byte-stable, so the prompt cache is
     kept. The stubs carry no full schemas, so they cost about 1.2–1.5k
     tokens for about 40 tools, not 9.6k.
   - **Moves:** a, b (4/4), Access (41 more tools), Awareness.
   - **Could break:** the model may call a stub with wrong arguments first
     (one retry). Every local parity golden changes and needs a live
     re-record. The prompt grows until item 6 lands.
   - **Size:** about 80–120 lines (`dynamic_tools.py`, a config flag), plus
     tests and goldens. Validate with the real-model reachability tier before
     any deploy.
   - **Alternative weighed:** select makes the tool callable for the rest of
     the run. The template puts `# Tools` ahead of the system text, so every
     select re-prefills all ~18.6k tokens: about 7.8 s cold on the 4090, per
     select. It also breaks the frozen-catalog rule. Not recommended.
4. **tool_search that finds what it means.**
   - **The changes:**
     - select resolves near-misses ("select cron_create", "agent explorer",
       "cron create");
     - a miss lists the 5 closest names, not all 192 (1,701 tokens);
     - each tool declares 1–2 `find_me` queries next to its definition, and
       search ranks on them.
   - **Moves:** Choice (52 × 3).
   - **Could break:** a near-miss resolving to the wrong tool (only resolve
     when exactly one tool name is a whole word of the query).
   - **Size:** about 60 lines, plus about 100 one-line declarations.
5. **Context windows, loud and consistent.**
   - **The changes:**
     - a startup WARNING, and `/api/status` entries, for overrides that match
       no served model;
     - match overrides by GGUF file name as well as full id;
     - a WARNING and a `source` field ("detected" / "override" / "fallback")
       wherever a window falls back, which Beacon can show;
     - one reply-room rule (the request's `max_tokens` everywhere: `/context`,
       compactor, pre-flight), documented;
     - Ollama on native `/api/chat` with `options.num_ctx` set to the window
       it budgets, read from `/api/ps` when loaded, else the configured window;
     - known windows for cloud models that fall back (at least grok-4.5 and
       kimi-k2.6), or a warning per model.
   - **Moves:** context-window checks (8).
   - **Could break:** the Ollama provider switch changes every Ollama request
     (the backends, the worker routes). Warnings may flag entries Will keeps
     on purpose.
   - **Size:** about 150 lines across `budget.py`, `compactor.py`,
     `output_caps.py`, `backends.py`, `ollama.py` and `server.py`, plus tests.
6. **Cut the overhead to the target.**
   - **The changes:**
     - drop the QwenFormatter prose list and JSON line on tier light (−1,440,
       and it removes the JSON-vs-XML contradiction);
     - include the deploy checkout's PROMETHEUS.md only when the session
       works in that repo (−2,828);
     - saved-files list as a count plus on-demand (−~1,000);
     - shorten the documents section and the `task_create` schema (−~700);
     - stop naming `lcm_*` in the base prompt when they are not callable
       (moot after item 3).
   - **Moves:** e.
   - **Could break:** every local prompt changes, so parity needs a re-record
     and bench (WP-1.2 gate). It also removes the JSON route that the 3
     genuine text-path calls used (moot after item 3).
   - **Size:** about 60 lines, plus goldens.
7. **Telemetry that tells the truth.** The lucky-guess marker and `tool_sets`
   record the tools actually sent in that round, not a recomputation.
   - **Moves:** nothing in the gate; it keeps the next audit honest.
   - **Could break:** readers that assume one set per run.
   - **Size:** about 30 lines.
8. **Live config, after 1–3 ship.** Will's pinned `always_loaded` and the
   live workaround (`agent` added, cap 4,096 by GGUF path) then become
   redundant, and the GGUF-path override is exactly the fragile shape item 5
   warns about. Decide then whether to keep the pins.

## 10. Notes

- The prior diagnosis is on branch `docs/local-tools-diagnosis-2026-10-16`,
  not on main. Its leads held, except "grammar or model": it's both (§4).
- The live workaround from this morning is in place (`agent` always loaded,
  cap 4,096 for the 27B by GGUF path). The fix branch
  `fix/local-tool-reach-0913` holds uncommitted fixes 1 and 2. Neither was
  touched by this audit.
- The daemon's user journal contains a Telegram bot token in plain text, in
  HTTP request-URL lines from early July. It was not checked whether newer
  lines still carry it. The value is not repeated here. Consider rotating the
  token and confirming the logging no longer prints it.
- The real-model reachability tier is meant to run against the 27B on
  scratch before each release (`scripts/local_gate/run_gate.py --scenarios
  r_cron,r_agent,r_history,r_wiki,r_tasks,r_sessions,r_todo`).
