# Force-search — per-call tool_choice (Sprint B follow-up to Piece 2)

**Branch:** `feat/force-search` off origin/main (`3a506fe`)
**Status:** PR-ready — **do NOT merge** without a go (brief authorized green, not a merge). TDD: the red spec `tests/test_force_search.py` is now **green (18/18)**, full suite **3012 passed, 0 failed**, no `--no-verify`.

## What it adds
Generalizes Piece-2's per-call `suppress_tools: bool` into a **four-state `tool_choice` lever** on `ApiMessageRequest`, mirroring the cloud API so one field drives both tiers:

| tool_choice | meaning | local (GBNF) | cloud |
|---|---|---|---|
| `"auto"` | today's agent path | boot grammar as-is | native auto |
| `"none"` | no tools (chat) | grammar dropped | no tools |
| `"required"` | must call some tool | **prose branch dropped** (`root ::= tool-call`) | native `required` |
| `{"tool": X}` | must call tool X | force a tool (local fallback¹) | native `{"tool": X}` |

`mode` stays sugar: `agent→auto`, `chat→none`. **The CRUX resolved clean:** the boot grammar is a top-level alternation (`root ::= tool-call | prose`, enforcer.py:130), so `required` is the clean inverse of `suppress_tools` — drop the `prose` branch, no root rewrite.

## Merge gate (the non-negotiable, held)
`auto` / `none` are **byte-identical** to before: `run_loop` resolves an explicit `tool_choice` else `mode` (unknown→auto, so it can never silently drop tools); the `tool_schema` gate and `suppress_tools` are set exactly as in Piece 2. `tool_choice` is a **per-call `run_loop` param, never on the shared `loop_context`** (a no-crosstalk test proves it). The entire existing suite passes (3 stale test-doubles widened for the new optional kwarg — signature only, no behavior change).

## Files
- `src/prometheus/api/tool_choice.py` *(new)* — the union vocabulary + `resolve_mode_to_tool_choice` + `normalize_tool_choice` (validates malformed / unknown-tool → ValueError).
- `src/prometheus/providers/base.py` — `ApiMessageRequest.tool_choice` (default `"auto"`).
- `src/prometheus/adapter/enforcer.py` — `generate_grammar(require_tool_use=…, only_tool=…)` variants + `grammar_admits_tool`.
- `src/prometheus/engine/agent_loop.py` — `run_loop(tool_choice=…)` resolution + sets `request.tool_choice`.
- `src/prometheus/web/server.py` + `ws_server.py` — parse + **validate** tool_choice against the live registry (unknown/malformed → 400 / WS error frame; absent → auto), thread through `dispatch_user_message`→`_handle_send_message`→`_run_agent`→`run_loop`.
- `src/prometheus/providers/llama_cpp.py` — `_grammar_for(request)`: per-call grammar SELECTION (auto=boot, none/suppress=dropped, required/{tool}=prose-dropped), cached, **never mutates the boot grammar**.
- Tests: `tests/test_force_search.py` *(the spec)*, `tests/conftest.py` *(harness: spy + cloud doubles, `make_request`/`run_turn`)*, `tests/test_force_search_provider.py` *(real-provider grammar selection)*.

## ⚠️ Live-wiring still pending (the spec's doubles stand in for these)
The contract + plumbing + validation + the **local `required`** path are live. Two pieces of real-provider wiring remain (each defined by a conftest double):
1. **Cloud native tool_choice + synthetic-prefill fallback** — the real anthropic/openai providers don't yet read `request.tool_choice`; a live cloud `required`/`{tool}` request won't set the native param yet. (`SpyCloudProvider` defines the contract.)
2. **True specific-tool grammar at the local tier** — `{"tool": X}` currently falls back to "force *some* tool" locally (prose dropped); a single-alternative grammar needs the enforcer + schemas wired onto `LlamaCppProvider`. ¹

Both are clean, well-scoped follow-ups; neither is gated by the spec.

## Verify
`python3 -m pytest -q` → **3012 passed, 0 failed** (prior 2990 + 18 spec + 4 provider; 7 pre-existing warnings, unchanged). `tests/test_force_search.py` 18/18; `test_per_message_mode.py` still 8/8.
