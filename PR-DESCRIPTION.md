# Per-message Agent | Chat mode (Sprint B / Piece 2)

**Branch:** `feat/per-message-mode` off origin/main (`66fd820`)
**Status:** PR-ready. MODERATE — real engine work on the chat generation path. **MERGE GATE: the dormant default is byte-identical to today's always-agentic path.**

## What it adds
A per-message `mode` (`"agent"` | `"chat"`) threaded from the send paths into `run_loop`. `mode="chat"` runs a plain **no-tools** turn; absent / anything-else = today's always-agentic default. Unlocks the Beacon Agent|Chat composer toggle (a later UI sprint). Force-search remains **deferred**.

## Phase 0 finding → scope expanded (per your call)
The survey's "empty the one `tool_schema` list = no-tools turn" seam was clean for **cloud** but **not local**: at tier light/full, tools also reach the model via a **GBNF grammar** generated from the full registry at boot and stored on the **shared** provider (`daemon.py:342` → `LlamaCppProvider._grammar`), and it's *actively applied to empty-tools requests* (`llama_cpp.py:212/232`). Emptying `tool_schema` didn't clear it, and clearing it per-turn = shared-state mutation (the concurrency violation). **Resolution (you chose "expand scope"):** a per-call `ApiMessageRequest.suppress_tools` flag the provider honors to drop the grammar — concurrency-safe and **structurally tool-free at all tiers**.

## How it works
- `run_loop` gains a keyword-only `mode: str = "agent"`. `tools_enabled = mode != "chat"` — **only `"chat"` disables tools; unknown/None → agent**, so an unrecognized value can never silently drop tools. When disabled: (a) the single `tool_schema` derivation (`agent_loop.py:669-673`) stays empty → no prompt injection, no payload tools; (b) `suppress_tools=True` on the request → llama.cpp drops the GBNF grammar. Default (agent) leaves **both** untouched → byte-identical.
- `mode` is a per-call **parameter**, threaded WS `send_message` / REST `send_chat` → `dispatch_user_message` → `_handle_send_message` → `_run_agent` → `run_loop`. **Never** stored on the shared `loop_context` (the concurrency rule — proven by a test).
- Send paths: **absent mode → "agent"** (never an error); **malformed explicit mode → 400** (REST) / error frame (WS).

## Files
- `providers/base.py` — `ApiMessageRequest.suppress_tools: bool = False` (default = today's behavior).
- `engine/agent_loop.py` — `run_loop` `mode` param + gated `tool_schema` derivation + `suppress_tools` on the request.
- `providers/llama_cpp.py` — drop the GBNF grammar when `suppress_tools` (the local-tier structural fix).
- `web/ws_server.py` — parse + validate mode (WS), thread through `dispatch_user_message`/`_handle_send_message`/`_run_agent` → `run_loop`.
- `web/server.py` — parse + validate mode on REST `/api/chat/send`.
- `tests/test_per_message_mode.py` *(new, 8 tests)*. Plus **3 stale test-doubles widened** to mirror the new optional param — **signature only, no assertion/behavior change**: `test_api_chat_send` (a bridge + a handler stub), `test_wire_contract` (a `fake_run_loop`, 2 occurrences).

## Merge gate & tests (no `--no-verify`)
- **Default byte-identical (PRIMARY):** no-mode turn → tools offered + `suppress_tools=False` (unit), and the **entire existing suite passes** (only the 3 doubles' signatures widened).
- chat → empty tools + `suppress_tools=True`; unknown mode → agent; **concurrency**: two `run_loop`s on the SAME shared context (one agent, one chat) don't cross-talk; llama.cpp drops the grammar iff `suppress_tools`; REST absent→agent / chat→threads / malformed→400.
- **Full suite: 2990 passed, 0 failed.**
- **Live (daemon on branch):** malformed→400; chat-mode turn = single plain assistant reply (no tool round); agent-mode turn = multi-round tool loop; agent+chat fired concurrently both 200. The structural tool-free guarantee is **unit-proven** (grammar-drop test); live confirms end-to-end wiring + the behavioral contrast.

## Follow-ups
- **Beacon composer-UI sprint** — the model switcher (Piece 1) + this Agent|Chat toggle, built as one composer row.
- **Force-search** deferred — no clean force seam (needs provider `tool_choice` / synthetic `tool_use`); separate sprint.

## Out of scope (honored)
- No loop-body / execution-path restructuring — only the one `tool_schema` derivation is gated, plus a per-call suppress flag. No mode on shared state. No Beacon work. Default (no-mode) behavior unchanged.
