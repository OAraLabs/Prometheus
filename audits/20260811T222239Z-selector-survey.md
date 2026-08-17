# Selector survey — the §2d shape, applied to profiles, tool_result_max, and the compactor

**Date:** 2026-08-11 · **Main:** 043977e · **Method:** for each selector that sits
between *what exists* and *what the consumer gets*, classify every test as
NEAR-side (the selector's own logic against fixtures) or FAR-side (asserts what
the consumer actually receives through the real path). The deferred-loading
failure (§2d, PR #152 → #153) is the reference: the selector was *well* tested
the whole time, and that thoroughness is what hid the gap — a reviewer asking
"is it covered?" got an honest yes and stopped. "There are tests for X" is not
an answer; the answer is what the tests assert and on which side they sit.

**Far-side exemplars already in the suite** (the pattern to copy):
`tests/support/advertisement.py::advertised_names()` (reads through the real
deferred-loading selector), and
`tests/test_context_compactor.py::test_run_loop_sends_compacted_render_view`
(drives `run_async`, asserts on `provider.requests[0].messages` — the actual
consumer artifact — with a negative twin).

## Verdicts at a glance

| # | Selector | Mechanism tests | Far side | Verdict |
|---|----------|-----------------|----------|---------|
| 1 | profiles | 4 files, thorough | **does not exist** | **ORPHAN** — the filter has no caller; §1, not §2d |
| 2 | tool_result_max + turn budget | thorough, incl. dispatch-level | wiring YES, **notice contract NO** | §2d on the contract dimension — three dead notices proven live |
| 3a | ContextCompactor | thorough | **YES** — true far-side pair | The good example. No action. |
| 3b | microcompaction | direct-call only | wiring yes, config threading **CLI-only** | §1b config keys dead on every daemon surface, masked by value coincidence |

---

## 1. profiles — an orphan selector wearing four test files

**The selector:** `config/profiles.py` — `AgentProfile` (tools allowlist,
exclude_tools, bootstrap_files, subsystems, max_tool_schemas) +
`filter_tools_by_profile(schemas, profile)`. Live config: `profiles.default:
full`, custom dir wired.

**What tests exist (all NEAR-side or UI-side):**
- `test_profiles.py` — ProfileStore load/override + `filter_tools_by_profile`
  logic on fixture schema lists. Selector logic only.
- `test_wiring.py:2240` — "filter_tools_by_profile actually filters a real
  schema list." It feeds the function a real list; it does not (cannot) assert
  any caller exists. A wiring test for an unwired function.
- `test_api_profiles.py` — GET/POST `/api/profiles` (these routes ARE wired —
  to the UI).
- `test_gateway_parity.py` — `/profile` command parity across tg/slack/discord.

**The far side, checked:**
- `filter_tools_by_profile` has **zero call sites in src/** — only tests call it.
- `/profile <name>` on all three gateways stores `self._active_profile_name`;
  **nothing reads it back** except the same command's display line. It is a label.
- `web/server.py:203` reads `profiles.default` into `app.state.active_profile`;
  consumers are the status payload, the `is_active` flag in the listing, and the
  set-route. UI state only.
- `profile.tools`, `bootstrap_files`, `subsystems`, `max_tool_schemas`:
  **no consumer anywhere.** Switching profiles changes what `/profile` prints
  and what Beacon highlights. The model's advertised toolset, bootstrap files
  and subsystems never change.

**Secondary finding — name drift with no far side to catch it:** the builtin
`coder` profile lists `file_read`, `file_write`, `file_edit`, `lsp`; the real
registry (51 tools, via `create_tool_registry({})`) has **none of those names**
(`read_file`/`write_file`/`edit_file`). If profiles were wired tomorrow, coder
would filter to a fraction of its intended set — §1d ("the documented name the
registry rejects"), undetectable today because there is no far side at all.

**Disposition (decision needed, not a test):** wire it or delete it. If wired:
the far-side test is `advertised_names()`-shaped — build the advertised set
through a non-full profile and assert the model-visible list; plus a guard that
every name in every builtin profile exists in the registry (kills the drift).
If deleted: the four test files go with it, and `/profile` should say it does
nothing rather than pretend.

---

## 2. tool_result_max + tool_results_turn_budget — wiring tested, contract untested

**The selector:** per-result `ToolResultTruncator` (live `tool_result_max:
4000` tokens ≈ 16,000 chars head-kept; strategy by tool name — bash tail,
read_file head+tail, grep top-20, **everything else head-only**) applied at
`agent_loop.py:2956` before injection; then `_apply_cross_result_budget`
(live 8,000 tokens across a turn's results, read-only trimmed first).

**NEAR-side:** `test_context.py` (truncator strategies, token estimation),
`test_cross_result_budget.py` (direct calls to `_apply_cross_result_budget`:
under/over budget, errors skipped, read-only first, zero-budget passthrough).

**FAR-side that exists:** `test_loop_hardening_3.py` (H1) goes through the real
`_execute_tool_call` and asserts the injected block is truncated when the cap
is set, errors included, plus the max=0 back-compat twin. Both daemon
constructions and the CLI pass the value (daemon.py:524, daemon.py:1446,
__main__.py:1374 — verified). The *wiring* question of §2d is answered here.

**FAR-side that does NOT exist — the notice contract.** Nothing anywhere
asserts that what survives truncation tells the model the truth or gives it a
workable next action. Proven live (2026-08-11 audit, pre-#154):

1. **Head-keep beheads every tool's own tail notice.** vault_read's 48k
   truncation notice was never seen by the model once — it trailed a payload
   cut at 16k. Fixed for vault_read only (#154: head-positioned notice + window
   sized under the cap + an identity test against `ToolResultTruncator(4000)`).
   Every other tool that appends tail-positioned state is still exposed, and no
   generic test exists.
2. **The trailer lies about size.** `[truncated at N tokens]` counts the
   payload it was handed, not the artifact: a 72k-char page pre-capped to 48k
   reported "truncated at 12041 tokens" for an ~18,000-token page. Untested.
3. **The turn-budget notice prescribes the impossible.** "use lcm_expand or
   re-read for full content": `lcm_expand` expands LCM summary nodes and cannot
   recover a tool result truncated before injection (it was never stored);
   re-read returns the same head. No test asserts a notice's advice is
   executable.
4. **The strategy table is name-keyed with no registry guard.** bash /
   read_file / grep all exist today; nothing pins the table to registry names,
   so a tool rename silently demotes it to head-only default — the same
   name-keyed-selector class as §1d, one drift away.

**What a far-side test looks like here:** run a tool through
`_execute_tool_call` whose output exceeds the cap and carries tail-positioned
state; assert the surviving text names the true artifact size and a next action
that, when taken, actually recovers content (execute the advice in the test).
Plus: assert every name in the truncator's strategy table is in
`create_tool_registry({})`.

---

## 3a. ContextCompactor — the far side done right (no action)

`context/compactor.py`, live `compaction.enabled: true`. The suite has the
true far-side pair: `test_run_loop_sends_compacted_render_view` drives
`run_async` with a capturing provider and asserts the provider-received
messages start with the summary marker, the compacted span is absent, and the
session's own list was never mutated; the negative twin asserts full history
without a compactor. Two-loop threading (the "one line, months of config-dark"
web-path fix) is in place at daemon.py:1455 and drift-guarded by
`test_web_bridge_loop_parity.py::test_no_new_drift_between_the_two_loops`.
Residual (accepted): the trigger depends on `estimate_tokens` (chars/4), so
threshold accuracy vs a real tokenizer is unasserted — a calibration concern,
not a selector gap.

## 3b. Microcompaction — wired everywhere, configurable almost nowhere

`_microcompact_old_results` (compact tool results older than N turns in the
live message list) is called from the shared `run_loop` (agent_loop.py:1023),
so the *mechanism* runs on every surface. `test_microcompact.py` covers its
logic by direct call (threshold, recency, errors skipped, LCM-not-ingested
long-keep) — NEAR-side, MagicMock context.

**The finding:** the three config keys exist in the live config
(`context.microcompact_after_turns: 3`, `microcompact_keep_chars: 200`,
`microcompact_keep_chars_no_lcm: 500`) and are threaded **only by the CLI
construction** (__main__.py:1376-78). `AgentLoop.__init__` cannot accept them
and neither daemon LoopContext construction passes them — every daemon surface
(Telegram, web/Beacon, Bridge) runs on the dataclass defaults. Today the
defaults happen to EQUAL the live config values (3/200/500), so behavior
matches by coincidence: edit the key and the CLI obeys while every daemon
surface silently ignores it. §1b (config key with no reader on the paths that
matter) masked by value coincidence — the parity drift-guard is structurally
blind to it because it compares the two daemon constructions to each other,
and both lack the kwargs equally.

Also unasserted on the far side: `microcompact_on_cloud: False` (tier-gated
skip — §9's config-tier dimension), and no test drives 4+ turns through
`run_loop` asserting old tool results arrive compacted in the provider payload
while recent ones survive.

**Fix shape:** thread the three keys through AgentLoop and both daemon
constructions (the #128-class one-liner, three times), extend the parity
guard's kwarg list so the next selector can't repeat this, and add one
far-side test: recording provider, 4 turns, assert turn-1's tool result
arrives stubbed and turn-4's arrives whole.

---

## The pattern, once more

Three selectors, three different failure geometries under one law: profiles
tests a selector that selects for nobody (§1); tool_result_max selects
correctly and *describes its selection falsely* (§2d on the contract, three
dead notices); microcompaction selects correctly everywhere but obeys its
config on one surface out of four (§1b/§9). The compactor shows the cure in
this same codebase: one test that stands where the consumer stands. Every
verdict above was reachable only by asking "who receives this?" — never by
asking "is this tested?"
