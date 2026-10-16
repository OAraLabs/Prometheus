# Local tools fix stack: state (2026-10-16)

The audit: `docs/audits/local-tools-audit-2026-10-16/README.md` (on #810).

## Merge queue

Will's standing approval covers **this stack only, in this order**. Each PR merges only when CI is green on its exact head SHA (squash, pinned with `--match-head-commit`), then the next is retargeted or updated. Anything outside this list, or a PR whose scope changes after Will has seen it, waits for him. No tag, release or deploy.

| Order | PR | What | Head | Checks |
|---|---|---|---|---|
| 1 | #810 | the gate and the audit | `48defb6d` | CI 13/13 green; Will is merging it |
| 2 | #811 | fix 1: every ending reaches Beacon | `5303d6e6` | CI 14/14 green (parity included); local full suite and parity clean. After #810: `gh pr edit 811 --base main`, then `gh pr update-branch 811` |
| 3 | #812 | parity: side calls match by kind, in any order | `2166180f` | local: replay 12/12, stability identical, full suite green (2 known host failures), 121 harness tests; CI running |
| 4 | #813 | fix 2: llama.cpp cap back to 4096, 10 goldens re-recorded | `18e6dfd7` | local: replay 12/12, stability identical, full suite green (2 known host failures); CI running |
| 5 | not opened | 5a: context windows loud and consistent | `6af54634` | local: replay 12/12, stability identical; full suite running |
| 6 | not opened | 5b: Ollama on native /api/chat with num_ctx (its own PR; a release may ship 5a without it) | `b3e84637` | real-model check on this box's Ollama 7b passed (tool call, plain reply, thinking control, over-window prompt refused before Ollama); 2 goldens re-recorded with Ollama live; local checks queued |
| 7 | not opened | 4: tool_search near-miss select, 5-name miss list, find_me | `4fc55aac` | local checks queued |
| 8 | not opened | 7: telemetry records the tools a round sent | `ebe7b614` | local checks queued |
| 9 | not opened | 3: every tool callable locally (stubs) | `d3be2a79` | hermetic gate 289 passed, 5 xfailed (fix 1's d, fix 6's e); 11 goldens re-recorded; local checks queued |
| 10 | not opened | 6: cut round-0 overhead | WIP `2515a694` | see below |

The two known host failures on this box are `test_proc_net_tcp6_addresses` and `test_the_real_session_bus_agrees_with_the_probe`.

## What's left

- **Fix 6:**
  - finish the 12-request real-model tier after cuts A, B, C, E, and compare it with the before run;
  - commit the thresholds;
  - re-record the goldens once, at the end;
  - run the full checks, and open the PR with both tier scores.
- **Fixes 5a, 5b, 4, 7, 3:** finish the local checks (replay, stability, full suite, ruff, mypy gate), then push and open each PR stacked on the one before.
- **Merges** in the order above.
- **Release check** on main with everything merged: the full gate plus the 12-request tier, and a one-page summary for Will.

## Decisions

- **Fix 3 is stubs**, not a `call_tool` dispatcher. In the A/B on the 27B (12 requests), the right tool was called 11/12 either way; the right arguments on the first try, 11/12 against 9/12.
  - Full schema for a fixed ten: bash, read_file, write_file, edit_file, grep, glob, web_search, web_fetch, agent, ask_user.
  - Every other tool is a stub, listing an enum's values when there are 8 or fewer.
  - The block is sorted and byte-identical across rounds.
  - `always_loaded` is inert under `local_catalog: stubs`, and applies only to `names`.
- **Parity:** main-turn requests keep any-order exact matching, which was intentional. Side calls (titles, summaries) match by kind.
- **Fix 6:**
  - **Cuts made:**
    - A: no Qwen prose tool list on tier light;
    - B: no deploy-directory project files in the daemon's prompt (a bound workspace still brings its own);
    - C: saved files as a count;
    - E: stub string types dropped, stub lines capped at 80 characters.
  - **D (Documents library) left as is:** it is behaviour guidance, not a listing.
  - **F skipped:** it would change cloud schemas and need a paid hosted re-record.
  - **Thresholds:** round 0 on Will's config ≤ 13,500, Prometheus-owned (gate) ≤ 7,000.
  - **Revisit after launch:** D, F and the 6,200/5,000 targets, with plan-behaviour tests and a cloud A/B first.
  - **Supersession note:** Will's earlier fix 6 message (D and F approved, owned ≤ 6,200) was superseded by his later answers once D proved to be behaviour guidance and F to need a paid re-record. The stack follows the later answers unless he says otherwise.
- **Measured (the 27B, daemon run from a deploy-like directory):**
  - Round 0 on Will's config: 19,198 before fix 6, 13,340 after A, B, C, E.
  - Gate Prometheus-owned: 8,901 before, 6,840 after.
  - Tier before fix 6: found 12/12, called 12/12, right tool first 11/12, right arguments 12/12, 0 schema refusals.
- **Deploy runbook (0.9.13):** retire the 10-16 live workaround.
  - Remove `agent` from `tools.deferred_loading.always_loaded`, and the per-model `max_output_tokens: 4096` override keyed by the 27B's GGUF path. That is 4 lines in all; the rest of the config stays.
  - Back up the config first, and restore that backup if 0.9.13 is rolled back.
