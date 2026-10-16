## What this fixes

Fix 6 of the local tool-calling audit: cut the overhead of round 0. **Stacked on fix 3 (FIX3_PR).**

With every tool offered (fix 3), round 0 of a local turn on the operator's config was 19,198 server tokens: 59% of the 4090's 32k window before a word of conversation. The approved cuts:

| Cut | What | Round-0 tokens |
|---|---|---|
| A | On tier light, QwenFormatter adds no prose tool list or JSON example. The tools ride the request natively and the template renders them, and the JSON example contradicted Qwen's XML format. | −1,383 |
| B | The daemon's boot prompt carries no project files from its own cwd. That is its deploy checkout, so this is ~2.7k tokens of Prometheus's developer guide. A session bound to a workspace (`/workspace`) still gets that workspace's files, appended per run, so a session working in the Prometheus repo still reads its PROMETHEUS.md. | −2,813 (the file and its section header) |
| C | Saved files appear as a count and a location, not a listing. The model lists them with glob when it needs a name, and the prompt no longer changes whenever a file does. | −1,111 |
| E | Stub descriptions are capped at 80 characters. | −77 |

Not done, per Will (revisited after launch with plan-behaviour tests and a cloud A/B):
- **D:** the Documents library section is behaviour guidance, not a listing.
- **F:** tightening `grep`/`write_file`/`bash` would change cloud schemas and need a paid hosted re-record.

**E also dropped the type from string stub arguments**, and the real-model tier caught it. Untyped, the grammar admits any JSON value for an argument, and the 27B packed every argument into the first one: `cron_create(name={name, schedule, command})`. Right arguments fell to 5/12, with 27 refusals. Stub arguments stay typed, and `test_every_stub_argument_is_typed` pins it.

## The 12-request real-model tier, before and after

The 4090's 27B, a scratch daemon behind the logging proxy, run from a deploy-like directory on the operator's config. The same 12 plain requests as fix 3's A/B. Scored on found, called, right tool first, right arguments (the first call to the right tool; an environment failure like no TTS engine is not a wrong argument), schema refusals, tokens and prefix-cache reuse.

| | Before (fix 3) | After (A, B, C, E) |
|---|---|---|
| Found | 12/12 | 12/12 |
| Right tool called | 12/12 | 12/12 |
| Right tool called first | 11/12 | **12/12** |
| Right arguments | 12/12 | 12/12 |
| Schema refusals | 0 | 0 |
| **Round 0, server tokens** | **19,198** | **13,813** |
| Total prompt tokens over the tier | 1,223,146 (66 rounds) | 941,237 (70 rounds) |
| Round-to-round prefix reuse ≥ 0.9 | 52/54 | 51/56 |

The one first-call miss before was r_cron checking `cron_list` before creating the job. The reuse misses are rounds after a sub-agent (one llama.cpp slot) or after older history was rewritten.

Round 0 after, by component (server tokens):
- **Operator's files:** Memory (MEMORY.md + USER.md) 3,232, ANATOMY.md 1,004, SOUL.md 1,001, AGENTS.md 599.
- **Prometheus's:**
  - Tool schemas 5,007, of which the ambient ten are full and the rest are stubs.
  - Base sections 1,339: Doing tasks, Using your tools, Skills, System, Environment, Executing actions with care, Tone, Reasoning.
  - Documents library 788, saved files 84.
- **Template framing:** 721.

## Gate thresholds (Will, 0.9.13)

- **Prometheus-owned round-0 overhead ≤ 7,400** (`PROMETHEUS_OWNED_TARGET`). Measured 7,253 on the gate's config, down from 8,901. This flips `test_e_prometheus_owned_overhead_is_within_target`.
- **Round 0 on the operator's config ≤ 14,000.** Measured 13,813. Checked by the real-model suite (`run_gate.py --overhead`, `round0_within_target`).
- The 5,000 target is revisited after launch with an A/B. `test_e_the_client_can_read_the_overhead_breakdown` stays xfail; it is not in this PR's scope.

## The gate's coding session is longer

The cuts take the hermetic gate's round 0 from 21,122 to 16,232 stand-in tokens. Before, the overhead plus three fresh results crossed the 26,214 compaction line by round 5. Now, once results are trimmed, the session grows about 200 tokens a round. The nine-module session ended at 26,001 without a pass, so `test_c_the_session_finishes_its_task_after_compacting` skipped.

The per-result size stays, because the early-compaction check needs each result under 2,458 tokens, and so does the config: the session gets longer instead. With 26 modules it compacts once, at module 18 (26,281 tokens), and reads the other 8 after compacting, with about 1.7k tokens of slack. The skip message now names that knob. The real-model coding scenario keeps its own nine modules (`scripts/local_gate/isolate.py`).

## Parity goldens

Every local request's system prompt and stubs change, so the 11 scenarios with a local request are re-recorded once, here. The primary was live (27B), and the Ollama side was live for model_switch and repaired_tool_call; every other side was a strict stand-in. compaction and repaired_tool_call each took a second attempt. hosted_route was not re-recorded: it replays to PARITY as committed.

## Checks run

- Hermetic gate (`test_local_gate_*`, `test_local_tool_reachability`): 292 passed, 1 xfailed (`test_e_the_client_can_read_the_overhead_breakdown`, not in this PR's scope). `test_e_prometheus_owned_overhead_is_within_target` flips to pass.
- Full suite (CI's command): 14,422 passed; 2 failures that fail the same way on main on this host (environment, not this change); coverage 81.9%. It ran before the gate-session commit and the rebase onto fix 3's doctor commit; after both, the gate, doctor-row, stubs and advertisement tests: 501 passed, 1 xfailed; parity 12/12, stability identical. CI runs the full suite on this head.
- Parity `replay`: 12/12 PARITY (on this head). `stability`: IDENTICAL across 2 runs, run twice.
- `ruff check src/`: clean. Mypy gate: OK.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
