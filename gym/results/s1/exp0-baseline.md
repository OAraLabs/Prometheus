# Gym report — series `s1`, experiment `exp0-baseline`

> Baseline for series s1 against gym/tasksets/v1.yaml, run on main after the invariants-vs-policy split (511ed2f) and the FTS5 sanitizer fix (c59fe34) landed — i.e. this baseline measures the CURRENT pipeline, not the one that produced the April–June telemetry. The production "before" for the collapse and task_create shapes is documented in FINDINGS-TOOLCALLING-2026-06-10.md.


## Totals

- Runs: **66** — passed 54/66 (82%)
- Adapter repairs: 0 · malformed drops: 0 · breaker trips: 0 · loop feedback retries: 0
- ⚠️ harness-level errors (timeouts/crashes): 2

## By category

| category | pass rate |
|---|---|
| argshape | 8/9 (89%) |
| control | 21/24 (88%) |
| lcm | 3/3 (100%) ⚠️thin |
| namespace | 6/6 (100%) |
| resilience | 12/12 (100%) |
| task_create | 4/12 (33%) |

## By task

| task | pass rate | sample failure |
|---|---|---|
| argshape_grep_two_params | 3/3 (100%) |  |
| argshape_task_list_status | 2/3 (67%) | harness timeout after 240s |
| argshape_write_json_content | 3/3 (100%) |  |
| control_bash_echo | 1/3 (33%) | final text lacks 'gym control alpha' (got: "OK. I've run the command.") |
| control_edit_file | 3/3 (100%) |  |
| control_glob | 3/3 (100%) |  |
| control_grep | 2/3 (67%) | final text lacks 'b.py' (got: '<|tool_call>call:bash{command:<|"|>grep -r "needle_7Q" /tmp |
| control_multi_step | 3/3 (100%) |  |
| control_no_arg_tool | 3/3 (100%) |  |
| control_read_file | 3/3 (100%) |  |
| control_write_file | 3/3 (100%) |  |
| lcm_query_punctuation | 3/3 (100%) |  |
| namespace_task_list_not_bash | 3/3 (100%) |  |
| namespace_tool_search | 3/3 (100%) |  |
| resilience_collapse_arc_replay | 3/3 (100%) |  |
| resilience_misnamed_tool | 3/3 (100%) |  |
| resilience_missing_file | 3/3 (100%) |  |
| resilience_nonexistent_tool | 3/3 (100%) |  |
| task_create_agent_explicit | 0/3 (0%) | no task_create call matched required args {'type': 'local_agent'} (got: [{'description': ' |
| task_create_agent_implicit | 0/3 (0%) | no successful 'task_create' call (tools attempted: ['bash', 'bash', 'bash', 'bash']) |
| task_create_bash_explicit | 1/3 (33%) | no task_create call matched required args {'type': 'local_bash'} (got: [{'command': 'sleep |
| task_create_watch | 3/3 (100%) |  |

_Sample bar: ≥30 runs/arm for a verdict; cells under 6 runs flagged thin. Deterministic predicate scoring; no LLM judging._