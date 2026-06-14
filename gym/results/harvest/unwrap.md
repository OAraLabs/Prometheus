# Gym report — series `harvest`, experiment `unwrap`

> Harvest run: drive the v1 task set with conservative dict-wrap unwrapping on, pair-capture configured to the gym harvest DB. Each phantom-nesting the model emits that the adapter losslessly unwraps becomes a schema_repair training pair. Yield ≈ the repair count (exp3 fired ~10 across 66 runs).


## Totals

- Runs: **66** — passed 51/66 (77%)
- Adapter repairs: 5 · malformed drops: 0 · breaker trips: 0 · loop feedback retries: 0

## By category

| category | pass rate |
|---|---|
| argshape | 5/9 (56%) |
| control | 22/24 (92%) |
| lcm | 3/3 (100%) ⚠️thin |
| namespace | 6/6 (100%) |
| resilience | 12/12 (100%) |
| task_create | 3/12 (25%) |

## By task

| task | pass rate | sample failure |
|---|---|---|
| argshape_grep_two_params | 1/3 (33%) | final text lacks 'x.py' (got: "No matches for 'TODO' were found in the directory `/tmp/pro |
| argshape_task_list_status | 1/3 (33%) | task_list.status should be a string, got dict: {'status': 'failed'} |
| argshape_write_json_content | 3/3 (100%) |  |
| control_bash_echo | 3/3 (100%) |  |
| control_edit_file | 3/3 (100%) |  |
| control_glob | 3/3 (100%) |  |
| control_grep | 1/3 (33%) | final text lacks 'b.py' (got: '') |
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
| task_create_agent_implicit | 0/3 (0%) | task_create.prompt is a serialized JSON blob, not plain text: '{"prompt":"Research practic |
| task_create_bash_explicit | 0/3 (0%) | task_create.command should be a string, got dict: {'command': 'sleep 1 && echo done'}; no  |
| task_create_watch | 3/3 (100%) |  |

_Sample bar: ≥30 runs/arm for a verdict; cells under 6 runs flagged thin. Deterministic predicate scoring; no LLM judging._