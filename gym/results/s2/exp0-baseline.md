# Gym report — series `s2`, experiment `exp0-baseline`

> Fresh baseline for series s2 against gym/tasksets/v2.yaml, with DUAL SCORING (emission vs execution) live. This is a NEW series — s1 numbers do not carry over (different predicates: re-tagged task_create, expect_tool_args_present, added download_file/browser dict-wrap shapes). All series-2 arms compare against THIS baseline only. Harvest capture is OFF for the baseline (it is on for the experiment arms per the sprint); the baseline measures the unmodified current pipeline on the frozen v2 set.


## Totals

- Runs: **72**
- Emission pass (raw model call): 57/72 (79%)
- Execution pass (post-adapter): 57/72 (79%)
- Adapter value (execution − emission): **+0%** (0 run(s) saved by repair/unwrap)
- Adapter repairs: 0 · malformed drops: 0 · breaker trips: 0 · loop feedback retries: 0
- ⚠️ harness-level errors (timeouts/crashes): 5

## By category (emission → execution, Δ = adapter value)

| category | emission | execution | Δ |
|---|---|---|---|
| argshape | 11/15 (73%) | 11/15 (73%) | +0% |
| control | 21/24 (88%) | 21/24 (88%) | +0% |
| lcm | 3/3 (100%) ⚠️thin | 3/3 (100%) | +0% |
| namespace | 6/6 (100%) | 6/6 (100%) | +0% |
| resilience | 12/12 (100%) | 12/12 (100%) | +0% |
| task_create | 4/12 (33%) | 4/12 (33%) | +0% |

## By task (emission → execution)

| task | emission | execution | sample failure |
|---|---|---|---|
| argshape_browser_action | 3/3 (100%) | 3/3 (100%) |  |
| argshape_download_destination | 0/3 (0%) | 0/3 (0%) | harness timeout after 240s |
| argshape_grep_two_params | 2/3 (67%) | 2/3 (67%) | final text lacks 'x.py' (got: '<|tool_call>call:bash{command:<|"|>grep -r "TODO" |
| argshape_task_list_status | 3/3 (100%) | 3/3 (100%) |  |
| argshape_write_json_content | 3/3 (100%) | 3/3 (100%) |  |
| control_bash_echo | 1/3 (33%) | 1/3 (33%) | final text lacks 'gym control alpha' (got: "OK. I've run the command.") |
| control_edit_file | 3/3 (100%) | 3/3 (100%) |  |
| control_glob | 3/3 (100%) | 3/3 (100%) |  |
| control_grep | 3/3 (100%) | 3/3 (100%) |  |
| control_multi_step | 3/3 (100%) | 3/3 (100%) |  |
| control_no_arg_tool | 3/3 (100%) | 3/3 (100%) |  |
| control_read_file | 3/3 (100%) | 3/3 (100%) |  |
| control_write_file | 2/3 (67%) | 2/3 (67%) | no successful 'write_file' call (tools attempted: ['bash']) |
| lcm_query_punctuation | 3/3 (100%) | 3/3 (100%) |  |
| namespace_task_list_not_bash | 3/3 (100%) | 3/3 (100%) |  |
| namespace_tool_search | 3/3 (100%) | 3/3 (100%) |  |
| resilience_collapse_arc_replay | 3/3 (100%) | 3/3 (100%) |  |
| resilience_misnamed_tool | 3/3 (100%) | 3/3 (100%) |  |
| resilience_missing_file | 3/3 (100%) | 3/3 (100%) |  |
| resilience_nonexistent_tool | 3/3 (100%) | 3/3 (100%) |  |
| task_create_agent_explicit | 1/3 (33%) | 1/3 (33%) | no task_create call matched required args {'type': 'local_agent'} (got: [{'descr |
| task_create_agent_implicit | 0/3 (0%) | 0/3 (0%) | no successful 'task_create' call (tools attempted: ['bash', 'bash', 'bash', 'bas |
| task_create_bash_explicit | 1/3 (33%) | 1/3 (33%) | harness timeout after 240s |
| task_create_watch | 2/3 (67%) | 2/3 (67%) | no successful task_create call supplied 'watch_pattern' |

_Sample bar: ≥30 runs/arm for a verdict; cells under 6 runs flagged thin. Dual deterministic scoring (emission vs execution); no LLM judging._