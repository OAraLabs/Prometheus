# Gym report — series `s2`, experiment `unwrap-corpus`

> adapter_unwrap arm on the 63-task wrap-prone corpus with dual scoring. Run with --harvest to also stamp gym_harvest-tagged pairs. The emission→execution delta here IS the adapter's measured value; the eval-set baseline (s2-exp0) showed ~0 delta because grammar suppresses wrap on natural tasks.


## Totals

- Runs: **63**
- Emission pass (raw model call): 26/63 (41%)
- Execution pass (post-adapter): 49/63 (78%)
- Adapter value (execution − emission): **+37%** (23 run(s) saved by repair/unwrap)
- Adapter repairs: 25 · malformed drops: 0 · breaker trips: 0 · loop feedback retries: 0

## By category (emission → execution, Δ = adapter value)

| category | emission | execution | Δ |
|---|---|---|---|
| harvest_dict_wrap_unwrap | 3/17 (18%) | 15/17 (88%) | +71% |
| harvest_fuzzy_rename | 2/14 (14%) | 5/14 (36%) | +21% |
| harvest_json_stuffed_string | 4/10 (40%) | 10/10 (100%) | +60% |
| harvest_missing_discriminator | 7/12 (58%) | 9/12 (75%) | +17% |
| harvest_type_coercion | 10/10 (100%) | 10/10 (100%) | +0% |

## By task (emission → execution)

| task | emission | execution | sample failure |
|---|---|---|---|
| hv_coerce_0000 | 1/1 (100%) | 1/1 (100%) |  |
| hv_coerce_0001 | 1/1 (100%) | 1/1 (100%) |  |
| hv_coerce_0002 | 1/1 (100%) | 1/1 (100%) |  |
| hv_coerce_0003 | 1/1 (100%) | 1/1 (100%) |  |
| hv_coerce_0004 | 1/1 (100%) | 1/1 (100%) |  |
| hv_coerce_0005 | 1/1 (100%) | 1/1 (100%) |  |
| hv_coerce_0006 | 1/1 (100%) | 1/1 (100%) |  |
| hv_coerce_0007 | 1/1 (100%) | 1/1 (100%) |  |
| hv_coerce_0008 | 1/1 (100%) | 1/1 (100%) |  |
| hv_coerce_0009 | 1/1 (100%) | 1/1 (100%) |  |
| hv_dictwrap_cmd_0000 | 0/1 (0%) | 1/1 (100%) |  |
| hv_dictwrap_cmd_0001 | 0/1 (0%) | 1/1 (100%) |  |
| hv_dictwrap_cmd_0002 | 0/1 (0%) | 1/1 (100%) |  |
| hv_dictwrap_cmd_0003 | 0/1 (0%) | 1/1 (100%) |  |
| hv_dictwrap_cmd_0004 | 0/1 (0%) | 1/1 (100%) |  |
| hv_dictwrap_cmd_0005 | 0/1 (0%) | 1/1 (100%) |  |
| hv_dictwrap_grep_0000 | 0/1 (0%) | 0/1 (0%) | no successful 'grep' call (tools attempted: ['grep', 'grep', 'grep']) |
| hv_dictwrap_grep_0001 | 0/1 (0%) | 1/1 (100%) |  |
| hv_dictwrap_grep_0002 | 1/1 (100%) | 1/1 (100%) |  |
| hv_dictwrap_grep_0003 | 1/1 (100%) | 1/1 (100%) |  |
| hv_dictwrap_grep_0004 | 1/1 (100%) | 1/1 (100%) |  |
| hv_dictwrap_grep_0005 | 0/1 (0%) | 0/1 (0%) | no successful 'grep' call (tools attempted: ['grep', 'grep', 'grep']) |
| hv_dictwrap_status_0000 | 0/1 (0%) | 1/1 (100%) |  |
| hv_dictwrap_status_0001 | 0/1 (0%) | 1/1 (100%) |  |
| hv_dictwrap_status_0002 | 0/1 (0%) | 1/1 (100%) |  |
| hv_dictwrap_status_0003 | 0/1 (0%) | 1/1 (100%) |  |
| hv_dictwrap_status_0004 | 0/1 (0%) | 1/1 (100%) |  |
| hv_fuzzy_0000 | 1/1 (100%) | 1/1 (100%) |  |
| hv_fuzzy_0001 | 0/1 (0%) | 1/1 (100%) |  |
| hv_fuzzy_0002 | 0/1 (0%) | 1/1 (100%) |  |
| hv_fuzzy_0003 | 0/1 (0%) | 1/1 (100%) |  |
| hv_fuzzy_0004 | 0/1 (0%) | 0/1 (0%) | no successful 'task_get' call (tools attempted: ['tool_search']) |
| hv_fuzzy_0005 | 0/1 (0%) | 0/1 (0%) | no successful 'task_stop' call (tools attempted: ['tool_search', 'tool_search',  |
| hv_fuzzy_0006 | 0/1 (0%) | 0/1 (0%) | no successful 'task_output' call (tools attempted: ['tool_search', 'tool_search' |
| hv_fuzzy_0007 | 0/1 (0%) | 0/1 (0%) | no successful 'sessions_list' call (tools attempted: ['tool_search']) |
| hv_fuzzy_0008 | 0/1 (0%) | 0/1 (0%) | no successful 'sessions_list' call (tools attempted: ['tool_search']) |
| hv_fuzzy_0009 | 0/1 (0%) | 0/1 (0%) | no successful 'web_search' call (tools attempted: ['tool_search', 'tool_search'] |
| hv_fuzzy_0010 | 0/1 (0%) | 0/1 (0%) | no successful 'web_search' call (tools attempted: ['tool_search', 'tool_search', |
| hv_fuzzy_0011 | 0/1 (0%) | 0/1 (0%) | no successful 'web_fetch' call (tools attempted: ['tool_search', 'tool_search',  |
| hv_fuzzy_0012 | 0/1 (0%) | 0/1 (0%) | no successful 'read_file' call (tools attempted: ['read_file']) |
| hv_fuzzy_0013 | 1/1 (100%) | 1/1 (100%) |  |
| hv_jsonstuff_0000 | 0/1 (0%) | 1/1 (100%) |  |
| hv_jsonstuff_0001 | 0/1 (0%) | 1/1 (100%) |  |
| hv_jsonstuff_0002 | 0/1 (0%) | 1/1 (100%) |  |
| hv_jsonstuff_0003 | 1/1 (100%) | 1/1 (100%) |  |
| hv_jsonstuff_0004 | 1/1 (100%) | 1/1 (100%) |  |
| hv_jsonstuff_0005 | 1/1 (100%) | 1/1 (100%) |  |
| hv_jsonstuff_0006 | 0/1 (0%) | 1/1 (100%) |  |
| hv_jsonstuff_0007 | 0/1 (0%) | 1/1 (100%) |  |
| hv_jsonstuff_0008 | 0/1 (0%) | 1/1 (100%) |  |
| hv_jsonstuff_0009 | 1/1 (100%) | 1/1 (100%) |  |
| hv_missingdisc_0000 | 1/1 (100%) | 1/1 (100%) |  |
| hv_missingdisc_0001 | 1/1 (100%) | 1/1 (100%) |  |
| hv_missingdisc_0002 | 0/1 (0%) | 0/1 (0%) | no successful 'task_create' call (tools attempted: ['glob', 'bash', 'bash', 'bas |
| hv_missingdisc_0003 | 0/1 (0%) | 1/1 (100%) |  |
| hv_missingdisc_0004 | 1/1 (100%) | 1/1 (100%) |  |
| hv_missingdisc_0005 | 1/1 (100%) | 1/1 (100%) |  |
| hv_missingdisc_0006 | 0/1 (0%) | 0/1 (0%) | no successful 'task_create' call (tools attempted: none) |
| hv_missingdisc_0007 | 1/1 (100%) | 1/1 (100%) |  |
| hv_missingdisc_0008 | 1/1 (100%) | 1/1 (100%) |  |
| hv_missingdisc_0009 | 0/1 (0%) | 1/1 (100%) |  |
| hv_missingdisc_0010 | 1/1 (100%) | 1/1 (100%) |  |
| hv_missingdisc_0011 | 0/1 (0%) | 0/1 (0%) | no successful 'task_create' call (tools attempted: ['tool_search', 'glob', 'glob |

_Sample bar: ≥30 runs/arm for a verdict; cells under 6 runs flagged thin. Dual deterministic scoring (emission vs execution); no LLM judging._