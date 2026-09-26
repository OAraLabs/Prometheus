# Tier sweep — `r09b-ornith`

- model: `Ornith-1.5-9B-Q4_K_M.gguf` via `llama_cpp`, quantization `Q4_K_M`; the daemon picks tier `full` for it
- suite `ladder-v1` (sha `45df225d3b36`), harness `3bb6703ffbfbf7a5681d01e7ccdcd8740a82a91c`
- run labels: `r09b-ornith-sweep-r1-full-20260926`, `r09b-ornith-sweep-r1-light-20260926`, `r09b-ornith-sweep-r1-off-20260926`, `r09b-ornith-sweep-r2-full-20260926`, `r09b-ornith-sweep-r2-light-20260926`, `r09b-ornith-sweep-r2-off-20260926`, `r09b-ornith-sweep-r3-full-20260926`, `r09b-ornith-sweep-r3-light-20260926`, `r09b-ornith-sweep-r3-off-20260926`
- thinking suppression: `supported`

Each tier is the daemon's own adapter for that tier (only the tier decision is forced).
Tier `off` is never what the daemon uses for a local model — it measures what the server's
own parser does with no adapter behind it. Runs the circuit breaker bumped to another
tier are hybrids, and runs with no verdict (a harness or endpoint error, unscored) are
not the adapter's; both are left out of the main figures (counted below).

## By tier

| tier | runs | task success (95% CI) | accuracy | wrong | format miss | parse-disagreement halts | other halts | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 202 | 194/202 (0.92–0.98) | 194/202 | 2 | 0 | 0 | 6 | 791/798 | 4.22 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 1 | 4.8 | 21041 / 605 | 6.4 |
| light | 203 | 191/203 (0.90–0.97) | 191/203 | 4 | 0 | 0 | 8 | 811/823 | 4.32 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 3 | 4.8 | 24345 / 614 | 6.6 |
| full | 204 | 130/204 (0.57–0.70) | 130/189 | 46 | 15 | 2 | 11 | 603/658 | 3.51 | 0.00 | 0.23 / 0.00 | 3.51 | 0.00 | 2.51 | 3 | 1 / 0 | 4.6 | 10018 / 754 | 7.5 |

### single_tool

| tier | runs | task success (95% CI) | accuracy | wrong | format miss | parse-disagreement halts | other halts | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 71 | 71/71 (0.95–1.00) | 71/71 | 0 | 0 | 0 | 0 | 124/125 | 1.83 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 0 | 2.8 | 10675 / 167 | 2.6 |
| light | 72 | 71/72 (0.93–1.00) | 71/72 | 1 | 0 | 0 | 0 | 125/127 | 1.78 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 0 | 2.8 | 12318 / 154 | 2.2 |
| full | 72 | 58/72 (0.70–0.88) | 58/61 | 2 | 11 | 1 | 0 | 107/108 | 1.50 | 0.00 | 0.00 / 0.00 | 1.50 | 0.00 | 0.69 | 0 | 0 / 0 | 2.7 | 3773 / 152 | 1.7 |

### multi_step

| tier | runs | task success (95% CI) | accuracy | wrong | format miss | parse-disagreement halts | other halts | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 66 | 64/66 (0.90–0.99) | 64/66 | 2 | 0 | 0 | 0 | 266/267 | 4.24 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 0 | 5.1 | 20608 / 487 | 5.2 |
| light | 66 | 62/66 (0.85–0.98) | 62/66 | 2 | 0 | 0 | 2 | 285/285 | 4.62 | 0.00 | 0.00 / 0.00 | 0.02 | 0.00 | 0.00 | 0 | 0 / 0 | 5.3 | 25461 / 594 | 6.4 |
| full | 66 | 41/66 (0.50–0.73) | 41/62 | 18 | 4 | 0 | 3 | 219/227 | 3.55 | 0.00 | 0.08 / 0.00 | 3.55 | 0.00 | 2.61 | 1 | 1 / 0 | 4.8 | 6879 / 635 | 5.8 |

### file_edit

| tier | runs | task success (95% CI) | accuracy | wrong | format miss | parse-disagreement halts | other halts | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 65 | 59/65 (0.81–0.96) | 59/65 | 0 | 0 | 0 | 6 | 401/406 | 6.82 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 1 | 6.6 | 32803 / 1204 | 11.9 |
| light | 65 | 58/65 (0.79–0.95) | 58/65 | 1 | 0 | 0 | 6 | 401/411 | 6.83 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 3 | 6.5 | 36534 / 1142 | 11.6 |
| full | 66 | 31/66 (0.35–0.59) | 31/66 | 26 | 0 | 1 | 8 | 277/323 | 5.67 | 0.00 | 0.64 / 0.02 | 5.68 | 0.00 | 4.39 | 2 | 0 / 0 | 6.6 | 19969 / 1530 | 15.5 |

## Paired by task (task success; bumped and no-verdict runs left out)

Every tier ran the same tasks, so each task is compared with itself: the mean per-task
difference in pass rate, a 95% bootstrap interval over tasks, and how many tasks flip.

| comparison | tasks | mean difference | 95% interval | tasks flipped |
|---|---:|---:|---|---:|
| light − off | 68 | -0.025 | -0.064 – +0.015 | 2 |
| full − light | 68 | -0.299 | -0.377 – -0.221 | 23 |
| full − off | 68 | -0.324 | -0.407 – -0.245 | 23 |

## Left out and infrastructure

| tier | bumped runs (left out) | no verdict (left out) | stopped by (main runs) | provider HTTP retries |
|---|---:|---:|---|---:|
| off | 2 | 0 | done 196, round_cap 6 | 0 |
| light | 1 | 0 | done 195, round_cap 8 | 0 |
| full | 0 | 0 | circuit_breaker 3, done 191, parse_disagreement 2, round_cap 7, tool_call_cap 1 | 0 |

Counters: *adapter retries / aborts* are the adapter's own decisions after a rejected call;
*calls from text* are tool calls the adapter recovered from the reply's text; *text calls
missed* are calls tier off left in the text that light/full would have recovered;
*XML-markup turns* are replies with no structured tool call that carry `<tool_call>` /
`<function=` markup — the replies the adapter is asked to read; a reply that carries
markup beside a structured call is not counted. Provider HTTP retries are the
transport's, not the adapter's.
