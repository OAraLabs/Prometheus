# Tier sweep — `r27b-pq2`

- model: `Ternary-Bonsai-2-27B-PQ2_0.gguf` via `llama_cpp`, quantization `PQ2_0`; the daemon picks tier `full` for it
- suite `ladder-v1` (sha `45df225d3b36`), harness `4b5c294b263604e8e26b4c26999bf84e3db0326b`
- run labels: `r27b-pq2-sweep-r1-full-20260925`, `r27b-pq2-sweep-r1-light-20260925`, `r27b-pq2-sweep-r1-off-20260925`, `r27b-pq2-sweep-r2-full-20260925`, `r27b-pq2-sweep-r2-light-20260925`, `r27b-pq2-sweep-r2-off-20260925`, `r27b-pq2-sweep-r3-full-20260925`, `r27b-pq2-sweep-r3-light-20260925`, `r27b-pq2-sweep-r3-off-20260925`
- thinking suppression: `supported`

Each tier is the daemon's own adapter for that tier (only the tier decision is forced).
Tier `off` is never what the daemon uses for a local model — it measures what the server's
own parser does with no adapter behind it. Runs the circuit breaker bumped to another
tier are hybrids and are left out of the main figures (counted below).

## By tier

| tier | runs | task success (95% CI) | accuracy | format miss | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 203 | 188/203 (0.88–0.95) | 188/203 | 0 | 846/863 | 4.57 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 5 | 5.0 | 21547 / 505 | 9.2 |
| light | 204 | 186/204 (0.86–0.94) | 186/204 | 0 | 847/854 | 4.50 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 2 / 0 | 4.9 | 24656 / 552 | 10.4 |
| full | 204 | 135/204 (0.59–0.72) | 135/199 | 5 | 496/521 | 2.66 | 0.00 | 0.11 / 0.01 | 2.66 | 0.00 | 1.90 | 2 | 7 / 0 | 4.5 | 6195 / 340 | 6.8 |

### single_tool

| tier | runs | task success (95% CI) | accuracy | format miss | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 72 | 69/72 (0.88–0.99) | 69/72 | 0 | 133/137 | 1.92 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 0 | 2.9 | 11533 / 121 | 3.2 |
| light | 72 | 68/72 (0.87–0.98) | 68/72 | 0 | 125/129 | 1.82 | 0.00 | 0.00 / 0.00 | 0.01 | 0.00 | 0.00 | 0 | 1 / 0 | 2.8 | 12359 / 136 | 3.5 |
| full | 72 | 64/72 (0.80–0.94) | 64/69 | 3 | 105/111 | 1.56 | 0.00 | 0.08 / 0.00 | 1.56 | 0.00 | 0.94 | 0 | 0 / 0 | 3.0 | 3331 / 109 | 2.9 |

### multi_step

| tier | runs | task success (95% CI) | accuracy | format miss | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 66 | 58/66 (0.78–0.94) | 58/66 | 0 | 319/321 | 5.08 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 0 | 5.8 | 23545 / 413 | 8.0 |
| light | 66 | 60/66 (0.82–0.96) | 60/66 | 0 | 311/313 | 4.95 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1 / 0 | 5.6 | 26099 / 416 | 8.3 |
| full | 66 | 46/66 (0.58–0.79) | 46/64 | 2 | 211/214 | 3.32 | 0.00 | 0.03 / 0.00 | 3.32 | 0.00 | 2.09 | 0 | 2 / 0 | 5.3 | 7116 / 346 | 7.0 |

### file_edit

| tier | runs | task success (95% CI) | accuracy | format miss | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 65 | 61/65 (0.85–0.98) | 61/65 | 0 | 394/405 | 7.00 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 5 | 6.4 | 30612 / 1024 | 17.2 |
| light | 66 | 58/66 (0.78–0.94) | 58/66 | 0 | 411/412 | 6.97 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 0 | 6.5 | 36627 / 1143 | 20.1 |
| full | 66 | 25/66 (0.27–0.50) | 25/66 | 0 | 180/196 | 3.21 | 0.00 | 0.21 / 0.03 | 3.21 | 0.00 | 2.76 | 2 | 5 / 0 | 5.4 | 8398 / 586 | 10.9 |

## Paired by task (task success; bumped runs left out)

Every tier ran the same tasks, so each task is compared with itself: the mean per-task
difference in pass rate, a 95% bootstrap interval over tasks, and how many tasks flip.

| comparison | tasks | mean difference | 95% interval | tasks flipped |
|---|---:|---:|---|---:|
| light − off | 68 | -0.015 | -0.064 – +0.034 | 6 |
| full − light | 68 | -0.250 | -0.338 – -0.172 | 19 |
| full − off | 68 | -0.265 | -0.353 – -0.176 | 21 |

## Left out and infrastructure

| tier | bumped runs (left out) | stopped by (main runs) | provider HTTP retries |
|---|---:|---|---:|
| off | 1 | done 191, repeat_halt 1, round_cap 11 | 0 |
| light | 0 | done 193, round_cap 10, tool_call_cap 1 | 0 |
| full | 0 | circuit_breaker 2, done 171, empty_response 7, parse_disagreement 21, round_cap 3 | 0 |

Counters: *adapter retries / aborts* are the adapter's own decisions after a rejected call;
*calls from text* are tool calls the adapter recovered from the reply's text; *text calls
missed* are calls tier off left in the text that light/full would have recovered;
*XML-markup turns* are replies carrying `<tool_call>` / `<function=` markup. Provider HTTP
retries are the transport's, not the adapter's.
