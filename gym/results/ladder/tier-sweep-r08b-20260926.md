# Tier sweep — `r08b`

- model: `Qwen3.5-9B-UD-Q4_K_XL.gguf` via `llama_cpp`, quantization `UD-Q4_K_XL`; the daemon picks tier `light` for it
- suite `ladder-v1` (sha `45df225d3b36`), harness `3bb6703ffbfbf7a5681d01e7ccdcd8740a82a91c`
- run labels: `r08b-sweep-r1-full-20260926`, `r08b-sweep-r1-light-20260926`, `r08b-sweep-r1-off-20260926`, `r08b-sweep-r2-full-20260926`, `r08b-sweep-r2-light-20260926`, `r08b-sweep-r2-off-20260926`, `r08b-sweep-r3-full-20260926`, `r08b-sweep-r3-light-20260926`, `r08b-sweep-r3-off-20260926`
- thinking suppression: `supported`

Each tier is the daemon's own adapter for that tier (only the tier decision is forced).
Tier `off` is never what the daemon uses for a local model — it measures what the server's
own parser does with no adapter behind it. Runs the circuit breaker bumped to another
tier are hybrids, and runs with no verdict (a harness or endpoint error, unscored) are
not the adapter's; both are left out of the main figures (counted below).

## By tier

| tier | runs | task success (95% CI) | accuracy | wrong | format miss | parse-disagreement halts | other halts | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 204 | 189/204 (0.88–0.95) | 189/204 | 9 | 0 | 0 | 6 | 764/799 | 4.12 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1 / 4 | 4.6 | 20650 / 556 | 6.0 |
| light | 204 | 187/204 (0.87–0.95) | 187/204 | 11 | 0 | 0 | 6 | 759/794 | 4.12 | 0.00 | 0.00 / 0.00 | 0.04 | 0.00 | 0.00 | 0 | 4 / 4 | 4.7 | 24007 / 527 | 5.9 |
| full | 204 | 155/204 (0.70–0.81) | 155/202 | 31 | 2 | 6 | 10 | 593/672 | 3.47 | 0.00 | 0.35 / 0.00 | 3.68 | 0.00 | 1.34 | 2 | 8 / 1 | 4.6 | 8206 / 555 | 5.6 |

### single_tool

| tier | runs | task success (95% CI) | accuracy | wrong | format miss | parse-disagreement halts | other halts | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 72 | 68/72 (0.87–0.98) | 68/72 | 4 | 0 | 0 | 0 | 96/100 | 1.40 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 0 | 2.4 | 9960 / 105 | 1.8 |
| light | 72 | 67/72 (0.85–0.97) | 67/72 | 4 | 0 | 0 | 1 | 96/99 | 1.40 | 0.00 | 0.00 / 0.00 | 0.04 | 0.00 | 0.00 | 0 | 0 / 0 | 2.4 | 11282 / 109 | 1.9 |
| full | 72 | 61/72 (0.75–0.91) | 61/71 | 9 | 1 | 0 | 1 | 86/98 | 1.39 | 0.00 | 0.17 / 0.00 | 1.39 | 0.00 | 0.14 | 0 | 0 / 0 | 2.4 | 4168 / 113 | 1.5 |

### multi_step

| tier | runs | task success (95% CI) | accuracy | wrong | format miss | parse-disagreement halts | other halts | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 66 | 62/66 (0.85–0.98) | 62/66 | 4 | 0 | 0 | 0 | 281/293 | 4.52 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 / 0 | 5.0 | 19738 / 464 | 5.0 |
| light | 66 | 60/66 (0.82–0.96) | 60/66 | 6 | 0 | 0 | 0 | 268/280 | 4.33 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1 / 0 | 5.0 | 22844 / 420 | 4.8 |
| full | 66 | 50/66 (0.64–0.84) | 50/65 | 11 | 1 | 2 | 2 | 252/285 | 4.45 | 0.00 | 0.41 / 0.00 | 4.45 | 0.00 | 1.70 | 0 | 1 / 0 | 5.6 | 7718 / 429 | 4.4 |

### file_edit

| tier | runs | task success (95% CI) | accuracy | wrong | format miss | parse-disagreement halts | other halts | tool-call success | calls / run | repairs / run | adapter retries / aborts per run | calls from text / run | text calls missed / run | XML-markup turns / run | breaker halts | denied / blocked | rounds | tokens in / out | time s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|
| off | 66 | 59/66 (0.80–0.95) | 59/66 | 1 | 0 | 0 | 6 | 387/406 | 6.70 | 0.00 | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1 / 4 | 6.6 | 33225 / 1141 | 11.5 |
| light | 66 | 60/66 (0.82–0.96) | 60/66 | 1 | 0 | 0 | 5 | 395/415 | 6.86 | 0.00 | 0.00 / 0.00 | 0.08 | 0.00 | 0.00 | 0 | 3 / 4 | 7.0 | 39051 / 1088 | 11.3 |
| full | 66 | 44/66 (0.55–0.77) | 44/66 | 11 | 0 | 4 | 7 | 255/289 | 4.76 | 0.00 | 0.50 / 0.00 | 5.41 | 0.00 | 2.29 | 2 | 7 / 1 | 6.1 | 13100 / 1164 | 11.1 |

## Paired by task (task success; bumped and no-verdict runs left out)

Every tier ran the same tasks, so each task is compared with itself: the mean per-task
difference in pass rate, a 95% bootstrap interval over tasks, and how many tasks flip.

| comparison | tasks | mean difference | 95% interval | tasks flipped |
|---|---:|---:|---|---:|
| light − off | 68 | -0.010 | -0.039 – +0.020 | 3 |
| full − light | 68 | -0.157 | -0.221 – -0.098 | 10 |
| full − off | 68 | -0.167 | -0.235 – -0.103 | 11 |

## Left out and infrastructure

| tier | bumped runs (left out) | no verdict (left out) | stopped by (main runs) | provider HTTP retries |
|---|---:|---:|---|---:|
| off | 0 | 0 | done 198, round_cap 6 | 0 |
| light | 0 | 0 | done 198, round_cap 6 | 0 |
| full | 0 | 0 | circuit_breaker 2, done 188, parse_disagreement 6, round_cap 7, tool_call_cap 1 | 0 |

Counters: *adapter retries / aborts* are the adapter's own decisions after a rejected call;
*calls from text* are tool calls the adapter recovered from the reply's text; *text calls
missed* are calls tier off left in the text that light/full would have recovered;
*XML-markup turns* are replies with no structured tool call that carry `<tool_call>` /
`<function=` markup — the replies the adapter is asked to read; a reply that carries
markup beside a structured call is not counted. Provider HTTP retries are the
transport's, not the adapter's.
