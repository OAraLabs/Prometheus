# Model ladder — r09b-ornith-smoke-v2-20260926

- suite: `ladder-v1` (sha `b71c4bc2c390`), run label `r09b-ornith-smoke-v2-20260926`
- model: `Ornith-1.5-9B-Q4_K_M.gguf` via `llama_cpp`
- quantization: `Q4_K_M` (gguf-filename)
- adapter: tier `full`, base strictness `MEDIUM`
- KV cache: k=unknown v=unknown (unreported)
- judge: `none` (pinned: False)
- runs: 54; live web: off

## By class

Accuracy counts only runs with a verdict on the answer (pass ÷ pass + fail). A reply
without its `ANSWER:` line is a **format miss**: scored anyway when its bare last line
still gives the answer (and counted in *answer line missing*), otherwise recorded as a
`format_miss` verdict — never as a wrong answer.

| class | runs | pass | fail | format miss | unscored | error | accuracy (95% CI) | answer line missing | tool-call success | repairs | rounds (mean) | tokens in / out (mean) | time s (mean) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|
| qa | 18 | 13 | 1 | 1 | 3 | 0 | 13/14 (0.69–0.99) | 1/15 | 3/3 | 0 | 1.3 | 1081 / 135 | 1.2 |
| single_tool | 15 | 13 | 0 | 2 | 0 | 0 | 13/13 (0.77–1.00) | 2/15 | 21/21 | 0 | 2.6 | 2462 / 134 | 1.3 |
| multi_step | 12 | 9 | 3 | 0 | 0 | 0 | 9/12 (0.47–0.91) | 0/3 | 38/38 | 0 | 4.5 | 5412 / 400 | 3.7 |
| file_edit | 9 | 3 | 6 | 0 | 0 | 0 | 3/9 (0.12–0.65) | — | 20/23 | 0 | 4.0 | 7536 / 603 | 5.7 |
| **all** | 54 | 38 | 10 | 3 | 3 | 0 | 38/48 (0.66–0.88) | 3/33 | 82/85 | 0 | 2.8 | 3502 / 272 | 2.6 |

## By difficulty (pass / decided)

| class | easy | medium | hard |
|---|---:|---:|---:|
| qa | 3/3 | 2/2 | 8/9 |
| single_tool | 9/9 | 2/2 | 2/2 |
| multi_step | 3/3 | 5/6 | 1/3 |
| file_edit | 1/3 | 1/3 | 1/3 |

## Runs

| task | class | difficulty | verdict | answer line | decided by | rounds | tool calls ok/all | repairs | tokens in / out | time s | why (if not pass) |
|---|---|---|---|:---:|---|---:|---:|---:|---|---:|---|
| qa-arith-mult | qa | easy | pass | ✓ | predicates | 1 | 0/0 | 0 | 801 / 23 | 0.4 |  |
| qa-arith-mult | qa | easy | pass | ✓ | predicates | 1 | 0/0 | 0 | 801 / 22 | 0.3 |  |
| qa-arith-mult | qa | easy | pass | ✓ | predicates | 1 | 0/0 | 0 | 801 / 22 | 0.2 |  |
| qa-explain-sky-blue | qa | medium | unscored | — | judge | 1 | 0/0 | 0 | 788 / 79 | 0.7 | judge unavailable: judge disabled for this run |
| qa-explain-sky-blue | qa | medium | unscored | — | judge | 1 | 0/0 | 0 | 788 / 90 | 0.8 | judge unavailable: judge disabled for this run |
| qa-explain-sky-blue | qa | medium | unscored | — | judge | 1 | 0/0 | 0 | 788 / 66 | 0.6 | judge unavailable: judge disabled for this run |
| qa-roman-numeral | qa | medium | pass | ✓ | predicates | 1 | 0/0 | 0 | 804 / 24 | 0.3 |  |
| qa-roman-numeral | qa | medium | pass | ✓ | predicates | 1 | 0/0 | 0 | 804 / 46 | 0.4 |  |
| qa-roman-numeral | qa | medium | format_miss | ✗ | predicates | 2 | 0/1 | 0 | 1740 / 234 | 2.4 | format miss: the committed answer cannot be read (no 'answer:' line; the last line reads '{"name": "python3", "arguments": {"code": "print(i |
| qa-weekday-of-date | qa | hard | pass | ✓ | predicates | 3 | 1/2 | 0 | 2861 / 155 | 1.6 |  |
| qa-weekday-of-date | qa | hard | pass | ✓ | predicates | 2 | 1/1 | 0 | 1758 / 96 | 1.1 |  |
| qa-weekday-of-date | qa | hard | pass | ✓ | predicates | 2 | 1/1 | 0 | 1727 / 66 | 0.9 |  |
| qa-py-mutable-default | qa | hard | pass | ✓ | predicates | 1 | 0/0 | 0 | 839 / 273 | 2.2 |  |
| qa-py-mutable-default | qa | hard | pass | ✓ | predicates | 1 | 0/0 | 0 | 839 / 314 | 2.5 |  |
| qa-py-mutable-default | qa | hard | fail | ✓ | predicates | 1 | 0/0 | 0 | 839 / 297 | 2.4 | ANSWER '1' is a wrong value (not /1[\s,]+2[\s,]+1[\s,]+3/) |
| qa-percent-up-down | qa | hard | pass | ✓ | predicates | 1 | 0/0 | 0 | 824 / 158 | 1.3 |  |
| qa-percent-up-down | qa | hard | pass | ✓ | predicates | 1 | 0/0 | 0 | 824 / 213 | 1.7 |  |
| qa-percent-up-down | qa | hard | pass | ✓ | predicates | 1 | 0/0 | 0 | 824 / 256 | 2.1 |  |
| st-read-config-port | single_tool | easy | pass | ✓ | predicates | 3 | 2/2 | 0 | 2687 / 127 | 1.3 |  |
| st-read-config-port | single_tool | easy | pass | ✓ | predicates | 3 | 2/2 | 0 | 2694 / 133 | 1.3 |  |
| st-read-config-port | single_tool | easy | pass | ✓ | predicates | 4 | 2/2 | 0 | 3850 / 126 | 1.5 |  |
| st-read-manifest-build-tag | single_tool | easy | pass | ✓ | predicates | 4 | 2/2 | 0 | 3897 / 129 | 1.5 |  |
| st-read-manifest-build-tag | single_tool | easy | pass | ✓ | predicates | 2 | 1/1 | 0 | 1788 / 80 | 0.8 |  |
| st-read-manifest-build-tag | single_tool | easy | pass | ✓ | predicates | 4 | 2/2 | 0 | 3895 / 135 | 1.5 |  |
| st-read-meeting-owner | single_tool | easy | pass | ✓ | predicates | 2 | 1/1 | 0 | 1883 / 62 | 0.7 |  |
| st-read-meeting-owner | single_tool | easy | pass | ✓ | predicates | 2 | 1/1 | 0 | 1883 / 82 | 0.9 |  |
| st-read-meeting-owner | single_tool | easy | pass | ✓ | predicates | 2 | 1/1 | 0 | 1882 / 76 | 0.8 |  |
| st-grep-defines-function | single_tool | medium | pass | ✓ | predicates | 3 | 2/2 | 0 | 3173 / 347 | 3.0 |  |
| st-grep-defines-function | single_tool | medium | format_miss | ✗ | predicates | 2 | 1/1 | 0 | 1834 / 125 | 1.2 | format miss: the committed answer cannot be read (no 'answer:' line; the last line reads 'The grep shows three occurrences. Let me examine e |
| st-grep-defines-function | single_tool | medium | pass | ✓ | predicates | 3 | 2/2 | 0 | 3017 / 270 | 2.4 |  |
| st-buried-log-503 | single_tool | hard | pass | ✓ | predicates | 2 | 1/1 | 0 | 1815 / 93 | 1.0 |  |
| st-buried-log-503 | single_tool | hard | format_miss | ✗ | predicates | 1 | 0/0 | 0 | 825 / 122 | 1.0 | format miss: the committed answer cannot be read (no 'answer:' line; the last line reads 'Let me look at the file first') |
| st-buried-log-503 | single_tool | hard | pass | ✓ | predicates | 2 | 1/1 | 0 | 1800 / 96 | 0.9 |  |
| ms-find-csv-sum-write | multi_step | easy | pass | — | predicates | 4 | 3/3 | 0 | 3927 / 211 | 2.0 |  |
| ms-find-csv-sum-write | multi_step | easy | pass | — | predicates | 6 | 5/5 | 0 | 6720 / 402 | 3.8 |  |
| ms-find-csv-sum-write | multi_step | easy | pass | — | predicates | 6 | 5/5 | 0 | 9235 / 855 | 7.7 |  |
| ms-semver-latest-codename | multi_step | medium | pass | — | predicates | 4 | 3/3 | 0 | 4177 / 363 | 3.4 |  |
| ms-semver-latest-codename | multi_step | medium | fail | — | predicates | 3 | 2/2 | 0 | 2865 / 193 | 1.8 | expected latest_codename.txt to be a regular file |
| ms-semver-latest-codename | multi_step | medium | pass | — | predicates | 5 | 4/4 | 0 | 5901 / 490 | 4.3 |  |
| ms-config-chain-3hop | multi_step | medium | pass | ✓ | predicates | 4 | 3/3 | 0 | 4012 / 241 | 2.4 |  |
| ms-config-chain-3hop | multi_step | medium | pass | ✓ | predicates | 4 | 3/3 | 0 | 4045 / 288 | 2.6 |  |
| ms-config-chain-3hop | multi_step | medium | pass | ✓ | predicates | 4 | 3/3 | 0 | 4021 / 256 | 2.4 |  |
| ms-csv-join-region-total | multi_step | hard | fail | — | predicates | 6 | 3/4 | 0 | 10140 / 960 | 9.0 | expected north_total.txt to be a regular file |
| ms-csv-join-region-total | multi_step | hard | pass | — | predicates | 4 | 3/3 | 0 | 5073 / 417 | 3.9 |  |
| ms-csv-join-region-total | multi_step | hard | fail | — | predicates | 4 | 1/1 | 0 | 4822 / 123 | 1.7 | the loop stopped the turn (parse_disagreement) |
| fe-fix-last-n | file_edit | easy | pass | — | acceptance | 6 | 3/5 | 0 | 10899 / 664 | 6.9 |  |
| fe-fix-last-n | file_edit | easy | fail | — | acceptance | 3 | 1/1 | 0 | 2944 / 219 | 2.2 | acceptance failed (5 run, 2 failed, 0 errors) |
| fe-fix-last-n | file_edit | easy | fail | — | acceptance | 2 | 1/1 | 0 | 1811 / 194 | 1.7 | acceptance failed (5 run, 2 failed, 0 errors) |
| fe-mutable-default-tags | file_edit | medium | fail | — | acceptance | 2 | 1/1 | 0 | 1889 / 94 | 0.9 | acceptance failed (4 run, 4 failed, 0 errors) |
| fe-mutable-default-tags | file_edit | medium | fail | — | acceptance | 3 | 2/2 | 0 | 3481 / 356 | 3.3 | acceptance failed (4 run, 4 failed, 0 errors) |
| fe-mutable-default-tags | file_edit | medium | pass | — | acceptance | 4 | 4/4 | 0 | 5629 / 582 | 5.3 |  |
| fe-lru-cache-recency | file_edit | hard | fail | — | acceptance | 4 | 1/1 | 0 | 4948 / 145 | 1.9 | the loop stopped the turn (parse_disagreement) |
| fe-lru-cache-recency | file_edit | hard | pass | — | acceptance | 11 | 7/10 | 0 | 35385 / 2137 | 20.6 |  |
| fe-lru-cache-recency | file_edit | hard | fail | — | acceptance | 1 | 0/0 | 0 | 834 / 1038 | 8.3 | acceptance failed (7 run, 4 failed, 0 errors) |

## Empty-field check

Required fields: `task_class`, `model`, `quantization`, `adapter_tier`, `adapter_strictness`, `success`, `tool_call_success`, `repairs`, `rounds`, `input_tokens`, `output_tokens`, `duration_ms`.

PASS — every required field is populated in at least one row.

| field | rows populated |
|---|---:|
| `task_class` | 54/54 |
| `model` | 54/54 |
| `quantization` | 54/54 |
| `adapter_tier` | 54/54 |
| `adapter_strictness` | 54/54 |
| `success` | 48/54 |
| `tool_call_success` | 37/54 |
| `repairs` | 54/54 |
| `rounds` | 54/54 |
| `input_tokens` | 54/54 |
| `output_tokens` | 54/54 |
| `duration_ms` | 54/54 |
