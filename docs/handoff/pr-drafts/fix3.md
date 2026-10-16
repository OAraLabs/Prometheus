## What this fixes

Fix 3 of the local tool-calling audit: every tool is callable on a local turn. **Stacked on fix 7 (FIX7_PR).**

Deferred tools have been unreachable on a local llama.cpp turn since #121 (deferral on for local tiers, 31 Jul). The model does not call a function missing from `tools`, and the server's grammar (Qwen's XML tool format) admits only names in `tools` and, for each, only its declared property names. `tool_search` select returned a schema as text, which never made a tool callable. On 0.9.12 the real 27B called 0 of 7 plain requests that need a deferred tool.

A deferred local run now offers every tool (`context/tool_stubs.py`):
- **Full schema** for the ambient ten: `bash`, `read_file`, `write_file`, `edit_file`, `grep`, `glob`, `web_search`, `web_fetch`, `agent`, `ask_user`.
- **Every other tool as a stub**: name, one line of description, and top-level property names with primitive types, all optional. An enum's values are included when there are at most 8; otherwise the schema-refusal path covers it.
- **Sorted by name, built once per run**, so the block is byte-identical across rounds. The template renders it ahead of the system text, where one changed byte re-prefills everything.
- **Validated against the real schema** when a call runs. A refusal carries the full schema and a minimal valid call, twice per tool per turn, then tells the model to stop.
- **Sanitized for llama.cpp's grammar converter**: local `$ref` inlined, `Optional` collapsed, and `title`/`examples`/`pattern`/`format` dropped as keywords, never as property names (`glob`/`grep` have a `pattern` argument, `dashboard` a `title`).
- `tools.deferred_loading.local_catalog: names` keeps the 0.9.12 catalog. `always_loaded` now applies only to that mode.

## Why stubs: the A/B

The real 27B on the 4090, through a scratch daemon behind the logging proxy, ran 12 plain requests, each needing a non-ambient tool and none naming it. One run per request per arm:

| | Skeleton stubs | `call_tool` dispatcher |
|---|---|---|
| Tool found | 12/12 | 12/12 |
| Right tool called | 11/12 | 11/12 |
| Right tool called first | 11/12 | 11/12 |
| Right arguments on the first try | **11/12** | 9/12 |
| Schema refusals (each fixed on the first retry) | 1 | 2 |
| Round-0 prompt, server tokens | 21,670 | 19,285 |
| Total prompt tokens | 1.48M / 75 rounds | 1.42M / 81 rounds |
| Prefix cache reuse between rounds | 1.0, except sub-agent turns (one slot) | same |

- **Misses:**
  - Stubs: the YouTube request used `web_fetch`. With this PR's final stubs it called `youtube_transcript` first in 3 of 3 re-runs, so the miss was noise and no wording change was made.
  - Dispatcher: the sub-agent request used `bash`; `agent` has its full schema in both arms.
- **The dispatcher's argument errors** were guessed names: `cron_create` without `name`, and `todo_write` with `text` for `item`.
- **Round-0 figures:** both include a second copy of the repo's PROMETHEUS.md (~2.7k), because the scratch daemon ran from a nested worktree. Live runs from the deploy directory: stubs come to about 18.9k there, and the tools block to 4,856 tokens for 55 tools, against 4,762 for 0.9.12's 12.
- **Baseline:** 0.9.12's catalog called 0 of the original 7 requests; with deferral off, 7 of 7.
- **Sanitizer bug:** the A/B ran with a bug this PR fixes (argument names equal to a schema keyword were dropped: `glob`/`grep` `pattern`, `dashboard` `title`). Neither arm's 12 requests called those tools.

## Gate checks it flips

- `test_local_gate_integration.py`:
  - a: the sub-agent is spawned;
  - b: cron, wiki, memory search and MCP are each found and called in the same turn, and now **called with the arguments the model meant**. The stand-in, like the server's grammar, lets out only declared property names.
- `test_local_tool_reachability.py`: access for every deferred tool, and awareness.
- `test_c_the_tools_block_is_sorted_and_byte_identical_across_rounds` (new).
- The two fallback checks (new; see Fallback rounds).

The awareness count check passes when nothing is deferred. Every tool being offered is a stronger guarantee than stating a count.

New: `tests/test_tool_stubs.py` (14), including "every registered tool keeps every argument". The real-model tier in `scripts/local_gate/run_gate.py` grows to the A/B's 12 requests, graded on arguments, retries, tokens and prefix-cache reuse.

## The doctor's Advertised tools row

Under stubs, a local run offers every tool and `always_loaded` is not consulted. The row still reported the `always_loaded` set ("14 of 51 registered offered" on the live config), and it judged that list: a pinned list missing a shipped tool warned about tools the run offers anyway.
- **`oara doctor`** counts what a local run offers: "51 of 51 registered offered to a local model (10 with their full schema, 41 as stubs)".
- **Chat `/doctor`** says every tool is offered and that `always_loaded` applies to `local_catalog: names` only, with status ok.
- **One rule** for which catalog a config selects (`tool_stubs.catalog_mode`), shared by the loader and both rows.

Tests: under stubs, each row is checked against five configs: `always_loaded` absent, the shipped list, the live box's pinned superset, a list missing two, and `[]`. Counts come from the registry. The 10 new tests fail without the commit.

## Fallback rounds (the last commit, revertible alone)

The catalog is decided after routing (#462), but a fallback changes the serving model after that. Before this commit, a cloud round that fell back to the local model sent it the cloud's full schemas, and a local round that fell back to cloud sent it stubs.
- **Per-round provider fallback** (a terminal failure: auth, billing): the fallback round carries the catalog of the model that serves it: the local catalog for a local fallback, every full schema for a cloud one. The next round asks the primary again with its own catalog. The round's offered set and its tool_sets row follow what was sent.
- **Circuit-breaker model fallback** (for the rest of the run): the catalog is rebuilt once for the new model, then frozen again.
- Under `local_catalog: names` too: a cloud run's local fallback round now sends the `always_loaded` set, where 0.9.12 sent the cloud's full list into the local window.

Tests, all failing without the commit:
- Hermetic gate, both directions: `test_fallback_a_cloud_to_local_fallback_round_gets_the_local_catalog` and `test_fallback_a_local_to_cloud_fallback_round_gets_every_full_schema`. Each compares the fallback round byte for byte with a plain turn on the serving model. The scripted stand-in gains `fail_script`: a 401 on that script's tool-bearing requests.
- `tests/test_stub_advertisement.py`:
  - `test_a_circuit_breaker_model_fallback_rebuilds_the_catalog_for_the_new_model`, in both directions;
  - `test_a_local_fallback_round_gets_the_local_catalog_and_records_no_lucky_guess`;
  - `test_a_fallback_round_records_the_set_it_sent` (names mode, where the sets differ by name).

## Tests pinned to `local_catalog: names`, and what covers them under stubs

27 tests reach the pin: 24 through one helper per file, plus the three doctor-row tests that the doctor commit pins. They keep guarding the names catalog. Every stubs-mode test named below is in `tests/test_stub_advertisement.py`.

| Pinned test | Covered under stubs by |
|---|---|
| **tests/test_deferred_tier_aware.py** | |
| `test_auto_disables_on_cloud`, `test_auto_enables_on_local_tiers`, `test_auto_without_adapter_advertises_everything`, `test_explicit_true_wins_even_on_cloud`, `test_explicit_false_wins_even_on_local`, `test_set_configured_round_trips` | Mode-independent: the deferral decision (`resolve_deferred`) does not read `local_catalog`, and stubs mode uses the same decision. End to end: `test_a_local_turn_offers_full_schemas_for_the_ambient_set_and_stubs_for_the_rest`, `test_a_cloud_turn_offers_every_full_schema`, `test_the_next_run_picks_up_a_flip` |
| `test_schemas_for_run_shapes` | The names-only shape (the `always_loaded` subset). The stubs catalog's shape: `tests/test_tool_stubs.py`, and `test_a_local_turn_offers_…` |
| `test_call_site_advertises_deferred_set_on_local` | `test_a_local_turn_offers_full_schemas_for_the_ambient_set_and_stubs_for_the_rest` |
| `test_call_site_advertises_full_catalog_on_cloud` | `test_a_cloud_turn_offers_every_full_schema` |
| `test_advertised_set_is_frozen_across_rounds_even_if_config_flips` | `test_the_catalog_is_frozen_across_rounds_even_if_config_flips`; gate `test_c_the_tools_block_is_sorted_and_byte_identical_across_rounds` |
| `test_next_run_picks_up_the_flip` | `test_the_next_run_picks_up_a_flip` |
| `test_tool_advertisement_row_is_recorded` | `test_a_local_turn_offers_…` (the row: deferred, advertised, stubs, catalog) |
| `test_tool_advertisement_row_full_catalog_case` | `test_a_cloud_turn_offers_every_full_schema` |
| **tests/test_advertisement_after_routing.py** | |
| `test_routed_local_turn_advertises_the_deferred_set` | `test_the_catalog_is_decided_after_routing`; mid-turn, the fallback tests above |
| `test_advertisement_row_names_the_post_routing_model` | `test_the_row_names_the_routed_model_and_its_stubs` |
| `test_advertisement_row_is_filed_under_the_turn_session` | `test_the_row_names_the_routed_model_and_its_stubs`, `test_a_local_turn_offers_…` (context session `web`, row filed under the turn's) |
| `test_unrouted_turn_behaves_as_before` | `test_a_local_turn_offers_…` (no router) |
| **Lucky guesses** (pinned to names, as decided) | |
| `tests/test_lucky_guess_marker.py`: `test_a_lucky_guess_writes_one_call_row_and_a_marker_run_row`, `test_the_marker_carries_no_session_on_an_ephemeral_turn`, `test_a_tool_the_run_advertised_writes_no_marker` | `test_a_local_round_never_records_a_lucky_guess` |
| `tests/test_telemetry_records_what_was_sent.py`: `test_a_tool_the_round_offered_is_no_lucky_guess_whatever_a_recomputation_says`, `test_a_tool_the_round_did_not_offer_is_a_lucky_guess` | `test_a_local_round_never_records_a_lucky_guess` |
| `tests/test_telemetry_records_what_was_sent.py`: `test_a_round_that_sends_another_set_records_it` | Its stubs-mode version, same name |
| `tests/test_fallback_attribution.py`: `test_a_degraded_rounds_lucky_guess_names_the_fallback` | `test_a_local_fallback_round_gets_the_local_catalog_and_records_no_lucky_guess` |
| **tests/test_always_loaded_follows_the_default.py** (doctor rows) | |
| `test_the_chat_doctor_row_says_which_case_applies` (7 cases), `test_oara_doctors_row_carries_the_same_case`, `test_the_chat_doctor_shows_the_row` | `test_under_stubs_the_chat_row_says_every_tool_is_offered`, `test_under_stubs_oara_doctors_row_counts_what_a_local_run_offers` (same file) |

## Parity goldens

The tools block is in every local request, so the 11 scenarios with a local request were re-recorded. The primary was live (27B), and the Ollama side was live for model_switch and repaired_tool_call. hosted_route was not re-recorded.

The advertisement row then gained `catalog` and `stubs`, so all 12 expected files were re-derived with `parity_harness.py rebaseline-from-exchanges`. That command is for a change to what is recorded only, and refuses if any model request changed: 0 refused, and each scenario's 2 replays were identical. The only new lines are the two fields; the rest of that diff is the same rows in replay order.

## Checks run

- Hermetic gate (`test_local_gate_*`, `test_local_tool_reachability`): 291 passed, 2 xfailed (gate e's two overhead checks, which fix 6 flips).
- Full suite (CI's command), on fix 3 before the rebase onto 7 and the parity rebaseline: 14,410 passed; 2 failures that fail the same way on main on this host (environment, not this change); coverage 81.9%. After the rebase: ruff, the mypy gate and fix 3's tests with the gate, lucky-guess and fallback-attribution tests: 381 passed. With the doctor commit: the doctor-row, agent-wiring and scrub tests with the gate, 497 passed; parity 12/12 on this head. CI runs the full suite on this head.
- Parity `replay`: 12/12 PARITY (on this head). `stability`: IDENTICAL across 2 runs.
- `ruff check src/`: clean. Mypy gate: OK.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
