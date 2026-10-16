# Local tools fix stack: state (2026-10-16, evening)

The audit: `docs/audits/local-tools-audit-2026-10-16/README.md` (merged with #810).

## Files on this branch

| Path | What |
|---|---|
| `docs/handoff/runbook-0.9.13/LIVE-CONFIG.md` | The 0.9.13 runbook section for the live config: the deploy edit, what to check after the restart, the rollback, and the day-after cleanup. |
| `docs/handoff/runbook-0.9.13/retire_workaround.py` | Deploy edit (4 lines): removes the 27B's per-model `max_output_tokens: 4096` and `- agent` from `always_loaded`. |
| `docs/handoff/runbook-0.9.13/cleanup_dead_config.py` | Day-after cleanup (11 lines): `context.reserved_output` and the 5 `model_overrides` entries that match no served model. |
| `docs/handoff/release-check/tier.sh` | The 12-request real-model tier on the operator's config (scratch daemon behind the logging proxy). Needs `CHECKOUT` and `WORK`; see its header. |
| `docs/handoff/release-check/tier_score.py` | Scores a tier run: `python3 tier_score.py "$WORK" <label>`. |
| `docs/handoff/release-check/idle.sh` | Exits 0 once the live daemon has shown no model activity for 45 s; `tier.sh` waits on it. |
| `docs/handoff/release-check/context7_stub_tools.json` | The Context7 tool pair the tier's MCP request needs. |
| `docs/handoff/pr-drafts/fix7.md`, `fix3.md`, `fix6.md` | PR bodies for the three PRs not yet opened. Placeholders left: `FIX7_PR` in fix 3 and `FIX3_PR` in fix 6 (the predecessor PR numbers). |

Both runbook scripts are guarded: each block must occur exactly once, uncommented, where named, with exactly the content expected, or the script refuses and writes nothing. Without `--write`, a script only prints its diff. Both were tested on a copy of the live config, never on the live file.

## Merge queue

Will's standing approval covers this stack only, in this order, now including fix 6 (added 2026-10-16).

Rules for each PR:
- Merge only when CI is green on its exact head SHA (squash, pinned with `--match-head-commit`).
- After any base change or branch update, every required check needs a non-cancelled run on that SHA before anyone reports "CI is running". Rerun cancelled duplicates with `gh run rerun`.
- Report the PR number and merge SHA.
- No tag, release or deploy.

| Fix | PR | State |
|---|---|---|
| gate + audit | #810 | merged `2cfdceea` |
| 1: every ending reaches Beacon | #811 | merged `c571e6bb` |
| parity: side calls by kind | #812 | merged `83132e29` |
| 2: llama.cpp cap 4096 | #813 | merged `9bc78930` |
| 5a: context windows loud and consistent | #814 | merged `8b97c113` |
| 5b: Ollama native /api/chat with num_ctx | #815 | merged `dc332849` |
| 4: tool_search finds what the model means | #816 | open, head `d6cb7f9a`, auto-merge pinned; all 7 required checks have a live run |
| 7: telemetry records what was sent | not opened | branch `fix/telemetry-records-what-was-sent` at `200c86ca`, on fix 4's pre-merge head `6736cda6` |
| 3: every tool callable locally (stubs) | not opened | `fix/every-tool-callable-locally` at `deea416a`, on fix 7 `200c86ca` |
| 6: cut round-0 overhead | not opened | `fix/cut-local-overhead` at `e9a3a438`, on fix 3 `deea416a` |

To open the next PR once its predecessor has squash-merged:
1. Rebase: `git rebase --onto origin/main <predecessor's pre-merge head> <branch>`. The predecessor heads are the "on ..." SHAs in the table.
2. Check that the tree is unchanged, i.e. the same tree that was checked.
3. Push, open the PR with its draft body, and pin auto-merge.

Each branch was pre-rebased this way and checked locally: ruff, the mypy gate, its tests, the hermetic gate, parity replay 12/12 and stability identical. Fix 3 and fix 6 also ran their full suites: 14,410 and 14,422 passed, each with only the 2 known host failures.

Required checks on main: test (3.11), test (3.12), quality, security-floors, firstlight (local, editable), firstlight (cloud, editable), firstlight (local, wheel). The two known host failures on this box are `test_proc_net_tcp6_addresses` and `test_the_real_session_bus_agrees_with_the_probe`.

## What's left

1. **Merge the rest of the stack:** #816 (fix 4), then fix 7, fix 3 and fix 6, in that order.
2. **Release check on main** with everything merged:
   - the full hermetic gate (`tests/test_local_gate_*`, `tests/test_local_tool_reachability.py`);
   - parity replay and stability;
   - the 12-request tier: `CHECKOUT=<main checkout> WORK=<dir> docs/handoff/release-check/tier.sh release`, then `tier_score.py`.

   The tier uses the 10-16 backup config: the live config without the workaround, which is what live becomes after the deploy edit.
3. **One-page summary for Will,** including the runbook's config change.

## Decisions

- **Fix 3 is stubs.**
  - Full schema for a fixed ten: bash, read_file, write_file, edit_file, grep, glob, web_search, web_fetch, agent, ask_user. Every other tool is a stub; enum values are listed when there are 8 or fewer.
  - The tools block is sorted and byte-identical across rounds.
  - `always_loaded` applies to `local_catalog: names` only.
  - Commits after the original fix:
    - stubs-mode tests for every names-pinned test (the map is in the PR body);
    - a parity rebaseline from the committed exchanges, for the advertisement row's new fields;
    - the doctor's Advertised tools row, which now counts what a stubs run offers;
    - **last, revertible alone:** a fallback round's tools follow the model that serves it, tested in both directions in the hermetic gate.
- **Fix 6** follows Will's later answers.
  - **Cuts:** A, B, C and E, where E means the 80-character stub line only. Stub argument types are kept: untyped stubs dropped right arguments to 5/12 on the 27B.
  - **Not done:** D (behaviour guidance) is left as is, and F is skipped.
  - **Thresholds:** Prometheus-owned ≤ 7,400 and round 0 on Will's config ≤ 14,000.
  - **Measured:** round 0 went from 19,198 to 13,813; the gate's Prometheus-owned overhead from 8,901 to 7,253. The tier, before → after:
    - found 12/12 → 12/12;
    - right tool first 11/12 → 12/12;
    - right arguments 12/12 → 12/12.
  - **Goldens:** re-recorded once, at the end (hosted_route unchanged).
  - **Gate:** the coding session grew to 26 modules so that check c still compacts.
  - **Revisit after launch:** D, F and the 6,200/5,000 targets.
- **Parity:** main-turn requests keep any-order exact matching. Side calls (titles, summaries) match by kind.
- **Deploy (Will's go only):** retire the workaround with the targeted edit in `runbook-0.9.13/`, never by restoring the 10-16 backup.
  - Live differs from that backup only by the workaround's 4 lines (re-diffed 2026-10-16).
  - The config cleanup (`reserved_output` and the 5 unmatched overrides) is a separate day-after step, not part of the deploy.
