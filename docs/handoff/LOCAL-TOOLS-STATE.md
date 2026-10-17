# Local tools fix stack: state (2026-10-16, stack merged, release check done)

The audit: `docs/audits/local-tools-audit-2026-10-16/README.md` (merged with #810).

## Files on this branch

| Path | What |
|---|---|
| `docs/handoff/RELEASE-CHECK-0.9.13.md` | The one-page release-check summary for Will. |
| `docs/handoff/runbook-0.9.13/RELEASE.md` | The 0.9.13 release sequence: the bump (#821), tag and publish, the Homebrew tap bump 0.9.8 → 0.9.13 (not only url + sha256: 0.9.9 swapped PyMuPDF for pypdfium2 + Pillow and added distro), the deploy, and the day after. |
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
| 4: tool_search finds what the model means | #816 | merged `63c66da4` |
| 7: telemetry records what was sent | #818 | merged `3bbb7bb7` |
| 3: every tool callable locally (stubs) | #819 | merged `eb0a988e` |
| 6: cut round-0 overhead | #820 | merged `1f85ccee` |

The stack is complete. Each PR went in only with every required check green and non-cancelled on its exact head SHA, squash-merged with pinned auto-merge.

Required checks on main: test (3.11), test (3.12), quality, security-floors, firstlight (local, editable), firstlight (cloud, editable), firstlight (local, wheel). The two known host failures on this box are `test_proc_net_tcp6_addresses` and `test_the_real_session_bus_agrees_with_the_probe`.

## What's left

The release check on main `1f85ccee` is done: see `RELEASE-CHECK-0.9.13.md`.
- Hermetic gate: 292 passed, 1 xfail (gate e's breakdown).
- Parity: 12/12, stability identical. CI is green.
- Tier: right tool 12/12, right tool first 12/12, right first-call arguments 11/12 (the scorer says 12/12), round 0 13,809.

The version-bump PR is open: #821 (`chore/release-0.9.13`), with the version strings and the 0.9.13 CHANGELOG section. #822 (docs: the README's fencing, env-scrubbing and listening-address claims corrected) merged as `1889d588`, and #821 was then updated onto main by merge (`gh pr update-branch`), so it carries the corrected README, which is PyPI's description. #825 (daemon hardening) then merged as `48e1b3cf`; #821 was updated onto main again and gained one CHANGELOG line for it. #821's head is now `468a1a57`. Not merged, no auto-merge, not tagged. Still Will's go: merging #821, the tag and release, and the deploy, including the live-config edit in `runbook-0.9.13/` and its day-after cleanup.

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
