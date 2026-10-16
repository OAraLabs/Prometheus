# 0.9.13 local tools: release check (2026-10-16)

**Verdict: ready for your go on version bump, tag, release and deploy.** None of those has been started. The whole stack is merged; main is at `1f85ccee`, and CI is green on it.

## What merged, in order

| PR | Fix | Merge |
|---|---|---|
| #810 | regression gate and audit | `2cfdceea` |
| #811 | 1: every ending reaches Beacon | `c571e6bb` |
| #812 | parity: side calls match by kind | `83132e29` |
| #813 | 2: llama.cpp output cap back to 4096 | `9bc78930` |
| #814 | 5a: context windows loud and consistent | `8b97c113` |
| #815 | 5b: Ollama on native /api/chat with num_ctx | `dc332849` |
| #816 | 4: tool_search finds what the model means | `63c66da4` |
| #818 | 7: telemetry records the tools a round sent | `3bbb7bb7` |
| #819 | 3: every tool callable locally (stubs), doctor row, fallback rounds | `eb0a988e` |
| #820 | 6: round 0 cut by 5.4k tokens | `1f85ccee` |

## Release check on main (`1f85ccee`)

- **Hermetic gate:** 292 passed, 1 xfail. The xfail is gate e's client-readable overhead breakdown, which this stack doesn't target.
- **Parity:** replay 12/12, stability identical. Ruff and the mypy gate are clean.
- **Real-model tier:** the 27B on the 4090, run on your 10-16 backup config (what live becomes after the deploy edit). It ran on fix 6's head, whose tree is byte-identical to main's.

| | 0.9.12 | Before fix 6 | Release |
|---|---|---|---|
| Right tool called | 0 of 7 | 12/12 | **12/12** |
| Right tool first | n/a | 11/12 | **12/12** |
| Right arguments on the first call | n/a | 12/12 | **11/12** (the scorer says 12/12; see below) |
| Schema refusals | n/a | 0 | 1, fixed on the retry |
| Round 0, server tokens (limit 14,000) | n/a | 19,198 | **13,809** |

Notes on the tier:
- **The 11/12:** r_memory's first `memory` call left out `entry`, which `replace` needs. The second call left out `operation` and got the schema refusal. The third call was right.
- **Environment failures, not wrong calls:** in the scratch daemon, Context7 had no grant, there was no TTS engine, and yt-dlp wasn't installed. In all three the model called the right tool with the right arguments.
- **Before you rely on TTS or YouTube transcripts:** check that yt-dlp and a TTS engine exist in the live venv. That wasn't verified here.

## Deploy runbook: the live config

The scripts are on branch `docs/local-tools-state`, in `docs/handoff/runbook-0.9.13/`.

**At the deploy** (`retire_workaround.py`: a targeted edit, not a restore of the backup):
1. Inside the window, before the restart, back up `prometheus.yaml`. That backup keeps the workaround, which 0.9.12 needs if you roll back.
2. Dry run, then `--write`.
3. Diff the backup against live: you should see exactly the 4 workaround lines.
   - the 27B's per-model `max_output_tokens: 4096`, because 0.9.13's default is 4096;
   - `- agent` in `always_loaded`, which stubs mode doesn't read; `agent` always has its full schema.

**After the restart:**
- The boot line should say 10 tools with full schemas and 41 as stubs.
- `oara doctor`'s Advertised tools row should read 51 of 51.
- Expected warnings: `reserved_output` is no longer read; 5 overrides match no served model; 5 `slash_commands` lines.

**Rollback:** restore that backup together with the code.

**Day after** (`cleanup_dead_config.py`, 11 lines): removes `reserved_output` and the 5 unmatched overrides. First check that the 0.9.13 boot log names the same five. Same pattern: backup, dry run, write, diff. No restart needed.

## After launch

Fix 6's cuts D and F, the 6,200 and 5,000 overhead targets (revisit with plan-behaviour tests and a cloud A/B), and gate e's client-readable overhead breakdown.
