# Distillation + flywheel log

Append-only, times in ET.

- 2026-10-30 16:58 — First-run setup: worktree `../Prometheus-distill` on `feat/distill-flywheel`, runbook committed (private business names scrubbed), draft PR #878 opened, `data/distill/` gitignored, Telegram start message sent (200, ok). Phase 0 begins.
- 2026-10-30 17:15 — Phase 0 (read-only) done. **GPU gate: BLOCKED** (4090 serves a customer-facing tunnel with no fallback; its nights are booked by another job; 3090 Ti backs the live daemon via Saluki; 4090 disk 75 GB). Rehearsal not run. Student stays Qwen3.5-9B (no Qwen3.6–3.8 8–10B Apache model). No setups mechanism exists. Open questions 2–6 logged. Continuing with no-GPU work: Phase 1 audit, setups CLI + read-only `business` capture.
- 2026-10-30 17:22 — Phase 1 started: three subagents (flywheel code audit, read-only trace corpus audit on the 3090 Ti host, setups CLI with read-only `save business`).
- 2026-10-30 17:27 — Phase 1A (code audit) done → `data/distill/phase1-partA.md`. Trace store: telemetry.db `tool_calls` (is_golden=1) + golden JSONL exports; traces hold raw messages, not the rendered template; labels come from no model judge. Started the Phase 4 scorer code agent (no GPU).
- 2026-10-30 17:40 — Setups CLI built (`scripts/distill/setups.py`, 41 tests). `save business` captured the live layout read-only; `status`: business MATCH, all services healthy. Locked the 4090 `llama-server` and the daemon's Saluki server as per-host never-stop (gitignored hosts.yaml) until OPEN-QUESTIONS 2/4 are answered; `apply flywheel` dry run now refuses. Adversarial review of the CLI running before commit.
- 2026-10-30 17:48 — Phase 4 scorer code written (`scripts/distill/score.py`: run/summarize/compare with the Phase 6 gate, 31 tests). Adversarial review of the scorer running before commit.
- 2026-10-30 18:05 — Phase 1 done → `DISTILL-PHASE1-AUDIT.md`. 10,195 deduped Max traces but only ~300 distinct tasks; coding 86%, chat-mode 0, compacted 0; labels are tool-success only (≥41% of linkable Max traces sit in abandoned/stopped turns). Phase 3 can't reach 5–10K under the 35% coding cap without Phase 2b. Phase 2/2b need the Max API (held on Q6).
