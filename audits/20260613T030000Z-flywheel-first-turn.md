# Repair-pair flywheel — first real turn (harvest → export)

**Branch:** `feat/gym-pair-harvest` · **Date:** 2026-06-13
**What:** built the pair-harvest gym mode (closeout follow-up #6 — the realistic 1,000-pair path)
and ran it end-to-end against the live model. **The flywheel turns.**

## The cycle, proven end-to-end

1. **Induce** — `gym_run.py --harvest --manifest gym/experiments/harvest-unwrap.yaml` drove the v1
   task set (22 tasks × 3 = 66 real `run_loop` turns) against `gemma4-26b` with the
   `adapter_unwrap` variant on. 51/66 runs passed; the 15 "failures" are the dict-wrapped
   `task_create`/`task_list` shapes — exactly the repair source.
2. **Capture** — `pair_capture` configured (in the gym process) at `~/.prometheus/data/gym-training
   .db`, separate from the daemon's live `training.db`. **9 pairs captured**, all unique (9 total /
   9 distinct `context_hash`):
   - `schema_repair` ×5 — `task_create` ×3, `task_list` ×2 (the adapter losslessly unwrapping the
     model's phantom nesting; immediate-capture path).
   - `self_correction` ×4 — `task_list` ×3, `grep` ×1 (failed-then-succeeded; pending-stash path).
   Both capture paths fired with zero agent-loop changes (run_loop lazily inits `pair_pending`;
   every `capture_pair` guards on `get_store()`).
3. **Export** — `export_training_pairs.py --db <gym-training.db> --out pairs.jsonl` → 9 DPO-format
   lines `{prompt, chosen, rejected, meta}`. Example (a real, useful pair):
   - rejected: `{"input": {"command": {"command": "sleep 1 && echo done"}, "description": …}}`
   - chosen:   `{"input": {"command": "sleep 1 && echo done", "description": …}}`

## Yield + the scaling finding (the decision input)

- **~9 unique pairs / 66 runs** (~0.14/run), ~25 min wall on the shared GPU slot.
- **Naive repetition does NOT scale linearly.** Pairs dedup on `context_hash` (derived from the
  prompt + the rejected call). The gym is near-deterministic on these fixed shapes, so re-running
  the same 66 tasks mostly re-captures the same ~9 contexts → deduped back to ~9. The mechanism is
  proven; **reaching hundreds/1,000 unique pairs needs task/variant DIVERSITY**, not just more
  hours: more task sets, more tools exercising the dict-wrap/fuzzy-name shapes, varied prompts,
  and additional repair-inducing variants — each a distinct `context_hash` space.
- Concretely, the 1,000-pair path = build out the harvest task corpus (e.g. parametrized prompt
  templates × the dict-wrap-prone tools × seeds), then run overnight. That corpus is the next
  piece of work; this turn proved the harvest+capture+export plumbing it would feed.

## Shipped

- `src/prometheus/gym/harvest.py` (`configure_harvest`, `pair_total`) + `gym_run.py --harvest`;
  `tests/test_gym_harvest.py` (5 tests: isolation from live `training.db`, capture lands + counts,
  off-by-default). `gym/experiments/harvest-unwrap.yaml` (dedicated series, never clobbers s1).
- Artifacts (not repo): `~/.prometheus/data/gym-training.db` (9 pairs), `gym/results/harvest/
  unwrap.md` (run report).
