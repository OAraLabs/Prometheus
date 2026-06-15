# Harvest Overnight Runbook (schedulable — run in a deliberate daemon-down window)

**Status:** STAGED, not executed. Trigger this in a SEPARATE session during a window where the
daemon (and therefore AutoDream / Telegram / Beacon) can be DOWN for sole-GPU use.
**Authorization:** this runbook harvests DATA only. The LoRA training run + the eval are a separate
go (see Non-goal).

## Read first — what this run actually produces (Phase-3 finding)

The verification harvest (`audits/20260613T060000Z-harvest-breadth-verification.md`) showed the
model's failures are **~88% dict_wrap_unwrap** and the other transitions (fuzzy-rename, missing-
discriminator, type-coercion) **do not induce naturally** — fuzzy-rename is grammar-blocked (the
local tool-calling path won't emit invalid tool names). So:

- **This overnight harvest will be dict-wrap-dominated (~88%).** That is the model's real failure
  distribution, not a corpus defect — a dict-wrap specialist LoRA trains on the real problem.
- **A breadth-BALANCED set requires synthetic injection FIRST** (generalize the pair-smoke
  `--inject` path to fabricate fuzzy/missing-disc/coercion pairs through the real adapter). That
  injection tool does not exist yet — it is the prerequisite for "balanced", and a separate task.
  Decide before you run: dict-wrap specialist (this runbook as-is) vs. wait for injection.

## Verified numbers (from Phase 3, recompute if the corpus changed)

- Density: **0.52 unique pairs / run** (held across #40 0.44 → verify 0.52).
- Per-run wall: ~38 s **contended** (daemon up). Sole-GPU (daemon stopped): estimate ~15–20 s/run.
- Target: **~1,000 NEW dict-wrap pairs ⇒ ~1,925 distinct tasks** at 0.52/run.
- Wall-clock sole-GPU: ~1,925 × ~17 s ≈ **~9 h** (refine after a 100-run pilot; do not trust a
  single point estimate overnight).

## PRE-FLIGHT GATES (all must pass before the run)

1. **Corpus capacity.** The current dict-wrap pools yield only ~410 distinct tasks — far short of
   ~1,925. EXPAND the value pools in `src/prometheus/gym/corpus.py` (`_CMD_VERBS`, `_CMD_ARGS`,
   `_GREP_PATTERNS`, `_GREP_PATHS`) ~5× so the cross-products exceed the target, then:
   ```
   uv run python scripts/gen_harvest_corpus.py --per-transition 700 --out gym/tasksets/harvest-overnight.yaml
   uv run python -c "from prometheus.gym.tasks import load_taskset as L; print(len(L('gym/tasksets/harvest-overnight.yaml').tasks),'tasks')"
   ```
   GATE: task count ≥ ~1,925 and `load_taskset` accepts it (no exception).
2. **Dedup small-sample check.** Confirm distinct prompts → distinct pairs on a 20-run pilot before
   the full run (catches a pool that silently repeats):
   ```
   uv run python scripts/gym_run.py --manifest <pilot-manifest> --harvest --harvest-db /tmp/pilot.db   # ~20 tasks
   sqlite3 /tmp/pilot.db "SELECT count(*), count(DISTINCT context_hash) FROM training_pairs"            # the two must be equal
   ```
   Also recompute density from the pilot and re-derive the task target; don't reuse 0.52 if it moved.
3. **Disk.** Pairs are tiny (~1–2 KB each); 1,000 pairs ≈ a few MB. Still check headroom:
   ```
   df -h ~/.prometheus/data        # GATE: > 1 GB free
   ```
4. **Daemon-down window confirmed.** Stopping the daemon takes AutoDream, the Telegram gateway, and
   Beacon OFFLINE for the whole run. Confirm that outage is acceptable for the window.
   ```
   systemctl --user stop prometheus.service        # frees the single GPU slot for sole use
   systemctl --user is-active prometheus.service   # expect: inactive
   ```
   RESTORE (run at the end, unconditionally): `systemctl --user start prometheus.service`

## RUN (detached, kill-proof, survives session death)

`gym_run.py` is a long foreground process; follow-up #13's lesson is that the watcher must not be
the thing that dies. Launch detached with `setsid`, write a PID file, and a COMPLETION SENTINEL the
owner can poll without the launching session alive:

```
cd ~/Prometheus
mkdir -p ~/.prometheus/harvest-runs
RUN=~/.prometheus/harvest-runs/overnight-$(date +%Y%m%dT%H%M%SZ)
setsid bash -c '
  uv run python scripts/gym_run.py \
    --manifest gym/experiments/harvest-overnight.yaml \
    --harvest --harvest-db ~/.prometheus/data/gym-training.db \
    > '"$RUN"'.log 2>&1
  echo "exit=$?" > '"$RUN"'.SENTINEL
' < /dev/null &
echo $! > "$RUN".pid
echo "detached; poll $RUN.SENTINEL for completion"
```

- Survives the launching session closing (`setsid` + `</dev/null`).
- Completion = `$RUN.SENTINEL` exists (contains the exit code). Poll that file, NOT the process.
- Harvests into the MAIN `gym-training.db` (accumulates with prior pairs; that's intended for the
  real corpus — use a separate `--harvest-db` only if you want an isolated measurement).

## POST-FLIGHT (so the owner wakes to a MEASURED corpus, not a mystery DB)

```
DB=~/.prometheus/data/gym-training.db
sqlite3 "$DB" "SELECT count(*), count(DISTINCT context_hash) FROM training_pairs"           # total + dedup loss
sqlite3 "$DB" "SELECT pair_source, count(*) FROM training_pairs GROUP BY pair_source"        # by source
sqlite3 "$DB" "SELECT tool_name, count(*) FROM training_pairs GROUP BY tool_name ORDER BY 2 DESC"  # by shape/tool
uv run python scripts/transition_histogram.py --db "$DB"                                     # by TRANSITION (the breadth metric)
uv run python scripts/export_training_pairs.py --db "$DB" --out ~/.prometheus/harvest-runs/overnight.jsonl   # export validates
wc -l ~/.prometheus/harvest-runs/overnight.jsonl
```

Report: total NEW pairs, dedup loss (should be ~0), by-shape AND by-transition distribution, and an
updated **pairs-to-1,000 projection at the run's measured density** (recomputed, not 0.52 if it
moved). Expect the transition histogram to remain ~88% dict-wrap — that is the known finding, not a
regression.

## NON-GOAL (explicit — a separate go each)

This runbook produces training DATA only. NOT in scope here, and NOT authorized by it:
- The **LoRA training run** itself.
- The **eval** proving the fine-tune actually fixes the failing shapes — which MUST be on
  **HELD-OUT** transitions/shapes the corpus did not drill (different params/tools/nouns), to guard
  against the Goodhart this whole sprint exists to prevent. A fine-tune that only aces the trained
  nouns is a failure even at 100% gym score.
- If a **balanced** (multi-transition) set is wanted, the synthetic-injection tool (Phase-3
  proposal #1) is a prerequisite and a separate task.
