# Flywheel corpus — Phase 0 survey

**Branch:** `feat/flywheel-corpus` (off `main @ f754257`). **Goal:** turn the proven harvest mechanism
(9 pairs/66-run cycle) into a path to 1,000 unique pairs by generating task DIVERSITY.

## Where pairs come from (the 9 harvested, by shape)

The model **nests a scalar argument inside a dict keyed by the param name**; the adapter unwraps it
(`schema_repair`) or the model self-corrects (`self_correction`). Exact shapes captured:

| tool | rejected (wrapped) input | source |
|---|---|---|
| `task_create` | `{"command": {"command": "sleep 1 && echo done"}}` | schema_repair |
| `task_create` | `{"prompt": {"prompt": …}, "description": …}` ×2 | schema_repair |
| `task_list` | `{"status": {"status": "failed"}}` ×2, `{"status": {}}`, `{"status": {"running": true}}` | both |
| `grep` | `{"root": {"path": "/tmp…"}}` | self_correction |

## Dedup mechanics (decides what "distinct" means)

`PairStore.add_pair`: `context_hash = sha256(context_json + "\x1e" + rejected_call_json)`
(`pair_capture.py:112`), UNIQUE column. So a pair is distinct iff the **context** OR the **rejected
call (tool + input)** differs. The target value flows into the rejected call → **varying the value
guarantees a distinct hash**, even for the same tool/param. This is the diversity lever.

## Ceiling per shape

- `task_create(command)` and `task_create(goal/prompt)` — **unbounded** distinct values → high yield.
- `grep(pattern × path)` — **unbounded** (cross-product) → high yield.
- `task_list(status)` — **low** (~handful of valid statuses) → include for completeness, not volume.
- Bonus wrap-prone tools in the adapter_unwrap set (sessions_list, download_file, browser) — scalar
  params, includable but secondary.

## Taskset schema (what the generator must emit)

`gym/tasks.py`: each task needs `id`, `category`, `prompt`, `score` (required); optional `seed`,
`setup_files`, `stub_tools`, `notes`. `score` keys must be in `ALLOWED_SCORE_KEYS` (`expect_tool`,
…). For harvest the predicate is near-irrelevant (we want induced REPAIRS, not passes) — a minimal
`expect_tool: <tool>` suffices. `stub_tools` the side-effecting tools (task_create spawns real
tasks; download_file/browser hit network) — the **unwrap still fires at the adapter/validation
layer, before execution**, so stubbing is safe and is exactly how v1 captured the 3 task_create
pairs (task_create is side-effect-stubbed in v1).

## Plan

1. `src/prometheus/gym/corpus.py`: `generate_harvest_corpus(per_shape)` → a deterministic taskset
   dict. Shapes = the high-ceiling proven ones (task_create command, task_create goal, grep
   pattern×path) + task_list status (capped). Values built from cross-products of curated word
   lists → many distinct scalars, no RNG. Side-effecting tools stubbed.
2. `scripts/gen_harvest_corpus.py` + a harvest manifest pointing at the generated taskset.
3. Bounded harvest run (e.g. ~50–80 distinct tasks × 1–2 runs) → measure **unique** pairs vs the
   fixed-22 baseline (9). runs_per_task is a weak multiplier (same prompt → same wrap → deduped);
   **distinct values is the lever**, so optimize task count over runs/task.
4. Report yield density + projection to 1,000 + overnight-run recommendation. (The LoRA train
   itself is a separate, larger commitment — out of scope here; this proves the corpus path.)

Test: generator is deterministic, emits schema-valid tasks, distinct ids + distinct values.
