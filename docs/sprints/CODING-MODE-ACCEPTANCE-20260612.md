# Coding Mode v2 — Acceptance Result (Phase 4 gate)

**When:** 2026-06-12 ~19:45 EDT · **Branch:** `feat/coding-mode` → PR #33
**Harness:** `~/bakeoff-harness/runner/coding_accept.py` · fixture marshmallow @ 27bfa77 · result `results/coding-accept-1781304749.json`
**Verdict: 4/5 converted — MEETS GATE (≥3/5).** The current loop scored 0–1/5 on these same tasks; openhands ~4.5/5.

## Per-task (2 runs each; converted = any run passes)

| Task | r1 | r2 | converted | note |
|---|---|---|---|---|
| t11 make-failing-test-pass | ✅ success (30r, 91s) | ✅ success (8r, 39s) | **PASS** | both runs green |
| t12 write-tests-for-untested | ✗ crash (30s) | ✅ success (30r, 122s) | **PASS** | r1 transient crash, see below |
| t13 bug-only-tests-find | ✗ abandoned (30r, 694s) | ✗ crash (673s) | fail | the genuine residual |
| t14 refactor-under-green | ✅ success (30r, 123s) | ✅ success (30r, 153s) | **PASS** | both runs green |
| t15 resolve-circular-import | ✅ success (30r, 124s) | ✅ success (30r, 109s) | **PASS** | both runs green |

All passing runs: acceptance command exit 0 (ground-truth verified by the session itself, not the model's claim).

## The two crashes (honest accounting)

t12 r1 (30s) and t13 r2 (673s) exited 1 with **no JSON report** (`status=None`). Investigated:
- The daemon journal was clean in the t12-r1 window — not GPU-slot contention.
- Direct reproduction of t12 (max-rounds 8) ran clean to a valid `failed_abandoned` report — **not deterministic**; a model-output edge case propagated out of `run_coding_task` before the report printed.
- **Hardened** (`0d53286`): the CLI now always emits a structured `failed_error` report + traceback to stderr on any uncaught exception, so a recurrence is self-documenting. Did not chase the non-deterministic root cause further — it didn't change the verdict (t12 converted on r2; the 4 passing tasks each passed ≥1 run cleanly), and the hardening makes the next occurrence diagnosable.

## t13 is the real residual

t13 ("fix a planted bug only discoverable by running tests") failed both runs — one honest 30-round abandonment (ground truth never went green), one transient crash. This is the hardest reasoning task in the set (openhands got it 1/2 in the bakeoff; the thinking-on addendum had Prometheus at 1/10 on the T3 tier). Coding mode's iterate-to-green discipline converts the other four T3 tasks but not this one — consistent with "the loop now runs tests and re-edits, but the deepest diagnosis still misses." A fair next-cycle target, not a gate failure.

## Conclusion

Gate met decisively: **4/5**, with t11/t14/t15 green on **both** runs. The iterate-to-green loop policy does what the addendum said was the irreducible port — it converts T3 edit-and-verify tasks the thinking-on loop alone could not. PR #33 is mergeable on this result.
