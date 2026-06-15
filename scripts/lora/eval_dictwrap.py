#!/usr/bin/env python3
"""Phase 3 — the held-out eval JUDGE (the whole ballgame).

Given two HELD-OUT harvest DBs (base server vs LoRA server, same held-out
taskset) and the control pass rates, decides PASS/FAIL against PRE-REGISTERED
bars. The bars are fixed here in code, before any LoRA number is seen:

  HELD-OUT  dict-wrap repair rate must drop ≥ 40% relative vs base
            (rate = dict_wrap_unwrap pairs captured / runs; the model wrapping
            LESS on shapes the train never drilled = genuine generalization).
  CONTROL   pass rate must not regress > 2 pts vs base.

Validity guard: if the BASE barely wraps the held-out shapes (rate < 0.20),
there is nothing to reduce — the eval is INCONCLUSIVE, not a pass. Said loudly.

    python scripts/lora/eval_dictwrap.py \
        --base-db heldout-base.db --lora-db heldout-lora.db --runs 40 \
        --control-base 47/50 --control-lora 46/50
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from prometheus.learning.transition_taxonomy import classify_transition

REL_DROP_BAR = 0.40          # ≥40% relative reduction on held-out dict-wrap
CONTROL_REGRESSION_BAR = 2.0  # control pass rate within 2 pts of base
BASE_VALIDITY_FLOOR = 0.20    # base must wrap held-out enough to measure a drop


def count_dictwrap(db: str) -> int:
    conn = sqlite3.connect(str(Path(db).expanduser()))
    try:
        rows = conn.execute(
            "SELECT pair_source, rejected, chosen FROM training_pairs").fetchall()
    finally:
        conn.close()
    n = 0
    for src, rej_s, cho_s in rows:
        try:
            rej = json.loads(rej_s) if rej_s else None
            cho = json.loads(cho_s) if cho_s else None
        except (ValueError, TypeError):
            rej, cho = None, None
        if classify_transition(src, rej, cho) == "dict_wrap_unwrap":
            n += 1
    return n


def _frac(s: str) -> float:
    a, b = s.split("/")
    return 100.0 * int(a) / int(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-db", required=True, help="held-out harvest, BASE server")
    ap.add_argument("--lora-db", required=True, help="held-out harvest, LoRA server")
    ap.add_argument("--runs", type=int, required=True, help="held-out run count (denominator)")
    ap.add_argument("--control-base", required=True, help="passed/total on control, base")
    ap.add_argument("--control-lora", required=True, help="passed/total on control, lora")
    args = ap.parse_args()

    base_n = count_dictwrap(args.base_db)
    lora_n = count_dictwrap(args.lora_db)
    base_rate = base_n / args.runs
    lora_rate = lora_n / args.runs
    rel_drop = (base_rate - lora_rate) / base_rate if base_rate > 0 else 0.0

    cb, cl = _frac(args.control_base), _frac(args.control_lora)
    control_delta = cl - cb

    print("══ Phase 3 — held-out dict-wrap eval ══")
    print(f"  held-out runs           : {args.runs}")
    print(f"  BASE dict-wrap repairs  : {base_n}  → rate {base_rate:.3f}")
    print(f"  LoRA dict-wrap repairs  : {lora_n}  → rate {lora_rate:.3f}")
    print(f"  relative reduction      : {rel_drop*100:5.1f}%   (bar ≥ {REL_DROP_BAR*100:.0f}%)")
    print(f"  control pass  base→lora : {cb:.1f}% → {cl:.1f}%   (Δ {control_delta:+.1f} pts, bar ≥ -{CONTROL_REGRESSION_BAR:.0f})")

    valid = base_rate >= BASE_VALIDITY_FLOOR
    held_pass = valid and rel_drop >= REL_DROP_BAR
    control_pass = control_delta >= -CONTROL_REGRESSION_BAR

    print("──")
    if not valid:
        print(f"  ⚠ INCONCLUSIVE: base rate {base_rate:.3f} < {BASE_VALIDITY_FLOOR} "
              "— held-out shapes don't induce wrapping in the base model; "
              "the eval cannot measure a reduction. Pick wrap-inducing held-out shapes.")
        return 2
    print(f"  held-out gate : {'PASS' if held_pass else 'FAIL'}")
    print(f"  control gate  : {'PASS' if control_pass else 'FAIL'}")
    verdict = held_pass and control_pass
    print(f"  ══ VERDICT: {'PASS — LoRA generalizes, no regression' if verdict else 'FAIL'} ══")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
