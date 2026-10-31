"""Stand-in for scripts/distill/feasibility_27b.py, for the tower job runner tests.

--execute --seqs N writes a one-length report after FAKE_FEAS_DELAY seconds; a length
over FAKE_FEAS_MAX_FIT is an OOM; the length FAKE_FEAS_STOPPED_AT reports a child stopped by
SIGTERM (as the real probe does when a restore stops its distill-feas27b-* scope). --measured merges and decides. Calls are logged to
$FAKE_STATE/feas-calls.log.
"""

import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base-model")
    ap.add_argument("--seqs", default="16384,12288,8192,4096")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--stop-at")
    ap.add_argument("--out", required=True)
    ap.add_argument("--measured")
    ap.add_argument("--no-scope", action="store_true")
    a = ap.parse_args()
    state = os.environ["FAKE_STATE"]
    seqs = [int(s) for s in a.seqs.split(",")]
    with open(os.path.join(state, "feas-calls.log"), "a") as f:
        f.write(("execute " if a.execute else "measured ") + a.seqs + "\n")
    if a.execute:
        assert a.stop_at and ("+" in a.stop_at[10:] or "-" in a.stop_at[10:]), a.stop_at
        time.sleep(float(os.environ.get("FAKE_FEAS_DELAY", "0.3")))
        seq = seqs[0]
        fits = seq <= int(os.environ.get("FAKE_FEAS_MAX_FIT", "12288"))
        res = {"max_seq": seq, "fits": fits, "peak_gb": 21.5 if fits else None,
               "tokens_per_s": 410.0 if fits else None, "tps_source": "measured" if fits else "assumed",
               "share_untruncated": 0.2, "wall_clock_h": 50.0 if fits else None,
               "error": None if fits else "cuda_oom"}
        if str(seq) == os.environ.get("FAKE_FEAS_STOPPED_AT"):  # its scope stopped by a restore
            res.update(fits=None, peak_gb=None, tokens_per_s=None, wall_clock_h=None,
                       error="stopped (signal 15): a restore?")
        rep = {"schema": "distill-feasibility-27b/1", "mode": "execute", "results": [res],
               "decision": {"verdict": "incomplete", "say": "one length"}}
    else:
        prev = json.load(open(a.measured))
        res = prev["results"]
        ok = [r for r in res if r.get("fits") and r["max_seq"] >= 8192]
        dec = ({"verdict": "local-viable", "max_seq": max(r["max_seq"] for r in ok),
                "say": f"{max(r['max_seq'] for r in ok)} fits locally"} if ok
               else {"verdict": "rent", "say": "no length >= 8k fits"})
        rep = {"schema": "distill-feasibility-27b/1", "mode": "measured", "results": res, "decision": dec}
    with open(a.out, "w") as f:
        json.dump(rep, f)
    print("decision:", rep["decision"]["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
