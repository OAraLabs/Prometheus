"""Stand-in for scripts/distill/teacher_ab_run.sh, for the tower job runner tests.

The job runner runs ``/bin/bash <code>/scripts/distill/teacher_ab_run.sh ...``; the test writes that
file as a two-line bash script that execs this one. Every call is logged to $FAKE_STATE/ab-calls.log
(one JSON list of its arguments). Without --execute it prints a dry run and exits 0. With --execute it
checks --until (ISO with an offset), works FAKE_AB_DELAY seconds (never past --until), then writes
ab_report.json + ab_summary.txt in --out and exits FAKE_AB_RC (default 0): 0 with a decision, 1 with
FAKE_AB_VERDICT (default incomplete), anything else without a report. FAKE_AB_UNTIL_EXIT=1 works until
--until and exits 1 (the time box). SIGTERM exits 143 at once (the driver's own on_signal).
"""

import json
import os
import signal
import sys
import time
from datetime import datetime


def main() -> int:
    argv = sys.argv[1:]
    state = os.environ["FAKE_STATE"]
    with open(os.path.join(state, "ab-calls.log"), "a") as f:
        f.write(json.dumps(argv) + "\n")
    if "--execute" not in argv:
        print("DRY RUN (nothing is started, stopped or sent; add --execute)")
        print("1. select: ... 2. arm A (IQ3) ... 3. arm B (Q4_K_M) and the judge ... 4. report")
        return 0
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    a = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--") and argv[i] != "--execute"}
    until = datetime.fromisoformat(a["--until"]).timestamp()
    assert a["--until"][19:] and a["--until"][19] in "+-", a["--until"]
    with open(os.path.join(state, "ab-pid"), "w") as f:
        f.write(f"{os.getpid()}\n")
    out = a["--out"]
    os.makedirs(out, exist_ok=True)
    if os.environ.get("FAKE_AB_UNTIL_EXIT") == "1":
        while time.time() < until:
            time.sleep(0.05)
        rc, verdict = 1, "incomplete"
    else:
        end = min(until, time.time() + float(os.environ.get("FAKE_AB_DELAY", "0.2")))
        while time.time() < end:
            time.sleep(0.05)
        rc = int(os.environ.get("FAKE_AB_RC", "0"))
        verdict = "keep IQ3" if rc == 0 else os.environ.get("FAKE_AB_VERDICT", "incomplete")
    if rc in (0, 1):
        with open(os.path.join(out, "ab_report.json"), "w") as f:
            json.dump({"decision": {"verdict": verdict, "a_minus_b_pts": -3.33, "pairs": 30}}, f)
        with open(os.path.join(out, "ab_summary.txt"), "w") as f:
            f.write(f"Teacher A/B decision: {verdict}\nPass: IQ3_XXS 20/30 vs Q4_K_M 21/30\n")
    else:
        print("error: a server would not start", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
