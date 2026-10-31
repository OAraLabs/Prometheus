"""Stand-in for scripts/distill/train_qlora.py --smoke, for the tower job runner tests.

The runner starts ``<train python> <code>/scripts/distill/train_qlora.py ... --smoke``; the test copies
this file there. Every call is logged to $FAKE_STATE/smoke-calls.log (a JSON object: argv and the
offline env). It checks --stop-at (ISO with an offset), writes <out>/run_state.json, works
FAKE_SMOKE_DELAY seconds (never past --stop-at), then writes <out>/manifest.json the way the trainer
does. FAKE_SMOKE_MODE: pass (default; exit 0, status smoke-train-pass), fail (exit 1,
smoke-train-fail), until (works until --stop-at, status stopped, exit 0: the time box), refuse (exit 3,
no manifest). SIGTERM: status stopped, exit 0 (the trainer checkpoints and ends cleanly).
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
    with open(os.path.join(state, "smoke-calls.log"), "a") as f:
        f.write(json.dumps({"argv": argv, "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE")}) + "\n")
    a = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    stop_at = a["--stop-at"]
    assert stop_at[19:] and stop_at[19] in "+-", stop_at
    until = datetime.fromisoformat(stop_at).timestamp()
    mode = os.environ.get("FAKE_SMOKE_MODE", "pass")
    out = a["--out"]
    if mode == "refuse":
        print("REFUSED: 9/9 rows are Max-derived", file=sys.stderr)
        return 3
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(state, "smoke-pid"), "w") as f:
        f.write(f"{os.getpid()}\n")
    with open(os.path.join(out, "run_state.json"), "w") as f:
        json.dump({"schema": "distill-train-run-state/1"}, f)

    def manifest(status: str, smoke: dict) -> None:
        with open(os.path.join(out, "manifest.json"), "w") as f:
            json.dump({"status": status, "max_derived": True, "tags": ["max-derived"], "smoke": smoke,
                       "adapter": os.path.join(out, "adapter-max-derived")}, f)

    stopped = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.append(1))
    end = until if mode == "until" else min(until, time.time() + float(os.environ.get("FAKE_SMOKE_DELAY", "0.2")))
    while time.time() < end and not stopped:
        time.sleep(0.05)
    if stopped or mode == "until":
        if "--smoke" not in argv:
            with open(os.path.join(out, "manifest.json"), "w") as f:
                json.dump({"status": "stopped", "max_derived": True, "step": 300, "tokens_seen": 3e7,
                           "reason": "stop-at"}, f)
            return 0
        manifest("stopped", {"steps": 12})
        print("smoke stopped (stop-at) at step 12: no verdict; finish it with --smoke --resume")
        return 0
    ok = mode == "pass"
    if "--smoke" not in argv:  # a full run (the train-full step): done, with its val history
        with open(os.path.join(out, "manifest.json"), "w") as f:
            json.dump({"status": "done" if ok else "failed", "max_derived": True, "tags": ["max-derived"],
                       "step": 900, "tokens_seen": 96_800_000, "reason": None if ok else "diverged: 3 steps",
                       "val_history": [{"step": 0, "val_loss": 1.64}, {"step": 900, "val_loss": 1.2}],
                       "adapter": os.path.join(out, "adapter-max-derived")}, f)
        return 0 if ok else 1
    manifest("smoke-train-pass" if ok else "smoke-train-fail", {
        "steps": 50, "loss_first10": 1.25, "loss_last10": 0.75 if ok else 1.26, "loss_decreased": ok,
        "checkpoint_round_trip": True})
    print(f"smoke (training): 50 steps, loss 1.2500 -> {0.75 if ok else 1.26:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
