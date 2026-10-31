"""Stand-in for scripts/distill/teacher_gen.py, for the generation-runner tests.

``run --help`` prints the options gen_runner.py looks for (``--teacher {local,max-payg,max-token-plan}``,
``--budget-usd``, ``--tp-start-concurrency``), or an older help without them when FAKE_TG_OLD=1. ``run ...`` logs its arguments to
$FAKE_STATE/tg-calls.log, checks the teacher (or the judge) through --probe-guard-url like the real
preflight (no answer: exit 3), then finishes one task every FAKE_TG_EVERY seconds until --until: two rows
appended to <out>/<name>.traces.jsonl, then a ``kept`` ledger event in <name>.tasks.jsonl (the real
order). A rerun resumes from the ledger. FAKE_TG_FAIL_LOCAL=N makes the first N local runs exit 3 at once;
FAKE_TG_MAX_RC=N makes every Max (--teacher max-payg / max-token-plan) run exit N at once; FAKE_TG_LOCAL_RC=N every local
run (after writing <name>.report.json only when FAKE_TG_LOCAL_REPORT=1, as a normal end does: a crash
writes none). A run that reaches --until writes <name>.report.json, like the real one.
Its PID goes to $FAKE_STATE/<local|max>-pid.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    state = Path(os.environ["FAKE_STATE"])
    if argv[:2] == ["run", "--help"]:
        print("usage: teacher_gen.py run [-h] --out OUT ...")
        if os.environ.get("FAKE_TG_OLD") != "1":
            print("  --teacher {local,max-payg,max-token-plan}")
            print("  --budget-usd BUDGET_USD")
            print("  --tp-start-concurrency TP_START_CONCURRENCY")
        print("  --probe-guard-url PROBE_GUARD_URL")
        return 0
    with open(state / "tg-calls.log", "a") as f:
        f.write(json.dumps(argv) + "\n")
    p = argparse.ArgumentParser()
    for opt in ("--teacher", "--budget-usd", "--out", "--name", "--until", "--tasks", "--teacher-url",
                "--probe-guard-url", "--judge-url", "--concurrency", "--injection-manifest", "--pack", "--repo",
                "--notify-cmd", "--sandbox-root", "--tp-start-concurrency"):
        p.add_argument(opt)
    p.add_argument("--eval-tools", action="append")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv[1:])
    payg = a.teacher in ("max-payg", "max-token-plan")
    kind = "max" if payg else "local"
    (state / f"{kind}-pid").write_text(f"{os.getpid()}\n")
    if payg and os.environ.get("FAKE_TG_MAX_RC"):
        print(f"stopped: the fake Max teacher exits {os.environ['FAKE_TG_MAX_RC']}")
        return int(os.environ["FAKE_TG_MAX_RC"])
    out = Path(a.out)
    name = a.name or ({"max-payg": "max-payg-gym", "max-token-plan": "max-tokenplan-gym"}.get(a.teacher)
                      or "qwen38-27b-gym")
    if not payg:
        n = int((state / "local-runs").read_text()) if (state / "local-runs").exists() else 0
        (state / "local-runs").write_text(str(n + 1))
        if n < int(os.environ.get("FAKE_TG_FAIL_LOCAL", "0")):
            print("error: 5 errors in a row")
            return 3
        if os.environ.get("FAKE_TG_LOCAL_RC"):
            if os.environ.get("FAKE_TG_LOCAL_REPORT") == "1":
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{name}.report.json").write_text('{"rows_total": 0}\n')
            else:
                print("Traceback (most recent call last): OSError: [Errno 28] No space left on device")
            return int(os.environ["FAKE_TG_LOCAL_RC"])
    if a.probe_guard_url:
        try:
            with urllib.request.urlopen(a.probe_guard_url + "/v1/models", timeout=5) as r:
                served = json.load(r)["data"][0]["id"]
        except Exception as e:  # noqa: BLE001
            print(f"error: the teacher is unreachable: {e}")
            return 3
    else:
        served = "qwen3.8-max"
    out.mkdir(parents=True, exist_ok=True)
    traces, ledger = out / f"{name}.traces.jsonl", out / f"{name}.tasks.jsonl"
    i = sum(1 for line in ledger.open() if line.strip()) if ledger.exists() else 0
    until = datetime.fromisoformat(a.until).timestamp()
    every = float(os.environ.get("FAKE_TG_EVERY", "0.1"))
    while time.time() < until:
        time.sleep(every)
        run_id = uuid.uuid4().hex
        meta = {"source": name, "run_id": run_id, "task_id": f"tg-{i}", "max_derived": payg,
                "model": "qwen3.8-max" if payg else "qwen3.8-27b", "served_model": served}
        rows = [{"messages": [{"role": "user", "content": f"task {i}"}, {"role": "assistant", "content": f"step {k}"}],
                 "_meta": dict(meta, step=k)} for k in (0, 1)]
        with traces.open("a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        with ledger.open("a") as f:
            f.write(json.dumps({"task_id": f"tg-{i}", "status": "kept", "run_id": run_id}) + "\n")
        i += 1
    (out / f"{name}.report.json").write_text(json.dumps({"rows_total": 2 * i, "stop": "until"}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
