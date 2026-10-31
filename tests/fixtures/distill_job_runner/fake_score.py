"""Stand-in for scripts/distill/score.py (run / summarize), for the tower job runner tests.

``run`` appends one row per case to --out (flushed), skipping cases whose last row is ok,
and needs the server's /health to answer 200 for each case (else an error row). Exit 0
when every case is ok, 3 otherwise. FAKE_SCORE_DELAY: seconds per case.
FAKE_SCORE_EXIT_ON_SET=<set>:<code> exits at once with that code for that set.
FAKE_SUMMARIZE_DELAY: seconds a summarize takes (touches $FAKE_STATE/summarize-started first).
Each call is logged to $FAKE_STATE/score-calls.log as "<cmd> <label> <set> <sent>".
"""

import argparse
import json
import os
import sys
import time
import urllib.request


def cases_of(paths):
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    out.append((c["set"], c["id"]))
    return out


def rows_of(path):
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                rows[(r["set"], r["id"])] = r
    return rows


def write_summary(cases, rows, label, out):
    sets = {}
    for s, i in cases:
        e = sets.setdefault(s, {"cases": 0, "scored": 0, "errors": 0, "missing": 0, "ok": 0})
        e["cases"] += 1
        r = rows.get((s, i))
        if r is None:
            e["missing"] += 1
        elif r["status"] != "ok":
            e["errors"] += 1
        else:
            e["scored"] += 1
            e["ok"] += 1
    key = {"sanity": "sanity_score", "eval-tools": "task_success",
           "eval-injection": "injection_resistance", "eval-longctx": "recall"}
    for s, e in sets.items():
        e["metrics"] = {key.get(s, "score"): {"rate": (e["ok"] / e["cases"]) * 0.5 if e["cases"] else 0}}
    with open(out, "w") as f:
        json.dump({"schema": "fake", "label": label, "sets": sets}, f)
    return sum(e["errors"] + e["missing"] for e in sets.values())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--cases", action="append", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--concurrency")
    r.add_argument("--timeout")
    s = sub.add_parser("summarize")
    s.add_argument("--results", required=True)
    s.add_argument("--cases", action="append", required=True)
    s.add_argument("--label")
    s.add_argument("--summary-out")
    s.add_argument("--markdown-out")
    a = ap.parse_args()
    state = os.environ["FAKE_STATE"]
    cases = cases_of(a.cases)
    set_name = cases[0][0] if cases else "?"
    if a.cmd == "summarize":
        delay = float(os.environ.get("FAKE_SUMMARIZE_DELAY", "0"))
        if delay:
            open(os.path.join(state, "summarize-started"), "a").close()
            time.sleep(delay)
        rows = rows_of(a.results)
        write_summary(cases, rows, a.label, a.summary_out)
        with open(os.path.join(state, "score-calls.log"), "a") as f:
            f.write(f"summarize {a.label} {len(set(c[0] for c in cases))}sets 0\n")
        return 0
    stop = os.environ.get("FAKE_SCORE_EXIT_ON_SET", "")
    if stop and stop.split(":")[0] == set_name:
        return int(stop.split(":")[1])
    rows = rows_of(a.out)
    delay = float(os.environ.get("FAKE_SCORE_DELAY", "0.05"))
    sent = 0
    with open(a.out, "a") as f:
        for s_, i in cases:
            prev = rows.get((s_, i))
            if prev and prev["status"] == "ok":
                continue
            time.sleep(delay)
            try:
                with urllib.request.urlopen(a.base_url + "/health", timeout=2) as resp:
                    ok = resp.status == 200
            except Exception:
                ok = False
            row = {"set": s_, "id": i, "label": a.label, "status": "ok" if ok else "error"}
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows[(s_, i)] = row
            sent += 1
    with open(os.path.join(state, "score-calls.log"), "a") as f:
        f.write(f"run {a.label} {set_name} {sent}\n")
    bad = write_summary(cases, rows, a.label, os.path.splitext(a.out)[0] + ".summary.json")
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
