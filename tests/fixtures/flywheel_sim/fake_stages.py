#!/usr/bin/env python3
"""Stand-ins for the weekly flywheel's stage executables (tests/test_flywheel_window_e2e.py).

One script, one subcommand per program the job starts or reads the output of.
Each writes what the real one writes, in its shape, and spends SIMULATED time
on a shared clock file instead of real time:

  traces    the week's producers: golden exports (the daemon's exporter),
            gym teacher traces (teacher_gen.py's shape), offspring traces
  build     scripts/distill/build.py build: the data policy (train sources,
            strict verification), splits, report.json with gates
  train     the trainer: checkpoint-<step>/ every --save-every steps, adapter/
            and manifest.json when done; SIGTERM saves a checkpoint and exits 75
  quantize  GGUFs into --out; FLYWHEEL_SIM_CRASH=quantize raises (exit 1)
  run       scripts/distill/score.py run (scores from FLYWHEEL_SIM_SCORES)
  compare   scripts/distill/score.py compare (the Phase 6 gate)
  deadman / apply   scripts/distill/setups.py

The clock (FLYWHEEL_SIM_CLOCK) moves only when a stand-in spends time: it
writes the new time, then waits until the job has read it twice (the job's
clock writes what it read, and a read count, to <clock>.seen). The job polls
its child against that clock, so it sends SIGTERM at the simulated stop time,
never misses a tick, and the child sees the signal on the step it was sent. Every call appends a line to FLYWHEEL_SIM_LOG. Standard library only;
never imports prometheus; ends by returning an exit code, never a signal crash.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

EX_TEMPFAIL = 75


# --------------------------------------------------------------------------- #
# The simulated clock and the call log
# --------------------------------------------------------------------------- #


def sim_now() -> datetime:
    return datetime.fromisoformat(Path(os.environ["FLYWHEEL_SIM_CLOCK"]).read_text().strip())


def spend(minutes: float) -> None:
    """Advance the clock by *minutes*, then wait until the job has finished the
    poll that read the new time.

    The job's clock writes "<time> <read count>" to <clock>.seen on every read.
    Waiting for a second read at or after the new time means the job's whole
    poll iteration in between (the SIGTERM it sends at the stop time
    included) is done before this process takes its next step.
    """
    if minutes <= 0:
        return
    clock = Path(os.environ["FLYWHEEL_SIM_CLOCK"])
    target = sim_now() + timedelta(minutes=minutes)
    tmp = clock.with_name(f".{clock.name}.{os.getpid()}")
    tmp.write_text(target.isoformat())
    os.replace(tmp, clock)
    seen = clock.with_name(clock.name + ".seen")
    first: int | None = None
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            when, count = seen.read_text().split()
            if datetime.fromisoformat(when) >= target:
                if first is None:
                    first = int(count)
                elif int(count) > first:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.0005)
    print("sim clock: the job never read the clock", flush=True)
    raise SystemExit(70)


def record(cmd: str, **fields: Any) -> None:
    path = os.environ.get("FLYWHEEL_SIM_LOG")
    if not path:
        return
    try:
        at = sim_now().isoformat()
    except (KeyError, OSError, ValueError):
        at = None
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"cmd": cmd, "at": at, **fields}) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def names_max(name: Any) -> bool:
    return isinstance(name, str) and "max" in re.split(r"[^a-z0-9]+", name.lower())


# --------------------------------------------------------------------------- #
# traces: what the week produces between windows
# --------------------------------------------------------------------------- #


def cmd_traces(a: argparse.Namespace) -> int:
    rows = []
    for i in range(a.first, a.first + a.count):
        call = {"id": f"call_{a.kind}_{i}", "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": f"ls /srv/{i}"})}}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are Prometheus."},
            {"role": "user", "content": f"{a.kind} task {i}: list the files in /srv/{i}"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
        ]
        tools = [{"type": "function", "function": {
            "name": "bash", "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}]
        meta: dict[str, Any] = {"tool_name": "bash", "timestamp": 1_790_000_000.0 + i,
                                "session_id": f"{a.kind}:{i}"}
        if a.kind == "golden":                    # the daemon's exporter
            meta["model"] = a.model or "qwen3.8-max"
        elif a.kind == "gym":                     # teacher_gen.py
            judge = {"verdict": "pass", "model": "qwen3.8-27b", "pass": True}
            meta.update({
                "model": a.model or "qwen3.8-27b", "served_model": "Qwen3.8-27B-IQ3_XXS.gguf",
                "source": "qwen38-27b-gym", "task_id": f"gym-{i}", "category": "other",
                "verdict": "pass", "max_derived": False,
                "verification": {"verified": True, "tool_success": True, "sandboxed": True,
                                 "side_effects": False, "judge": judge},
            })
        else:                                     # offspring: the served student
            meta.update({"model": "qwen3.5-9b", "served_model": a.served, "source": "shadow"})
            if a.judged_by:
                meta.update({"verdict": "pass", "verified_by": a.judged_by})
        rows.append({"messages": messages, "tools": tools, "_meta": meta})
    write_jsonl(Path(a.out), rows, append=True)
    return 0


# --------------------------------------------------------------------------- #
# build: scripts/distill/build.py build
# --------------------------------------------------------------------------- #


def strict_pass(meta: dict[str, Any], judge_models: list[str]) -> bool:
    """build.py's strict verification of a train-source row (every part required)."""
    rec = meta.get("verification")
    if not isinstance(rec, dict) or rec.get("verified") is not True:
        return False
    judge = rec.get("judge")
    return (rec.get("tool_success") is True and rec.get("sandboxed") is True
            and rec.get("side_effects") is False and isinstance(judge, dict)
            and judge.get("verdict") == "pass" and not names_max(judge.get("model"))
            and any(fnmatch.fnmatchcase(str(judge.get("model", "")).lower(), p.lower())
                    for p in judge_models))


def cmd_build(a: argparse.Namespace) -> int:
    spend(a.sim_minutes)
    policy = a.train_sources != ["none"]
    rows: list[dict[str, Any]] = []
    withheld = 0
    for kind, files in (("golden", a.traces or []), ("extra", a.extra_traces or [])):
        for name in files:
            for lineno, trace in enumerate(read_jsonl(Path(name)), 1):
                meta = trace.get("_meta") or {}
                if meta.get("model") == "flywheel-withheld":
                    withheld += 1
                    continue
                trace_id = f"{Path(name).name}:{lineno}"
                tag = meta.get("source") if isinstance(meta.get("source"), str) else None
                source = tag or ("max-golden" if kind == "golden" else "extra")
                model = str(meta.get("model") or "")
                offspring = source.startswith("flywheel/offspring/")
                if not policy:
                    train_ok = model == a.teacher_model
                    label = "tool-success"
                elif offspring and a.offspring_label:
                    train_ok, label = True, a.offspring_label      # a faulty builder
                else:
                    listed = any(fnmatch.fnmatchcase(source, p) for p in a.train_sources)
                    train_ok = (listed and not names_max(source) and not names_max(model)
                                and meta.get("max_derived") is False
                                and any(fnmatch.fnmatchcase(model.lower(), p.lower())
                                        for p in a.train_models)
                                and strict_pass(meta, a.train_models))
                    label = "judge-pass" if train_ok else "tool-success"
                task = str(meta.get("task_id") or trace_id)
                rows.append({
                    "id": "p3-" + hashlib.sha256(trace_id.encode()).hexdigest()[:12],
                    "task_id": task, "group_id": task, "category": meta.get("category", "coding"),
                    "source": source, "label": label, "model": model or None,
                    "max_derived": not train_ok or names_max(model) or names_max(source),
                    "train_ok": train_ok, "messages": trace.get("messages"),
                    "tools": trace.get("tools"),
                })
    ok = {r["id"] for r in rows if r.pop("train_ok")}
    trainable = [r for r in rows if r["id"] in ok]
    val = [r for i, r in enumerate(trainable) if i % 8 == 7]
    train = [r for i, r in enumerate(trainable) if i % 8 != 7]
    held = [r for r in rows if r["id"] not in ok]
    for split, part in (("train", train), ("val", val), ("eval-tools", held)):
        for r in part:
            r["split"] = split
    out = Path(a.out)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "val-cases.jsonl", [{"id": r["id"], "set": "val"} for r in val])
    write_jsonl(out / "eval-tools.jsonl", [{"id": r["id"], "set": "eval-tools"} for r in held])
    models = {r["model"] for r in train + val}
    if not policy:
        teacher = a.teacher_model
    elif len(models) == 1 and not any(r["max_derived"] for r in train + val) \
            and not names_max(next(iter(models))):
        teacher = str(next(iter(models)))
    else:
        teacher = a.teacher_model
    gates: dict[str, Any] = {
        "train_size": {"rows": len(train), "min": a.min_train, "pass": len(train) >= a.min_train},
        "data_policy": {"pass": not policy or not any(r["max_derived"] for r in train + val)},
        "spot_check": ({"status": "PASS"} if a.spot_check_server else {"status": "NOT RUN"}),
    }
    failed = [k for k, v in gates.items() if v.get("pass") is False]
    gates["overall"] = ("FAIL" if failed else "PASS" if a.spot_check_server else "INCOMPLETE")
    (out / "report.json").write_text(json.dumps({
        "schema": "distill-build-report/sim",
        "settings": {"teacher_model": teacher, "train_sources": a.train_sources,
                     "max_model": a.teacher_model},
        "inputs": {"written": {"train.jsonl": len(train), "val.jsonl": len(val),
                               "eval-tools.jsonl": len(held)}, "withheld_stubs": withheld},
        "gates": gates,
    }, indent=1))
    record("build", train=len(train), val=len(val), eval_tools=len(held), teacher=teacher,
           withheld=withheld, overall=gates["overall"])
    print(f"build: train {len(train)} val {len(val)} eval-tools {len(held)} "
          f"({gates['overall']})", flush=True)
    return 0 if gates["overall"] == "PASS" else 1


# --------------------------------------------------------------------------- #
# train: a resumable trainer that checkpoints on SIGTERM
# --------------------------------------------------------------------------- #


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for child in run_dir.glob("checkpoint-*") if run_dir.is_dir() else []:
        m = re.fullmatch(r"checkpoint-(\d+)", child.name)
        if m and child.is_dir():
            found.append((int(m.group(1)), child))
    return sorted(found)


def cmd_train(a: argparse.Namespace) -> int:
    data, run_dir = Path(a.data), Path(a.out)
    rows = len(read_jsonl(data))
    sha = _sha(data)
    have = _checkpoints(run_dir)
    if have and not a.resume:
        print("refusing: the run directory holds a train; pass --resume", flush=True)
        record("train", refused="state without --resume")
        return 2
    step = 0
    if a.resume and have:
        state = json.loads((have[-1][1] / "trainer_state.json").read_text())
        if state["data_sha256"] != sha:
            print("refusing: the data changed since the checkpoint", flush=True)
            record("train", refused="data changed")
            return 3
        step = int(state["step"])
    record("train", start_step=step, steps=rows, data=str(data), stop_at=a.stop_at,
           resume=bool(a.resume))
    stop = {"asked": False}

    def on_term(signum: int, frame: Any) -> None:
        stop["asked"] = True

    signal.signal(signal.SIGTERM, on_term)

    def save(at: int) -> Path:
        ck = run_dir / f"checkpoint-{at}"
        ck.mkdir(parents=True, exist_ok=True)
        (ck / "trainer_state.json").write_text(json.dumps({"step": at, "data_sha256": sha}))
        return ck

    run_dir.mkdir(parents=True, exist_ok=True)
    while step < rows:
        spend(a.minutes_per_row)
        step += 1
        print(f"step {step}/{rows} loss {2.0 / (1 + step):.4f}", flush=True)
        if step % a.save_every == 0:
            save(step)
        if stop["asked"]:
            ck = save(step)
            (run_dir / "manifest.json").write_text(json.dumps({"status": "stopped",
                                                               "step": step}))
            print(f"SIGTERM: saved {ck.name}, exiting {EX_TEMPFAIL}", flush=True)
            record("train-stopped", step=step, checkpoint=ck.name)
            return EX_TEMPFAIL
    adapter = run_dir / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 32,
                                                             "rows": rows}))
    (adapter / "adapter_model.safetensors").write_bytes(b"\0" * 64)
    (run_dir / "manifest.json").write_text(json.dumps({"status": "done", "steps": rows}))
    record("train-done", steps=rows)
    return 0


# --------------------------------------------------------------------------- #
# quantize, score, setups
# --------------------------------------------------------------------------- #


def cmd_quantize(a: argparse.Namespace) -> int:
    record("quantize", version=a.version)
    if "quantize" in os.environ.get("FLYWHEEL_SIM_CRASH", ""):
        spend(4)
        raise RuntimeError("simulated llama-quantize crash: imatrix calibration ran out of memory")
    if not (Path(a.adapter) / "adapter_model.safetensors").is_file():
        print(f"no adapter at {a.adapter}", flush=True)
        return 2
    spend(15)
    out = Path(a.out)
    calibration = len(read_jsonl(Path(a.data) / "train.jsonl"))
    for q in ("Q4_K_M", "Q5_K_M"):
        (out / f"{a.version}-{q}.gguf").write_bytes(b"GGUF" + b"\0" * 60)
    (out / "quantize.json").write_text(json.dumps({"adapter": a.adapter, "imatrix": "train",
                                                   "calibration_rows": calibration}))
    return 0


def _metrics(scores: dict[str, float], cases: int) -> dict[str, Any]:
    def rate(r: float) -> dict[str, Any]:
        return {"k": round(r * cases), "n": cases, "rate": r}

    return {
        "eval-tools": {"cases": cases, "scored": cases, "errors": 0,
                       "metrics": {"task_success": rate(scores["task_success"])}},
        "eval-longctx": {"metrics": {"recall_by_depth": {
            "32k": rate(scores["recall_32k"]), "64k": rate(scores["recall_64k"])}}},
        "eval-injection": {"metrics": {"resistance": rate(scores["injection"])}},
        "sanity": {"metrics": {"sanity": rate(scores["sanity"])}},
    }


def cmd_score_run(a: argparse.Namespace) -> int:
    served = Path(os.environ["FLYWHEEL_SIM_SERVED"]).read_text().strip()
    scores = json.loads(Path(os.environ["FLYWHEEL_SIM_SCORES"]).read_text())[a.label]
    cases = sum(len(read_jsonl(Path(c))) for c in a.cases)
    record("score", label=a.label, served=served, cases=cases)
    spend(2 * cases)
    write_jsonl(Path(a.out), [{"id": f"case-{i}", "label": a.label, "served_model": served,
                               "status": "ok"} for i in range(cases)])
    summary = {"schema": "distill-score-summary/2", "label": a.label, "served_model": served,
               "sets": _metrics(scores, cases)}
    Path(a.summary_out).write_text(json.dumps(summary, indent=1))
    Path(a.markdown_out).write_text(f"# {a.label}\n\ntask success {scores['task_success']}\n")
    return 0


def _rates(summary: dict[str, Any]) -> dict[str, float]:
    sets = summary["sets"]
    out = {"task": sets["eval-tools"]["metrics"]["task_success"]["rate"],
           "inj": sets["eval-injection"]["metrics"]["resistance"]["rate"],
           "sanity": sets["sanity"]["metrics"]["sanity"]["rate"]}
    for depth, r in sets["eval-longctx"]["metrics"]["recall_by_depth"].items():
        out[f"recall@{depth}"] = r["rate"]
    return out


def cmd_score_compare(a: argparse.Namespace) -> int:
    base = _rates(json.loads(Path(a.base).read_text()))
    cand = _rates(json.loads(Path(a.candidate).read_text()))
    checks = {
        "task success >= base + 10 pts": cand["task"] >= base["task"] + 0.10 - 1e-9,
        "injection >= base": cand["inj"] >= base["inj"] - 1e-9,
        "sanity within 3 pts": cand["sanity"] >= base["sanity"] - 0.03 - 1e-9,
        **{f"{k} within 3 pts": cand[k] >= base[k] - 0.03 - 1e-9
           for k in base if k.startswith("recall@")},
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    Path(a.markdown_out).write_text("\n".join(f"- {k}: {'ok' if v else 'FAIL'}"
                                              for k, v in checks.items()) + "\n")
    record("compare", verdict=verdict)
    return 0 if verdict == "PASS" else 1


def cmd_deadman(a: argparse.Namespace) -> int:
    record("deadman", setup=a.setup, at=a.at)
    return 0


def cmd_apply(a: argparse.Namespace) -> int:
    spend(3)
    record("apply", setup=a.setup)
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="fake_stages")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("traces")
    t.add_argument("--kind", choices=("golden", "gym", "offspring"), required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--first", type=int, default=0)
    t.add_argument("--count", type=int, required=True)
    t.add_argument("--model", default="")
    t.add_argument("--served", default="")
    t.add_argument("--judged-by", default="")
    t.set_defaults(fn=cmd_traces)

    b = sub.add_parser("build")
    b.add_argument("--pack")
    b.add_argument("--tokenizer-dir")
    b.add_argument("--traces", action="append")
    b.add_argument("--extra-traces", action="append")
    b.add_argument("--out", required=True)
    b.add_argument("--train-sources", nargs="+", default=["qwen38-27b-gym"])
    b.add_argument("--train-models", nargs="+", default=["qwen3.8-27b*"])
    b.add_argument("--teacher-model", default="qwen3.8-max")
    b.add_argument("--min-train", type=int, default=5)
    b.add_argument("--spot-check-server", default="")
    b.add_argument("--offspring-label", default="")
    b.add_argument("--sim-minutes", type=float, default=20)
    b.set_defaults(fn=cmd_build)

    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True)
    tr.add_argument("--val", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--stop-at", default="")
    tr.add_argument("--resume", action="store_true")
    tr.add_argument("--minutes-per-row", type=float, default=5)
    tr.add_argument("--save-every", type=int, default=10)
    tr.set_defaults(fn=cmd_train)

    q = sub.add_parser("quantize")
    for flag in ("--adapter", "--data", "--out", "--version"):
        q.add_argument(flag, required=True)
    q.set_defaults(fn=cmd_quantize)

    r = sub.add_parser("run")
    r.add_argument("--base-url", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--cases", action="append", required=True)
    r.add_argument("--summary-out", required=True)
    r.add_argument("--markdown-out", required=True)
    r.set_defaults(fn=cmd_score_run)

    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--markdown-out", required=True)
    c.set_defaults(fn=cmd_score_compare)

    d = sub.add_parser("deadman")
    d.add_argument("setup")
    d.add_argument("--at", required=True)
    d.add_argument("--execute", action="store_true")
    d.set_defaults(fn=cmd_deadman)

    ap = sub.add_parser("apply")
    ap.add_argument("setup")
    ap.add_argument("--execute", action="store_true")
    ap.set_defaults(fn=cmd_apply)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
