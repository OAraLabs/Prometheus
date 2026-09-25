"""Command-line entry for the parity harness (see scripts/parity_harness.py)."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

from parity import traces
from parity.instance import DEFAULT_ROOT, HarnessError, RootLock
from parity.model_server import COMPLETIONS_PATHS
from parity.runner import RunOutput, Runner, build_config
from parity.scenarios import BY_NAME, SCENARIOS

SRC_ROOT = Path(__file__).resolve().parents[2]


def _sha(src_root: Path) -> str:
    out = subprocess.run(["git", "-C", str(src_root), "rev-parse", "--short=12", "HEAD"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def _model_identity(out: RunOutput) -> dict:
    ident: dict[str, str] = {}
    for ex in out.exchanges:
        if ex.method != "GET":
            continue
        try:
            body = json.loads(ex.body)
        except (json.JSONDecodeError, TypeError):
            continue
        if ex.path == "/v1/models" and isinstance(body, dict):
            ids = [m.get("id") or m.get("name") for m in body.get("data", []) or body.get("models", [])]
            if ids:
                ident.setdefault(ex.upstream, str(ids[0]).rsplit("/", 1)[-1])
        if ex.path == "/props" and isinstance(body, dict) and body.get("build_info"):
            ident[f"{ex.upstream}_server"] = f"llama.cpp {body['build_info']}"
    served = [ex.upstream for ex in out.exchanges
              if ex.method == "POST" and ex.path in COMPLETIONS_PATHS]
    ident["completions_by_backend"] = json.dumps({u: served.count(u) for u in sorted(set(served))})
    return ident


def cmd_record(args: argparse.Namespace) -> int:
    names = args.scenario or [s.name for s in SCENARIOS]
    upstreams = {"primary": args.upstream_primary, "alt": args.upstream_alt}
    failed = 0
    with RootLock(args.root):
        for name in names:
            scenario = BY_NAME[name]
            config_text = build_config(SRC_ROOT, scenario)
            print(f"[record] {name}: {scenario.covers}", flush=True)
            out = Runner(scenario, mode="record", root=args.root, src_root=SRC_ROOT,
                         config_text=config_text, upstreams=upstreams).run()
            problems = list(out.errors)
            if scenario.require is not None and not out.errors:
                problems += scenario.require(out.evidence())
            n_comp = sum(1 for ex in out.exchanges
                         if ex.method == "POST" and ex.path in COMPLETIONS_PATHS)
            if problems:
                failed += 1
                print(f"[record] {name}: NOT SAVED — " + "; ".join(problems), flush=True)
                print(f"[record]   daemon log: {out.daemon_log}", flush=True)
                if args.keep_failed:
                    raw = args.root.parent / f"{args.root.name}.{name}.failed.json"
                    raw.write_text(json.dumps({"steps": out.steps, "exchanges": [
                        e.to_json() for e in out.exchanges]}, indent=1, default=str))
                    print(f"[record]   raw: {raw}", flush=True)
                continue
            meta = {
                "date": _dt.date.today().isoformat(),
                "prometheus_sha": _sha(SRC_ROOT),
                "models": _model_identity(out),
                "completions": n_comp,
            }
            from parity.compare import expected_from
            tpath = traces.write_trace(SRC_ROOT, scenario, config_text=config_text,
                                       exchanges=out.exchanges, meta=meta)
            epath = traces.write_expected(SRC_ROOT, name, expected_from(out, args.root))
            # The recording's RAW observables, outside the repo: a normalization
            # change re-derives expected.json from these (`rebaseline`) instead
            # of asking the model again.
            raw_dir = _raw_dir(args.root)
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{name}.json").write_text(
                json.dumps({"steps": out.steps, "stores": out.stores}, default=str))
            print(f"[record] {name}: saved {n_comp} completions -> {tpath.name}, "
                  f"{epath.name} (boot {out.boot_seconds:.1f}s, {out.shutdown})", flush=True)
    return 1 if failed else 0


def _raw_dir(root: Path) -> Path:
    return root.parent / f"{root.name}.recorded-raw"


def cmd_rebaseline(args: argparse.Namespace) -> int:
    """Re-derive expected.json from a recording's raw observables."""
    from parity.normalize import normalize_observables
    names = args.scenario or traces.available(SRC_ROOT)
    for name in names:
        raw = _raw_dir(args.root) / f"{name}.json"
        if not raw.exists():
            print(f"[rebaseline] {name}: no raw recording at {raw} — re-record it", flush=True)
            return 2
        traces.write_expected(SRC_ROOT, name, normalize_observables(json.loads(raw.read_text())))
        traces.renormalize_trace(SRC_ROOT, name)
        print(f"[rebaseline] {name}: expected.json re-derived from the recording; "
              f"trace requests re-normalized", flush=True)
    return 0


def replay_one(name: str, root: Path) -> RunOutput:
    trace = traces.load_trace(SRC_ROOT, name)
    scenario = traces.scenario_from_trace(trace, BY_NAME.get(name))
    return Runner(scenario, mode="replay", root=root, src_root=SRC_ROOT,
                  config_text=trace["config"], recorded=traces.exchanges_from_trace(trace),
                  turn_timeout=180.0).run()


def cmd_replay(args: argparse.Namespace) -> int:
    from parity.compare import compare, render_report
    names = args.scenario or traces.available(SRC_ROOT)
    if not names:
        print("[replay] no traces under tests/fixtures/parity — nothing was checked", flush=True)
        return 2
    worst = 0
    with RootLock(args.root):
        for name in names:
            out = replay_one(name, args.root)
            result = compare(out, traces.load_expected(SRC_ROOT, name), args.root)
            print(render_report(name, result), flush=True)
            if args.dump:
                args.dump.mkdir(parents=True, exist_ok=True)
                (args.dump / f"{name}.actual.json").write_text(
                    json.dumps(result.actual, indent=1, sort_keys=True, default=str))
            worst = max(worst, result.exit_code)
    return worst


def _digest(obj) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cmd_stability(args: argparse.Namespace) -> int:
    """Replay every trace twice on unchanged code; both runs must match the
    recording AND each other, byte for byte after normalization."""
    from parity.compare import compare
    names = args.scenario or traces.available(SRC_ROOT)
    runs: list[dict[str, tuple[int, str, str]]] = []
    with RootLock(args.root):
        for r in range(args.runs):
            this: dict[str, tuple[int, str, str]] = {}
            for name in names:
                out = replay_one(name, args.root)
                res = compare(out, traces.load_expected(SRC_ROOT, name), args.root)
                reqs = [s.request for s in out.served]
                this[name] = (res.exit_code, _digest(res.actual), _digest(reqs))
                print(f"[stability] run {r + 1} {name}: exit {res.exit_code} "
                      f"observables {this[name][1]} requests {this[name][2]}", flush=True)
            runs.append(this)
    ok = True
    print("\n| scenario | " + " | ".join(f"run {i + 1}" for i in range(len(runs)))
          + " | identical |\n|---|" + "---|" * (len(runs) + 1))
    for name in names:
        cells = [runs[i][name] for i in range(len(runs))]
        same = len({(c[1], c[2]) for c in cells}) == 1 and all(c[0] == 0 for c in cells)
        ok &= same
        print(f"| {name} | " + " | ".join(
            f"{'PARITY' if c[0] == 0 else 'DIFF'} `{c[1]}`/`{c[2]}`" for c in cells)
            + f" | {'yes' if same else 'NO'} |")
    print(f"\n[stability] {'IDENTICAL' if ok else 'NOT identical'} across {len(runs)} runs "
          f"(cells: observables digest / model-requests digest)", flush=True)
    return 0 if ok else 1


def cmd_bench(args: argparse.Namespace) -> int:
    """Daemon overhead per round (p50/p95) and RSS, over repeated replays."""
    import platform
    import statistics
    from parity import bench
    from parity.compare import compare
    names = args.scenario or traces.available(SRC_ROOT)
    per_run: list[dict] = []
    with RootLock(args.root):
        for r in range(args.runs):
            pooled: list[float] = []
            rss: list[dict] = []
            by_scenario: dict[str, list[float]] = {}
            invalid: list[str] = []
            for name in names:
                out = replay_one(name, args.root)
                res = compare(out, traces.load_expected(SRC_ROOT, name), args.root)
                if res.exit_code != 0:
                    invalid.append(name)
                samples, notes = bench.samples(out)
                for n in notes:
                    print(f"[bench] note: {n}", flush=True)
                ms = [s.overhead_ms for s in samples]
                pooled += ms
                by_scenario[name] = ms
                rss += out.rss
            st = bench.run_stats(pooled, rss)
            per_run.append({"run": r + 1, "rounds": st.rounds, "p50_ms": st.p50_ms,
                            "p95_ms": st.p95_ms, "mean_ms": st.mean_ms,
                            "max_hwm_mb": st.max_hwm_kb / 1024, "max_rss_mb": st.max_rss_kb / 1024,
                            "invalid": invalid,
                            "by_scenario_p50": {k: bench._pct(v, 0.5) for k, v in by_scenario.items()}})
            print(f"[bench] run {r + 1}: {st.rounds} rounds, p50 {st.p50_ms:.1f} ms, "
                  f"p95 {st.p95_ms:.1f} ms, mean {st.mean_ms:.1f} ms, peak RSS "
                  f"{st.max_hwm_kb / 1024:.0f} MB" + (f", NOT AT PARITY: {invalid}" if invalid else ""),
                  flush=True)
    if any(r["invalid"] for r in per_run):
        print("[bench] a run was not at parity — its timings describe different behaviour; "
              "no baseline is reported", flush=True)
        return 1
    summary = {
        "host": platform.node() and "the recording host", "python": platform.python_version(),
        "runs": len(per_run), "scenarios": names,
        "p50_ms": bench.band([r["p50_ms"] for r in per_run]),
        "p95_ms": bench.band([r["p95_ms"] for r in per_run]),
        "mean_ms": bench.band([r["mean_ms"] for r in per_run]),
        "peak_rss_mb": bench.band([r["max_hwm_mb"] for r in per_run]),
        "rounds_per_run": statistics.median([r["rounds"] for r in per_run]),
        "per_run": per_run,
    }
    print("\n[bench] daemon overhead per round (model and tool time excluded), "
          f"{len(per_run)} runs x {summary['rounds_per_run']:.0f} rounds:")
    for key in ("p50_ms", "p95_ms", "mean_ms", "peak_rss_mb"):
        print(f"  {key:12s} {summary[key]}")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))
        print(f"[bench] wrote {args.out}")
    return 0


def cmd_normalizations(args: argparse.Namespace) -> int:
    from parity.normalize import all_rules
    for rule in all_rules():
        print(f"- {rule.name} [{rule.family}]\n    replaces: {rule.replaces}\n"
              f"    why:      {rule.why}\n    cost:     {rule.cost}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="parity_harness")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="isolated root for the daemon under test (fixed path; see instance.py)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record scenarios against a real model")
    r.add_argument("--scenario", action="append", choices=sorted(BY_NAME))
    r.add_argument("--upstream-primary", required=True,
                   help="llama.cpp server URL the primary model is proxied to")
    r.add_argument("--upstream-alt", required=True,
                   help="second backend (Ollama) URL, for the model switch")
    r.add_argument("--keep-failed", action="store_true")
    r.set_defaults(fn=cmd_record)

    rp = sub.add_parser("replay", help="replay the recorded traces and diff")
    rp.add_argument("--scenario", action="append")
    rp.add_argument("--dump", type=Path, help="write each run's normalized observables here")
    rp.set_defaults(fn=cmd_replay)

    st = sub.add_parser("stability", help="replay twice; results must be identical")
    st.add_argument("--scenario", action="append")
    st.add_argument("--runs", type=int, default=2)
    st.set_defaults(fn=cmd_stability)

    b = sub.add_parser("bench", help="daemon overhead per round and RSS over N replays")
    b.add_argument("--scenario", action="append")
    b.add_argument("--runs", type=int, default=10)
    b.add_argument("--out", type=Path)
    b.set_defaults(fn=cmd_bench)

    rb = sub.add_parser("rebaseline", help="re-derive expected.json from the raw recording")
    rb.add_argument("--scenario", action="append")
    rb.set_defaults(fn=cmd_rebaseline)

    n = sub.add_parser("normalizations", help="print every normalization rule")
    n.set_defaults(fn=cmd_normalizations)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except HarnessError as exc:
        print(f"[parity] HARNESS ERROR (not a pass): {exc}", file=sys.stderr, flush=True)
        return 2
