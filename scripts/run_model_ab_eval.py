#!/usr/bin/env python3
"""
Model A/B Eval Orchestrator  —  Gemma-4-26B  vs  Qwen3.6-35B-A3B
===============================================================

Runs the SAME Prometheus scored-eval pipeline against two locally-served models,
one at a time, with a SHARED NEUTRAL JUDGE, and produces a side-by-side report.
Answers: "what does a Prometheus user actually get with model A vs model B?"

Every constant here is bound to what the live system exposes (survey-confirmed),
NOT guessed:

  • Scored-eval entrypoint = scripts/run_nightly_evals.py -> EvalRunner -> the
    three DeepEval metrics (ToolUsage/TaskCompletion/NoHallucination). It runs
    IN-PROCESS, reading model.base_url for the contestant; swapping the model on
    the server is therefore sufficient. Args: --config / --output-dir / --tier /
    --verbose. --output-dir is a DIRECTORY; the runner writes results_<ts>.json
    into it. There is NO --out and NO --judge flag.
  • Result schema (per task): {"results": [{... "metrics": [{"metric_name",
    "score", "threshold", "passed", "reasoning"}], "error", ...}]}. The real
    metric_name values are "Tool Usage" / "Task Completion" / "No Hallucination".
  • Everything runs under `uv run` — the eval + prometheus.* packages live in
    the uv venv, not system python3.

NEUTRAL JUDGE — the crux of a fair A/B:
  run_nightly_evals reads evals.judge_base_url from config, defaulting to the
  model endpoint. If that stays pointed at the contestant endpoint, swapping the
  model makes each model judge itself (Qwen by Qwen, Gemma by Gemma) — the exact
  bias to avoid. So the ephemeral eval config here repoints the judge at a
  DIFFERENT, independent OpenAI-compatible endpoint via OARA_JUDGE_BASE_URL
  (default the mini's local 3090 ollama at localhost:11434), which is:
    - independent of BOTH GPU-box contestants,
    - untouched by the production stop (it lives on the mini),
    - confirmed to honor OpenAI response_format:json_schema, so the judge's
      constrained-decode (no-parse-failures) guarantee still holds.
  The judge MODEL is pinned (OARA_JUDGE_MODEL, default llama3.1:8b — non-Qwen,
  non-Gemma) because a multi-model endpoint's auto-detect is non-deterministic.
  NOTE: the judge speaks the OpenAI protocol; pointing it at the Anthropic API
  is a separate, larger change (not done here).

Production is OFFLINE during the eval window (two ~20 GB models won't co-reside
on 24 GB). This stops the production daemon + the GPU-box llama-server, runs
ephemeral eval servers, and RESTORES both on exit (success, failure, AND
Ctrl-C). The neutral judge on the mini stays up. USE --dry-run FIRST. A real run
is a deliberate, production-stopping action — not something to do casually.

Honesty/fairness guarantees:
  - Same harness commit, same golden tasks, SAME NEUTRAL JUDGE, same thinking
    setting (config suppress_thinking), same hardware for both legs.
  - Recipes are DERIVED FROM THE LIVE llama-server unit, so they can't drift
    from production. Only the GGUF (and the mmproj-drop Qwen requires) differs —
    that delta + the quant asymmetry (Gemma Q4_K_M vs Qwen UD-Q4_K_XL) are
    written to the per-leg manifest, so any accidental difference is visible.

RUN UNDER UV:
    uv run python scripts/run_model_ab_eval.py --dry-run

Required env (sourced, so no hostnames/IPs live in this committed file):
    OARA_4090_SSH         ssh target for the GPU box (alias or user@host)
    OARA_LLAMA_BASE_URL   what config model.base_url points at, e.g. http://HOST:8080
  Optional:
    OARA_QWEN_GGUF        Qwen GGUF (basename resolved against Gemma's model dir,
                          or an absolute path on the GPU box)
    OARA_JUDGE_BASE_URL   default http://localhost:11434
    OARA_JUDGE_MODEL      default llama3.1:8b
    OARA_EVAL_OUT         results root (default ~/.prometheus/evals/ab)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ════════════════════════════════════════════════════════════════════════
# wiring (survey-confirmed against the live system)
# ════════════════════════════════════════════════════════════════════════
# {cfg} = ephemeral eval config (judge overridden); {dir} = per-leg output dir.
EVAL_CMD_TEMPLATE = (
    "uv run python scripts/run_nightly_evals.py --config {cfg} --output-dir {dir}"
)
SMOKE_CMD = "uv run python scripts/smoke_test_tool_calling.py --test basic"

JUDGE_BASE_URL = os.environ.get("OARA_JUDGE_BASE_URL", "http://localhost:11434")
JUDGE_MODEL = os.environ.get("OARA_JUDGE_MODEL", "llama3.1:8b")

LLAMA_UNIT = "llama-server"      # systemd --user unit on the GPU box
PROMETHEUS_UNIT = "prometheus"   # production daemon on the mini (local)

SSH_TARGET = os.environ.get("OARA_4090_SSH")
LLAMA_BASE = os.environ.get("OARA_LLAMA_BASE_URL")   # config model.base_url
QWEN_GGUF = os.environ.get("OARA_QWEN_GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
RESULTS_DIR = Path(os.environ.get(
    "OARA_EVAL_OUT", str(Path.home() / ".prometheus" / "evals" / "ab")
))

HEALTH_TIMEOUT_S = 300   # cold-load of a 21 GB MoE + large-ctx KV alloc is slow
HEALTH_POLL_S = 3
# metric_name keys exactly as the runner serializes them (runner.py)
METRIC_KEYS = {
    "tool_usage": "Tool Usage",
    "task_completion": "Task Completion",
    "no_hallucination": "No Hallucination",
}
LOG = lambda m: print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════
# ssh / shell / http helpers
# ════════════════════════════════════════════════════════════════════════
def ssh(cmd: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    if not SSH_TARGET:
        sys.exit("OARA_4090_SSH not set — refusing to guess the GPU-box address.")
    return subprocess.run(["ssh", SSH_TARGET, cmd], check=check, text=True,
                          capture_output=capture)

def sh(cmd: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, check=check, text=True, capture_output=capture)

def http_json(url: str, timeout: float = 10.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


# ════════════════════════════════════════════════════════════════════════
# recipe derivation — read the LIVE unit, derive both models from it
# ════════════════════════════════════════════════════════════════════════
def read_live_execstart() -> str:
    """Canonical launcher = the production unit's ExecStart, verbatim.

    Handles the multi-line (backslash-continued) ExecStart the real unit uses —
    a naive 'first line that starts with ExecStart=' collapses the recipe to
    just the binary path.
    """
    out = ssh(f"systemctl --user cat {LLAMA_UNIT}").stdout
    parts: list[str] = []
    capturing = False
    for raw in out.splitlines():
        s = raw.strip()
        if not capturing:
            if s.startswith("ExecStart="):
                capturing = True
                s = s.split("ExecStart=", 1)[1].strip()
            else:
                continue
        cont = s.endswith("\\")
        parts.append(s[:-1].strip() if cont else s)
        if not cont:
            break
    if not parts:
        sys.exit(f"Could not find ExecStart in {LLAMA_UNIT} unit.")
    cmd = " ".join(p for p in parts if p).strip()
    if not cmd:
        sys.exit(f"{LLAMA_UNIT} ExecStart parsed empty — refusing to serve a "
                 f"recipe with no flags.")
    return cmd

def _gguf_path(tokens: list[str]) -> str | None:
    for i, t in enumerate(tokens):
        if t in ("-m", "--model") and i + 1 < len(tokens):
            return tokens[i + 1]
    return None

def _remote_qwen_path(gemma_model_path: str | None) -> str:
    """Resolve the Qwen GGUF ON THE GPU BOX. An absolute env path is used as-is;
       otherwise the basename is placed in Gemma's model dir (same GPU-box dir),
       so we never accidentally point at a path on the mini."""
    if QWEN_GGUF.startswith("/"):
        return QWEN_GGUF
    if gemma_model_path:
        return str(Path(gemma_model_path).parent / Path(QWEN_GGUF).name)
    return Path(QWEN_GGUF).name  # caller should set an absolute OARA_QWEN_GGUF

def derive_recipes(canonical: str) -> dict[str, list[str]]:
    """Gemma recipe = canonical launcher verbatim.
       Qwen recipe  = canonical with -m -> Qwen GGUF and --mmproj dropped
                      (Qwen 35B is text-only; the Gemma vision projector is wrong).
       Everything else (--jinja, --reasoning-budget, -c, -ngl, port, host,
       flash-attn, parallel, batch sizes) is INHERITED, so the field stays even."""
    toks = shlex.split(canonical)
    gemma = list(toks)
    qwen_path = _remote_qwen_path(_gguf_path(toks))
    qwen: list[str] = []
    skip_next = False
    for t in toks:
        if skip_next:
            skip_next = False
            continue
        if t in ("-m", "--model"):
            qwen += [t, qwen_path]
            skip_next = True             # skip the original gguf path
        elif t == "--mmproj":
            skip_next = True             # drop flag + its value
        elif t.startswith("--mmproj="):
            continue                     # drop inline form
        else:
            qwen.append(t)
    return {"gemma": gemma, "qwen": qwen}


# ════════════════════════════════════════════════════════════════════════
# ephemeral eval config — the judge override, without mutating the real file
# ════════════════════════════════════════════════════════════════════════
def write_eval_config(run_dir: Path) -> Path:
    import yaml  # in the uv venv
    src = Path("config/prometheus.yaml")
    try:
        cfg = yaml.safe_load(src.read_text())
    except yaml.YAMLError as e:
        sys.exit(f"Could not parse {src} for judge override: {e}")
    evals = cfg.setdefault("evals", {})
    evals["judge_base_url"] = JUDGE_BASE_URL
    evals["judge_model"] = JUDGE_MODEL
    out = run_dir / "eval_config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out

def judge_preflight() -> list[str]:
    """Confirm the neutral judge endpoint is up and HAS the pinned model.
       Empty list = clean."""
    models = http_json(f"{JUDGE_BASE_URL.rstrip('/')}/v1/models", timeout=8)
    if models is None:
        return [f"judge endpoint {JUDGE_BASE_URL} unreachable"]
    ids = [m.get("id") for m in (models.get("data") or [])]
    if JUDGE_MODEL not in ids:
        return [f"judge model '{JUDGE_MODEL}' not on {JUDGE_BASE_URL} "
                f"(have: {ids}); pull it first (e.g. `ollama pull {JUDGE_MODEL}`)"]
    return []


# ════════════════════════════════════════════════════════════════════════
# server lifecycle (ephemeral; production restored on exit)
# ════════════════════════════════════════════════════════════════════════
class ProductionGuard:
    """Stops production for the eval window; ALWAYS restores it (success,
       failure, KeyboardInterrupt). The neutral judge on the mini is untouched."""
    def __init__(self, manual: bool):
        self.manual = manual
        self.stopped = False
        self.eval_pid: str | None = None
        self.eval_gguf: str | None = None

    def __enter__(self):
        if self.manual:
            return self
        LOG("Stopping production daemon + GPU-box llama-server (eval needs VRAM)…")
        sh(f"systemctl --user stop {PROMETHEUS_UNIT}", check=False)
        ssh(f"systemctl --user stop {LLAMA_UNIT}", check=False)
        self.stopped = True
        return self

    def serve(self, recipe: list[str], label: str):
        self.kill_eval_server()
        self.eval_gguf = _gguf_path(recipe)
        cmd = " ".join(shlex.quote(t) for t in recipe)
        LOG(f"[{label}] launching eval server:\n    {cmd}")
        launch = f"nohup {cmd} > /tmp/eval-{label}.log 2>&1 & echo $!"
        self.eval_pid = ssh(launch).stdout.strip()
        LOG(f"[{label}] eval server pid={self.eval_pid} (log: GPU-box:/tmp/eval-{label}.log)")

    def kill_eval_server(self):
        if self.eval_pid:
            ssh(f"kill {self.eval_pid} 2>/dev/null || true", check=False)
            self.eval_pid = None
        # net: kill any llama-server holding THIS leg's model file. Production is
        # stopped during the window, so matching the gguf path is unambiguous.
        if self.eval_gguf:
            ssh(f"pkill -f {shlex.quote(self.eval_gguf)} 2>/dev/null || true", check=False)
            self.eval_gguf = None

    def _wait_port_free(self, timeout: float = 30.0):
        """Don't restart production until the eval server has released the port."""
        if not LLAMA_BASE:
            return
        deadline = time.time() + timeout
        while time.time() < deadline:
            if http_json(f"{LLAMA_BASE}/health", timeout=2) is None and \
               http_json(f"{LLAMA_BASE}/v1/models", timeout=2) is None:
                return
            time.sleep(1)

    def __exit__(self, *exc):
        self.kill_eval_server()
        if self.stopped:
            LOG("Restoring production llama-server + daemon…")
            self._wait_port_free()
            ssh(f"systemctl --user start {LLAMA_UNIT}", check=False)
            time.sleep(5)
            sh(f"systemctl --user start {PROMETHEUS_UNIT}", check=False)
        return False  # never swallow exceptions


def wait_healthy(label: str) -> dict | None:
    """Poll /health (or /v1/models) then /props. Returns parsed /props or None."""
    if not LLAMA_BASE:
        sys.exit("OARA_LLAMA_BASE_URL not set — refusing to guess the endpoint.")
    deadline = time.time() + HEALTH_TIMEOUT_S
    while time.time() < deadline:
        if http_json(f"{LLAMA_BASE}/health") is not None or \
           http_json(f"{LLAMA_BASE}/v1/models") is not None:
            props = http_json(f"{LLAMA_BASE}/props")
            if props is not None:
                LOG(f"[{label}] server healthy.")
                return props
        time.sleep(HEALTH_POLL_S)
    LOG(f"[{label}] TIMEOUT waiting for healthy server (>{HEALTH_TIMEOUT_S}s) — "
        f"check GPU-box:/tmp/eval-{label}.log (OOM on a large inherited ctx is "
        f"the likely cause for the bigger Qwen weights on a 24 GB card).")
    return None


# ════════════════════════════════════════════════════════════════════════
# preflight: /props sanity (three-state: ok / problem / UNKNOWN-on-missing)
# ════════════════════════════════════════════════════════════════════════
def props_sanity(label: str, props: dict, expect_modality: str) -> list[str]:
    """Problems list; empty = clean. Missing props -> the UNKNOWN result (never
       a fabricated pass)."""
    if not props:
        return ["could not read /props (treat result as UNKNOWN, not clean)"]
    problems: list[str] = []
    caps = props.get("chat_template_caps", {}) or {}
    if not caps.get("supports_tool_calls", False):
        problems.append("server reports NO tool-call support — is --jinja set?")
    mods = props.get("modalities", {}) or {}
    has_vision = bool(mods.get("vision"))
    if expect_modality == "text-only" and has_vision:
        problems.append("vision projector present on a text-only model — "
                        "drop --mmproj (this is the Qwen trap).")
    if expect_modality == "vision" and not has_vision:
        problems.append("expected vision modality but server reports none — "
                        "--mmproj missing?")
    return problems


# ════════════════════════════════════════════════════════════════════════
# gates + scored run
# ════════════════════════════════════════════════════════════════════════
def run_smoke(label: str, verbose: bool) -> bool:
    LOG(f"[{label}] SMOKE GATE…")
    cmd = SMOKE_CMD + (" --verbose" if verbose else "")
    rc = subprocess.run(cmd, shell=True).returncode
    ok = rc == 0
    LOG(f"[{label}] smoke {'PASS' if ok else 'FAIL'} (exit {rc}).")
    return ok

def run_scored(label: str, leg_dir: Path, eval_config: Path,
               verbose: bool) -> tuple[bool, Path | None]:
    LOG(f"[{label}] SCORED RUN (judge={JUDGE_MODEL} @ {JUDGE_BASE_URL})…")
    leg_dir.mkdir(parents=True, exist_ok=True)
    cmd = EVAL_CMD_TEMPLATE.format(cfg=shlex.quote(str(eval_config)),
                                   dir=shlex.quote(str(leg_dir)))
    if verbose:
        cmd += " --verbose"
    rc = subprocess.run(cmd, shell=True).returncode
    # runner names files results_<ts>.json; rc==1 if ANY task errored (results
    # are still written), so gate on the file, not on rc alone.
    files = sorted(leg_dir.glob("results_*.json"))
    results = files[-1] if files else None
    ok = results is not None
    LOG(f"[{label}] scored run {'OK' if ok else 'FAILED'} "
        f"(exit {rc}, results={'<none>' if results is None else results.name}).")
    return ok, results


# ════════════════════════════════════════════════════════════════════════
# manifest — the audit trail that keeps the A/B honest
# ════════════════════════════════════════════════════════════════════════
def git_sha() -> str:
    try:
        return sh("git rev-parse HEAD").stdout.strip()
    except Exception:
        return "UNKNOWN"

def remote_gguf_sha(recipe: list[str]) -> str:
    gguf = _gguf_path(recipe)
    if not gguf:
        return "UNKNOWN"
    out = ssh(f"sha256sum {shlex.quote(gguf)} 2>/dev/null | cut -c1-12",
              check=False).stdout.strip()
    return out or "UNREADABLE"

def gguf_quant(recipe: list[str]) -> str:
    gguf = _gguf_path(recipe)
    if not gguf:
        return "UNKNOWN"
    name = Path(gguf).name.upper()
    for tag in ("UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q4_K_M", "Q4_K_M", "Q4_K_S",
                "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "IQ4_XS", "IQ4_NL"):
        if tag in name:
            return tag
    return "UNRECOGNIZED"

def write_manifest(label: str, recipe: list[str], props: dict, smoke_ok: bool,
                   scored_ok: bool, results_path: Path | None, eval_config: Path,
                   run_dir: Path) -> dict:
    p = props or {}
    dgs = p.get("default_generation_settings") or {}
    manifest = {
        "label": label,
        "timestamp": dt.datetime.now().isoformat(),
        "harness_git_sha": git_sha(),
        "judge_base_url": JUDGE_BASE_URL,
        "judge_model": JUDGE_MODEL,
        "eval_config": str(eval_config),
        "served_command": " ".join(recipe),
        "gguf_sha256_12": remote_gguf_sha(recipe),
        "quant": gguf_quant(recipe),
        "server_props": {
            "model": p.get("model_path") or p.get("model_alias") or p.get("model"),
            "modalities": p.get("modalities"),
            "tool_calls": (p.get("chat_template_caps") or {}).get("supports_tool_calls"),
            "n_ctx": p.get("n_ctx") or dgs.get("n_ctx"),  # top-level is null on this build
        },
        "smoke_gate_passed": smoke_ok,
        "scored_run_completed": scored_ok,
        "results_file": str(results_path) if results_path else None,
        "pinned": {
            "thinking": "suppressed (config suppress_thinking)",
            "judge_independent_of_both_contestants": True,
            "same_harness_commit": True,
        },
    }
    mpath = run_dir / f"{label}_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    LOG(f"[{label}] manifest -> {mpath}")
    return manifest


# ════════════════════════════════════════════════════════════════════════
# comparison report
# ════════════════════════════════════════════════════════════════════════
def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return round(sum(xs) / len(xs), 4) if xs else None

def _metric(task: dict, metric_name: str):
    for m in task.get("metrics") or []:
        if m.get("metric_name") == metric_name:
            return m.get("score")
    return None

def _task_passed(task: dict) -> bool:
    ms = task.get("metrics") or []
    return bool(ms) and not task.get("error") and all(m.get("passed") for m in ms)

def build_report(run_dir: Path, manifests: dict[str, dict]) -> Path:
    summaries: dict[str, dict] = {}
    for label, man in manifests.items():
        rp = Path(man["results_file"]) if man.get("results_file") else None
        data = json.loads(rp.read_text()) if rp and rp.exists() else {}
        tasks = data.get("results") or data.get("tasks") or []
        summaries[label] = {
            "tasks": len(tasks),
            "errors": sum(1 for t in tasks if t.get("error")),
            "smoke": man["smoke_gate_passed"],
            "pass_rate": _mean([1.0 if _task_passed(t) else 0.0 for t in tasks]),
            **{k: _mean([_metric(t, name) for t in tasks])
               for k, name in METRIC_KEYS.items()},
        }

    # A/B validity: HARD drift (a non-model variable changed -> may be invalid)
    # vs SOFT note (quant differs -> expected for a real-deploy comparison).
    a, b = list(manifests.values())
    hard: list[str] = []
    soft: list[str] = []
    if a["harness_git_sha"] != b["harness_git_sha"]:
        hard.append("harness git SHA differs between legs")
    if a["judge_base_url"] != b["judge_base_url"] or a["judge_model"] != b["judge_model"]:
        hard.append("JUDGE differs between legs")
    if a.get("quant") != b.get("quant"):
        soft.append(f"quant differs ({a.get('quant')} vs {b.get('quant')}) — "
                    f"EXPECTED for a real-deploy A/B; a slice of any delta is "
                    f"quant, not architecture")

    lines = ["# Model A/B — Gemma vs Qwen", "",
             f"_Judge: `{a['judge_model']}` @ `{a['judge_base_url']}` "
             f"(neutral, independent of both contestants)._", ""]
    if hard:
        lines += ["> ⚠️ **A/B VALIDITY WARNING** — a non-model variable changed "
                  "between legs; the comparison may be invalid:"]
        lines += [f"> - {d}" for d in hard] + [""]
    if soft:
        lines += ["> ℹ️ **Expected-for-real-deploy notes:**"]
        lines += [f"> - {d}" for d in soft] + [""]
    lines += ["| metric | " + " | ".join(summaries) + " |",
              "|---|" + "|".join(["---"] * len(summaries)) + "|"]
    for metric in ("tasks", "errors", "smoke", "pass_rate",
                   "tool_usage", "task_completion", "no_hallucination"):
        cells = " | ".join(str(summaries[l].get(metric)) for l in summaries)
        lines.append(f"| {metric} | {cells} |")
    lines += ["", "## Served commands (audit)", ""]
    for label, man in manifests.items():
        lines += [f"**{label}** (quant `{man.get('quant', '?')}`, "
                  f"gguf `{man['gguf_sha256_12']}`):",
                  "```", man["served_command"], "```", ""]
    report = run_dir / "AB_REPORT.md"
    report.write_text("\n".join(lines))
    LOG(f"comparison report -> {report}")
    return report


# ════════════════════════════════════════════════════════════════════════
# per-leg + main
# ════════════════════════════════════════════════════════════════════════
def eval_one(label: str, recipe: list[str], guard: ProductionGuard,
             expect_modality: str, eval_config: Path, run_dir: Path, args) -> dict | None:
    if args.manual_serve:
        input(f"\n>>> Bring up the **{label}** server at {LLAMA_BASE}, then Enter…\n"
              f"    recipe: {' '.join(recipe)}\n")
    else:
        guard.serve(recipe, label)

    props = wait_healthy(label)
    problems = props_sanity(label, props or {}, expect_modality)
    if problems:
        LOG(f"[{label}] /props sanity problems:")
        for p in problems:
            LOG(f"    - {p}")
        if not args.force:
            LOG(f"[{label}] HALTING this leg — fix serving config, or pass --force "
                f"to score anyway (NOT recommended; numbers would be a config "
                f"artifact, not capability).")
            return None

    smoke_ok = run_smoke(label, args.verbose)
    scored_ok, results_path = False, None
    if smoke_ok or args.force:
        scored_ok, results_path = run_scored(label, run_dir / f"{label}_eval",
                                             eval_config, args.verbose)
    else:
        LOG(f"[{label}] smoke FAILED — skipping scored run (would bank garbage). "
            f"Use --force to override.")

    return write_manifest(label, recipe, props or {}, smoke_ok, scored_ok,
                          results_path, eval_config, run_dir)


def main():
    ap = argparse.ArgumentParser(description="Gemma vs Qwen A/B eval on Prometheus")
    ap.add_argument("--dry-run", action="store_true",
                    help="derive recipes + write the eval config + print the plan; "
                         "touch nothing on the GPU box, leave production running.")
    ap.add_argument("--manual-serve", action="store_true",
                    help="don't SSH-control the GPU box; you bring up each server.")
    ap.add_argument("--only", choices=["gemma", "qwen"], help="run a single leg.")
    ap.add_argument("--force", action="store_true",
                    help="score even if /props sanity or smoke fails (DANGER).")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="skip the production-downtime confirmation prompt.")
    args = ap.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    canonical = read_live_execstart()
    recipes = derive_recipes(canonical)
    eval_config = write_eval_config(run_dir)
    LOG("Derived recipes from the LIVE unit:")
    for k, v in recipes.items():
        LOG(f"  {k}: {' '.join(v)}")
    LOG(f"eval config (judge -> {JUDGE_MODEL} @ {JUDGE_BASE_URL}): {eval_config}")

    jp = judge_preflight()
    if jp:
        LOG("Judge preflight problems:")
        for p in jp:
            LOG(f"    - {p}")
        if not (args.dry_run or args.force):
            sys.exit("Refusing to start — the neutral judge is not ready.")

    if args.dry_run:
        LOG(f"dry-run: recipes + eval config written under {run_dir}; "
            f"production untouched. Exiting.")
        return

    if not args.manual_serve and not args.yes:
        ans = input("\nThis stops PRODUCTION (daemon + GPU-box llama-server) for "
                    "the eval window and restores it on exit. Continue? [y/N] ")
        if ans.strip().lower() != "y":
            sys.exit("Aborted.")

    LOG(f"results dir: {run_dir}")
    legs = [("gemma", "vision"), ("qwen", "text-only")]
    if args.only:
        legs = [l for l in legs if l[0] == args.only]

    manifests: dict[str, dict] = {}
    with ProductionGuard(manual=args.manual_serve) as guard:  # restore-on-exit
        for label, modality in legs:
            man = eval_one(label, recipes[label], guard, modality, eval_config,
                           run_dir, args)
            if man:
                manifests[label] = man

    if len(manifests) == 2:
        report = build_report(run_dir, manifests)
        LOG(f"\nDONE. Open: {report}")
    else:
        LOG("\nDONE (single leg or a leg halted) — no comparison generated.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        main()
    except KeyboardInterrupt:
        LOG("Interrupted — production restore handled by guard.")
        sys.exit(130)
