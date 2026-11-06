#!/usr/bin/env python3
"""Gym fidelity, phase 0: read-only measurements (DISTILL-GYM-FIDELITY-PLAN §1.3).

    PYTHONPATH=src python3 scripts/distill/gym_fidelity.py measure m1 --cases <eval-tools.jsonl> [--store <repo-store>.git]
    PYTHONPATH=src python3 scripts/distill/gym_fidelity.py measure m2 --cases <eval-tools.jsonl> \\
        --store <repo-store>.git --work <scratch>/work [--jail require]
    PYTHONPATH=src python3 scripts/distill/gym_fidelity.py measure m3 --arg-equiv <full.json> \\
        --candidates <candidates.json> [--report <acceptable-tools report.json>]
    PYTHONPATH=src python3 scripts/distill/gym_fidelity.py measure m4 <round-2 traces.jsonl> ...
    PYTHONPATH=src python3 scripts/distill/gym_fidelity.py measure avail --cases <eval-tools.jsonl> \\
        --trace-index <trace_index.jsonl> --store <repo-store>.git --lcm-snapshot <lcm.db>

m1     reference census, static: executes nothing. The expected (reference) call of every
       case: its tool, whether it has arguments, the context source, hard cases, whether
       the history holds tool results, whether the case's environment maps into the gym,
       the class of every path it names (and whether the history names it too), and how
       the gym's shell grammar (``teacher_gen._prescan`` / ``_lex`` and its program sets)
       would classify a bash reference. With ``--store``, also whether each cwd path
       exists in the fixture commit (``git ls-tree``, read-only).
m2     what runs today: every reference with arguments through today's
       ``arg_equiv.GymSandboxExecutor`` (bash and read_file; every other tool is
       counted ``not_executable``, never run). Writes only under ``--work``, which it
       removes at the end (``cleanup``).
m3     the reasons already stored: argument-equivalence verdicts by tier and reason
       (``arg_equiv.py rescore`` JSON), outcome-check states and reasons
       (``acceptable_tools.py`` ``candidates.json``) and the judge's decisions.
m4     the GRPO pool's reference calls: round-2 gym rows by target area, tool and
       ``_meta.call_result`` status (no_call / injected / planted / success / error),
       and whether the row is compacted.
avail  what phase 1 needs next, without opening any database: per case, whether its
       ``trace_id`` joins the phase-1 trace index (and so has a telemetry-snapshot rowid)
       and when the trace was made against the repo store's commit range; and the LCM
       snapshot file's size and age (``stat`` only).

Every output is aggregate counts: no message, call, path or result text.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import posixpath
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "distill-gym-fidelity-measure/1"
_DIR = Path(__file__).resolve().parent

#: Path-valued argument names of the gym's tools (teacher_gen.PATH_ARGS, plus the
#: names other registry tools use).
PATH_KEYS = frozenset({"path", "root", "cwd", "file_path", "directory", "dir", "notebook_path"})
TMP_ROOTS = ("/tmp", "/var/tmp", "/private/tmp")
PATH_CLASSES = ("cwd", "virtual", "tmp", "dot_prometheus", "home_other", "abs_other", "unmapped")
#: Worst first: a bash reference takes the first class any of its commands has.
BASH_CLASS_ORDER = ("refused", "unavailable_program", "simulated_host", "virtual_read", "git_refused",
                    "git_read", "allowed")


def _load(name: str, file: str) -> Any:
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def EQ() -> Any:
    return _load("distill_arg_equiv", "arg_equiv.py")


def S() -> Any:
    return _load("distill_score", "score.py")


def TG() -> Any:
    return EQ()._load_teacher_gen()


# ---------------------------------------------------------------------------
# Static classification
# ---------------------------------------------------------------------------


def classify_bash(cmd: str, tg: Any) -> tuple[str, str]:
    """``(class, detail)`` for a bash command under the gym's grammar, without a
    sandbox: ``refused`` (detail: ``refusal_kind``), ``unavailable_program``
    (detail: the program, or ``python``), ``simulated_host`` (an ops program the
    gym answers from a fixture), ``virtual_read`` (a virtual host file read by a
    text utility), ``git_refused`` / ``git_read``, ``allowed``. Path confinement
    and per-program option rules are left to the dynamic run (m2)."""
    try:
        tg._prescan(cmd)
        toks = tg._lex(cmd)
    except tg.SandboxRefusal as exc:
        return "refused", tg.refusal_kind(str(exc))
    if not toks:
        return "refused", "empty"
    # (the separator before it, its words): redirect targets and fd numbers dropped
    segments: list[tuple[str, list[Any]]] = [("", [])]
    skip = False
    for t in toks:
        if skip:
            skip = False
            continue
        words = segments[-1][1]
        if t.op:
            if t.text in tg._SEP:
                segments.append((t.text, []))
            elif t.text in tg._REDIR:
                # an unquoted number written against the operator is its fd (2>&1, 2>/dev/null)
                if words and not words[-1].quoted and words[-1].text.isdigit() and words[-1].end == t.start:
                    words.pop()
                skip = True
            else:
                msg = tg._OP_REFUSALS.get(t.text, f"`{t.text}` is not available in this session")
                return "refused", tg.refusal_kind(msg)
            continue
        words.append(t)
    found: list[tuple[str, str]] = []
    for sep, words in segments:
        if not words:
            return "refused", "other"
        seg = [w.text for w in words]
        prog = seg[0]
        if "=" in prog:
            found.append(("refused", "shell_syntax"))
        elif "/" in prog or prog not in tg.ALLOWED_PROGRAMS | tg.OPS_PROGRAMS | {"git"}:
            kind = tg.refusal_kind(f"`{prog}` is not available in this session")
            found.append(("unavailable_program", "python" if kind == "python" else posixpath.basename(prog)))
        elif prog in tg.OPS_PROGRAMS:
            # a host command answers only at the head of a pipeline (a list splits into pipelines)
            found.append(("refused", "program") if sep == "|" else ("simulated_host", prog))
        elif prog == "git":
            rest = seg[1:]
            while rest and rest[0].startswith("-"):
                rest = rest[2:] if rest[0] in ("-C", "-c") else rest[1:]
            sub = rest[0] if rest else ""
            found.append(("git_read", sub) if sub in tg._GIT_READ else ("git_refused", sub or "-"))
        elif tg.virtual_paths_in(" ".join(seg[1:])):
            found.append(("virtual_read", prog))
        else:
            found.append(("allowed", prog))
    return min(found, key=lambda f: BASH_CLASS_ORDER.index(f[0]))


def _bash_path_words(cmd: str, tg: Any) -> list[str]:
    """Absolute and ``~`` words of a command (``--opt=/x`` included); ``/dev/null`` left out."""
    try:
        tg._prescan(cmd)
        toks = tg._lex(cmd)
    except tg.SandboxRefusal:
        return re.findall(r"(?<![\w.:/])(?:~/|/)[^\s'\"`;&|<>()$]+", cmd or "")
    out: list[str] = []
    for t in toks:
        if t.op or not t.text or (t.text.startswith("-") and "=" not in t.text):
            continue
        text = t.text
        if "=" in text and not text.startswith(("/", "~")):
            text = text.split("=", 1)[1]
        if text.startswith(("/", "~/")) or text == "~":
            if "://" in text or text == "/dev/null":
                continue
            out.append(text)
    return out


def reference_paths(tool: str, args: dict[str, Any], tg: Any) -> list[str]:
    """Every path the reference names: path-valued arguments, and for bash the
    absolute or ``~`` words of its command (relative words run in the cwd)."""
    out = [v for k, v in args.items() if k in PATH_KEYS and isinstance(v, str) and v]
    if tool == "bash" and isinstance(args.get("command"), str):
        out += _bash_path_words(args["command"], tg)
    return out


def path_class(raw: str, env: Any, eq: Any, tg: Any) -> tuple[str, str | None]:
    """``(class, normalized absolute path or None)``."""
    norm = eq.norm_path(raw, env)
    if not isinstance(norm, str) or not norm.startswith("/"):
        return "unmapped", None
    norm = posixpath.normpath(norm)
    if norm in tg.VIRTUAL_FILES:
        return "virtual", norm
    if any(norm == r or norm.startswith(r + "/") for r in TMP_ROOTS):
        return "tmp", norm
    if env.home and (norm == env.home + "/.prometheus" or norm.startswith(env.home + "/.prometheus/")):
        return "dot_prometheus", norm
    if env.cwd and (norm == env.cwd or norm.startswith(env.cwd + "/")):
        return "cwd", norm
    if env.home and (norm == env.home or norm.startswith(env.home + "/")):
        return "home_other", norm
    return "abs_other", norm


#: The repository the gym's fixture is a clone of, as a directory under the home
#: (a checkout) or under ``~/projects`` (a checkout or worktree).
REPO_DIR = "Prometheus"


def home_target(norm: str, env: Any) -> tuple[str, str | None]:
    """For a path under the case's home: ``(target, path inside the repo or None)``.
    ``repo``: a checkout or worktree of this repository (``~/Prometheus*``,
    ``~/projects/Prometheus*``, or a worktree under its ``.claude/worktrees``: the path
    inside is relative to the worktree); ``other_project``: another directory under
    ``~/projects``; ``home_misc``: anything else under the home. No directory
    name is reported."""
    rest = norm[len(env.home) + 1:].split("/") if env.home and norm.startswith(env.home + "/") else []
    inner: list[str] | None = None
    if rest and rest[0].startswith(REPO_DIR):
        inner = rest[1:]
    elif len(rest) > 1 and rest[0] == "projects":
        if not rest[1].startswith(REPO_DIR):
            return "other_project", None
        inner = rest[2:]
    if inner is None:
        return "home_misc", None
    if inner[:2] == [".claude", "worktrees"] and len(inner) > 2:
        inner = inner[3:]  # a worktree of the repository: the repository at another branch
    return "repo", "/".join(inner)


def path_target(raw: str, env: Any, eq: Any, tg: Any) -> tuple[str, str | None]:
    """A path's class with ``home_other`` and ``cwd`` split by ``home_target``."""
    cls, norm = path_class(raw, env, eq, tg)
    if cls == "cwd" and norm and env.cwd:
        return "cwd", norm[len(env.cwd):].strip("/")  # the executor maps the cwd to the fixture
    if cls == "home_other" and norm:
        return home_target(norm, env)
    return cls, None


#: Worst first: a reference whose paths fall in several targets reports the first.
TARGET_ORDER = ("unmapped", "other_project", "abs_other", "dot_prometheus", "home_misc", "tmp", "virtual", "repo",
                "cwd")


def history_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        if isinstance(c, str):
            parts.append(c)
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            a = fn.get("arguments")
            parts.append(a if isinstance(a, str) else json.dumps(a, ensure_ascii=False))
    return "\n".join(parts)


def seen_in(raw: str, norm: str | None, env: Any, text: str) -> bool:
    forms = {raw.rstrip("/")}
    if norm:
        forms.add(norm)
        if env.home and norm.startswith(env.home + "/"):
            forms.add("~" + norm[len(env.home):])
        if env.cwd and norm.startswith(env.cwd + "/"):
            forms.add(norm[len(env.cwd) + 1:])
    return any(f and len(f) > 1 and f in text for f in forms)


def env_mapped(env: Any) -> bool:
    return bool(env.home and env.cwd and (env.cwd == env.home or env.cwd.startswith(env.home + "/")))


def fixture_files(store: str | None) -> set[str] | None:
    """Every path (files and their parent directories) of the store's HEAD commit."""
    if not store:
        return None
    out = subprocess.run(["git", "--git-dir", store, "ls-tree", "-r", "--name-only", "HEAD"],
                         check=True, capture_output=True, text=True).stdout.split("\n")
    paths: set[str] = set()
    for p in out:
        if not p:
            continue
        paths.add(p)
        while "/" in p:
            p = p.rsplit("/", 1)[0]
            paths.add(p)
    return paths


def _ref(case: Any) -> tuple[str | None, Any]:
    exp = case.expect or {}
    return exp.get("tool"), exp.get("args")


def _has_args(args: Any) -> bool:
    return isinstance(args, dict) and bool(args)


# ---------------------------------------------------------------------------
# M1: reference census
# ---------------------------------------------------------------------------


def m1(cases: list[Any], *, store: str | None = None) -> dict[str, Any]:
    eq, tg = EQ(), TG()
    files = fixture_files(store)
    C: dict[str, Counter[Any]] = defaultdict(Counter)
    for case in cases:
        tool, args = _ref(case)
        meta = case.raw.get("meta") or {}
        name = tool or "(no call)"
        has = _has_args(args)
        C["tool"][name] += 1
        C["tool_has_args"][f"{name}|{'args' if has else 'no_args'}"] += 1
        C["context_source"][str(meta.get("context_source"))] += 1
        C["hard_case"][bool(meta.get("hard_case"))] += 1
        C["context_x_hard"][f"{meta.get('context_source')}|hard={bool(meta.get('hard_case'))}"] += 1
        C["history_tool_results"][any(m.get("role") == "tool" for m in case.messages)] += 1
        env = eq.case_env(case.messages)
        mapped = env_mapped(env)
        C["env_mapped"][mapped] += 1
        if env.cwd and env.home:
            rel = env.cwd[len(env.home):].strip("/")
            C["cwd_shape"]["home" if not rel else "home/<1 dir>" if "/" not in rel else "home/<deeper>"] += 1
        if not has or tool is None:
            continue
        text = history_text(case.messages)
        classes: set[str] = set()
        for raw in reference_paths(tool, args, tg):
            cls, norm = path_class(raw, env, eq, tg)
            seen = seen_in(raw, norm, env, text)
            classes.add(cls)
            C["path_class"][cls] += 1
            C["path_class_seen"][f"{cls}|{'seen' if seen else 'unseen'}"] += 1
            tgt, rel = path_target(raw, env, eq, tg)
            C["path_target"][tgt] += 1
            if tgt in ("cwd", "repo") and files is not None and rel is not None:
                C[f"{tgt}_path_in_fixture"]["root" if not rel else ("yes" if rel in files else "no")] += 1
        C["ref_path_classes"]["|".join(sorted(classes)) or "(no absolute path)"] += 1
        C["ref_target"][ref_target(tool, args, env, eq, tg)] += 1
        if tool == "bash":
            cls, detail = classify_bash(str(args.get("command") or ""), tg)
            C["bash_class"][cls] += 1
            C["bash_class_detail"][f"{cls}:{detail}"] += 1
            if args.get("cwd"):
                C["bash_cwd_arg"]["set"] += 1
    out = {k: dict(v.most_common()) for k, v in C.items()}
    out["cases"] = len(cases)
    out["refs_with_args"] = sum(1 for c in cases if _ref(c)[0] and _has_args(_ref(c)[1]))
    return out


# ---------------------------------------------------------------------------
# M2: what runs today
# ---------------------------------------------------------------------------


def ref_target(tool: str, args: dict[str, Any], env: Any, eq: Any, tg: Any) -> str:
    """The worst ``path_target`` of the paths a reference names (``TARGET_ORDER``)."""
    targets = {path_target(r, env, eq, tg)[0] for r in reference_paths(tool, args, tg)}
    return min(targets, key=TARGET_ORDER.index) if targets else "(no absolute path)"


def first_blocker(cls: str, detail: str, status: str, a: Any, target: str) -> str:
    """Why a runnable-tool reference does not run cleanly, the static grammar first
    (the executor checks paths before the grammar, so a refused command with a
    missing path reports ``missing_path``): ``grammar:<kind>``, ``simulated``,
    ``missing_path:<target>``, ``clean`` / ``clean_empty``, or the run's status."""
    if cls == "refused":
        return f"grammar:{detail}"
    if cls == "unavailable_program":
        return "grammar:python" if detail == "python" else "grammar:program"
    if cls == "git_refused":
        return "grammar:git_write"
    if cls == "simulated_host" or status.endswith(":simulated"):
        return "simulated"
    if status == "clean":
        return "clean_empty" if a is not None and a.empty and not a.effect else "clean"
    if status.endswith(":missing_path"):
        return f"missing_path:{target}"
    return status


def m2(cases: list[Any], ex: Any) -> dict[str, Any]:
    eq, tg = EQ(), TG()
    per_tool: dict[str, Counter[str]] = defaultdict(Counter)
    bash_x: Counter[str] = Counter()
    path_x: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    t0 = time.monotonic()
    for case in cases:
        tool, args = _ref(case)
        if tool is None or not _has_args(args):
            totals["no_args_or_no_call"] += 1
            continue
        c = per_tool[tool]
        c["refs"] += 1
        totals["refs_with_args"] += 1
        env = eq.case_env(case.messages)
        if tool not in eq.EXECUTABLE_TOOLS:
            c["not_executable"] += 1
            totals["not_executable"] += 1
            side = eq.has_side_effects(tool, args)
            blockers[f"not_routed:{'side_effect' if side else 'read_only'}"] += 1
            continue
        res = ex.run(tool, args, env)
        a, b = res.get("A"), res.get("B")
        sa = a.status if a else "-"
        sb = b.status if b else "-"
        c[f"A:{sa}"] += 1
        c[f"B:{sb}"] += 1
        clean = sa == "ok" and sb == "ok"
        if clean:
            c["clean"] += 1
            totals["clean"] += 1
            if a.effect:
                c["clean_effect"] += 1
            if a.empty and not a.effect:
                c["clean_empty_no_effect"] += 1
                totals["clean_empty_no_effect"] += 1
        status = "clean" if clean else (f"A:{sa}" if sa != "ok" else f"B:{sb}")
        totals[f"exec:{status}"] += 1
        cls, detail = classify_bash(str(args.get("command") or ""), tg) if tool == "bash" else ("allowed", "")
        if tool == "bash":
            bash_x[f"{cls}|{status}"] += 1
        blockers[first_blocker(cls, detail, status, a, ref_target(tool, args, env, eq, tg))] += 1
        classes = {path_class(r, env, eq, tg)[0] for r in reference_paths(tool, args, tg)}
        path_x[f"{'|'.join(sorted(classes)) or '(no absolute path)'}|{status}"] += 1
    return {"per_tool": {k: dict(v.most_common()) for k, v in sorted(per_tool.items())},
            "totals": dict(totals.most_common()), "bash_class_x_status": dict(bash_x.most_common()),
            "path_classes_x_status": dict(path_x.most_common()),
            "first_blocker": dict(blockers.most_common()),
            "executor": ex.describe(), "seconds": round(time.monotonic() - t0, 1)}


# ---------------------------------------------------------------------------
# M3: stored reasons
# ---------------------------------------------------------------------------


def m3(arg_equiv: dict[str, Any] | None, candidates: dict[str, Any] | None,
       report: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if arg_equiv is not None:
        tiers: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        per_model: dict[str, dict[str, int]] = {}
        for label, m in (arg_equiv.get("models") or {}).items():
            mt: Counter[str] = Counter()
            for e in (m.get("per_case") or {}).values():
                v = e.get("arg_equiv")
                if not v:
                    continue
                tiers[v["tier"]] += 1
                mt[v["tier"]] += 1
                reasons[f"{v['tier']}|{v.get('reason', '')}"] += 1
            per_model[label] = dict(mt.most_common())
        out["arg_equiv"] = {"method": arg_equiv.get("method"), "models": len(per_model),
                            "tiers": dict(tiers.most_common()), "per_model": per_model,
                            "reasons": dict(reasons.most_common())}
    if candidates is not None:
        calls = candidates.get("calls") or []
        states = Counter(str((c.get("outcome") or {}).get("state")) for c in calls)
        reasons = Counter(f"{(c.get('outcome') or {}).get('state')}|{(c.get('outcome') or {}).get('reason')}"
                          for c in calls)
        ref_bad = [c for c in calls if (c.get("outcome") or {}).get("state") == "reference_not_ok"]
        out["outcome_check"] = {
            "candidate_calls": len(calls), "states": dict(states.most_common()),
            "reasons": dict(reasons.most_common()),
            "reference_not_ok_cases": len({c.get("case") for c in ref_bad}),
            "reference_not_ok_by_candidate_tool": dict(Counter(c.get("tool") for c in ref_bad).most_common()),
            "executor_runs": (candidates.get("executor") or {}).get("runs")}
    if report is not None:
        ann = report.get("annotation") or {}
        out["judge"] = {"decisions": ann.get("decisions"), "outcome_states": ann.get("outcome_states"),
                        "candidate_calls": ann.get("candidate_calls")}
    return out


# ---------------------------------------------------------------------------
# M4: the GRPO pool's reference calls
# ---------------------------------------------------------------------------


def call_status(meta: dict[str, Any]) -> str:
    cr = meta.get("call_result")
    if not meta.get("tool_name") or not isinstance(cr, dict):
        return "no_call"
    if cr.get("injected"):
        return "injected"
    if cr.get("planted"):
        return "planted"
    return str(cr.get("status") or "unknown")


def m4(paths: Iterable[str]) -> dict[str, Any]:
    C: dict[str, Counter[Any]] = defaultdict(Counter)
    per_file: dict[str, int] = {}
    rows = 0
    for p in paths:
        n = 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                meta = json.loads(line).get("_meta") or {}
                n += 1
                area = str(meta.get("target_area"))
                st = call_status(meta)
                comp = bool(meta.get("compacted"))
                C["area"][area] += 1
                C["tool"][str(meta.get("tool_name"))] += 1
                C["status"][st] += 1
                C["compacted"][comp] += 1
                C["pad"][meta.get("compaction_pad") is not None] += 1
                C["area_x_status"][f"{area}|{st}"] += 1
                C["tool_x_status"][f"{meta.get('tool_name')}|{st}"] += 1
                C["status_x_compacted"][f"{st}|compacted={comp}"] += 1
                C["generator"][str(meta.get("generator"))] += 1
                C["teacher"][str(meta.get("teacher") or meta.get("model"))] += 1
                if st == "success" and not comp:
                    C["ok_uncompacted_area"][area] += 1
                    C["ok_uncompacted_tool"][str(meta.get("tool_name"))] += 1
                    C["ok_uncompacted_executable_today"][meta.get("tool_name") in EQ().EXECUTABLE_TOOLS] += 1
        per_file[Path(p).name] = n
        rows += n
    out = {k: dict(v.most_common()) for k, v in C.items()}
    out["rows"] = rows
    out["files"] = per_file
    out["ok_uncompacted"] = sum(C["ok_uncompacted_area"].values())
    return out


# ---------------------------------------------------------------------------
# What phase 1 needs next (no database opened)
# ---------------------------------------------------------------------------


def store_range(store: str) -> dict[str, Any]:
    ts = [int(x) for x in subprocess.run(["git", "--git-dir", store, "log", "--all", "--format=%ct"],
                                         check=True, capture_output=True, text=True).stdout.split()]
    head = int(subprocess.run(["git", "--git-dir", store, "log", "-1", "--format=%ct", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip())
    return {"commits": len(ts), "oldest": min(ts), "newest": max(ts), "head": head}


def avail(cases: list[Any], trace_index: str | None, store: str | None,
          lcm_snapshot: str | None, now: float | None = None) -> dict[str, Any]:
    now = time.time() if now is None else now
    out: dict[str, Any] = {}
    if lcm_snapshot:
        p = Path(lcm_snapshot).expanduser()
        if p.exists():
            st = p.stat()
            out["lcm_snapshot"] = {"exists": True, "bytes": st.st_size,
                                   "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st.st_mtime)),
                                   "age_days": round((now - st.st_mtime) / 86400, 1),
                                   "wal_present": p.with_name(p.name + "-wal").exists()}
        else:
            out["lcm_snapshot"] = {"exists": False}
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if trace_index:
        with open(Path(trace_index).expanduser(), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    by_file[r["file"]].append({"ts": r.get("ts"), "rowid": r.get("rowid"), "tool": r.get("tool")})
    rng = store_range(store) if store else None
    if rng:
        out["store"] = {k: (time.strftime("%Y-%m-%d", time.gmtime(v)) if k != "commits" else v)
                        for k, v in rng.items()}
    C: dict[str, Counter[Any]] = defaultdict(Counter)
    for case in cases:
        meta = case.raw.get("meta") or {}
        tid = str(meta.get("trace_id") or "")
        file, _, line = tid.rpartition(":")
        row = None
        if file in by_file and line.isdigit() and 0 < int(line) <= len(by_file[file]):
            row = by_file[file][int(line) - 1]
        src = "fail_traces" if file.startswith("fail_traces") else "golden" if file.startswith("golden") else "other"
        C["source"][src] += 1
        C["trace_index_join"][f"{src}|{'joined' if row else 'not_joined'}"] += 1
        if row is None:
            continue
        # the index is in file and line order: the joined row must be the case's own call
        C["join_tool_matches"][row.get("tool") == (case.expect or {}).get("tool")] += 1
        C["telemetry_rowid"][row.get("rowid") is not None] += 1
        ts = row.get("ts")
        if ts is None:
            continue
        C["trace_month"][time.strftime("%Y-%m", time.gmtime(ts))] += 1
        if rng:
            where = ("before_oldest_commit" if ts < rng["oldest"] else
                     "after_fixture_head" if ts > rng["head"] else "within_store_range")
            C["trace_vs_store"][where] += 1
            C["days_after_head"][
                "<=0" if ts <= rng["head"] else "1-7" if ts - rng["head"] <= 7 * 86400
                else "8-14" if ts - rng["head"] <= 14 * 86400 else ">14"] += 1
    out.update({k: dict(v.most_common()) for k, v in C.items()})
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_json(p: str | None) -> Any:
    return json.loads(Path(p).expanduser().read_text(encoding="utf-8")) if p else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    top = ap.add_subparsers(dest="cmd", required=True)
    meas = top.add_parser("measure", help="phase-0 read-only measurements")
    sub = meas.add_subparsers(dest="which", required=True)
    p1 = sub.add_parser("m1", help="reference census (static)")
    p1.add_argument("--cases", required=True)
    p1.add_argument("--store", help="the gym's repo store (cwd paths checked against its HEAD)")
    p2 = sub.add_parser("m2", help="run every reference with arguments in today's executor")
    p2.add_argument("--cases", required=True)
    p2.add_argument("--store", required=True)
    p2.add_argument("--work", required=True, help="a scratch directory the executor owns")
    p2.add_argument("--jail", default="require", choices=("auto", "require", "off"))
    p3 = sub.add_parser("m3", help="tally stored reasons")
    p3.add_argument("--arg-equiv")
    p3.add_argument("--candidates")
    p3.add_argument("--report")
    p4 = sub.add_parser("m4", help="round-2 pool reference calls")
    p4.add_argument("traces", nargs="+")
    p5 = sub.add_parser("avail", help="trace dates, trace-index joins, LCM snapshot age")
    p5.add_argument("--cases", required=True)
    p5.add_argument("--trace-index")
    p5.add_argument("--store")
    p5.add_argument("--lcm-snapshot")
    for p in (p1, p2, p3, p4, p5):
        p.add_argument("--json-out")
    args = ap.parse_args(argv)
    if args.which == "m1":
        res = m1(S().load_cases([args.cases]), store=args.store)
    elif args.which == "m2":
        ex = EQ().GymSandboxExecutor(Path(args.store), Path(args.work), jail=args.jail)
        try:
            res = m2(S().load_cases([args.cases]), ex)
        finally:
            ex.cleanup()
    elif args.which == "m3":
        res = m3(_read_json(args.arg_equiv), _read_json(args.candidates), _read_json(args.report))
    elif args.which == "m4":
        res = m4(args.traces)
    else:
        res = avail(S().load_cases([args.cases]), args.trace_index, args.store, args.lcm_snapshot)
    doc = {"schema": SCHEMA, "measure": args.which, "result": res}
    text = json.dumps(doc, indent=1, sort_keys=False, default=str)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
