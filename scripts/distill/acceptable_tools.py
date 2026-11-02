#!/usr/bin/env python3
"""Acceptable-tool sets for eval-tools: which first tools are an acceptable first step, case by case.

    PYTHONPATH=src python scripts/distill/acceptable_tools.py prepare --work <dir> \\
        --cases <evalsets>/eval-tools.jsonl --results student-q4=<dir>/eval-tools.jsonl ... \\
        [--subset student-bf16] [--pair student-q4:base-same-recipe ...] \\
        [--exec-store <sandbox-root>/repo-store/<sha>.git --exec-work <scratch> --jail require] \\
        [--annotation-out <evalsets>/eval-tools.acceptable-tools.json]
    PYTHONPATH=src python scripts/distill/judge.py --workdir <dir>/judge run --provider local \\
        --judge-url http://<tailnet host>:<port> --judge-key-file <0600 file> --view tool-choice --concurrency 1
    PYTHONPATH=src python scripts/distill/acceptable_tools.py finish --work <dir>
    PYTHONPATH=src python scripts/distill/acceptable_tools.py night --work <dir> --judge-url ... \\
        --judge-key-file ... --until 04:55 [--execute]
    python scripts/distill/acceptable_tools.py section --report <dir>/report.json \\
        --scorecard docs/sprints/DISTILL-PHASE7-SCORECARD.md

The old wrong-tool metric (``score.py``) has ONE right answer per case: Max's first tool (``expect.tool``).
A model whose first call names any other offered tool is "wrong", even when that tool is a perfectly good
first step (reading a file with ``read_file`` instead of ``cat``). This module builds, per case, the SET of
acceptable first tools, and rescores every model's wrong-tool rate against it.

The rule (decided per case, from the stored first calls of every scored model)
--------------------------------------------------------------------------------
* Max's expected tool is always in the set (``reference``).
* Candidates: every distinct first call any scored model made to an OFFERED tool other than Max's (the
  27B's included; a name the case does not offer is never a candidate: it stays an "other miss"). One
  candidate per distinct (tool, arguments).
* A candidate CALL is acceptable when
    - ``outcome``: the gym can run both it and Max's reference call (``bash`` / ``read_file``, the
      execution tier of ``arg_equiv.py``: the gym sandbox over the repo fixture, snapshots A and B, A run
      twice), both exit 0 in both snapshots with no workspace effect, and the candidate's output CONTAINS
      the reference's (non-empty, compared line by line with trailing whitespace ignored) in both: it
      obtained at least the information Max's step obtained, whatever the files' length (snapshot B pads
      every file, so ``read_file``'s 200-line default does not contain a full ``cat`` there); or
    - ``judge``: the calibrated local judge (the 4090's Qwen3.8-27B UD-Q4_K_XL, ``judge.py`` view
      ``tool-choice``, same context and offered tools as the case) scores ``right_tool`` 2 (clearly an
      acceptable tool for this step) and neither ``sensible_order`` nor ``task_progress`` 0. Its
      ``valid_args`` / ``no_fabrication`` scores are recorded (``call_acceptable``: the strict verdict) but
      do not decide tool choice: argument quality is what exact match and argument equivalence measure.
  The outcome check is positive evidence only: a candidate that ran and got different information, or
  failed in the gym, is left to the judge (the gym is one commit of one repo; a grep with no match exits 1;
  Phase 7 found every executed same-tool pair "differs").
* A TOOL is in the case's set when at least one of its candidate calls is acceptable; it is ``rejected``
  when every one of its calls has a judge ruling and none is acceptable; else ``undecided`` (no ruling
  yet). Each tool records the mechanism(s) that admitted it, the models that called it, and its calls'
  counts.
* The judge route is used only after the view passes its agreement check (below); a view that fails it
  leaves every judge-only tool ``undecided`` (``--accept-view-anyway`` overrides, recorded).

The judge view's agreement check
--------------------------------
The 85.4% calibration measured the judge's TRACE view against Max; the tool-choice view is new and not
covered by it. ``prepare`` adds controls to the same queue: ``positive`` = Max's own reference call on a
seeded sample of cases (exact-args cases, every non-bash expected tool first), which should be ruled
acceptable ~always; ``negative`` = an obviously wrong tool for a seeded sample of cases (text-to-speech,
image or video generation, a YouTube transcript: offered, schema-valid, unrelated to the task), which should
be ruled unacceptable. The check passes at >= ``POSITIVE_MIN`` positives accepted AND >= ``NEGATIVE_MIN``
negatives rejected (point estimates; Wilson intervals reported).

Outputs (``--work``; ids, tool names, verdicts and counts only, never call, reply or trace text, except the
judge queue itself, which carries each case's context to the judge and stays on this host):
  prepare.json        the inputs (paths, sha256s, labels, pairs) the other steps read
  candidates.json     every candidate call: key, case, tool, the models that made it, its outcome check
  judge/              the judge.py workdir (judge-queue.jsonl + judge-queue.meta.json, mode tool-choice)
  acceptable-tools.json  the annotation (``score.py summarize --acceptable-tools`` reads it), also copied to
                      ``--annotation-out`` (beside the eval set)
  agreement.json      the view's agreement check
  report.json / report.md   old vs new wrong-tool rates per model (Wilson 95%), the decomposition, paired
                      student-vs-base deltas on both metrics (Newcombe 95%)
  scorecard-section.md      the scorecard section (``section`` renders it; aggregates only)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")

ANNOTATION_SCHEMA = "distill-acceptable-tools/1"  # == score.ACCEPTABLE_TOOLS_SCHEMA
REPORT_SCHEMA = "distill-acceptable-tools-report/1"
PREPARE_FILE = "prepare.json"
CANDIDATES_FILE = "candidates.json"
ANNOTATION_FILE = "acceptable-tools.json"
AGREEMENT_FILE = "agreement.json"
REPORT_FILE = "report.json"
REPORT_MD = "report.md"
SECTION_FILE = "scorecard-section.md"
JUDGE_DIR = "judge"

#: The view's agreement check (point estimates): Max's reference accepted, obviously wrong tools rejected.
POSITIVE_MIN = 0.90
NEGATIVE_MIN = 0.95
DEFAULT_POSITIVES = 80
DEFAULT_NEGATIVES = 60
DEFAULT_SEED = 20261102

#: Obviously wrong first steps for a coding / daemon case: offered, schema-valid, unrelated.
NEGATIVE_CALLS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("tts", {"text": "Working on it now."}),
    ("image_generate", {"prompt": "A diagram of this project's files"}),
    ("video_generate", {"prompt": "A short clip of a terminal window"}),
    ("youtube_transcript", {"url": "https://www.youtube.com/watch?v=aaaaaaaaaaa"}),
)
#: A case whose conversation mentions one of these is never given a negative control (it might be about it).
_NEGATIVE_TOPICS = re.compile(r"\b(image|video|speech|audio|voice|youtube|tts|transcript|picture|photo)", re.I)

#: Scorecard placeholder markers (``section --scorecard`` replaces what is between them).
SECTION_BEGIN = "<!-- acceptable-tools:begin -->"
SECTION_END = "<!-- acceptable-tools:end -->"

ACCEPTABLE, REJECTED, UNDECIDED, REFERENCE = "acceptable", "rejected", "undecided", "reference"
OUTCOME_SUPPORTS = "supports"


# ---------------------------------------------------------------------------
# Modules beside this file
# ---------------------------------------------------------------------------


def _load(name: str, file: str) -> Any:
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def S() -> Any:
    """score.py (shared with whoever loaded it first as ``distill_score``)."""
    return _load("distill_score", "score.py")


def EQ() -> Any:
    return _load("distill_arg_equiv", "arg_equiv.py")


def J() -> Any:
    return _load("distill_judge", "judge.py")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def call_key(case_id: str, tool: str, arguments: Any) -> str:
    """A candidate call's id: its case, tool and arguments (canonical JSON; raw text when unreadable)."""
    blob = json.dumps([case_id, tool, arguments], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_specs(specs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for spec in specs:
        label, eq, path = spec.partition("=")
        if not eq or not label or not path:
            raise SystemExit(f"--results wants LABEL=PATH, got {spec!r}")
        if label in out:
            raise SystemExit(f"--results: label {label!r} given twice")
        out[label] = path
    return out


# ---------------------------------------------------------------------------
# First calls and candidates
# ---------------------------------------------------------------------------


def first_calls(cases: list[Any], rows: dict[tuple[str, str], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per scored case: ``{"outcome", "first_tool", "arguments"}`` (arguments only for a wrong tool: the
    parsed object, or the raw text when it did not parse)."""
    sc = S()
    out: dict[str, dict[str, Any]] = {}
    for c in cases:
        r = rows.get(c.key)
        if r is None or not sc.row_matches(c, r) or r.get("status") != "ok":
            continue
        v = sc.score_case(c, r)
        e: dict[str, Any] = {"outcome": v["tool_outcome"], "first_tool": v.get("first_tool")}
        if v["tool_outcome"] == "wrong_tool":
            calls, _ = sc.reply_calls(c, r.get("message") or {})
            first = calls[0]
            e["arguments"] = first["arguments"] if first["arguments"] is not None else first["arguments_raw"]
        out[c.id] = e
    return out


def collect_candidates(cases: list[Any], per_model: dict[str, dict[str, dict[str, Any]]]
                       ) -> dict[str, dict[str, Any]]:
    """Every distinct wrong-tool first call (an offered tool other than Max's): key -> {case, tool,
    arguments, labels}."""
    by_id = {c.id: c for c in cases}
    out: dict[str, dict[str, Any]] = {}
    for label, firsts in per_model.items():
        for cid, e in firsts.items():
            if e["outcome"] != "wrong_tool" or cid not in by_id:
                continue
            key = call_key(cid, e["first_tool"], e["arguments"])
            cand = out.setdefault(key, {"key": key, "case": cid, "tool": e["first_tool"],
                                        "arguments": e["arguments"], "labels": []})
            cand["labels"].append(label)
    for cand in out.values():
        cand["labels"] = sorted(set(cand["labels"]))
    return out


# ---------------------------------------------------------------------------
# The outcome check (arg_equiv's execution tier)
# ---------------------------------------------------------------------------


def _norm_output(text: str) -> str:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def make_outcome_executor(store: Path, work: Path, jail: str = "require", **kw: Any) -> Any:
    """``arg_equiv.GymSandboxExecutor`` that also keeps each clean run's output text IN MEMORY (keyed like its
    own cache; never written anywhere), so two calls of different tools can be compared by what they read.
    The text is the one ``_run_once`` digests (``arg_equiv._digest`` on the output with the sandbox root
    replaced): this wraps that one call rather than copying the sandbox code."""
    eq = EQ()

    class OutcomeExecutor(eq.GymSandboxExecutor):  # type: ignore[misc, name-defined]
        def __post_init__(self) -> None:
            super().__post_init__()
            self.texts: dict[tuple[str, str], str] = {}

        def _run_once(self, tool: str, args: dict[str, Any], env: Any, snap: str) -> Any:
            seen: list[str] = []
            real = eq._digest

            def spy(obj: Any) -> str:
                if isinstance(obj, str):
                    seen.append(obj)
                return real(obj)

            eq._digest = spy
            try:
                res = super()._run_once(tool, args, env, snap)
            finally:
                eq._digest = real
            if res.status == "ok" and seen:
                self.texts[(real([tool, args, env.cwd, env.home]), snap)] = seen[-1]
            return res

        def text(self, tool: str, args: dict[str, Any], env: Any, snap: str) -> str | None:
            return self.texts.get((eq._digest([tool, args, env.cwd, env.home]), snap))

    return OutcomeExecutor(Path(store), Path(work), jail=jail, **kw)


def outcome_check(ex: Any, env: Any, ref_tool: str, ref_args: Any, tool: str, args: Any) -> dict[str, str]:
    """``{"state", "reason"}``: ``supports`` (the candidate obtained at least the reference's information in
    both snapshots), ``differs`` (both ran cleanly; it did not), or a state that decides nothing:
    ``not_run`` (a tool the gym cannot run, or arguments that are not an object), ``reference_not_ok`` /
    ``candidate_not_ok`` (refused, simulated, a missing path, a nonzero exit, a timeout, cut, nondeterminism),
    ``side_effects``, ``empty`` (the reference printed nothing)."""
    runnable = EQ().EXECUTABLE_TOOLS
    if ref_tool not in runnable or tool not in runnable:
        return {"state": "not_run", "reason": "a tool the gym cannot run"}
    if not isinstance(ref_args, dict) or not isinstance(args, dict):
        return {"state": "not_run", "reason": "arguments are not an object"}
    rr, rc = ex.run(ref_tool, ref_args, env), ex.run(tool, args, env)
    for name, res in (("reference", rr), ("candidate", rc)):
        a, b = res.get("A"), res.get("B")
        if a is None or a.status != "ok" or b is None or b.status != "ok":
            got = a.status if a is not None and a.status != "ok" else (b.status if b is not None else "-")
            return {"state": f"{name}_not_ok", "reason": f"{name}: {got}"}
    if rr["A"].effect or rc["A"].effect or rr["B"].effect or rc["B"].effect:
        return {"state": "side_effects", "reason": "a call changed the workspace"}
    contains = True
    for snap in ("A", "B"):
        rt, ct = ex.text(ref_tool, ref_args, env, snap), ex.text(tool, args, env, snap)
        if rt is None or ct is None:
            return {"state": "candidate_not_ok", "reason": f"no output kept for snapshot {snap}"}
        rn, cn = _norm_output(rt), _norm_output(ct)
        if not rn.strip():
            return {"state": "empty", "reason": "the reference printed nothing"}
        contains = contains and rn in cn
    if contains:
        return {"state": OUTCOME_SUPPORTS, "reason": "the candidate's output contains the reference's in A and B"}
    return {"state": "differs", "reason": "both ran cleanly; the candidate's output lacks the reference's"}


def run_outcome_checks(cases: list[Any], candidates: dict[str, dict[str, Any]], ex: Any | None
                       ) -> dict[str, dict[str, str]]:
    eq = EQ()
    by_id = {c.id: c for c in cases}
    out: dict[str, dict[str, str]] = {}
    for key, cand in sorted(candidates.items()):
        c = by_id[cand["case"]]
        if ex is None:
            out[key] = {"state": "not_run", "reason": "no executor"}
            continue
        try:
            out[key] = outcome_check(ex, eq.case_env(c.messages), c.expect.get("tool"),
                                     c.expect.get("args"), cand["tool"], cand["arguments"])
        except Exception as exc:  # one call the sandbox chokes on decides nothing; never the whole step
            out[key] = {"state": "candidate_not_ok", "reason": f"error: {type(exc).__name__}"}
    return out


# ---------------------------------------------------------------------------
# The judge queue
# ---------------------------------------------------------------------------


def _item(item_id: str, label: str, category: str, case: Any, name: str, arguments: Any,
          source: dict[str, Any]) -> dict[str, Any]:
    return {"id": item_id, "label": label, "category": category, "stratum_weight": 1.0,
            "source": {"case": case.id, "tool": name, **source},
            "messages": case.messages, "tools": case.tools,
            "candidate": {"name": name, "arguments": arguments},
            "meta": {"case_category": (case.raw.get("meta") or {}).get("category")}, "outcome": None}


def positive_controls(cases: list[Any], n: int, seed: int) -> list[Any]:
    """Max's reference call on *n* cases with exact-match arguments: every non-bash expected tool first (the
    tools the wrong-tool pairs involve), then a seeded draw of the rest."""
    exact = [c for c in cases if isinstance(c.expect.get("args"), dict) and c.expect.get("tool")
             and c.expect.get("args_match", "exact") == "exact"]
    rng = random.Random(f"{seed}:positives")
    rare = sorted((c for c in exact if c.expect["tool"] != "bash"), key=lambda c: c.id)
    rest = sorted((c for c in exact if c.expect["tool"] == "bash"), key=lambda c: c.id)
    rng.shuffle(rare)
    rng.shuffle(rest)
    return (rare + rest)[:n]


def _conversation_text(case: Any) -> str:
    return " ".join(str(m.get("content") or "") for m in case.messages if m.get("role") == "user")


def negative_controls(cases: list[Any], n: int, seed: int) -> list[tuple[Any, str, dict[str, Any]]]:
    rng = random.Random(f"{seed}:negatives")
    pool = sorted((c for c in cases if not _NEGATIVE_TOPICS.search(_conversation_text(c))), key=lambda c: c.id)
    rng.shuffle(pool)
    out: list[tuple[Any, str, dict[str, Any]]] = []
    for c in pool:
        offered = {S().tool_name(t) for t in c.tools}
        options = [(t, a) for t, a in NEGATIVE_CALLS if t in offered and t != c.expect.get("tool")]
        if not options:
            continue
        name, args = options[len(out) % len(options)]
        out.append((c, name, dict(args)))
        if len(out) >= n:
            break
    return out


def lead_call(calls: list[dict[str, Any]], reference_label: str | None) -> dict[str, Any]:
    """The call judged first for a (case, tool): the reference model's (the 27B) when it made one, else the
    one most models made, then by key."""
    return sorted(calls, key=lambda k: (reference_label not in k["labels"], -len(k["labels"]), k["key"]))[0]


def build_queue(cases: list[Any], candidates: dict[str, dict[str, Any]], *, positives: int, negatives: int,
                seed: int, reference_label: str | None) -> list[dict[str, Any]]:
    """The judge queue: first one call per (case, tool) with controls spread among them (so a night cut
    short still has a ruling on every pair and the agreement check), then the remaining calls. Items of one
    case sit together (their prompts share everything before the candidate)."""
    by_id = {c.id: c for c in cases}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates.values():
        groups[(cand["case"], cand["tool"])].append(cand)
    leads, rest = [], []
    for gkey in sorted(groups):
        lead = lead_call(groups[gkey], reference_label)
        leads.append(lead)
        rest += sorted((k for k in groups[gkey] if k is not lead), key=lambda k: k["key"])

    def cand_item(k: dict[str, Any]) -> dict[str, Any]:
        return _item(f"cand-{k['key']}", "candidate", "candidate", by_id[k["case"]], k["tool"], k["arguments"],
                     {"call": k["key"], "labels": k["labels"]})

    controls = [_item(f"pos-{c.id}", "pass", "control-positive", c, c.expect["tool"], c.expect["args"],
                      {"control": "max-reference"})
                for c in positive_controls(cases, positives, seed)]
    controls += [_item(f"neg-{c.id}-{name}", "fail", "control-negative", c, name, args,
                       {"control": "obviously-wrong"})
                 for c, name, args in negative_controls(cases, negatives, seed)]
    controls.sort(key=lambda it: (it["source"]["case"], it["id"]))
    first = [cand_item(k) for k in leads]
    merged: list[dict[str, Any]] = []
    if controls:
        step = max(1, round(len(first) / len(controls))) if first else 1
        ci = 0
        for i, it in enumerate(first):
            merged.append(it)
            if (i + 1) % step == 0 and ci < len(controls):
                merged.append(controls[ci])
                ci += 1
        merged += controls[ci:]
    else:
        merged = first
    return merged + [cand_item(k) for k in rest]


# ---------------------------------------------------------------------------
# Judge rulings
# ---------------------------------------------------------------------------


def tool_acceptable(rubric: dict[str, int]) -> bool:
    """The tool-choice rule on a judge reply's rubric: right_tool 2, sensible_order and task_progress >= 1."""
    return rubric.get("right_tool") == 2 and rubric.get("sensible_order", 0) >= 1 \
        and rubric.get("task_progress", 0) >= 1


def judge_rulings(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """item id -> the latest ok tool-choice ruling: ``{"tool_acceptable", "call_acceptable", "rubric"}``."""
    j = J()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if (r.get("status") != "ok" or r.get("view") != j.TOOL_CHOICE_VIEW
                or r.get("rubric_version") != j.RUBRIC_VERSION or not isinstance(r.get("rubric"), dict)):
            continue
        out[r["id"]] = {"tool_acceptable": tool_acceptable(r["rubric"]),
                        "call_acceptable": r.get("verdict") == "pass", "rubric": r["rubric"],
                        "model": r.get("model"), "judge_quant": r.get("judge_quant"),
                        "judge_sha256": r.get("judge_sha256")}
    return out


def judge_load(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the judge pass took from the server it shared with that night's generation: calls (every row,
    errors included), tokens, the sum of request times (slot-seconds at most) and the first/last row time."""
    ts = sorted(str(r["ts"]) for r in rows if r.get("ts"))
    return {"rows": len(rows), "ok": sum(1 for r in rows if r.get("status") == "ok"),
            "input_tokens": sum(int(r.get("input_tokens") or 0) for r in rows),
            "output_tokens": sum(int(r.get("output_tokens") or 0) for r in rows),
            "request_seconds": round(sum(float(r.get("elapsed_s") or 0.0) for r in rows), 1),
            "first": ts[0] if ts else None, "last": ts[-1] if ts else None}


def _wilson(k: int, n: int) -> dict[str, Any]:
    return S().rate(k, n)


def agreement(queue: list[dict[str, Any]], rulings: dict[str, dict[str, Any]],
              categories: dict[str, str]) -> dict[str, Any]:
    """The view's agreement check on its controls (and how many items have a ruling)."""
    out: dict[str, Any] = {"queued": len(queue), "ruled": sum(1 for it in queue if it["id"] in rulings)}
    for cat, want in (("control-positive", True), ("control-negative", False)):
        items = [it for it in queue if it["category"] == cat]
        ruled = [it for it in items if it["id"] in rulings]
        hits = [it for it in ruled if rulings[it["id"]]["tool_acceptable"] is want]
        strict = [it for it in ruled if rulings[it["id"]]["call_acceptable"] is want]
        by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for it in ruled:
            b = by_cat[categories.get(it["source"]["case"], "?")]
            b[1] += 1
            b[0] += rulings[it["id"]]["tool_acceptable"] is want
        out[cat] = {"queued": len(items), "agree": _wilson(len(hits), len(ruled)),
                    "agree_strict_verdict": _wilson(len(strict), len(ruled)),
                    "by_case_category": {k: {"k": v[0], "n": v[1]} for k, v in sorted(by_cat.items())},
                    "misses": sorted(it["id"] for it in ruled if rulings[it["id"]]["tool_acceptable"] is not want)}
    pos, neg = out["control-positive"]["agree"], out["control-negative"]["agree"]
    if not pos["n"] or not neg["n"]:
        out["check"] = "incomplete"
    elif pos["rate"] >= POSITIVE_MIN and neg["rate"] >= NEGATIVE_MIN:
        out["check"] = "pass"
    else:
        out["check"] = "fail"
    out["thresholds"] = {"positive_min": POSITIVE_MIN, "negative_min": NEGATIVE_MIN}
    return out


# ---------------------------------------------------------------------------
# The annotation
# ---------------------------------------------------------------------------


def call_verdict(key: str, outcome: dict[str, str], ruling: dict[str, Any] | None, judge_ok: bool
                 ) -> tuple[bool | None, list[str]]:
    """(acceptable: True / False / None for no ruling, the mechanisms that admitted it)."""
    by = []
    if outcome.get("state") == OUTCOME_SUPPORTS:
        by.append("outcome")
    if ruling is not None and judge_ok and ruling["tool_acceptable"]:
        by.append("judge")
    if by:
        return True, by
    if ruling is not None and judge_ok:
        return False, []
    return None, []


def build_annotation(cases: list[Any], candidates: dict[str, dict[str, Any]],
                     outcomes: dict[str, dict[str, str]], rulings: dict[str, dict[str, Any]], *,
                     judge_ok: bool, provenance: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the annotation, per-call verdicts). Every case gets an entry (Max's tool alone when no model called
    another offered tool)."""
    per_call: dict[str, dict[str, Any]] = {}
    by_case: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for key, cand in candidates.items():
        ok, by = call_verdict(key, outcomes.get(key, {}), rulings.get(f"cand-{key}"), judge_ok)
        ruling = rulings.get(f"cand-{key}")
        per_call[key] = {"case": cand["case"], "tool": cand["tool"], "labels": cand["labels"],
                         "outcome": outcomes.get(key, {}).get("state", "not_run"),
                         "judge": None if ruling is None else {
                             "tool_acceptable": ruling["tool_acceptable"],
                             "call_acceptable": ruling["call_acceptable"], "rubric": ruling["rubric"]},
                         "acceptable": ok, "by": by}
        by_case[cand["case"]][cand["tool"]].append({"key": key, **per_call[key]})
    entries: dict[str, Any] = {}
    for c in cases:
        exp = c.expect.get("tool")
        e: dict[str, Any] = {"sha": c.sha, "expected": exp,
                             "category": (c.raw.get("meta") or {}).get("category"),
                             "acceptable": [exp] if exp else [], "rejected": [], "undecided": [],
                             "tools": {}}
        if exp:
            e["tools"][exp] = {"decision": REFERENCE, "by": [REFERENCE]}
        for tool, calls in sorted(by_case.get(c.id, {}).items()):
            oks = [k for k in calls if k["acceptable"] is True]
            if oks:
                decision = ACCEPTABLE
            elif all(k["acceptable"] is False for k in calls):
                decision = REJECTED
            else:
                decision = UNDECIDED
            e[decision].append(tool)
            e["tools"][tool] = {
                "decision": decision, "by": sorted({m for k in oks for m in k["by"]}),
                "models": sorted({lab for k in calls for lab in k["labels"]}),
                "calls": len(calls), "calls_acceptable": len(oks),
                "calls_judged": sum(1 for k in calls if k["judge"] is not None),
                "outcome": dict(Counter(k["outcome"] for k in calls)),
            }
        entries[c.id] = e
    undecided = sum(len(e["undecided"]) for e in entries.values())
    ann = {"schema": ANNOTATION_SCHEMA, "set": "eval-tools", "cases": entries,
           "complete": undecided == 0, "undecided_tools": undecided,
           "rule": ("Max's expected tool, plus every offered tool with at least one candidate first call ruled "
                    "acceptable by the outcome check (contains the reference's output in both gym snapshots) "
                    "or by the calibrated judge's tool-choice view (right_tool 2, sensible_order and "
                    "task_progress >= 1)"),
           "judge_route": "applied" if judge_ok else "not applied (the view's agreement check did not pass)",
           **provenance}
    return ann, per_call


# ---------------------------------------------------------------------------
# Rescoring: old vs new, per model and paired
# ---------------------------------------------------------------------------


def model_report(cases: list[Any], firsts: dict[str, dict[str, Any]], ann: dict[str, Any],
                 per_call: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sc = S()
    by_id = ann["cases"]
    n = len(firsts)
    old = sum(1 for e in firsts.values() if e["outcome"] == "wrong_tool")
    per_case: dict[str, dict[str, Any]] = {}
    split = Counter()
    mech = Counter()
    own = 0
    for cid, e in firsts.items():
        so = sc.set_tool_outcome(e["outcome"], e.get("first_tool"), by_id[cid])
        per_case[cid] = {"old": e["outcome"], "new": so}
        if e["outcome"] != "wrong_tool":
            continue
        split[so] += 1
        if so == "acceptable_tool":
            mech["+".join(by_id[cid]["tools"][e["first_tool"]]["by"])] += 1
            k = per_call.get(call_key(cid, e["first_tool"], e["arguments"]))
            own += bool(k and k["acceptable"] is True)
    by_cat: dict[str, Counter[str]] = defaultdict(Counter)
    for cid, pc in per_case.items():
        if pc["old"] == "wrong_tool":
            by_cat[by_id[cid].get("category") or "?"][pc["new"]] += 1
    return {
        "scored": n,
        "old_wrong_tool": sc.rate(old, n),
        "new_wrong_tool": sc.rate(split["wrong_tool"], n),
        "new_wrong_tool_upper": sc.rate(split["wrong_tool"] + split["undecided_tool"], n),
        "wrong_but_acceptable": split["acceptable_tool"], "truly_wrong": split["wrong_tool"],
        "undecided": split["undecided_tool"],
        "acceptable_by": dict(mech),
        "acceptable_own_call": own,
        "by_case_category": {k: dict(v) for k, v in sorted(by_cat.items())},
        "per_case": per_case,
    }


def paired_delta(a: dict[str, Any], b: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Candidate *a* minus base *b* on "wrong tool" (``old`` or ``new``) over the cases both scored;
    Newcombe 95% interval in points (score.paired_report)."""
    sc = S()
    pa, pb = a["per_case"], b["per_case"]
    ids = sorted(set(pa) & set(pb))
    if not ids:
        return None
    t = [0, 0, 0, 0]
    for i in ids:
        x, y = pb[i][key] == "wrong_tool", pa[i][key] == "wrong_tool"
        t[0 if x and y else 1 if x else 2 if y else 3] += 1
    rep = sc.paired_report(sc.PairedCounts(*t), f"acceptable-tools/{key}", bootstrap=False)
    return {**rep, "base_rate": round(100 * (t[0] + t[1]) / len(ids), 2),
            "candidate_rate": round(100 * (t[0] + t[2]) / len(ids), 2)}


def annotation_summary(ann: dict[str, Any], per_call: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tools = [(cid, t, d) for cid, e in ann["cases"].items() for t, d in e["tools"].items()
             if d["decision"] != REFERENCE]
    dec = Counter(d["decision"] for _, _, d in tools)
    by = Counter("+".join(d["by"]) for _, _, d in tools if d["decision"] == ACCEPTABLE)
    pairs = Counter((ann["cases"][cid]["expected"], t, d["decision"]) for cid, t, d in tools)
    oc = Counter(k["outcome"] for k in per_call.values())
    cross = Counter((k["outcome"] == OUTCOME_SUPPORTS, None if k["judge"] is None else k["judge"]["tool_acceptable"])
                    for k in per_call.values() if k["outcome"] in (OUTCOME_SUPPORTS, "differs"))
    by_cat: dict[str, Counter[str]] = defaultdict(Counter)
    for cid, _, d in tools:
        by_cat[ann["cases"][cid].get("category") or "?"][d["decision"]] += 1
    return {
        "cases": len(ann["cases"]), "cases_with_candidates": len({cid for cid, _, _ in tools}),
        "candidate_tools": len(tools), "decisions": dict(dec), "acceptable_by": dict(by),
        "candidate_calls": len(per_call),
        "calls_judged": sum(1 for k in per_call.values() if k["judge"] is not None),
        "calls_acceptable": sum(1 for k in per_call.values() if k["acceptable"] is True),
        "outcome_states": dict(oc),
        "outcome_vs_judge": [{"outcome_supports": a, "judge_tool_acceptable": b, "calls": n}
                             for (a, b), n in sorted(cross.items(), key=lambda x: str(x[0]))],
        "by_case_category": {k: dict(v) for k, v in sorted(by_cat.items())},
        "pairs": [{"expected": e, "tool": t, "decision": d, "n": n}
                  for (e, t, d), n in sorted(pairs.items(), key=lambda x: (-x[1], x[0]))],
        "set_size_hist": dict(Counter(len(e["acceptable"]) for e in ann["cases"].values())),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _pct(r: dict[str, Any] | None, counts: bool = True) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    body = f"{100 * r['rate']:.1f}%"
    if counts:
        body += f" ({r['k']})"
    return body + f" [{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}]"


def _signed(x: float) -> str:
    text = f"{x:+.1f}"
    return "+0.0" if text == "-0.0" else text


def _delta(d: dict[str, Any] | None) -> tuple[str, str]:
    if not d:
        return "n/a", "n/a"
    sc = S()
    return sc._diff_text(d), sc._ci_text(d)


def render_report(rep: dict[str, Any]) -> str:
    partial = any(m["undecided"] for m in rep["models"].values())
    new_head = ("Wrong tool, new: LOWER bound (undecided counted not wrong)" if partial
                else "Wrong tool, new (acceptable set)")
    lines = [f"| Model | n | Wrong tool, old (single reference) | {new_head} "
             "| Wrong but acceptable | Truly wrong | Undecided | New, upper bound |",
             "|---|---|---|---|---|---|---|---|"]
    for label, m in rep["models"].items():
        lines.append(f"| {rep['display'].get(label, label)} | {m['scored']} | {_pct(m['old_wrong_tool'])} "
                     f"| **{_pct(m['new_wrong_tool'])}** | {m['wrong_but_acceptable']} | {m['truly_wrong']} "
                     f"| {m['undecided']} | {_pct(m['new_wrong_tool_upper'], counts=False)} |")
    lines += ["", "| Model | Admitted by outcome | by judge | by both | Own call acceptable |", "|---|---|---|---|---|"]
    for label, m in rep["models"].items():
        by = m["acceptable_by"]
        lines.append(f"| {rep['display'].get(label, label)} | {by.get('outcome', 0)} | {by.get('judge', 0)} "
                     f"| {by.get('judge+outcome', 0)} | {m['acceptable_own_call']} of {m['wrong_but_acceptable']} |")
    if rep.get("pairs"):
        lines += ["", "| Comparison | Wrong tool, old: Δ | Paired 95% CI | Wrong tool, new: Δ | Paired 95% CI |",
                  "|---|---|---|---|---|"]
        for p in rep["pairs"]:
            od, oci = _delta(p["old"])
            nd, nci = _delta(p["new"])
            o, n = p["old"] or {}, p["new"] or {}
            lines.append(f"| {rep['display'].get(p['candidate'], p['candidate'])} vs "
                         f"{rep['display'].get(p['base'], p['base'])} "
                         f"| {od} ({o.get('base_rate', 0):.1f} → {o.get('candidate_rate', 0):.1f}) | {oci} "
                         f"| **{nd}** ({n.get('base_rate', 0):.1f} → {n.get('candidate_rate', 0):.1f}) | {nci} |")
    return "\n".join(lines) + "\n"


def _agree_line(a: dict[str, Any], key: str) -> str:
    r = a.get(key, {}).get("agree") or {}
    if not r.get("n"):
        return "not judged yet"
    return f"{r['k']}/{r['n']} = {_pct(r, counts=False)}"


def render_section(rep: dict[str, Any], *, date: str) -> str:
    """The scorecard section: aggregates only (tool names, counts, rates)."""
    a, s = rep["agreement"], rep["annotation"]
    dec = s["decisions"]
    pos = a.get("control-positive", {})
    neg = a.get("control-negative", {})
    pending = not rep.get("annotation_complete")
    head = [
        f"## Acceptable-tool wrong-tool rate (added {date})",
        "",
        "Operator ask (Mon ~08:00): replace the single-reference wrong-tool metric with an acceptable-tool set "
        "per case, built from Max + 27B + judge agreement, plus an outcome check where the gym can run it. "
        "Same stored eval-tools replies as above (329 cases; Student BF16 on the 100-case subset); no new "
        "model requests.",
        "",
    ]
    if pending:
        head += [f"**Partial:** {rep['annotation']['decisions'].get('undecided', 0)} candidate tools still have "
                 "no ruling; the \"new\" column counts them as not wrong, the upper bound as wrong.", ""]
    body = [
        "**The rule.** Per case, the acceptable set is Max's expected tool plus every offered tool for which at "
        "least one model's first call was ruled acceptable:",
        "- **outcome:** the gym ran both that call and Max's reference (`bash` / `read_file`, arg_equiv's "
        "execution tier, both snapshots, the fixture run twice), both exited 0 with no workspace effect, and the "
        "call's output contains the reference's in both snapshots;",
        "- **judge:** the calibrated judge (4090 Qwen3.8-27B UD-Q4_K_XL, temperature 0, thinking off) in a new "
        "tool-choice view (the case's full context and offered tools, one candidate call: \"is this an "
        "acceptable next step here?\") scored right_tool 2 and neither sensible_order nor task_progress 0. "
        "Argument scores are recorded but don't decide tool choice.",
        "",
        "A tool is rejected when every one of its calls was ruled not acceptable. Candidates are the distinct "
        "first calls to an offered tool other than Max's by any scored model (27B included); names the case "
        "doesn't offer stay \"other misses\". The outcome check only admits: a call that got different "
        "information, or failed in the gym, goes to the judge. Max's tool keeps the case's task-success scoring: "
        "nothing here changes task success or the Phase 6 gate.",
        "",
        f"**The judge view's check** (it isn't covered by the 85.4% trace-view calibration). Controls in the same "
        f"queue: Max's own reference call on {pos.get('queued', 0)} cases, and an obviously wrong tool "
        f"(text-to-speech, image or video generation, a YouTube transcript) on {neg.get('queued', 0)}. Bar: "
        f"≥{POSITIVE_MIN:.0%} of references accepted and ≥{NEGATIVE_MIN:.0%} of wrong tools rejected.",
        f"- Max's reference accepted: {_agree_line(a, 'control-positive')}",
        f"- Obviously wrong tool rejected: {_agree_line(a, 'control-negative')}",
        f"- Result: **{a.get('check', 'incomplete')}**"
        + ("" if a.get("check") == "pass" else "; the judge route was not applied, so judge-only tools stay "
           "undecided") + ".",
        "",
        f"**The sets.** {s['candidate_calls']} distinct candidate calls ({s['candidate_tools']} case-tool pairs in "
        f"{s['cases_with_candidates']} cases): {dec.get('acceptable', 0)} tools acceptable, "
        f"{dec.get('rejected', 0)} rejected, {dec.get('undecided', 0)} undecided. Admitted by: "
        + (", ".join(f"{k} {v}" for k, v in sorted(s["acceptable_by"].items())) or "none")
        + ". Outcome check states over the candidate calls: "
        + ", ".join(f"{k} {v}" for k, v in sorted(s["outcome_states"].items())) + ".",
        "",
        "### Old vs new, per model",
        "",
        render_report(rep).rstrip("\n"),
        "",
        "Rates are of all scored cases, with Wilson 95% intervals. \"Wrong but acceptable\" are old wrong tools "
        "whose tool is in the case's set; \"own call acceptable\" counts those whose own call was ruled "
        "acceptable (the rest got in through another model's call to the same tool). Paired Δ is the student "
        "minus the base (Newcombe 95%); positive means more wrong tools.",
        "",
        "### Most common expected → called pairs",
        "",
        "| Max's tool | Called | Decision | Case-tool pairs |",
        "|---|---|---|---|",
        *[f"| {p['expected']} | {p['tool']} | {p['decision']} | {p['n']} |" for p in s["pairs"][:12]],
        "",
        "**Load on the shared server.** The judge pass ran beside that night's generation on the same 4x32768 "
        f"server: {rep.get('judge_load', {}).get('rows', 0)} requests, "
        f"{rep.get('judge_load', {}).get('input_tokens', 0):,} prompt and "
        f"{rep.get('judge_load', {}).get('output_tokens', 0):,} output tokens, "
        f"{rep.get('judge_load', {}).get('request_seconds', 0) / 3600:.2f} slot-hours of request time "
        f"({rep.get('judge_load', {}).get('first') or '?'} to {rep.get('judge_load', {}).get('last') or '?'}).",
        "",
        f"Sources: `~/distill/acceptable-tools/` on the 3090 Ti host (`report.json`, `agreement.json`, "
        f"`acceptable-tools.json`: ids, tool names and verdicts only; the judge queue stays there). The annotation "
        f"is also beside the eval set the runs used (`phase3/build-v2/eval-tools.acceptable-tools.json`). "
        f"Cases sha256 `{rep['cases_sha256'][:16]}`.",
    ]
    return "\n".join(head + body) + "\n"


def replace_section(scorecard: str, section: str) -> str:
    """The scorecard with the text between the markers replaced (markers kept); refuses without them."""
    i, j = scorecard.find(SECTION_BEGIN), scorecard.find(SECTION_END)
    if i < 0 or j < i:
        raise SystemExit(f"the scorecard has no {SECTION_BEGIN} ... {SECTION_END} placeholder")
    return scorecard[: i + len(SECTION_BEGIN)] + "\n" + section + scorecard[j:]


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def load_inputs(prep: dict[str, Any]) -> tuple[list[Any], dict[str, dict[str, dict[str, Any]]]]:
    sc = S()
    cases = [c for c in sc.load_cases([prep["cases"]]) if c.set == "eval-tools"]
    got = sc.canonical_sha(sorted(c.sha for c in cases))
    if got != prep["cases_sha256"]:
        raise SystemExit(f"{prep['cases']}: cases sha256 {got[:16]} is not the prepared {prep['cases_sha256'][:16]}")
    per_model: dict[str, dict[str, dict[str, Any]]] = {}
    for label, m in prep["models"].items():
        if sha256_file(Path(m["path"])) != m["sha256"]:
            raise SystemExit(f"{m['path']} changed since prepare (sha256)")
        rows, _labels = sc.latest_rows(Path(m["path"]))
        per_model[label] = first_calls(cases, rows)
    return cases, per_model


def cmd_prepare(args: argparse.Namespace) -> int:
    sc = S()
    work = Path(args.work).expanduser()
    work.mkdir(mode=0o700, parents=True, exist_ok=True)
    (work / JUDGE_DIR).mkdir(mode=0o700, exist_ok=True)
    os.chmod(work / JUDGE_DIR, 0o700)  # its queue carries each case's context
    jq = work / JUDGE_DIR / J().QUEUE_FILE
    if jq.exists() and not args.force:
        print(f"{jq} exists; --force to rewrite it (the judge's rows for unchanged ids are kept and reused)",
              file=sys.stderr)
        return 2
    cases_path = str(Path(args.cases).expanduser().resolve())
    cases = [c for c in sc.load_cases([cases_path]) if c.set == "eval-tools"]
    cases_sha = sc.canonical_sha(sorted(c.sha for c in cases))
    if args.expect_cases_sha256 and not cases_sha.startswith(args.expect_cases_sha256):
        print(f"error: {cases_path}: cases sha256 {cases_sha[:16]}, not {args.expect_cases_sha256} (the set the "
              "runs scored)", file=sys.stderr)
        return 2
    specs = parse_specs(args.results)
    models: dict[str, Any] = {}
    per_model: dict[str, dict[str, dict[str, Any]]] = {}
    for label, path in specs.items():
        p = Path(path).expanduser().resolve()
        rows, _labels = sc.latest_rows(p)
        firsts = first_calls(cases, rows)
        if not firsts:
            print(f"error: {p}: no row answers these cases (a different eval set?)", file=sys.stderr)
            return 2
        per_model[label] = firsts
        models[label] = {"path": str(p), "sha256": sha256_file(p), "scored": len(firsts),
                         "subset": len(firsts) < len(cases),
                         "wrong_tool": sum(1 for e in firsts.values() if e["outcome"] == "wrong_tool")}
        print(f"{label}: {len(firsts)} scored, {models[label]['wrong_tool']} wrong tool", file=sys.stderr)
    if args.reference_label and args.reference_label not in specs:
        print(f"error: --reference-label {args.reference_label} is not a --results label", file=sys.stderr)
        return 2
    pairs = []
    for spec in args.pair:
        cand, _, base = spec.partition(":")
        if cand not in specs or base not in specs:
            print(f"error: --pair {spec}: unknown label", file=sys.stderr)
            return 2
        pairs.append([cand, base])
    candidates = collect_candidates(cases, per_model)
    ex = None
    if args.exec_store:
        if not args.exec_work:
            print("error: --exec-store needs --exec-work", file=sys.stderr)
            return 2
        ex = make_outcome_executor(Path(args.exec_store).expanduser(), Path(args.exec_work).expanduser(),
                                   jail=args.jail)
    try:
        outcomes = run_outcome_checks(cases, candidates, ex)
    finally:
        executor = ex.describe() if ex is not None else None
        if ex is not None:
            ex.cleanup()
    queue = build_queue(cases, candidates, positives=args.positives, negatives=args.negatives, seed=args.seed,
                        reference_label=args.reference_label)
    write_text(jq, "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in queue))
    write_json(work / JUDGE_DIR / J().QUEUE_META_FILE, {
        "mode": J().MODE_TOOL_CHOICE, "view": J().TOOL_CHOICE_VIEW, "items": len(queue),
        "by_category": dict(Counter(it["category"] for it in queue)), "seed": args.seed,
        "cases_sha256": cases_sha})
    write_json(work / CANDIDATES_FILE, {
        "cases_sha256": cases_sha,
        "calls": [{"key": k["key"], "case": k["case"], "tool": k["tool"], "labels": k["labels"],
                   "outcome": outcomes[k["key"]]} for k in sorted(candidates.values(), key=lambda k: k["key"])],
        "executor": executor})
    prep = {"cases": cases_path, "cases_file_sha256": sha256_file(Path(cases_path)), "cases_sha256": cases_sha,
            "models": models, "pairs": pairs, "display": dict(d.split("=", 1) for d in args.display),
            "reference_label": args.reference_label, "annotation_out": args.annotation_out,
            "positives": args.positives, "negatives": args.negatives, "seed": args.seed,
            "prepared": datetime.now(ET).isoformat(timespec="seconds")}
    write_json(work / PREPARE_FILE, prep)
    oc = Counter(o["state"] for o in outcomes.values())
    print(json.dumps({"candidate_calls": len(candidates),
                      "case_tool_pairs": len({(k["case"], k["tool"]) for k in candidates.values()}),
                      "outcome": dict(oc), "queue": dict(Counter(it["category"] for it in queue)),
                      "queue_items": len(queue)}, indent=1))
    return 0


def finish(work: Path, *, accept_view_anyway: bool = False, date: str | None = None) -> dict[str, Any]:
    """build + report + section from what is in *work* now (resumable: run again after more verdicts)."""
    j = J()
    prep = read_json(work / PREPARE_FILE)
    cand_doc = read_json(work / CANDIDATES_FILE)
    cases, per_model = load_inputs(prep)
    candidates = collect_candidates(cases, per_model)
    keys = {k["key"] for k in cand_doc["calls"]}
    if set(candidates) != keys:
        raise SystemExit("the candidate calls differ from prepare's (results changed?): prepare again")
    outcomes = {k["key"]: k["outcome"] for k in cand_doc["calls"]}
    queue = j.read_jsonl(work / JUDGE_DIR / j.QUEUE_FILE)
    judge_rows = j.read_jsonl(work / JUDGE_DIR / j.RESULTS_FILE)
    rulings = judge_rulings(judge_rows)
    categories = {c.id: (c.raw.get("meta") or {}).get("category") or "?" for c in cases}
    agree = agreement(queue, rulings, categories)
    judge_ok = agree["check"] == "pass" or accept_view_anyway
    judges = sorted({(r.get("model"), r.get("judge_quant"), r.get("judge_sha256")) for r in rulings.values()},
                    key=str)
    provenance = {"cases_file": Path(prep["cases"]).name, "cases_file_sha256": prep["cases_file_sha256"],
                  "cases_sha256": prep["cases_sha256"],
                  "models": {k: {"sha256": v["sha256"], "scored": v["scored"]} for k, v in prep["models"].items()},
                  "judge": [{"model": m, "quant": q, "sha256": s} for m, q, s in judges],
                  "judge_view": j.TOOL_CHOICE_VIEW, "view_check": agree["check"],
                  "accept_view_anyway": bool(accept_view_anyway),
                  "built": datetime.now(ET).isoformat(timespec="seconds")}
    ann, per_call = build_annotation(cases, candidates, outcomes, rulings, judge_ok=judge_ok, provenance=provenance)
    write_json(work / ANNOTATION_FILE, ann)
    write_json(work / AGREEMENT_FILE, agree)
    if prep.get("annotation_out") and agree["check"] != "incomplete":
        # beside the eval set only once the judge has ruled (an all-undecided set would read as "no wrong tools")
        write_json(Path(prep["annotation_out"]).expanduser(), ann)
    models = {label: model_report(cases, firsts, ann, per_call) for label, firsts in per_model.items()}
    pairs = [{"candidate": a, "base": b, "old": paired_delta(models[a], models[b], "old"),
              "new": paired_delta(models[a], models[b], "new")} for a, b in prep["pairs"]]
    rep = {"schema": REPORT_SCHEMA, "cases_sha256": prep["cases_sha256"], "display": prep.get("display") or {},
           "annotation_complete": ann["complete"], "view_check": agree["check"], "judge_route": ann["judge_route"],
           "agreement": agree, "annotation": annotation_summary(ann, per_call),
           "models": {k: {kk: vv for kk, vv in v.items() if kk != "per_case"} for k, v in models.items()},
           "pairs": pairs, "judge_load": judge_load(judge_rows),
           "created": datetime.now(ET).isoformat(timespec="seconds")}
    write_json(work / REPORT_FILE, rep)
    rep_full = {**rep, "models": models}
    write_text(work / REPORT_MD, render_report(rep_full))
    write_text(work / SECTION_FILE, render_section(rep, date=date or datetime.now(ET).strftime("%Y-%m-%d")))
    return rep


def cmd_finish(args: argparse.Namespace) -> int:
    rep = finish(Path(args.work).expanduser(), accept_view_anyway=args.accept_view_anyway, date=args.date)
    print(render_report(rep), end="")
    print(json.dumps({"view_check": rep["view_check"], "annotation_complete": rep["annotation_complete"],
                      "decisions": rep["annotation"]["decisions"]}))
    return 0


def cmd_section(args: argparse.Namespace) -> int:
    rep = read_json(Path(args.report).expanduser())
    text = render_section(rep, date=args.date or rep.get("created", "")[:10])
    if args.scorecard:
        p = Path(args.scorecard)
        p.write_text(replace_section(p.read_text(encoding="utf-8"), text), encoding="utf-8")
        print(f"replaced the section in {p}")
    else:
        print(text, end="")
    return 0


# ---------------------------------------------------------------------------
# night: one window's judge pass, then finish
# ---------------------------------------------------------------------------


def deadline_today(hhmm: str, now: datetime) -> datetime:
    h, m = (int(x) for x in hhmm.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def server_check(url: str, key_file: str, min_slots: int, factory: Callable[..., Any] | None = None) -> str | None:
    """None when the judge server is the calibrated one in a shape with >= *min_slots* slots; else why not.
    ``JudgeNotReady`` propagates (the caller waits and tries again)."""
    j = J()
    make = factory or j.RemoteJudgeClient
    client = make(url, key_file=key_file, ledger=j.spend.Ledger(), phase=j.PHASE_TOOL_CHOICE)
    client.preflight()
    if client.slots < min_slots:
        return (f"the judge server has {client.slots} slot(s), not the standard profile's {min_slots}: tonight is "
                "not a standard night (a judge pass would share the generation's slot)")
    return None


def notify(cmd: str, lines: list[str], execute: bool) -> None:
    text = "\n".join(lines[:5])
    if not execute:
        print("[dry run] would notify:\n" + text)
        return
    path = Path(cmd).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        subprocess.run([str(path), "5"], input=text, text=True, timeout=60, check=False)
    else:
        print(f"notify command {path} missing: {text}", file=sys.stderr)


def run_judge_until(argv: list[str], until: datetime, env: dict[str, str]) -> int | None:
    """judge.py run until it exits or *until* (then SIGTERM: a local run's rows so far stay); its exit code,
    or None when it was stopped at the deadline."""
    proc = subprocess.Popen(argv, env=env)
    try:
        return proc.wait(timeout=max(1.0, (until - datetime.now(ET)).total_seconds()))
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return None


def cmd_night(args: argparse.Namespace) -> int:
    j = J()
    work = Path(args.work).expanduser().resolve()
    now = datetime.now(ET)
    until = deadline_today(args.until, now)
    log = lambda msg: print(f"{datetime.now(ET):%F %T %Z} acceptable-tools night: {msg}", flush=True)  # noqa: E731
    queue = j.read_jsonl(work / JUDGE_DIR / j.QUEUE_FILE)
    done = j.done_ids(j.read_jsonl(work / JUDGE_DIR / j.RESULTS_FILE), j.TOOL_CHOICE_VIEW)
    left = [it for it in queue if it["id"] not in done]
    log(f"{len(left)} of {len(queue)} items without a ruling; judge until {until:%a %H:%M %Z}")
    if not left and (work / REPORT_FILE).exists():
        log("nothing left to judge and the report exists: nothing to do")
        return 0
    judge_state = "nothing to judge"
    if left:
        start_by = now + timedelta(minutes=args.wait_min)
        why = None
        while True:
            try:
                why = server_check(args.judge_url, args.judge_key_file, args.min_slots)
                break
            except j.JudgeNotReady as exc:
                if datetime.now(ET) >= start_by:
                    why = f"the judge server was not ready by {start_by:%H:%M} ({exc})"
                    break
                log(f"judge server not ready ({exc}); waiting")
                time.sleep(args.poll_s)
            except (j.JudgeSetupError, j.JudgeCallError) as exc:
                why = f"not the calibrated judge server: {exc}"
                break
        if why:
            log(why)
            notify(args.notify, ["acceptable-tools judge pass: not run tonight", why[:280],
                                 f"{len(left)} items wait in {work / JUDGE_DIR}"], args.execute)
            return 3
        argv = [args.py, str(HERE / "judge.py"), "--workdir", str(work / JUDGE_DIR), "run", "--provider", "local",
                "--judge-url", args.judge_url, "--judge-key-file", args.judge_key_file, "--view", j.TOOL_CHOICE_VIEW,
                "--concurrency", str(args.concurrency), "--timeout", str(args.timeout), "--max-pause", "1800"]
        if not args.execute:
            log("[dry run] would run: " + " ".join(shlex.quote(a) for a in argv))
            return 0
        env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
        rc = run_judge_until(argv, until, env)
        if rc == j.EXIT_INCOMPLETE and datetime.now(ET) < until - timedelta(minutes=5):
            log("unparseable or failed items left: one more pass")
            rc = run_judge_until(argv, until, env)
        judge_state = {None: "stopped at the deadline", 0: "done"}.get(rc, f"judge.py exit {rc}")
        log(judge_state)
    if not args.execute:
        log("[dry run] then: finish (annotation, agreement, report, scorecard section)")
        return 0
    rep = finish(work)
    a = rep["agreement"]
    pick = [p for p in rep["pairs"] if p.get("new")][:2]
    notify(args.notify, [
        f"acceptable-tools judge pass: {judge_state}; {a['ruled']}/{a['queued']} ruled",
        f"view check {a['check']}: refs accepted {_agree_line(a, 'control-positive')}; "
        f"wrong rejected {_agree_line(a, 'control-negative')}",
        "annotation " + ("complete" if rep["annotation_complete"] else
                         f"partial ({rep['annotation']['decisions'].get('undecided', 0)} undecided)"),
        "; ".join(f"{p['candidate']} vs {p['base']} new Δ {_delta(p['new'])[0]}" for p in pick) or "no pairs",
        f"report: {work / REPORT_MD}",
    ], args.execute)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="candidates, the outcome check and the judge queue")
    p.add_argument("--work", required=True)
    p.add_argument("--cases", required=True, help="the eval-tools case file the runs scored")
    p.add_argument("--expect-cases-sha256", default="", help="refuse unless the cases sha256 starts with this")
    p.add_argument("--results", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--reference-label", default=None, help="the 27B's label: its call is judged first per tool")
    p.add_argument("--pair", action="append", default=[], metavar="CANDIDATE:BASE")
    p.add_argument("--display", action="append", default=[], metavar="LABEL=NAME", help="a label's table name")
    p.add_argument("--exec-store", help="the gym's repo store (bare clone): enables the outcome check")
    p.add_argument("--exec-work", help="a scratch directory the executor owns")
    p.add_argument("--jail", default="require", choices=("auto", "require", "off"))
    p.add_argument("--positives", type=int, default=DEFAULT_POSITIVES)
    p.add_argument("--negatives", type=int, default=DEFAULT_NEGATIVES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--annotation-out", default=None, help="also write the annotation here (beside the eval set)")
    p.add_argument("--force", action="store_true")
    f = sub.add_parser("finish", help="annotation + agreement + report + scorecard section from the verdicts so far")
    f.add_argument("--work", required=True)
    f.add_argument("--accept-view-anyway", action="store_true",
                   help="apply the judge route even though the view failed its agreement check (recorded)")
    f.add_argument("--date", default=None)
    s = sub.add_parser("section", help="render the scorecard section from report.json")
    s.add_argument("--report", required=True)
    s.add_argument("--scorecard", default=None, help="replace the placeholder between the markers in this file")
    s.add_argument("--date", default=None)
    n = sub.add_parser("night", help="one window's judge pass (the standard profile only), then finish")
    n.add_argument("--work", required=True)
    n.add_argument("--judge-url", required=True)
    n.add_argument("--judge-key-file", required=True)
    n.add_argument("--until", required=True, help="HH:MM ET: stop judging then")
    n.add_argument("--min-slots", type=int, default=4, help="refuse a server with fewer slots (not standard)")
    n.add_argument("--concurrency", type=int, default=1)
    n.add_argument("--timeout", type=float, default=600.0)
    n.add_argument("--wait-min", type=float, default=45.0, help="how long to wait for the server to come up")
    n.add_argument("--poll-s", type=float, default=60.0)
    n.add_argument("--py", default=sys.executable)
    n.add_argument("--notify", default="~/distill/notify.sh")
    n.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "finish": cmd_finish, "section": cmd_section, "night": cmd_night}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
