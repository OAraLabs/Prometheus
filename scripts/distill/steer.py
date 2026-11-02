#!/usr/bin/env python3
"""Round-2 steering: the share of generation each target area gets (stdlib only).

    python3 steer.py defaults [--out ~/distill/round2/steer.json]          # the operator's priorities
    python3 steer.py steer --scorecard '~/distill/phase7/<run>/summaries/q4*.summary.json' \\
        [--cases ~/distill/phase3/build-v4b-max-derived/eval-tools.jsonl --cases .../sanity.jsonl] \\
        [--base '~/distill/phase7/<run>/summaries/base*.summary.json'] [--out ~/distill/round2/steer.json]
    python3 steer.py show [--config ~/distill/round2/steer.json]           # the flattened scheduling shares

The config is what ``teacher_gen.py`` reads for a round-2 taskset (``--steer``, default
``DEFAULT_PATH``; re-read whenever the file changes, so a new scorecard steers a running
generation at its next task). Until a scorecard exists the defaults hold.

Areas (``AREAS``) and the operator's default priorities (2026-11-01 ~01:20 ET):

1. ``injection-resistance``: real tasks with adversarial content in a tool result;
2. ``error-recovery``: real tasks whose sandbox fails a call with a real failure pattern;
3. ``long-context``: real prompts after 32K-128K of real repo files already read;
4. ``daily-agent``: the non-coding categories (sentinel/daemon, memory/wiki, other,
   chat-mode), split by ``daily_split``;

and ``coding`` (plain coding tasks) at a small share, never above ``CODING_MAX``.

``tool-choice`` (operator 2026-11-01 ~23:00 ET, from the Phase 7 scorecard: the student's
eval-tools misses are wrong-tool choices, not syntax): real tasks whose right first tool is
known from a Max trace and has near-neighbours the evals confuse it with. It takes the
largest default share (``DEFAULT_WEIGHTS``). A config written before it existed (no
``tool-choice`` weight) still loads: the area gets 0 there.

``steer`` maps a phase-7 scorecard (``score.py`` summaries) onto the areas:

* ``eval-injection`` -> injection-resistance (``injection_resistance``);
* ``eval-longctx`` -> long-context (``recall_by_depth`` for the 32k/64k/128k buckets: the
  buckets round 2 generates; 256k is outside the teacher's context);
* ``eval-tools`` cases -> by ``meta.category`` (``--cases``: the case files give each id its
  category): coding -> coding, the others -> daily-agent's own category; a case whose
  request holds a failed tool result after its last user message also counts for
  error-recovery;
* ``sanity`` -> daily-agent's chat-mode;
* every ``eval-tools`` case -> tool-choice too (its outcome needs the right tool first).

An area's weakness is ``1 - score`` (plus the drop below ``--base`` when a base summary is
given). The new share is the default share times ``1 + STRENGTH * (weakness - mean) /
mean``, clamped to ``[MIN_FACTOR, MAX_FACTOR]`` x the default, coding capped at
``CODING_MAX``, then normalised; an area with no measurement keeps its default factor 1.
The file records every input (paths, sha256), the scores, the weaknesses and the reasoning
line per area.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "distill-steer/1"
DEFAULT_PATH = "~/distill/round2/steer.json"
AREAS = ("injection-resistance", "error-recovery", "long-context", "daily-agent", "coding", "tool-choice")
DAILY_CATEGORIES = ("sentinel/daemon", "memory/wiki", "other", "chat-mode")
#: the operator's order (2026-11-01 ~01:20 ET): injection, error recovery, long context, daily agent; coding
#: small; then (~23:00 ET) tool CHOICE first: 0.30, the other shares scaled to the remaining 0.70
DEFAULT_WEIGHTS = {"tool-choice": 0.30, "injection-resistance": 0.21, "error-recovery": 0.175,
                   "long-context": 0.126, "daily-agent": 0.14, "coding": 0.049}
#: areas a config written before them may leave out (they get a share of 0 there)
LATER_AREAS = ("tool-choice",)
DEFAULT_DAILY_SPLIT = {"sentinel/daemon": 0.30, "memory/wiki": 0.30, "other": 0.20, "chat-mode": 0.20}
CODING_MAX = 0.10
STRENGTH = 1.0
MIN_FACTOR, MAX_FACTOR = 0.5, 2.0
LONGCTX_BUCKETS = ("32k", "64k", "128k")


class SteerError(ValueError):
    pass


def default_config() -> dict[str, Any]:
    return {"schema": SCHEMA, "basis": "defaults", "weights": dict(DEFAULT_WEIGHTS),
            "daily_split": dict(DEFAULT_DAILY_SPLIT), "coding_max": CODING_MAX,
            "reasoning": ["operator priorities 2026-11-01: injection-resistance, error-recovery, "
                          "long-context, daily-agent; coding at a small share (no scorecard yet)",
                          "operator 2026-11-01 ~23:00 ET: tool choice first (0.30); the others scaled by 0.70"]}


def _normalise(d: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, v) for v in d.values())
    if total <= 0:
        raise SteerError("weights sum to zero")
    return {k: round(max(0.0, v) / total, 4) for k, v in d.items()}


def validate(cfg: Any) -> dict[str, Any]:
    """A config read back, checked: every area present, shares >= 0, coding within its cap."""
    if not isinstance(cfg, dict) or cfg.get("schema") != SCHEMA:
        raise SteerError(f"not a {SCHEMA} config")
    w = cfg.get("weights")
    if isinstance(w, dict):
        w = dict(w, **{k: 0.0 for k in LATER_AREAS if k not in w})
    if not isinstance(w, dict) or set(w) != set(AREAS) or not all(
            isinstance(v, (int, float)) and v >= 0 for v in w.values()):
        raise SteerError(f"weights must give every area of {AREAS} a share >= 0")
    split = cfg.get("daily_split")
    if not isinstance(split, dict) or set(split) != set(DAILY_CATEGORIES) or not all(
            isinstance(v, (int, float)) and v >= 0 for v in split.values()):
        raise SteerError(f"daily_split must give every category of {DAILY_CATEGORIES} a share >= 0")
    cmax = float(cfg.get("coding_max", CODING_MAX))
    w = _normalise({k: float(v) for k, v in w.items()})
    if w["coding"] > cmax + 1e-6:
        raise SteerError(f"coding share {w['coding']} is above coding_max {cmax}")
    return dict(cfg, weights=w, daily_split=_normalise({k: float(v) for k, v in split.items()}))


def load(path: str | Path | None = None) -> dict[str, Any]:
    """The config at *path* (default ``DEFAULT_PATH``), or the defaults when it does not exist."""
    p = Path(os.path.expanduser(str(path or DEFAULT_PATH)))
    if not p.exists():
        return validate(default_config())
    return validate(json.loads(p.read_text(encoding="utf-8")))


def bucket_of(area: str, category: str) -> str:
    """The scheduling bucket of a round-2 task: its area, or ``daily-agent:<category>``."""
    return f"daily-agent:{category}" if area == "daily-agent" else area


def flatten(cfg: dict[str, Any]) -> dict[str, float]:
    """Scheduling shares per bucket (``bucket_of``): area share, daily-agent split by category."""
    cfg = validate(cfg)
    out: dict[str, float] = {}
    for area, share in cfg["weights"].items():
        if area == "daily-agent":
            for cat, sub in cfg["daily_split"].items():
                out[bucket_of(area, cat)] = round(share * sub, 5)
        else:
            out[area] = share
    return out


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def _expand(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.expanduser(pat)))
        if not hits:
            raise SteerError(f"{pat}: no file matched")
        out += [Path(x) for x in hits]
    return out


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def case_facts(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """id -> {set, category, error_recovery} from score.py case files (eval-tools, sanity)."""
    facts: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh.read().split("\n"):
                if not line.strip():
                    continue
                c = json.loads(line)
                msgs = c.get("messages") or []
                last_user = max((k for k, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
                err = any(m.get("role") == "tool" and (m.get("is_error") is True
                                                      or re.match(r"\s*(error|.* failed:)", _text(m.get("content")),
                                                                  re.I))
                          for m in msgs[last_user + 1:])
                facts[str(c.get("id"))] = {"set": c.get("set"), "category": (c.get("meta") or {}).get("category"),
                                           "error_recovery": err}
    return facts


def _rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def area_scores(summaries: list[dict[str, Any]], facts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pass rates per area (and per daily-agent category) from score.py summaries. Several
    summaries pool their outcomes (one model, several runs or quants of the same student)."""
    tally: dict[str, list[int]] = {}

    def add(key: str, ok: bool) -> None:
        t = tally.setdefault(key, [0, 0])
        t[0] += bool(ok)
        t[1] += 1

    for sm in summaries:
        sets = sm.get("sets") or {}
        inj = sets.get("eval-injection") or {}
        for _cid, ok in (inj.get("outcomes") or {}).items():
            add("injection-resistance", ok)
        lc = sets.get("eval-longctx") or {}
        buckets = lc.get("case_bucket") or {}
        for cid, ok in (lc.get("outcomes") or {}).items():
            if buckets.get(cid) in LONGCTX_BUCKETS:
                add("long-context", ok)
                add(f"long-context:{buckets[cid]}", ok)
        for cid, ok in ((sets.get("eval-tools") or {}).get("outcomes") or {}).items():
            f = facts.get(cid) or {}
            cat = f.get("category")
            if cat == "coding":
                add("coding", ok)
            elif cat in DAILY_CATEGORIES:
                add("daily-agent", ok)
                add(f"daily-agent:{cat}", ok)
            if f.get("error_recovery"):
                add("error-recovery", ok)
            add("tool-choice", ok)
        for _cid, ok in ((sets.get("sanity") or {}).get("outcomes") or {}).items():
            add("daily-agent", ok)
            add("daily-agent:chat-mode", ok)
    return {k: {"k": v[0], "n": v[1], "rate": _rate(*v)} for k, v in sorted(tally.items())}


def steer(scores: dict[str, Any], base: dict[str, Any] | None = None,
          prior: dict[str, Any] | None = None) -> dict[str, Any]:
    """A config reweighted toward the weakest measured areas (module docstring)."""
    prior = validate(prior or default_config())
    pw, ps = prior["weights"], prior["daily_split"]
    reasoning: list[str] = []

    def weakness(key: str) -> float | None:
        s = (scores.get(key) or {}).get("rate")
        if s is None:
            return None
        w = 1.0 - s
        b = ((base or {}).get(key) or {}).get("rate")
        if b is not None and b > s:
            w += b - s
        return round(w, 4)

    def reweight(keys: list[str], shares: dict[str, float], label: str) -> dict[str, float]:
        weak = {k: weakness(k if label == "area" else f"daily-agent:{k}") for k in keys}
        known = [v for v in weak.values() if v is not None]
        mean = sum(known) / len(known) if known else 0.0
        out: dict[str, float] = {}
        for k in keys:
            w = weak[k]
            if w is None or mean <= 0:
                factor = 1.0
                why = "no measurement" if w is None else "every measured area is at 100%"
            else:
                factor = min(MAX_FACTOR, max(MIN_FACTOR, 1.0 + STRENGTH * (w - mean) / mean))
                key = k if label == "area" else f"daily-agent:{k}"
                sc = scores[key]
                why = (f"score {sc['rate']:.1%} ({sc['k']}/{sc['n']}), weakness {w:.3f} vs mean {mean:.3f}"
                       + (f", base {base[key]['rate']:.1%}" if base and (base.get(key) or {}).get("rate") is not None
                          else ""))
            out[k] = shares[k] * factor
            reasoning.append(f"{label} {k}: default {shares[k]:.3f} x {factor:.2f} ({why})")
        return out

    w = reweight(list(AREAS), pw, "area")
    cmax = float(prior.get("coding_max", CODING_MAX))
    w = _normalise(w)
    if w["coding"] > cmax:
        rest = 1.0 - cmax
        others = {k: v for k, v in w.items() if k != "coding"}
        tot = sum(others.values())
        w = {k: (v / tot * rest if k != "coding" else cmax) for k, v in w.items()}
        reasoning.append(f"coding capped at {cmax:.2f} (plain coding stays a small share)")
    split = _normalise(reweight(list(DAILY_CATEGORIES), ps, "daily-agent category"))
    return validate({"schema": SCHEMA, "basis": "scorecard", "weights": _normalise(w), "daily_split": split,
                     "coding_max": cmax, "reasoning": reasoning, "scores": scores,
                     "base_scores": base or None})


def _load_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for p in paths:
        sm = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(sm, dict) or "sets" not in sm:
            raise SteerError(f"{p}: not a score.py summary (no 'sets')")
        out.append(sm)
    return out


def _write(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="steer.py", description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("defaults", help="write the operator's default priorities")
    d.add_argument("--out", default=DEFAULT_PATH)
    s = sub.add_parser("steer", help="reweight toward the weakest areas of a scorecard")
    s.add_argument("--scorecard", action="append", required=True, help="score.py summary JSON (glob; repeatable)")
    s.add_argument("--base", action="append", default=[], help="a base model's summaries (glob; repeatable)")
    s.add_argument("--cases", action="append", default=[], help="case files mapping ids to categories (glob)")
    s.add_argument("--prior", help="the config to reweight (default: the operator's defaults)")
    s.add_argument("--out", default=DEFAULT_PATH)
    s.add_argument("--dry-run", action="store_true")
    sh = sub.add_parser("show")
    sh.add_argument("--config", default=DEFAULT_PATH)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "defaults":
            cfg = validate(default_config())
            cfg["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            _write(Path(os.path.expanduser(a.out)), cfg)
            print(json.dumps({"out": a.out, "buckets": flatten(cfg)}, indent=1))
            return 0
        if a.cmd == "show":
            cfg = load(a.config)
            print(json.dumps({"basis": cfg.get("basis"), "weights": cfg["weights"],
                              "daily_split": cfg["daily_split"], "buckets": flatten(cfg),
                              "reasoning": cfg.get("reasoning")}, indent=1))
            return 0
        sc_paths = _expand(a.scorecard)
        case_paths = _expand(a.cases) if a.cases else []
        facts = case_facts(case_paths)
        scores = area_scores(_load_summaries(sc_paths), facts)
        base = area_scores(_load_summaries(_expand(a.base)), facts) if a.base else None
        prior = load(a.prior) if a.prior else None
        cfg = steer(scores, base, prior)
        cfg["inputs"] = {"scorecard": [{"path": str(p), "sha256": _sha256(p)} for p in sc_paths],
                         "base": [{"path": str(p), "sha256": _sha256(p)} for p in _expand(a.base)] if a.base else [],
                         "cases": [{"path": str(p), "sha256": _sha256(p)} for p in case_paths]}
        if not case_paths:
            cfg["reasoning"].append("no --cases: eval-tools outcomes could not be mapped to categories "
                                    "(coding, daily-agent and error-recovery keep their defaults)")
        cfg["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if not a.dry_run:
            _write(Path(os.path.expanduser(a.out)), cfg)
        print(json.dumps({"out": None if a.dry_run else a.out, "weights": cfg["weights"],
                          "daily_split": cfg["daily_split"], "reasoning": cfg["reasoning"]}, indent=1))
        return 0
    except (SteerError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
