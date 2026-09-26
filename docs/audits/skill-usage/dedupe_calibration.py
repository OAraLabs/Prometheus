"""Calibrate SkillCreator's near-duplicate gate, and count what it would have rejected.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<checkout>/src:~/prometheus-deploy/src \\
        python3 dedupe_calibration.py <snapshot-dir> <bge-small dir> [--chat-only]

For each of the auto skills ever written (live and archived, backups
excluded), in creation order, it takes the skill's ``name + description`` —
the text ``prometheus.skills.similarity.skill_text`` builds — and finds its
nearest neighbour among the skills that existed when it was written: the
builtins, the user-directory skills, and the earlier auto skills still served
then. Encoder: the audit's bge-small (``missed.py``), which is the one the gate
pins. It also applies the gate's other check, a description that is literally
``name: …``.

Aggregate output: the distribution of nearest-neighbour cosines and, for a
range of thresholds, how many skills the gate would have rejected.
``--chat-only`` adds per-skill rows (names) for hand-labelling the threshold;
never commit that output.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import load_catalog  # noqa: E402
from missed import Encoder  # noqa: E402

MALFORMED = re.compile(r"^\s*name\s*:", re.IGNORECASE)


def text_of(s: dict) -> str:
    try:
        from prometheus.skills.similarity import skill_text
    except ImportError:  # the deploy tree predates the module: same formula as missed.py
        return f"{s['name'].replace('-', ' ').replace('_', ' ')}: {s['description']}"
    return skill_text(s["name"], s["description"])


def main() -> None:
    snap, models = Path(sys.argv[1]), Path(sys.argv[2])
    chat = "--chat-only" in sys.argv
    cat = [s for s in load_catalog(snap) if not s["is_backup"]]
    autos = sorted((s for s in cat if s["source"].startswith("auto")), key=lambda s: s["born"])
    enc = Encoder(models)
    vecs = enc.encode([text_of(s) for s in cat])
    index = {id(s): i for i, s in enumerate(cat)}

    def existed_at(s: dict, ts: float) -> bool:
        if s["source"] in ("builtin", "user"):
            return s["source"] == "builtin" or s["mtime"] <= ts
        if s["born"] >= ts:
            return False
        return s["served"] or (s["archived_at"] is not None and s["archived_at"] > ts)

    rows = []
    for a in autos:
        others = [s for s in cat if s is not a and existed_at(s, a["born"])]
        if not others:
            rows.append((a, 0.0, None))
            continue
        sims = vecs[[index[id(s)] for s in others]] @ vecs[index[id(a)]]
        k = int(np.argmax(sims))
        rows.append((a, float(sims[k]), others[k]))
    best = np.array([r[1] for r in rows])
    malformed = [r for r in rows if MALFORMED.match(r[0]["description"])]
    print(f"== {len(rows)} auto skills; nearest neighbour at creation, cosine quantiles: "
          + " ".join(f"p{p}={np.percentile(best, p):.3f}" for p in (10, 25, 50, 75, 90)))
    print(f"   malformed 'name: …' descriptions: {len(malformed)}")
    for t in (0.80, 0.82, 0.85, 0.87, 0.88, 0.90, 0.92, 0.95):
        dup = [r for r in rows if r[1] >= t]
        either = {id(r[0]) for r in dup} | {id(r[0]) for r in malformed}
        by_src = {}
        for r in dup:
            src = r[2]["source"] if r[2] else "-"
            by_src[src] = by_src.get(src, 0) + 1
        print(f"   T={t:.2f}: near-duplicate {len(dup):2} (nearest is {by_src}); "
              f"rejected by either check {len(either):2} of {len(rows)}")
    if chat:
        print("#### CHAT-ONLY OUTPUT — do not commit ####")
        for a, score, near in sorted(rows, key=lambda r: -r[1]):
            print(f"[{score:.3f}] {a['name']}  ->  {near['source'] if near else '-'}: "
                  f"{near['name'] if near else '-'}")


def via_gate(snap: Path, models: Path) -> None:
    """The same 57, through the real write path: ``SkillCreator.persist_skill_content``
    with the pinned encoder (``SimilarityChecker``) and, for each skill, the catalog
    as it stood when that skill was written. Counts rejections by reason."""
    import asyncio
    import collections
    import tempfile
    from unittest.mock import MagicMock

    from prometheus.learning.skill_creator import SkillCreator
    from prometheus.skills.similarity import SimilarityChecker, skill_text

    cat = [s for s in load_catalog(snap) if not s["is_backup"]]
    autos = sorted((s for s in cat if s["source"].startswith("auto")), key=lambda s: s["born"])
    checker = SimilarityChecker(models)
    assert checker.available, checker.unavailable_reason

    class _Tel:
        def __init__(self) -> None:
            self.reasons: collections.Counter = collections.Counter()

        def record_run(self, subsystem, operation, outcome, summary=None, **kw):
            if (subsystem, operation, outcome) == ("skill_creator", "quality_gate", "skipped"):
                self.reasons[(summary or {}).get("reason")] += 1

        def record_silent_failure(self, *a, **k):
            pass

    def existed_at(s: dict, ts: float) -> bool:
        if s["source"] in ("builtin", "user"):
            return s["source"] == "builtin" or s["mtime"] <= ts
        if s["born"] >= ts:
            return False
        return s["served"] or (s["archived_at"] is not None and s["archived_at"] > ts)

    tel = _Tel()
    written = 0
    for a in autos:
        catalog = [(s["name"], skill_text(s["name"], s["description"]))
                   for s in cat if s is not a and existed_at(s, a["born"])]
        with tempfile.TemporaryDirectory() as auto_dir:
            creator = SkillCreator(MagicMock(), auto_dir=Path(auto_dir), telemetry=tel,
                                   similarity=checker, catalog=lambda c=catalog: c)
            path = asyncio.run(creator.persist_skill_content(a["content"], trigger="replay",
                                                             on_collision="skip"))
            written += path is not None
    print(f"== via SkillCreator.persist_skill_content: {len(autos)} skills, written {written}, "
          f"rejected {len(autos) - written}: {dict(tel.reasons)}")


if __name__ == "__main__":
    if "--via-gate" in sys.argv:
        via_gate(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        main()
