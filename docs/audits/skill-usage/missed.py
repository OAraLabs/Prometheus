"""Missed opportunities: human turns an existing skill clearly matched, where no skill was loaded.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=~/prometheus-deploy/src \\
        python3 missed.py <snapshot-dir> <models-dir> [--threshold T] [--chat-only calib|borderline]

Encoder: the Instinct kNN baseline's, reproduced exactly — BAAI/bge-small-en-v1.5
at revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a, its own ONNX export on
onnxruntime's CPU provider, the repo tokenizer truncated to 256 tokens, CLS
pooling, L2-normalised, no query instruction. Files are checked against their
SHA-256 pins before use.

Turns: every LCM row a person sent (``_common.human_text``, the daemon's
``_human_message_from`` rule) on a user surface. Skills: name + description of
every skill the registry held at the turn's time — born before it and not yet
archived. A turn "loaded a skill" when a ``skill`` tool_use follows it in the
same session before the next human turn.

Default output is aggregate. ``--chat-only calib`` prints a stratified sample
of (score, request, skill) pairs and ``--chat-only borderline`` the pairs
nearest the threshold, for the operator's terminal only: never commit them.
"""

from __future__ import annotations

import collections
import hashlib
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
from catalog import load_catalog  # noqa: E402

PINS = {
    "onnx/model.onnx": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
}
BANDS = [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]


class Encoder:
    def __init__(self, directory: Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        for rel, sha in PINS.items():
            if hashlib.sha256((directory / rel).read_bytes()).hexdigest() != sha:
                raise SystemExit(f"{rel}: sha256 does not match the pin")
        self.tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tok.enable_truncation(max_length=256)
        self.tok.enable_padding()
        self.sess = ort.InferenceSession(str(directory / "onnx" / "model.onnx"),
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts: list[str], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            hidden = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            pooled = hidden[:, 0, :].astype(np.float32)
            out.append(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12))
        return np.concatenate(out)


_SECRET = __import__("re").compile(
    r"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_\-]{16,}|xox[abprs]-[A-Za-z0-9\-]{10,}"
    r"|AKIA[0-9A-Z]{16}|\b\d{1,3}(?:\.\d{1,3}){3}\b|[A-Fa-f0-9]{32,})")


def shown(text: str, n: int) -> str:
    """Chat-only display: truncated, with secret-shaped strings and IPv4s masked."""
    return repr(_SECRET.sub("<redacted>", text[:n]))


def skill_text(s: dict) -> str:
    return f"{s['name'].replace('-', ' ').replace('_', ' ')}: {s['description']}"


def main() -> None:
    snap, models = Path(sys.argv[1]), Path(sys.argv[2])
    threshold = float(sys.argv[sys.argv.index("--threshold") + 1]) if "--threshold" in sys.argv else None
    mode = sys.argv[sys.argv.index("--chat-only") + 1] if "--chat-only" in sys.argv else None

    cat = [s for s in load_catalog(snap) if not s["is_backup"]]
    turns = []  # (sid, rid, ts, surface, text)
    loads: set[tuple[str, int]] = set()
    current: dict[str, int] = {}
    for row in C.lcm_rows(C.lcm(snap)):
        sid = row["session_id"]
        text = C.human_text(row["role"], row["provenance"], row["content"], row["content_json"])
        if text is not None:
            current[sid] = row["rid"]
            if C.is_user_surface(sid):
                turns.append((sid, row["rid"], row["timestamp"], C.surface(sid), text))
            continue
        for b in C.blocks(row["content_json"]):
            if b.get("type") == "tool_use" and b.get("name") == "skill" and sid in current:
                loads.add((sid, current[sid]))

    enc = Encoder(models)
    S = enc.encode([skill_text(s) for s in cat])
    T = enc.encode([t[4] for t in turns])
    sims = T @ S.T  # cosine: both sides are L2-normalised

    def alive(s: dict, ts: float) -> bool:
        if s["born"] > ts:
            return False
        if s["served"]:
            return True
        return s["archived_at"] is not None and s["archived_at"] > ts

    alive_mask = np.array([[alive(s, t[2]) for s in cat] for t in turns])
    masked = np.where(alive_mask, sims, -1.0)
    order = np.argsort(-masked, axis=1)
    top1 = masked[np.arange(len(turns)), order[:, 0]]
    top2 = masked[np.arange(len(turns)), order[:, 1]]

    print(f"== {len(turns)} human turns on user surfaces, {len(cat)} skills "
          f"({collections.Counter(s['source'] for s in cat)}); "
          f"median skills alive per turn {int(np.median(alive_mask.sum(axis=1)))}")
    print(f"   turns that loaded a skill: {sum((t[0], t[1]) in loads for t in turns)}")
    print("== top-1 cosine per turn (bge-small), histogram")
    hist = np.histogram(top1, bins=BANDS)[0]
    for lo, hi, n in zip(BANDS[:-1], BANDS[1:], hist):
        print(f"   [{lo:.2f}, {min(hi, 1.0):.2f}) {n:4}")
    print("== turns whose top-1 skill clears a threshold, by the skill's source; and how many loaded it")
    for thr in (0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85):
        hit = top1 >= thr
        src = collections.Counter(cat[order[i, 0]]["source"] for i in np.where(hit)[0])
        loaded = sum(1 for i in np.where(hit)[0] if (turns[i][0], turns[i][1]) in loads)
        two = int(((top1 >= thr) & (top2 >= thr)).sum())
        uniq_sk = len({order[i, 0] for i in np.where(hit)[0]})
        print(f"   >= {thr:.2f}: {int(hit.sum()):4} turns ({hit.mean():.1%}), top-2 also clears {two}; "
              f"distinct skills {uniq_sk}; by source {dict(src)}; loaded a skill {loaded}")
    # Calibration on KNOWN matches: an auto skill against the request it was
    # generated from (skill_created's trigger_task, the message's first 200
    # chars). The skill literally codifies that request, so these pairs are
    # the clearest matches that exist; their scores say where "clear" sits.
    import json as _json

    stems = {s["stem"]: i for i, s in enumerate(cat)} | {s["name"]: i for i, s in enumerate(cat)}
    pairs = []
    for row in C.telemetry(snap).execute(
            "SELECT payload FROM signal_events WHERE signal_type='skill_created'"):
        p = _json.loads(row["payload"])
        i = stems.get(Path(str(p.get("skill_path", ""))).stem, stems.get(str(p.get("skill_name", ""))))
        if i is not None and str(p.get("trigger_task", "")).strip():
            pairs.append((str(p["trigger_task"]), i))
    if pairs:
        E = enc.encode([t for t, _ in pairs])
        own = np.array([float(E[k] @ S[i]) for k, (_, i) in enumerate(pairs)])
        other = np.array([float(np.max(np.delete(E[k] @ S.T, i))) for k, (_, i) in enumerate(pairs)])
        rank1 = int(sum(o > b for o, b in zip(own, other)))
        q = lambda a: " ".join(f"{p}%={np.percentile(a, p):.3f}" for p in (10, 25, 50, 75, 90))  # noqa: E731
        print(f"== calibration: {len(pairs)} auto skills vs their own trigger request")
        print(f"   own-skill cosine:        {q(own)}")
        print(f"   best OTHER skill cosine: {q(other)}")
        print(f"   own skill ranks first among all {len(cat)} skills: {rank1} of {len(pairs)}")
        for thr in (0.70, 0.72, 0.75, 0.78, 0.80):
            print(f"   >= {thr:.2f}: own {float((own >= thr).mean()):.0%}, best-other {float((other >= thr).mean()):.0%}")

    # Option B cost: a per-turn block (header + top-1, + top-2 when it also
    # clears) in the same "- **name**: description" line format the prompt uses.
    qtok = models.parent / "qwen3" / "tokenizer.json"
    if qtok.is_file():
        from tokenizers import Tokenizer

        qt = Tokenizer.from_file(str(qtok))
        count = lambda t: len(qt.encode(t, add_special_tokens=False).ids)  # noqa: E731
        header = count("## Skills that may fit this request (load with the skill tool)")
        line = [count(f"- **{c['name']}**: {c['description']}") for c in cat]
        print("== relevance-picked block: prompt tokens (qwen3) it would add")
        for thr in (0.72, 0.75, 0.78, 0.80):
            per = []
            for i in range(len(turns)):
                toks = 0
                if top1[i] >= thr:
                    toks = header + line[order[i, 0]] + (line[order[i, 1]] if top2[i] >= thr else 0)
                per.append(toks)
            fired = [t for t in per if t]
            print(f"   T={thr:.2f}: fires on {len(fired)} of {len(turns)} turns; "
                  f"{np.mean(per):.1f} tokens per turn on average, "
                  f"{(np.mean(fired) if fired else 0):.0f} on a turn where it fires")
    import time as _time

    lat = []
    for t in [t[4] for t in turns[:200]]:
        t0 = _time.perf_counter()
        enc.encode([t])
        lat.append((_time.perf_counter() - t0) * 1000)
    print(f"== bge-small single-request encode on this CPU: p50 {np.percentile(lat, 50):.1f} ms, "
          f"p95 {np.percentile(lat, 95):.1f} ms (+ a {len(cat)}-row dot product)")

    if threshold is not None:
        hit = np.where(top1 >= threshold)[0]
        by_surface = collections.Counter(turns[i][3] for i in hit)
        by_month = collections.Counter(C.month(turns[i][2]) for i in hit)
        missed = [i for i in hit if (turns[i][0], turns[i][1]) not in loads]
        print(f"== at T={threshold:.2f}: matched {len(hit)} of {len(turns)} turns; missed {len(missed)}; "
              f"by surface {dict(by_surface)}; by month {sorted(by_month.items())}")
        # Was the skill tool in front of the model on those turns? The run's
        # advertisement row precedes the turn's LCM rows (persisted at turn
        # end) in the same session, or in the pre-#458 "web" namespace.
        ads = collections.defaultdict(list)
        for row in C.telemetry(snap).execute(
                "SELECT timestamp, session_id, summary_json FROM subsystem_runs"
                " WHERE subsystem='agent_loop' AND operation='tool_advertisement'"):
            ads[row["session_id"]].append((row["timestamp"], _json.loads(row["summary_json"] or "{}")))
        shapes = collections.Counter()
        shape_of: dict[int, str] = {}
        for i in hit:
            sid, ts = turns[i][0], turns[i][2]
            cands = [a for a in ads.get(sid, []) + ads.get("web", []) if ts - 3600 <= a[0] <= ts]
            if not cands:
                shape_of[i] = "no advertisement row (before 2026-07-31, or not found)"
            else:
                d = max(cands, key=lambda a: a[0])[1]
                shape_of[i] = ("skill advertised" if not d.get("deferred_active") and d.get("advertised") == d.get(
                    "registered_total") and (d.get("registered_total") or 0) > 20 else "skill NOT advertised")
            shapes[shape_of[i]] += 1
        print(f"   the flagged turns' runs: {dict(shapes)}")
        if mode == "above":
            print("   #### CHAT-ONLY: per flagged turn (score, day, surface, top skill source, run shape)")
            for i in sorted(hit, key=lambda i: -top1[i]):
                print(f"   [{top1[i]:.3f}] {C.day(turns[i][2])} {turns[i][3]:16} "
                      f"{cat[order[i, 0]]['source']:16} {shape_of[i]}")
        print(f"   base rate by surface: {dict(collections.Counter(t[3] for t in turns))}")

    if mode:
        rng = random.Random(20260926)
        print("\n#### CHAT-ONLY OUTPUT — do not commit ####")
        if mode == "calib":
            for lo, hi in zip(BANDS[3:-1], BANDS[4:]):
                idx = [i for i in range(len(turns)) if lo <= top1[i] < hi]
                for i in rng.sample(idx, min(6, len(idx))):
                    s = cat[order[i, 0]]
                    print(f"[{top1[i]:.3f}] {shown(turns[i][4], 200)}\n        -> {s['source']}: "
                          f"{s['name']} — {shown(s['description'], 160)}")
        elif mode == "band":
            lo = float(sys.argv[sys.argv.index("--lo") + 1])
            hi = float(sys.argv[sys.argv.index("--hi") + 1])
            n = int(sys.argv[sys.argv.index("--n") + 1])
            idx = [i for i in range(len(turns)) if lo <= top1[i] < hi]
            print(f"band [{lo}, {hi}): {len(idx)} turns; showing {min(n, len(idx))}")
            for i in sorted(random.Random(7).sample(idx, min(n, len(idx))), key=lambda i: -top1[i]):
                s = cat[order[i, 0]]
                print(f"[{top1[i]:.3f}] {turns[i][3]} {shown(turns[i][4], 200)}\n"
                      f"        -> {s['source']}: {s['name']} — {shown(s['description'], 140)}")
        elif mode in ("borderline", "above") and threshold is not None:
            if mode == "above":
                idx = [i for i in range(len(turns)) if top1[i] >= threshold]
            else:
                idx = sorted(range(len(turns)), key=lambda i: abs(top1[i] - threshold))[:12]
            for i in sorted(idx, key=lambda i: -top1[i]):
                s = cat[order[i, 0]]
                print(f"[{top1[i]:.3f}] {C.day(turns[i][2])} {turns[i][3]} {shown(turns[i][4], 240)}\n"
                      f"        -> {s['source']}: {s['name']} — {shown(s['description'], 160)}")


if __name__ == "__main__":
    main()
