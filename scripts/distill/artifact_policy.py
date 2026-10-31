#!/usr/bin/env python3
"""Max-derived artifacts: one rule for names, manifests and publishing.

DISTILL-OPEN-QUESTIONS Q6b (operator ruling, 2026-10-30): the harvested Max
traces may train a PRIVATE, internal-only model. Anything trained on them, or
holding them, is Max-derived:

* it never leaves our machines for a public place (Hugging Face, a release,
  a public bucket): :func:`assert_publishable` refuses it, and so does
  ``publish_guard.py check``;
* it is tagged: ``max_derived: true`` plus ``tags: ["max-derived"]`` in its
  manifest (:func:`policy_fields`), and ``max-derived`` in its name
  (:func:`tag_name`);
* the rental tool (``modal_train.py``) refuses it outright: renting uploads the
  training data off our machines.

Stdlib only, so every distill script can load it from its own directory (the
way ``build.py`` loads ``render.py``).

Provenance chain
----------------
:func:`provenance` reads an artifact's own manifest, then follows the paths it
records upstream: build.py's dataset report (``report.json``,
``distill-phase3-report/1``) and dataset manifest (``dataset-manifest.json``)
-> train_qlora.py's run manifest (``manifest.json``, whose ``data.path`` names
the training file) -> the adapter directory inside that run (its parent's
manifest) -> quantize.py's ``quantize-manifest.json`` (``inputs.adapter.dir``)
for the merged and quantized models beside it. A pre-tokenized directory's
``meta.json`` counts its own rows. A path whose name (or any parent's name)
carries the tag is Max-derived whatever its manifests say.

Two questions, kept apart:

* ``max_derived`` -- trained on Max-derived rows (a model, an adapter, a run),
  or Max-derived rows in a dataset's train/val (what a training run inherits).
  ``None`` when nothing shows either way; :func:`is_max_derived` counts that as
  Max-derived (fail closed, like the trainer's row guard).
* ``holds_max_rows`` -- the artifact holds Max-derived rows anywhere: a
  dataset's eval-tools set is built from Max traces even when its train split
  is clean. Such a file is not published either, but a model trained on that
  dataset's clean train split does not inherit it.

Publishing
----------
No publish path for model or dataset artifacts exists in this repo (package
releases ship ``src/prometheus`` and the sdist allowlist, never ``data/``).
Any future publish step -- ``huggingface_hub`` uploads (``push_to_hub``,
``upload_folder``, ``upload_file``), ``hf upload``, ``gh release upload`` of a
model or dataset, a copy to any public location -- must first call
:func:`assert_publishable` on every artifact it ships, or run
``python3 scripts/distill/publish_guard.py check <path>...`` and stop on a
non-zero exit. Unknown provenance is refused too.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

TAG = "max-derived"
RULING = "DISTILL-OPEN-QUESTIONS Q6b (operator, 2026-10-30)"
#: build.py's explicit allow: both flags together, the reference recorded verbatim.
BUILD_ALLOW_FLAG = "--allow-max-derived"
BUILD_APPROVAL_FLAG = "--max-derived-approval"
RENTAL_REFUSAL = (
    "Max-derived data never rents: renting uploads the training data off our machines, and "
    "Max-derived data stays on them (DISTILL-OPEN-QUESTIONS #6, Q6b ruling). No flag overrides "
    "this. Train it locally (train_qlora.py), or rent with data whose rows are not Max-derived.")

REPORT_SCHEMA = "distill-phase3-report/1"
DATASET_MANIFEST_SCHEMA = "distill-dataset-manifest/1"
TRAIN_MANIFEST_SCHEMA = "distill-train-manifest/1"
PRETOK_SCHEMA = "distill-pretok/1"
QUANTIZE_MANIFEST_SCHEMA = "distill-quantize-manifest/1"
SCHEMAS = frozenset({REPORT_SCHEMA, DATASET_MANIFEST_SCHEMA, TRAIN_MANIFEST_SCHEMA, PRETOK_SCHEMA,
                     QUANTIZE_MANIFEST_SCHEMA})
#: The files a directory's provenance is read from (each must carry one of SCHEMAS).
MANIFEST_NAMES = ("dataset-manifest.json", "report.json", "manifest.json", "meta.json",
                  "quantize-manifest.json")
DATASET_MANIFEST = "dataset-manifest.json"

#: "max-derived" / "max_derived" as a word of a name (not inside another word).
TAG_RE = re.compile(r"(?<![a-z0-9])max[-_]derived(?![a-z0-9])", re.I)
#: A provenance value naming Max: "qwen3.8-max", "max-golden", "max-gym-<date>".
MAX_RE = re.compile(r"(?<![a-z0-9])max(?![a-z0-9])", re.I)
#: Suffixes the tag goes in front of (``train.jsonl`` -> ``train-max-derived.jsonl``).
KNOWN_SUFFIXES = frozenset({
    ".jsonl", ".json", ".gguf", ".safetensors", ".bin", ".pt", ".pth", ".npz", ".npy", ".onnx",
    ".tar", ".gz", ".tgz", ".zst", ".xz", ".zip", ".csv", ".parquet", ".md", ".txt", ".log"})
MAX_UP = 3  # parent levels searched for a manifest
MAX_DEPTH = 8  # upstream links followed
MAX_TREE_ENTRIES = 50_000  # a directory bigger than this is refused unchecked


class NotPublishable(Exception):
    """The artifact is Max-derived, holds Max-derived rows, or its provenance is unknown."""

    def __init__(self, path: Path | str, reasons: list[str]) -> None:
        self.path = str(path)
        self.reasons = list(reasons)
        super().__init__(f"{path}: not publishable ({RULING}): " + "; ".join(self.reasons))


# ---------------------------------------------------------------------------
# Names and manifest fields
# ---------------------------------------------------------------------------


def has_tag(name: str) -> bool:
    """*name* carries the ``max-derived`` tag as a word."""
    return bool(TAG_RE.search(name or ""))


def path_has_tag(path: Path | str) -> bool:
    """Any part of *path* (the file and every parent named in it) carries the tag."""
    return any(has_tag(part) for part in Path(path).parts)


def tag_name(name: str) -> str:
    """*name* with ``max-derived`` in it, before its known suffixes; unchanged
    when it already carries the tag. ``adapter`` -> ``adapter-max-derived``,
    ``train.jsonl`` -> ``train-max-derived.jsonl``, ``student.Q4_K_M.gguf`` ->
    ``student.Q4_K_M-max-derived.gguf``, ``qwen3.5-9b`` -> ``qwen3.5-9b-max-derived``."""
    if not name or name in (".", "..") or "/" in name:
        raise ValueError(f"not a base name: {name!r}")
    if has_tag(name):
        return name
    stem, suffixes = name, ""
    while True:
        dot = stem.rfind(".")
        if dot <= 0 or stem[dot:].lower() not in KNOWN_SUFFIXES:
            break
        stem, suffixes = stem[:dot], stem[dot:] + suffixes
    return f"{stem}-{TAG}{suffixes}"


def tag_path(path: Path | str) -> Path:
    """*path* with its last part tagged (:func:`tag_name`)."""
    p = Path(path)
    return p.with_name(tag_name(p.name))


def name_for(name: str, max_derived: bool) -> str:
    """*name*, tagged when *max_derived*."""
    return tag_name(name) if max_derived else name


def tags_for(max_derived: bool) -> list[str]:
    return [TAG] if max_derived else []


def policy_fields(max_derived: bool) -> dict[str, Any]:
    """The fields every artifact manifest carries: the flag, the tag list and
    whether the artifact is private (never published, never rented)."""
    return {"max_derived": bool(max_derived), "tags": tags_for(max_derived),
            "private": bool(max_derived),
            "publish": (f"refused: Max-derived ({RULING})" if max_derived
                        else "only after publish_guard.py check passes")}


def approval_record(ref: str | None) -> dict[str, Any]:
    """build.py's record of its Max-derived allow (``data_policy.max_derived_allow``):
    allowed only with both flags, the reference kept verbatim."""
    allowed = isinstance(ref, str) and bool(ref.strip())
    return {"allowed": allowed, "approval": ref if allowed else None,
            "flags": [BUILD_ALLOW_FLAG, BUILD_APPROVAL_FLAG] if allowed else [],
            "ruling": RULING}


def dataset_approval(report: Any) -> str | None:
    """The approval reference a dataset recorded (build.py's report or dataset
    manifest), or None: no allow was given, or the record is malformed."""
    if not isinstance(report, dict):
        return None
    rec = report.get("max_derived_allow")
    dp = report.get("data_policy")
    if rec is None and isinstance(dp, dict):
        rec = dp.get("max_derived_allow")
    if not isinstance(rec, dict) or rec.get("allowed") is not True:
        return None
    ref = rec.get("approval")
    return ref if isinstance(ref, str) and ref.strip() else None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass
class Provenance:
    max_derived: bool | None  # None: nothing shows either way
    holds_max_rows: bool = False
    evidence: list[str] = field(default_factory=list)

    @property
    def publishable(self) -> bool:
        return self.max_derived is False and not self.holds_max_rows

    def as_dict(self) -> dict[str, Any]:
        return {"max_derived": self.max_derived, "holds_max_rows": self.holds_max_rows,
                "publishable": self.publishable, "evidence": list(self.evidence)}


def _or(a: bool | None, b: bool | None) -> bool | None:
    """True if either says so; else unknown if either is; else False."""
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


def _flag(value: Any) -> bool:
    """A manifest's ``max_derived`` value: a bool as given; quantize.py's
    calibration provenance dict by its Max-derived row count; anything else
    (null, a string, a number) counts as Max-derived (fail closed)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        n = value.get("calibration_max_derived")
        return not (isinstance(n, int) and not isinstance(n, bool) and n == 0)
    return True


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass
class _DocVerdict:
    max_derived: bool | None
    holds: bool
    evidence: list[str]
    upstream: list[tuple[str, bool]]  # (path, decisive): a decisive link settles an unknown


def _doc_verdict(doc: dict[str, Any], where: str) -> _DocVerdict:
    """One manifest on its own, plus the upstream paths it records."""
    ev: list[str] = []
    up: list[tuple[str, bool]] = []
    holds = False
    flag: bool | None = None  # what the manifest says outright
    derived: bool | None = None  # what its own counts say
    if "max_derived" in doc:
        flag = _flag(doc["max_derived"])
        ev.append(f"{where}: max_derived={flag}")
    tags = doc.get("tags")
    if isinstance(tags, list) and any(isinstance(t, str) and has_tag(t) for t in tags):
        flag = True
        ev.append(f"{where}: tagged {TAG}")
    schema = doc.get("schema")
    if schema == REPORT_SCHEMA:
        splits = doc.get("splits") if isinstance(doc.get("splits"), dict) else {}
        counts = {k: _count(v.get("max_derived")) for k, v in splits.items() if isinstance(v, dict)}
        tv = [counts.get(k) for k in ("train", "val")]
        if all(n is not None for n in tv):
            total = sum(n or 0 for n in tv)
            derived = total > 0
            ev.append(f"{where}: {total} Max-derived rows in train/val")
        else:  # an older report: its teacher_model is the trainer's key
            settings = doc.get("settings") if isinstance(doc.get("settings"), dict) else {}
            teacher = settings.get("teacher_model")
            if isinstance(teacher, str) and teacher.strip():
                derived = bool(MAX_RE.search(teacher))
                ev.append(f"{where}: teacher_model {'names' if derived else 'does not name'} Max")
        holds = doc.get("holds_max_rows") is True or any(bool(n) for n in counts.values())
    elif schema == DATASET_MANIFEST_SCHEMA:
        holds = doc.get("holds_max_rows") is not False
    elif schema == TRAIN_MANIFEST_SCHEMA:
        data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
        n = _count(data.get("max_derived_rows"))
        if n is not None:
            derived = n > 0
            ev.append(f"{where}: {n} Max-derived training rows")
        if isinstance(data.get("path"), str) and data["path"]:
            up.append((data["path"], flag is None and derived is None))
    elif schema == PRETOK_SCHEMA:
        n = _count(doc.get("max_derived_rows"))
        if n is not None:
            derived = n > 0
            ev.append(f"{where}: {n} Max-derived rows")
        elif flag is None:
            derived = True  # no count, no flag: fail closed
            ev.append(f"{where}: no Max-derived row count")
        if isinstance(doc.get("source"), str) and doc["source"]:
            up.append((doc["source"], False))
    elif schema == QUANTIZE_MANIFEST_SCHEMA:
        inputs = doc.get("inputs") if isinstance(doc.get("inputs"), dict) else {}
        adapter = inputs.get("adapter")
        if isinstance(adapter, dict) and isinstance(adapter.get("dir"), str):
            up.append((adapter["dir"], True))  # the model is the adapter's: it decides
    if flag is True or derived is True:
        md: bool | None = True
    elif flag is None:
        md = derived
    else:
        md = False
    if schema == PRETOK_SCHEMA:
        holds = md is not False
    if holds:
        ev.append(f"{where}: holds Max-derived rows")
    return _DocVerdict(md, holds, ev, up)


def _load_doc(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) and doc.get("schema") in SCHEMAS else None


def _docs_at(d: Path) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for name in MANIFEST_NAMES:
        p = d / name
        if p.is_file():
            doc = _load_doc(p)
            if doc is not None:
                out.append((p, doc))
    return out


def row_verdict(row: Any) -> bool | None:
    """One data row: True when a provenance field names Max or a
    ``max_derived`` is anything but ``false``; False only with an explicit
    ``max_derived: false``; None when nothing shows either way."""
    if not isinstance(row, dict):
        return None
    holders = [h for h in (row, row.get("_meta"), row.get("meta")) if isinstance(h, dict)]
    for h in holders:
        for key in ("model", "source", "served_model"):
            v = h.get(key)
            if isinstance(v, str) and MAX_RE.search(v):
                return True
    cleared = False
    for h in holders:
        if "max_derived" in h:
            if h["max_derived"] is not False:
                return True
            cleared = True
    return False if cleared else None


def scan_rows(path: Path) -> Provenance:
    """A JSONL file's rows (stops at the first Max-derived row)."""
    unknown = rows = 0
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rows += 1
                try:
                    v = row_verdict(json.loads(line))
                except ValueError:
                    v = None
                if v is True:
                    return Provenance(True, True, [f"{path.name}: row {rows} is Max-derived"])
                unknown += v is None
    except (OSError, UnicodeDecodeError) as exc:
        return Provenance(None, False, [f"{path.name}: unreadable ({type(exc).__name__})"])
    if unknown:
        return Provenance(None, False, [f"{path.name}: {unknown} of {rows} rows show no provenance"])
    return Provenance(False, False, [f"{path.name}: {rows} rows, none Max-derived"])


def provenance(target: Any, *, base: Path | None = None,
               _seen: set[str] | None = None, _depth: int = 0) -> Provenance:
    """Where *target* (a path, or a manifest dict) stands: Max-derived, not,
    or unknown (``None``), following the chain upstream (module docstring).
    Relative upstream paths in a dict resolve against *base* (default: cwd)."""
    seen = _seen if _seen is not None else set()
    if isinstance(target, dict):
        where = Path(base or Path.cwd()) / "<manifest>"
        return _merge([(where, target)], seen=seen, depth=_depth, named=False, rows=None)
    p = Path(str(target)).expanduser()
    if base is not None and not p.is_absolute():
        p = base / p
    try:
        rp = p.resolve(strict=True)
    except (OSError, RuntimeError):
        return Provenance(None, False, [f"{p.name}: not found"])
    if str(rp) in seen or _depth > MAX_DEPTH:
        return Provenance(None, False, [f"{rp.name}: provenance loop or chain too long"])
    seen.add(str(rp))
    rows: Provenance | None = None
    docs: list[tuple[Path, dict[str, Any]]] = []
    level = rp
    if rp.is_file():
        level = rp.parent
        if rp.suffix == ".json":
            own = _load_doc(rp)
            if own is not None:
                docs.append((rp, own))
        elif rp.suffix == ".jsonl":
            rows = scan_rows(rp)
    if not docs:
        for _ in range(MAX_UP + 1):  # the artifact's own manifest, else its parents'
            docs = _docs_at(level)
            if docs or level.parent == level:
                break
            level = level.parent
    return _merge(docs, seen=seen, depth=_depth, named=path_has_tag(rp), rows=rows)


def _merge(docs: list[tuple[Path, dict[str, Any]]], *, seen: set[str], depth: int, named: bool,
           rows: Provenance | None) -> Provenance:
    """The manifests' verdicts (each with its upstream chain), a data file's
    own rows, and the name, into one :class:`Provenance`."""
    ev: list[str] = []
    docs_md: bool | None = None
    docs_holds = False
    for i, (path, doc) in enumerate(docs):
        v = _doc_verdict(doc, path.name)
        ev += v.evidence
        docs_holds = docs_holds or v.holds
        own = v.max_derived
        for ref, decisive in v.upstream:
            up = provenance(ref, base=path.parent, _seen=seen, _depth=depth + 1)
            if up.max_derived is True:
                own = True
                ev += [f"upstream {Path(ref).name}: {e}" for e in up.evidence[:3]]
            elif decisive:
                own = up.max_derived if own is None else _or(own, up.max_derived)
                ev.append(f"upstream {Path(ref).name}: "
                          + ("unknown" if up.max_derived is None else "not Max-derived"))
        docs_md = own if i == 0 else _or(docs_md, own)
    if rows is not None:
        # A data file answers for its own rows; its directory's manifests can
        # only add a Max-derived verdict (a report saying train/val hold Max
        # rows), never clear rows that show no provenance.
        md = True if docs_md is True else rows.max_derived
        holds = rows.holds_max_rows
        ev = list(rows.evidence) + ev
    else:
        md, holds = docs_md, docs_holds
    if named:
        md = True
        ev.insert(0, f"name carries {TAG}")
    if md is None and not ev:
        ev.append("no manifest or provenance found")
    return Provenance(md, holds, ev)


def is_max_derived(target: Any) -> bool:
    """*target* (a manifest dict or a path) is Max-derived, or nothing shows it
    is not (fail closed): :func:`provenance` is not exactly False."""
    return provenance(target).max_derived is not False


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def _tree_problems(root: Path) -> list[str]:
    """What inside directory *root* blocks publishing it: a tagged name, a
    symbolic link (to a file or a directory), a manifest that is Max-derived
    (or unknown), a JSONL file with a Max-derived row or rows without
    provenance.

    A link is refused whatever it points at: an upload tool or archiver that
    follows links would ship its target (a Max-derived adapter's weights, say),
    which this check never looked at under the directory's own manifest."""
    problems: list[str] = []
    links: set[str] = set()
    n = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        for name in sorted(dirnames) + sorted(filenames):
            n += 1
            if n > MAX_TREE_ENTRIES:
                return problems + [f"more than {MAX_TREE_ENTRIES} entries: not checked"]
            rel = (d / name).relative_to(root)
            if has_tag(name):
                problems.append(f"{rel}: name carries {TAG}")
            if os.path.islink(d / name):
                links.add(str(d / name))
                problems.append(f"{rel}: a symbolic link (refused: an upload that follows it "
                                "ships a target this check does not vouch for; copy the file in "
                                "or check the target itself)")
        for name in sorted(filenames):
            f = d / name
            rel = f.relative_to(root)
            if str(f) in links:
                continue  # refused above; its target is not read
            if f.suffix == ".jsonl":
                v = scan_rows(f)
                if v.max_derived is not False:
                    problems.append(f"{rel}: " + "; ".join(v.evidence))
            elif f.suffix == ".json" and _load_doc(f) is not None:
                v = provenance(f)  # the manifest with its upstream chain
                if v.max_derived is not False or v.holds_max_rows:
                    problems.append(f"{rel}: " + ("; ".join(v.evidence) or "provenance unknown"))
    return problems


def check_publishable(path: Path | str) -> tuple[Provenance, list[str]]:
    """(provenance, reasons it may not be published); no reasons = publishable."""
    p = Path(path).expanduser()
    if not p.exists():
        return Provenance(None, False, [f"{p}: not found"]), [f"{p}: not found"]
    prov = provenance(p)
    reasons: list[str] = []
    if prov.max_derived is True:
        reasons.append("Max-derived: " + "; ".join(prov.evidence))
    elif prov.max_derived is None:
        reasons.append("provenance unknown (refused, fail closed): " + "; ".join(prov.evidence))
    if prov.holds_max_rows and prov.max_derived is not True:
        reasons.append("holds Max-derived rows")
    if p.is_dir():
        reasons += _tree_problems(p.resolve())
    return prov, reasons


def assert_publishable(path: Path | str) -> Provenance:
    """Raise :class:`NotPublishable` unless *path* is shown not to be
    Max-derived and holds no Max-derived rows (a directory: nothing in it
    either). Every publish step calls this on every artifact it ships."""
    prov, reasons = check_publishable(path)
    if reasons:
        raise NotPublishable(path, reasons)
    return prov


def assert_all_publishable(paths: Iterable[Path | str]) -> list[Provenance]:
    return [assert_publishable(p) for p in paths]
