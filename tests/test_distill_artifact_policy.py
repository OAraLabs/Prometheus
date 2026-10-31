"""Tests for scripts/distill/artifact_policy.py and publish_guard.py: the Q6b
rule for Max-derived artifacts (names, manifest fields, the provenance chain
and the publish refusal). Synthetic files only; no network.

The dataset -> run -> adapter -> quantized-model chain built by the real
build.py and train_qlora.py is in tests/test_distill_build.py
(test_q6b_tags_flow_end_to_end); these are the rule's own cases.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_DIR = Path(__file__).resolve().parent.parent / "scripts" / "distill"


def _load(name: str, file: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AP = _load("distill_artifact_policy", "artifact_policy.py")
PG = _load("distill_publish_guard", "publish_guard.py")


def write_json(path: Path, doc: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def clean_run(root: Path) -> Path:
    """A run whose manifest and training rows show no Max-derived row."""
    data = write_jsonl(root / "data" / "train.jsonl",
                       [{"id": f"r{i}", "source": "qwen38-27b-gym", "max_derived": False}
                        for i in range(3)])
    run = root / "run"
    write_json(run / "manifest.json", {"schema": AP.TRAIN_MANIFEST_SCHEMA, "max_derived": False,
                                       "tags": [], "data": {"path": str(data),
                                                            "max_derived_rows": 0}})
    (run / "adapter").mkdir()
    (run / "adapter" / "adapter_config.json").write_text("{}")
    return run


# ── names ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name, tagged", [
    ("adapter", "adapter-max-derived"),
    ("train.jsonl", "train-max-derived.jsonl"),
    ("student.Q4_K_M.gguf", "student.Q4_K_M-max-derived.gguf"),
    ("qwen3.5-9b", "qwen3.5-9b-max-derived"),
    ("model.tar.gz", "model-max-derived.tar.gz"),
    ("adapter-max-derived", "adapter-max-derived"),
    ("rows_max_derived.jsonl", "rows_max_derived.jsonl"),
    ("Phase3-MAX-DERIVED", "Phase3-MAX-DERIVED"),
])
def test_tag_name_puts_the_tag_before_known_suffixes(name: str, tagged: str) -> None:
    assert AP.tag_name(name) == tagged
    assert AP.has_tag(AP.tag_name(name)) and AP.tag_name(AP.tag_name(name)) == AP.tag_name(name)


def test_tag_words_and_paths() -> None:
    for name in ("maxderived", "max-derivedness", "nomax-derived", "max", "adapter"):
        assert not AP.has_tag(name), name
    assert AP.path_has_tag(Path("runs/9b-max-derived/adapter"))
    assert not AP.path_has_tag(Path("runs/9b/adapter"))
    assert AP.tag_path(Path("a/b/train.jsonl")) == Path("a/b/train-max-derived.jsonl")
    assert AP.name_for("adapter", False) == "adapter" and AP.name_for("adapter", True) != "adapter"
    for bad in ("", ".", "a/b"):
        with pytest.raises(ValueError):
            AP.tag_name(bad)


def test_policy_fields_and_the_recorded_approval() -> None:
    assert AP.policy_fields(True) == {"max_derived": True, "tags": ["max-derived"],
                                      "private": True, "publish": AP.policy_fields(True)["publish"]}
    assert "refused" in AP.policy_fields(True)["publish"]
    assert AP.policy_fields(False)["tags"] == [] and AP.policy_fields(False)["private"] is False
    rec = AP.approval_record("operator Q6b 2026-10-30")
    assert rec["allowed"] and rec["approval"] == "operator Q6b 2026-10-30"
    assert rec["flags"] == [AP.BUILD_ALLOW_FLAG, AP.BUILD_APPROVAL_FLAG]
    for ref in (None, "", "   "):
        assert AP.approval_record(ref)["allowed"] is False
    assert AP.dataset_approval({"data_policy": {"max_derived_allow": rec}}) == rec["approval"]
    assert AP.dataset_approval({"max_derived_allow": rec}) == rec["approval"]  # a dataset manifest
    for bad in (None, {}, {"data_policy": {}}, {"data_policy": {"max_derived_allow": "yes"}},
                {"data_policy": {"max_derived_allow": {"allowed": "true", "approval": "x"}}},
                {"data_policy": {"max_derived_allow": {"allowed": True, "approval": " "}}},
                {"data_policy": {"max_derived_allow": {"allowed": True, "approval": 7}}}):
        assert AP.dataset_approval(bad) is None, bad


@pytest.mark.parametrize("row, verdict", [
    ({"model": "qwen3.8-max"}, True),
    ({"_meta": {"source": "max-golden"}}, True),
    ({"meta": {"served_model": "Qwen3.8-MAX"}, "max_derived": False}, True),
    ({"max_derived": True}, True),
    ({"max_derived": "false"}, True),
    ({"meta": {"max_derived": None}}, True),
    ({"max_derived": False, "source": "qwen38-27b-gym"}, False),
    ({"source": "qwen38-27b-gym"}, None),
    ({"model": "maximal-7b"}, None),
    ("not a row", None),
])
def test_row_verdicts_fail_closed(row: Any, verdict: bool | None) -> None:
    assert AP.row_verdict(row) is verdict


# ── manifests and the chain ───────────────────────────────────────────────


def test_manifest_dicts_by_schema(tmp_path: Path) -> None:
    P = AP.provenance
    train = {"schema": AP.TRAIN_MANIFEST_SCHEMA, "data": {"max_derived_rows": 0}}
    assert P(train).max_derived is False
    assert P({**train, "data": {"max_derived_rows": 3}}).max_derived is True
    assert P({**train, "max_derived": False, "data": {"max_derived_rows": 3}}).max_derived is True
    assert P({**train, "tags": ["max-derived"]}).max_derived is True
    assert P({**train, "max_derived": None}).max_derived is True  # not a bool: fail closed
    # an older dataset report: its teacher_model is the trainer's key
    old = {"schema": AP.REPORT_SCHEMA, "settings": {"teacher_model": "qwen3.8-max"}}
    assert P(old).max_derived is True
    assert P({**old, "settings": {"teacher_model": "qwen3.8-27b"}}).max_derived is False
    counted = {"schema": AP.REPORT_SCHEMA, "splits": {
        "train": {"max_derived": 0}, "val": {"max_derived": 0}, "eval-tools": {"max_derived": 5}}}
    v = P(counted)
    assert v.max_derived is False and v.holds_max_rows and not v.publishable
    assert P({"schema": AP.PRETOK_SCHEMA}).max_derived is True  # no count, no flag
    assert P({"schema": AP.PRETOK_SCHEMA, "max_derived_rows": 0}).max_derived is False
    q = {"schema": AP.QUANTIZE_MANIFEST_SCHEMA,
         "max_derived": {"calibration_max_derived": 2}}
    assert P(q).max_derived is True  # its calibration used Max-derived rows
    missing = {"schema": AP.QUANTIZE_MANIFEST_SCHEMA, "max_derived": {"calibration_max_derived": 0},
               "inputs": {"adapter": {"dir": str(tmp_path / "gone")}}}
    assert P(missing).max_derived is None  # the adapter decides, and it is not here
    assert AP.is_max_derived(missing) and AP.is_max_derived({"schema": "something-else"})


def test_the_chain_reads_up_from_an_adapter_and_a_quantized_model(tmp_path: Path) -> None:
    run = clean_run(tmp_path / "clean")
    ckpt = run / "checkpoint-000200" / "adapter"
    ckpt.mkdir(parents=True)
    q = tmp_path / "q"
    write_json(q / "quantize-manifest.json", {
        "schema": AP.QUANTIZE_MANIFEST_SCHEMA, "inputs": {"adapter": {"dir": str(run / "adapter")}},
        "max_derived": {"calibration_max_derived": 0}})
    (q / "s.gguf").write_bytes(b"GGUF")
    for p in (run, run / "adapter", ckpt, q, q / "s.gguf"):
        assert AP.provenance(p).max_derived is False, p
        AP.assert_publishable(p)
    # the run's data turns out to hold a Max row: everything downstream follows
    data = Path(json.loads((run / "manifest.json").read_text())["data"]["path"])
    write_jsonl(data, [{"id": "r0", "max_derived": False}, {"id": "r1", "model": "qwen3.8-max"}])
    for p in (run, run / "adapter", ckpt, q, q / "s.gguf"):
        assert AP.provenance(p).max_derived is True, p
        with pytest.raises(AP.NotPublishable, match="Max-derived"):
            AP.assert_publishable(p)


def test_a_tagged_name_is_max_derived_whatever_its_manifest_says(tmp_path: Path) -> None:
    run = clean_run(tmp_path / "c")
    tagged = run.parent / "run-max-derived"
    run.rename(tagged)
    v = AP.provenance(tagged / "adapter")
    assert v.max_derived is True and "name carries" in v.evidence[0]
    # a clean directory holding a tagged entry is refused as a whole
    box = tmp_path / "box"
    (box / "adapter-max-derived").mkdir(parents=True)
    write_json(box / "manifest.json", {"schema": AP.TRAIN_MANIFEST_SCHEMA, "max_derived": False,
                                       "data": {"max_derived_rows": 0}})
    _, reasons = AP.check_publishable(box)
    assert any("adapter-max-derived: name carries" in r for r in reasons)


def test_unknown_provenance_is_refused(tmp_path: Path) -> None:
    lone = tmp_path / "lone" / "model.safetensors"
    lone.parent.mkdir()
    lone.write_bytes(b"\0")
    assert AP.provenance(lone).max_derived is None and AP.is_max_derived(lone)
    with pytest.raises(AP.NotPublishable, match="provenance unknown"):
        AP.assert_publishable(lone)
    with pytest.raises(AP.NotPublishable, match="not found"):
        AP.assert_publishable(tmp_path / "nothing-here")
    rows = write_jsonl(tmp_path / "rows" / "gym.jsonl", [{"id": 1, "source": "qwen38-27b-gym"}])
    assert AP.provenance(rows).max_derived is None  # rows that show no provenance
    # a clean manifest beside it does not clear rows that say nothing
    write_json(rows.parent / "report.json", {"schema": AP.REPORT_SCHEMA, "max_derived": False,
                                             "splits": {"train": {"max_derived": 0},
                                                        "val": {"max_derived": 0}}})
    assert AP.provenance(rows).max_derived is None


def test_a_directory_is_checked_whole(tmp_path: Path) -> None:
    run = clean_run(tmp_path / "c")
    AP.assert_publishable(run)
    write_jsonl(run / "extra" / "notes.jsonl", [{"id": 1, "_meta": {"model": "qwen3.8-max"}}])
    with pytest.raises(AP.NotPublishable, match="notes.jsonl"):
        AP.assert_publishable(run)


def test_a_link_inside_a_published_directory_is_refused(tmp_path: Path) -> None:
    """A clean run's staging dir holding a link to a Max-derived adapter (a
    file link or a directory link): an upload that follows links would ship the
    Max weights under the clean manifest, so any link inside is refused."""
    max_adapter = tmp_path / "runs" / "9b-q6b" / "adapter-max-derived"
    max_adapter.mkdir(parents=True)
    (max_adapter / "adapter_model.safetensors").write_bytes(b"\0")
    (max_adapter / "adapter_config.json").write_text("{}")
    staged = clean_run(tmp_path / "c")
    release = staged / "release"
    release.mkdir()
    AP.assert_publishable(staged)
    (release / "model.safetensors").symlink_to(max_adapter / "adapter_model.safetensors")
    _prov, reasons = AP.check_publishable(staged)
    assert any("release/model.safetensors: a symbolic link" in r for r in reasons), reasons
    (release / "model.safetensors").unlink()
    (release / "adapter").symlink_to(max_adapter, target_is_directory=True)
    _prov, reasons = AP.check_publishable(release)
    assert reasons and "adapter: a symbolic link" in reasons[-1]
    assert PG.main(["check", str(staged)]) == 1
    # a link whose name says nothing and whose target is clean is refused too (fail closed)
    (release / "adapter").unlink()
    (release / "config.json").symlink_to(staged / "adapter" / "adapter_config.json")
    with pytest.raises(AP.NotPublishable, match="symbolic link"):
        AP.assert_publishable(staged)
    (release / "config.json").unlink()
    AP.assert_publishable(staged)
    # the top-level path given as a link still resolves to its target and is judged by it
    link = tmp_path / "published"
    link.symlink_to(max_adapter, target_is_directory=True)
    with pytest.raises(AP.NotPublishable, match="max-derived"):
        AP.assert_publishable(link)


def test_a_provenance_loop_ends(tmp_path: Path) -> None:
    a, b = tmp_path / "a" / "manifest.json", tmp_path / "b" / "manifest.json"
    write_json(a, {"schema": AP.TRAIN_MANIFEST_SCHEMA, "data": {"path": str(b)}})
    write_json(b, {"schema": AP.TRAIN_MANIFEST_SCHEMA, "data": {"path": str(a)}})
    assert AP.provenance(a).max_derived is None  # no count anywhere: unknown, refused
    with pytest.raises(AP.NotPublishable):
        AP.assert_publishable(a.parent)


# ── publish_guard.py ───────────────────────────────────────────────────────


def test_publish_guard_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = clean_run(tmp_path / "c")
    assert PG.main(["check", str(run), str(run / "adapter")]) == 0
    assert capsys.readouterr().out.count("ok: ") == 2
    bad = tmp_path / "d" / "adapter-max-derived"
    bad.mkdir(parents=True)
    assert PG.main(["check", str(run), str(bad)]) == 1
    captured = capsys.readouterr()
    assert "REFUSED" in captured.err and str(bad) in captured.err and "Q6b" in captured.err
    assert PG.main(["check", "--json", str(bad), str(tmp_path / "missing")]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert [r["publishable"] for r in doc["results"]] == [False, False]
    assert doc["results"][0]["provenance"]["max_derived"] is True
    with pytest.raises(SystemExit) as e:
        PG.main(["check"])
    assert e.value.code == 2
