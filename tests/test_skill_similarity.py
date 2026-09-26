"""prometheus.skills.similarity — the near-duplicate encoder SkillCreator's gate uses.

It pins the encoder the skill-usage audit calibrated on (BAAI/bge-small-en-v1.5
at a fixed revision, its own ONNX export, CLS pooling, L2-normalised), checks
every file's SHA-256 before use, and degrades to "unavailable, and here is
why" when the optional runtime (onnxruntime, tokenizers, numpy) or the model
files are missing — never an exception into the caller.
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import pytest

from prometheus.skills import similarity as sim


def test_the_pin_is_the_audits_encoder():
    pin = sim.BGE_SMALL_EN_V15
    assert pin.repo == "BAAI/bge-small-en-v1.5"
    assert pin.revision == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    assert {f.path: f.sha256 for f in pin.files} == {
        "onnx/model.onnx": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
        "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    }
    assert (pin.pooling, pin.max_length) == ("cls", 256)


def test_skill_text_is_the_audits_formula():
    assert sim.skill_text("release-check_v2", "Check a release") == "release check v2: Check a release"


def test_model_dir_is_under_the_config_dir(tmp_path):
    d = sim.model_dir(root=tmp_path)
    assert d == tmp_path / "BAAI__bge-small-en-v1.5" / sim.BGE_SMALL_EN_V15.revision


_FAKE_PIN = sim.ModelPin(
    repo="example/tiny", revision="0" * 40,
    files=(sim.PinnedFile("a.bin", hashlib.sha256(b"abc").hexdigest(), 3),),
)


class TestUnavailableSaysWhy:
    def test_missing_files(self, tmp_path):
        checker = sim.SimilarityChecker(tmp_path / "nowhere", pin=_FAKE_PIN)
        assert checker.available is False
        assert "missing" in checker.unavailable_reason
        assert checker.nearest("x", [("a", "y")]) is None

    def test_a_wrong_hash_is_refused(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"abd")
        checker = sim.SimilarityChecker(tmp_path, pin=_FAKE_PIN)
        assert checker.available is False
        assert "sha256" in checker.unavailable_reason

    def test_a_missing_runtime(self, tmp_path, monkeypatch):
        (tmp_path / "a.bin").write_bytes(b"abc")

        def no_runtime():
            raise ImportError("No module named 'onnxruntime'")

        monkeypatch.setattr(sim, "_import_runtime", no_runtime)
        checker = sim.SimilarityChecker(tmp_path, pin=_FAKE_PIN)
        assert checker.available is False
        assert "onnxruntime" in checker.unavailable_reason


def _fake_encode(texts):
    import numpy as np

    table = {"new": [1.0, 0.0], "near": [0.9, math.sqrt(1 - 0.81)], "far": [0.0, 1.0]}
    return np.array([table[t] for t in texts], dtype=np.float32)


class TestNearest:
    def test_returns_the_most_similar_entry_and_its_cosine(self):
        pytest.importorskip("numpy")
        checker = sim.SimilarityChecker(encode=_fake_encode)
        score, name = checker.nearest("new", [("b", "far"), ("a", "near")])
        assert name == "a"
        assert score == pytest.approx(0.9, abs=1e-6)

    def test_an_empty_catalog_has_no_nearest(self):
        pytest.importorskip("numpy")
        assert sim.SimilarityChecker(encode=_fake_encode).nearest("new", []) is None


class TestFetch:
    def test_downloads_and_verifies(self, tmp_path):
        urls: list[str] = []

        def opener(url):
            urls.append(url)
            return io.BytesIO(b"abc")

        target = sim.fetch(_FAKE_PIN, root=tmp_path, opener=opener)
        assert (target / "a.bin").read_bytes() == b"abc"
        assert urls == [f"https://huggingface.co/example/tiny/resolve/{'0' * 40}/a.bin"]
        sim.verify(_FAKE_PIN, target)

    def test_a_bad_download_is_refused_and_leaves_nothing(self, tmp_path):
        with pytest.raises(sim.PinError):
            sim.fetch(_FAKE_PIN, root=tmp_path, opener=lambda url: io.BytesIO(b"xyz"))
        assert not any(p.is_file() for p in tmp_path.rglob("*"))

    def test_a_verified_file_is_not_downloaded_again(self, tmp_path):
        sim.fetch(_FAKE_PIN, root=tmp_path, opener=lambda url: io.BytesIO(b"abc"))

        def must_not_download(url):
            raise AssertionError("fetched a file that was already verified")

        sim.fetch(_FAKE_PIN, root=tmp_path, opener=must_not_download)


def test_default_checker_is_built_once(monkeypatch, tmp_path):
    built: list[Path] = []

    class _Stub:
        def __init__(self, directory=None, **kw):
            built.append(directory)
            self.available = False
            self.unavailable_reason = "stub"

    monkeypatch.setattr(sim, "SimilarityChecker", _Stub)
    monkeypatch.setattr(sim, "_DEFAULT", None)
    a = sim.default_checker()
    b = sim.default_checker()
    assert a is b
    assert len(built) == 1
