"""Skill-to-skill similarity for SkillCreator's near-duplicate gate.

The encoder is the one the skill-usage audit calibrated on
(docs/audits/SKILL-USAGE.md): BAAI/bge-small-en-v1.5 at a pinned revision,
the repo's own ONNX export on onnxruntime's CPU provider, its tokenizer
truncated to 256 tokens, CLS pooling, L2-normalised — so a cosine here means
what the audit's numbers mean. The threshold, :data:`DEFAULT_THRESHOLD`, is
that calibration's answer (``docs/audits/skill-usage/dedupe_calibration.py``).

Optional by design. The runtime (numpy, onnxruntime, tokenizers — the
``skills`` extra; the ``voice`` extra already brings all three) and the model
files (~134 MB) may be absent, and then :class:`SimilarityChecker` reports
itself unavailable with the reason instead of raising — skill creation must
never depend on an optional download. The files are fetched only by an
explicit operator action::

    python -m prometheus.skills.similarity fetch

which downloads the pinned revision over HTTPS and checks every file's
SHA-256 before keeping it; every later load checks them again.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

log = logging.getLogger(__name__)

#: Cosine at or above which a new skill is a near-duplicate of an existing one.
#: Calibrated on the 57 auto skills the mini ever wrote, each against the
#: catalog that existed when it was written: 0.80 flags the two later copies of
#: a same-named skill, one paraphrase of an existing skill and one restatement
#: of the builtin debugging skill; below it, genuine duplicates (0.76–0.78) sit
#: among clearly distinct skills (0.77–0.78), so a lower bar rejects both.
DEFAULT_THRESHOLD = 0.80

_CHUNK = 1 << 20


class PinError(RuntimeError):
    """A model file is missing, or is not what its pin says."""


@dataclass(frozen=True)
class PinnedFile:
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelPin:
    repo: str
    revision: str
    files: tuple[PinnedFile, ...]
    pooling: str = "cls"
    max_length: int = 256


BGE_SMALL_EN_V15 = ModelPin(
    repo="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    files=(
        PinnedFile("onnx/model.onnx",
                   "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
                   133_093_490),
        PinnedFile("tokenizer.json",
                   "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
                   711_396),
    ),
)


def skill_text(name: str, description: str) -> str:
    """The text a skill is compared by: its name as words, then its description."""
    return f"{name.replace('-', ' ').replace('_', ' ')}: {description}"


def model_dir(pin: ModelPin = BGE_SMALL_EN_V15, root: Path | None = None) -> Path:
    """``<config dir>/models/<repo with __>/<revision>`` unless *root* is given."""
    if root is None:
        from prometheus.config.paths import get_config_dir

        root = get_config_dir() / "models"
    return Path(root) / pin.repo.replace("/", "__") / pin.revision


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(pin: ModelPin, directory: Path) -> None:
    """Check every pinned file's size and SHA-256. Raises :class:`PinError`."""
    for f in pin.files:
        path = Path(directory) / f.path
        if not path.is_file():
            raise PinError(f"{f.path}: missing from {directory}")
        if path.stat().st_size != f.size:
            raise PinError(f"{f.path}: size {path.stat().st_size} != pinned {f.size}")
        if _sha256(path) != f.sha256:
            raise PinError(f"{f.path}: sha256 does not match the pin")


def fetch(
    pin: ModelPin = BGE_SMALL_EN_V15,
    root: Path | None = None,
    *,
    opener: Callable[[str], IO[bytes]] | None = None,
) -> Path:
    """Download whatever is missing from the pinned revision; verify everything.

    A file is kept only after its size and SHA-256 match (written to ``.part``
    first, so a bad download leaves nothing behind).
    """
    target = model_dir(pin, root)
    open_url = opener or (lambda url: urllib.request.urlopen(url, timeout=60))
    for f in pin.files:
        path = target / f.path
        if path.is_file() and path.stat().st_size == f.size and _sha256(path) == f.sha256:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        url = f"https://huggingface.co/{pin.repo}/resolve/{pin.revision}/{f.path}"
        digest, size = hashlib.sha256(), 0
        try:
            with open_url(url) as response, part.open("wb") as out:
                for chunk in iter(lambda: response.read(_CHUNK), b""):
                    size += len(chunk)
                    if size > f.size:
                        raise PinError(f"{f.path}: larger than pinned {f.size} B")
                    digest.update(chunk)
                    out.write(chunk)
            if size != f.size or digest.hexdigest() != f.sha256:
                raise PinError(f"{f.path}: download does not match the pin")
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
    verify(pin, target)
    return target


def _import_runtime() -> tuple[Any, Any, Any]:
    """numpy, onnxruntime, tokenizers.Tokenizer — the optional runtime."""
    import numpy
    import onnxruntime
    from tokenizers import Tokenizer

    return numpy, onnxruntime, Tokenizer


class SimilarityChecker:
    """Nearest existing skill by cosine, or unavailable-with-a-reason.

    Built lazily: nothing is imported, hashed or loaded until the first
    :meth:`nearest`. ``encode`` replaces the model (tests); it takes a list of
    texts and returns L2-normalised row vectors.
    """

    def __init__(
        self,
        directory: Path | None = None,
        *,
        pin: ModelPin = BGE_SMALL_EN_V15,
        encode: Callable[[list[str]], Any] | None = None,
    ) -> None:
        self._pin = pin
        self._directory = Path(directory) if directory is not None else None
        self._encode = encode
        self._reason: str | None = None
        self._loaded = encode is not None

    @property
    def available(self) -> bool:
        self._load()
        return self._encode is not None

    @property
    def unavailable_reason(self) -> str | None:
        self._load()
        return self._reason

    def nearest(self, text: str, catalog: Sequence[tuple[str, str]]) -> tuple[float, str] | None:
        """``(cosine, name)`` of the catalog entry closest to *text*.

        *catalog* is ``[(name, skill_text), …]``. None when the checker is
        unavailable or the catalog is empty.
        """
        if not catalog or not self.available:
            return None
        assert self._encode is not None
        vectors = self._encode([text] + [t for _, t in catalog])
        sims = vectors[1:] @ vectors[0]
        best = int(sims.argmax())
        return float(sims[best]), catalog[best][0]

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        directory = self._directory or model_dir(self._pin)
        try:
            verify(self._pin, directory)
        except PinError as exc:
            self._reason = (f"encoder files not usable ({exc}); fetch them with "
                            "`python -m prometheus.skills.similarity fetch`")
            return
        try:
            np, ort, Tokenizer = _import_runtime()
        except ImportError as exc:
            self._reason = (f"encoder runtime missing ({exc}); install the `skills` extra "
                            "(numpy, onnxruntime, tokenizers)")
            return
        try:
            tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
            tokenizer.enable_truncation(max_length=self._pin.max_length)
            tokenizer.enable_padding()
            session = ort.InferenceSession(str(directory / "onnx" / "model.onnx"),
                                           providers=["CPUExecutionProvider"])
        except Exception as exc:  # noqa: BLE001 — any load failure means "unavailable"
            self._reason = f"encoder failed to load: {exc}"
            return
        inputs = {i.name for i in session.get_inputs()}

        def encode(texts: list[str]) -> Any:
            enc = tokenizer.encode_batch(texts)
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            hidden = session.run(None, {k: v for k, v in feed.items() if k in inputs})[0]
            pooled = hidden[:, 0, :].astype(np.float32)  # CLS pooling
            norms = np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            return pooled / norms

        self._encode = encode


_DEFAULT: SimilarityChecker | None = None


def default_checker() -> SimilarityChecker:
    """The process-wide checker (built once; files are hashed on first use only)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SimilarityChecker()
    return _DEFAULT


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["fetch"]:
        path = fetch()
        print(f"verified {BGE_SMALL_EN_V15.repo}@{BGE_SMALL_EN_V15.revision[:12]} at {path}")
        return 0
    if args[:1] == ["verify"]:
        checker = default_checker()
        print("available" if checker.available else f"unavailable: {checker.unavailable_reason}")
        return 0 if checker.available else 1
    print("usage: python -m prometheus.skills.similarity {fetch|verify}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
