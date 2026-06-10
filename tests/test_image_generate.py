"""Tests for ImageGenerateTool — write-boundary guard (audit finding C2).

ImageGenerateTool used to declare ``is_read_only`` → True unconditionally
while writing image bytes to a model-supplied arbitrary absolute path with no
guard. SecurityGate never saw the path (it only inspects ``file_path`` /
``command``), so a prompt — including instructions injected via fetched web
content — could overwrite any user-writable file with image bytes, ungated.

These tests pin the fix: the read-only flag is now honest, and writes are
confined to ``~/.prometheus/`` (+ the agent workspace) at two layers — an
early check in ``execute`` and the ``_save_image_bytes`` sink.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin import image_generate as ig
from prometheus.tools.builtin.image_generate import (
    ImageGenerateInput,
    ImageGenerateTool,
    _save_image_bytes,
)

_PNG = b"\x89PNG\r\n\x1a\nfake-image-bytes"


@pytest.fixture
def confined_roots(tmp_path, monkeypatch):
    """Confine the tool's allowed write roots to an isolated tmp dir."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(ig, "_allowed_image_roots", lambda: [allowed])
    return allowed


# ---------------------------------------------------------------------------
# is_read_only is conditional (audit fix part 1)
# ---------------------------------------------------------------------------


class TestIsReadOnly:
    def test_read_only_when_no_output_path(self) -> None:
        tool = ImageGenerateTool()
        assert tool.is_read_only(ImageGenerateInput(prompt="a cat")) is True

    def test_not_read_only_when_output_path_set(self) -> None:
        # A write to a model-chosen path must NOT be classified read-only —
        # that would skip permission handling and use the parallel dispatch
        # path.
        tool = ImageGenerateTool()
        args = ImageGenerateInput(prompt="a cat", output_path="/tmp/cat.png")
        assert tool.is_read_only(args) is False


# ---------------------------------------------------------------------------
# _save_image_bytes write-boundary guard (audit fix part 2)
# ---------------------------------------------------------------------------


class TestSaveImageBytesGuard:
    def test_output_path_under_allowed_root_writes(self, confined_roots) -> None:
        dest = confined_roots / "sub" / "out.png"
        saved = _save_image_bytes(_PNG, ext=".png", override_path=str(dest))
        assert Path(saved) == dest
        assert dest.read_bytes() == _PNG

    def test_output_path_outside_roots_rejected_and_writes_nothing(
        self, confined_roots, tmp_path
    ) -> None:
        outside = tmp_path / "outside" / "evil.png"
        outside.parent.mkdir()
        with pytest.raises(ValueError):
            _save_image_bytes(_PNG, ext=".png", override_path=str(outside))
        assert not outside.exists()

    def test_traversal_escape_rejected(self, confined_roots) -> None:
        # ../ that escapes the allow-list is rejected even though the literal
        # string starts under an allowed root (resolve-then-check).
        escape = str(confined_roots / ".." / "outside" / "evil.png")
        with pytest.raises(ValueError):
            _save_image_bytes(_PNG, ext=".png", override_path=escape)

    def test_no_output_path_uses_media_cache(self, monkeypatch) -> None:
        calls = {}

        def _fake_cache(data, ext=".jpg"):
            calls["data"] = data
            calls["ext"] = ext
            return "/cached/image.png"

        monkeypatch.setattr(
            "prometheus.gateway.media_cache.cache_image_from_bytes", _fake_cache
        )
        saved = _save_image_bytes(_PNG, ext=".png", override_path=None)
        assert saved == "/cached/image.png"
        assert calls == {"data": _PNG, "ext": ".png"}


# ---------------------------------------------------------------------------
# execute() early gate — rejects before any generation / write
# ---------------------------------------------------------------------------


class TestExecuteEarlyGate:
    async def test_execute_rejects_out_of_bounds_path_without_generating(
        self, confined_roots, tmp_path, monkeypatch
    ) -> None:
        # A pre-existing file outside the roots must be left untouched, and no
        # backend call may happen (the guard fires before generation).
        victim = tmp_path / "home" / ".bashrc"
        victim.parent.mkdir()
        victim.write_text("export ORIGINAL=1\n")

        def _boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("backend was invoked despite an invalid path")

        monkeypatch.setattr(ig, "_resolve_backend", _boom)
        monkeypatch.setattr(ig, "_generate_via_pollinations", _boom)
        monkeypatch.setattr(ig, "_generate_via_comfyui", _boom)

        tool = ImageGenerateTool()
        result = await tool.execute(
            ImageGenerateInput(prompt="overwrite it", output_path=str(victim)),
            ToolExecutionContext(cwd=tmp_path, metadata={}),
        )
        assert result.is_error is True
        assert "allowed image directories" in result.output
        assert victim.read_text() == "export ORIGINAL=1\n"  # untouched
