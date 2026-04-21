"""Tests for Sprint 15 GRAFT: VisionTool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.tools.builtin.vision import VisionTool, VisionInput, _detect_mime
from prometheus.tools.base import ToolExecutionContext, ToolResult


class TestVisionTool:

    def test_detect_mime_jpeg(self):
        assert _detect_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"

    def test_detect_mime_png(self):
        assert _detect_mime(b"\x89PNG\r\n\x1a\n") == "image/png"

    def test_detect_mime_gif(self):
        assert _detect_mime(b"GIF89a") == "image/gif"

    def test_detect_mime_unknown(self):
        assert _detect_mime(b"random") == "image/jpeg"

    def test_file_not_found(self):
        tool = VisionTool()
        result = asyncio.run(
            tool.execute(
                VisionInput(image_path="/nonexistent/image.jpg"),
                ToolExecutionContext(cwd=Path.cwd()),
            )
        )
        assert result.is_error
        assert "not found" in result.output.lower()

    def test_no_provider(self, tmp_path):
        # Create a minimal JPEG file
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        tool = VisionTool()
        result = asyncio.run(
            tool.execute(
                VisionInput(image_path=str(img)),
                ToolExecutionContext(cwd=Path.cwd(), metadata={}),
            )
        )
        assert result.is_error
        assert "provider" in result.output.lower()

    def test_is_read_only(self):
        tool = VisionTool()
        assert tool.is_read_only(VisionInput(image_path="/x.jpg"))
