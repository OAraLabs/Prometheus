"""Tests for Sprint 15 GRAFT: media cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.gateway.media_cache import (
    cache_audio_from_bytes,
    cache_document_from_bytes,
    cache_image_from_bytes,
    cleanup_cache,
    extract_text_from_document,
    sniff_image_extension,
)


class TestMediaCache:

    def test_cache_image(self, tmp_path, monkeypatch):
        monkeypatch.setattr("prometheus.gateway.media_cache.get_config_dir", lambda: tmp_path)
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # JPEG header
        path = cache_image_from_bytes(data, ext=".jpg")
        assert Path(path).exists()
        assert Path(path).read_bytes() == data
        assert "img_" in Path(path).name

    def test_cache_audio(self, tmp_path, monkeypatch):
        monkeypatch.setattr("prometheus.gateway.media_cache.get_config_dir", lambda: tmp_path)
        data = b"OggS" + b"\x00" * 100
        path = cache_audio_from_bytes(data, ext=".ogg")
        assert Path(path).exists()
        assert "audio_" in Path(path).name

    def test_cache_document(self, tmp_path, monkeypatch):
        monkeypatch.setattr("prometheus.gateway.media_cache.get_config_dir", lambda: tmp_path)
        data = b"hello world"
        path = cache_document_from_bytes(data, "readme.txt")
        assert Path(path).exists()
        assert "doc_" in Path(path).name
        assert "readme.txt" in Path(path).name

    def test_extract_text_from_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, World!")
        result = extract_text_from_document(str(f))
        assert result == "Hello, World!"

    def test_extract_text_from_binary(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 binary content")
        result = extract_text_from_document(str(f))
        assert result is None

    def test_sniff_extension(self):
        assert sniff_image_extension("photos/abc.png") == ".png"
        assert sniff_image_extension("photos/abc.webp") == ".webp"
        assert sniff_image_extension(None) == ".jpg"
        assert sniff_image_extension("no_ext") == ".jpg"

    def test_cleanup_cache(self, tmp_path, monkeypatch):
        import time
        monkeypatch.setattr("prometheus.gateway.media_cache.get_config_dir", lambda: tmp_path)
        data = b"test"
        path = cache_image_from_bytes(data, ext=".jpg")
        # Set mtime to 48 hours ago
        old_time = time.time() - 48 * 3600
        import os
        os.utime(path, (old_time, old_time))
        removed = cleanup_cache("images", max_age_hours=24)
        assert removed == 1
        assert not Path(path).exists()
