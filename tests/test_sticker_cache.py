"""Tests for Sprint 15 GRAFT: sticker cache."""

from __future__ import annotations

import pytest

from prometheus.gateway.sticker_cache import (
    build_animated_sticker_injection,
    build_sticker_injection,
    cache_sticker_description,
    get_cached_description,
)


class TestStickerCache:

    def test_cache_and_retrieve(self, tmp_path, monkeypatch):
        monkeypatch.setattr("prometheus.gateway.sticker_cache.get_config_dir", lambda: tmp_path)
        cache_sticker_description("unique123", "A happy cat waving", emoji="😺", set_name="CatPack")
        result = get_cached_description("unique123")
        assert result is not None
        assert result["description"] == "A happy cat waving"
        assert result["emoji"] == "😺"
        assert result["set_name"] == "CatPack"
        assert "cached_at" in result

    def test_cache_miss(self, tmp_path, monkeypatch):
        monkeypatch.setattr("prometheus.gateway.sticker_cache.get_config_dir", lambda: tmp_path)
        result = get_cached_description("nonexistent")
        assert result is None

    def test_build_sticker_injection(self):
        text = build_sticker_injection("A cat waving", emoji="😺", set_name="CatPack")
        assert "cat waving" in text
        assert "😺" in text
        assert "CatPack" in text

    def test_build_animated_injection(self):
        text = build_animated_sticker_injection("🎉")
        assert "animated" in text.lower()
        assert "🎉" in text

    def test_build_animated_injection_no_emoji(self):
        text = build_animated_sticker_injection()
        assert "animated" in text.lower()
