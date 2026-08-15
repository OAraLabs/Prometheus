"""Tests for GRAFT-VISION-DETECT: vision detection and graceful fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from prometheus.providers.base import ModelProvider
from prometheus.providers.llama_cpp import LlamaCppProvider


# ── Helpers ─────────────────────────────────────────────────────────

def _mock_response(json_data: dict, status_code: int = 200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


# ── ModelProvider ABC ───────────────────────────────────────────────

def test_model_provider_default_vision_false():
    assert ModelProvider.supports_vision is False

def test_llama_provider_default_vision_false():
    p = LlamaCppProvider()
    assert p.supports_vision is False


# ── LlamaCpp detect_vision ──────────────────────────────────────────

async def test_llama_detect_vision_multimodal_true():
    p = LlamaCppProvider()
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               return_value=_mock_response({"multimodal": True})):
        result = await p.detect_vision()
    assert result is True
    assert p.supports_vision is True

async def test_llama_detect_vision_multimodal_false():
    p = LlamaCppProvider()
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               return_value=_mock_response({"multimodal": False})):
        result = await p.detect_vision()
    assert result is False

async def test_llama_detect_vision_modalities_true():
    """Current llama-server builds report capability under `modalities`."""
    p = LlamaCppProvider()
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               return_value=_mock_response(
                   {"modalities": {"vision": True, "audio": False}})):
        result = await p.detect_vision()
    assert result is True
    assert p.supports_vision is True


async def test_llama_detect_vision_modalities_false():
    p = LlamaCppProvider()
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               return_value=_mock_response(
                   {"modalities": {"vision": False, "audio": False}})):
        result = await p.detect_vision()
    assert result is False


async def test_llama_detect_vision_live_payload_shape_without_mmproj():
    """The exact /props shape the 4090 returns today — no `multimodal` key.

    Regression guard for the trap this fixes: reading only the legacy keys
    made detection return False on this shape no matter what was loaded, so
    an mmproj restart would have changed nothing visible.
    """
    p = LlamaCppProvider()
    live = {
        "model_path": "/home/x/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "modalities": {"vision": False, "video": False, "audio": False},
        "default_generation_settings": {"n_ctx": 32768},
        "total_slots": 1,
    }
    assert "multimodal" not in live, "sanity: the live payload has no legacy key"
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               return_value=_mock_response(live)):
        assert await p.detect_vision() is False

    # ...and the SAME shape with a projector loaded must flip to True.
    live_with_mmproj = dict(live)
    live_with_mmproj["modalities"] = {
        "vision": True, "video": False, "audio": False,
    }
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               return_value=_mock_response(live_with_mmproj)):
        assert await p.detect_vision() is True


async def test_llama_detect_vision_nested_legacy():
    p = LlamaCppProvider()
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               return_value=_mock_response({"default_generation_settings": {"multimodal": True}})):
        result = await p.detect_vision()
    assert result is True

async def test_llama_detect_vision_server_down():
    p = LlamaCppProvider()
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               side_effect=httpx.ConnectError("refused")):
        result = await p.detect_vision()
    assert result is False
    assert p.supports_vision is False

async def test_llama_detect_vision_bad_json():
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.side_effect = ValueError("bad json")
    p = LlamaCppProvider()
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
               return_value=resp):
        result = await p.detect_vision()
    assert result is False
