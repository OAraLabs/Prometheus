"""SPRINT G2 — reply/behaviour pins for the Telegram media-helper extraction.

G1 pattern (see tests/test_gateway_command_pins.py): these tests hard-code
the EXACT behaviour of Telegram's media helpers BEFORE the G2 refactor
(``_describe_image`` / ``_transcribe_audio`` / ``_truncate_for_context``
moved from telegram.py into the shared ``prometheus.gateway.media_services``
module so the Discord adapter can reuse them). They were written and run
green against the pre-refactor tree, then kept green through the refactor —
proving the extraction preserved Telegram-visible behaviour byte-for-byte.

No real adapters, tokens, or network: tools are monkeypatched fakes and the
adapter is constructed with a fake token; nothing starts polling.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.platform_base import SendResult
from prometheus.gateway.telegram import TelegramAdapter
from prometheus.tools.base import ToolRegistry


def _make_adapter(**kwargs) -> TelegramAdapter:
    config = PlatformConfig(platform=Platform.TELEGRAM, token="test")
    agent_loop = kwargs.pop("agent_loop", None)
    if agent_loop is None:
        agent_loop = AsyncMock()
        agent_loop._model_router = None
    adapter = TelegramAdapter(
        config=config,
        agent_loop=agent_loop,
        tool_registry=kwargs.pop("tool_registry", ToolRegistry()),
        system_prompt=kwargs.pop("system_prompt", "You are Prometheus."),
        model_name="test-model-v1",
        model_provider="llama_cpp",
        prometheus_config=kwargs.pop("prometheus_config", None),
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id=1))
    return adapter


class _FakeToolResult:
    def __init__(self, output: str, is_error: bool = False):
        self.output = output
        self.is_error = is_error


def _fake_tool_cls(recorder: dict, result: _FakeToolResult | Exception):
    """A stand-in for VisionTool / WhisperSTTTool recording execute() args."""

    class _FakeTool:
        def __init__(self):
            pass

        async def execute(self, tool_input, context):
            recorder["input"] = tool_input
            recorder["context"] = context
            if isinstance(result, Exception):
                raise result
            return result

    return _FakeTool


# ---------------------------------------------------------------------------
# _describe_image — vision analysis
# ---------------------------------------------------------------------------


class TestDescribeImagePins:
    @pytest.mark.asyncio
    async def test_describe_image_success_pin(self, monkeypatch):
        import prometheus.tools.builtin.vision as vision_mod

        rec: dict = {}
        monkeypatch.setattr(
            vision_mod, "VisionTool",
            _fake_tool_cls(rec, _FakeToolResult("a red fox on snow")),
        )
        adapter = _make_adapter()
        out = await adapter._describe_image("/tmp/img_abc.jpg")
        assert out == "a red fox on snow"
        # Exact question string + image path forwarded to VisionInput.
        assert rec["input"].image_path == "/tmp/img_abc.jpg"
        assert rec["input"].question == "Describe this image in detail."
        # The agent loop's provider rides along for multimodal routing.
        assert rec["context"].metadata["provider"] is adapter.agent_loop._provider

    @pytest.mark.asyncio
    async def test_describe_image_error_result_pin(self, monkeypatch):
        import prometheus.tools.builtin.vision as vision_mod

        rec: dict = {}
        monkeypatch.setattr(
            vision_mod, "VisionTool",
            _fake_tool_cls(rec, _FakeToolResult("boom", is_error=True)),
        )
        adapter = _make_adapter()
        assert await adapter._describe_image("/tmp/x.png") is None

    @pytest.mark.asyncio
    async def test_describe_image_exception_pin(self, monkeypatch):
        import prometheus.tools.builtin.vision as vision_mod

        rec: dict = {}
        monkeypatch.setattr(
            vision_mod, "VisionTool",
            _fake_tool_cls(rec, RuntimeError("no multimodal model")),
        )
        adapter = _make_adapter()
        assert await adapter._describe_image("/tmp/x.png") is None

    @pytest.mark.asyncio
    async def test_describe_image_empty_output_pin(self, monkeypatch):
        import prometheus.tools.builtin.vision as vision_mod

        rec: dict = {}
        monkeypatch.setattr(
            vision_mod, "VisionTool", _fake_tool_cls(rec, _FakeToolResult("")),
        )
        adapter = _make_adapter()
        # Empty output is falsy — helper returns None, not "".
        assert await adapter._describe_image("/tmp/x.png") is None


# ---------------------------------------------------------------------------
# _transcribe_audio — Whisper STT
# ---------------------------------------------------------------------------


class TestTranscribeAudioPins:
    @pytest.mark.asyncio
    async def test_transcribe_success_pin(self, monkeypatch):
        import prometheus.tools.builtin.whisper_stt as stt_mod

        rec: dict = {}
        monkeypatch.setattr(
            stt_mod, "WhisperSTTTool",
            _fake_tool_cls(rec, _FakeToolResult("hello prometheus")),
        )
        adapter = _make_adapter()
        out = await adapter._transcribe_audio("/tmp/audio_ab.ogg")
        assert out == "hello prometheus"
        assert rec["input"].audio_path == "/tmp/audio_ab.ogg"

    @pytest.mark.asyncio
    async def test_transcribe_failure_pin(self, monkeypatch):
        import prometheus.tools.builtin.whisper_stt as stt_mod

        rec: dict = {}
        monkeypatch.setattr(
            stt_mod, "WhisperSTTTool",
            _fake_tool_cls(rec, RuntimeError("whisper not installed")),
        )
        adapter = _make_adapter()
        assert await adapter._transcribe_audio("/tmp/audio_ab.ogg") is None


# ---------------------------------------------------------------------------
# _truncate_for_context — document budget math
# ---------------------------------------------------------------------------


class TestTruncateForContextPins:
    def test_short_text_untouched_pin(self):
        adapter = _make_adapter(
            prometheus_config={"context": {"effective_limit": 24000}},
        )
        adapter.agent_loop._provider = None
        assert adapter._truncate_for_context("short doc") == "short doc"

    def test_long_text_truncation_pin(self):
        """Pin the exact truncation formula + suffix format."""
        import json

        from prometheus.context.token_estimation import estimate_tokens

        system_prompt = "You are Prometheus."
        registry = ToolRegistry()
        adapter = _make_adapter(
            system_prompt=system_prompt,
            tool_registry=registry,
            prometheus_config={
                "context": {"effective_limit": 3000, "reserved_output": 100},
            },
        )
        # No live provider → server_context_size is absent on AsyncMock?
        # AsyncMock would fabricate one; pin the config-only path explicitly.
        adapter.agent_loop._provider = None

        text = "x" * 40_000  # 10k tokens at 4 chars/token — over budget
        out = adapter._truncate_for_context(text)

        prompt_tokens = estimate_tokens(system_prompt)
        tool_tokens = len(json.dumps(registry.list_schemas())) // 4
        overhead = prompt_tokens + tool_tokens + 100 + 500
        max_doc_tokens = max(2000, 3000 - overhead)
        doc_tokens = estimate_tokens(text)
        expected = text[: max_doc_tokens * 4] + (
            f"\n\n[... truncated to fit context window "
            f"({max_doc_tokens} of {doc_tokens} tokens)]"
        )
        assert out == expected

    def test_server_context_caps_limit_pin(self):
        adapter = _make_adapter(
            prometheus_config={"context": {"effective_limit": 24000}},
        )
        provider = MagicMock()
        provider.server_context_size = 2600
        adapter.agent_loop._provider = provider
        text = "y" * 40_000
        out = adapter._truncate_for_context(text)
        # ctx becomes min(24000, 2600); overhead pushes budget to the
        # 2000-token floor → 8000 chars kept.
        assert out.startswith("y" * 8000 + "\n\n[... truncated")
        assert "(2000 of 10000 tokens)]" in out


# ---------------------------------------------------------------------------
# End-to-end photo flow pin (handler + helper wiring)
# ---------------------------------------------------------------------------


class TestPhotoFlowPin:
    @pytest.mark.asyncio
    async def test_photo_event_text_pin(self, monkeypatch, tmp_path):
        """_handle_photo builds the exact '[Image: …]\\ncaption' user text."""
        import prometheus.gateway.media_cache as mc
        import prometheus.tools.builtin.vision as vision_mod

        rec: dict = {}
        monkeypatch.setattr(
            vision_mod, "VisionTool",
            _fake_tool_cls(rec, _FakeToolResult("a whiteboard diagram")),
        )
        cached = tmp_path / "img_pin.jpg"
        cached.write_bytes(b"jpegbytes")
        monkeypatch.setattr(
            mc, "cache_image_from_bytes", lambda data, ext=".jpg": str(cached),
        )

        adapter = _make_adapter()
        events = []

        async def _capture(event):
            events.append(event)

        adapter.on_message = _capture  # type: ignore[method-assign]

        file_obj = AsyncMock()
        file_obj.file_path = "photos/file_1.jpg"
        file_obj.download_as_bytearray = AsyncMock(return_value=bytearray(b"jpegbytes"))
        photo = MagicMock()
        photo.get_file = AsyncMock(return_value=file_obj)

        update = MagicMock()
        update.effective_chat.id = 123
        update.effective_user.id = 456
        update.effective_user.username = "tester"
        update.message.photo = [photo]
        update.message.caption = "what is this?"
        update.message.message_id = 7

        await adapter._handle_photo(update, MagicMock())

        assert len(events) == 1
        ev = events[0]
        assert ev.text == "[Image: a whiteboard diagram]\nwhat is this?"
        assert ev.media_urls == [str(cached)]
        assert ev.media_types == ["image/jpg"]
        assert ev.caption == "what is this?"
