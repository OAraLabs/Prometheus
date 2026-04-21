"""Tests for Sprint 15 GRAFT: WhisperSTT."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from prometheus.tools.builtin.whisper_stt import WhisperSTTTool, WhisperSTTInput, _detect_whisper_engine
from prometheus.tools.base import ToolExecutionContext


class TestWhisperSTT:

    def test_file_not_found(self):
        tool = WhisperSTTTool()
        result = asyncio.run(
            tool.execute(
                WhisperSTTInput(audio_path="/nonexistent/audio.ogg"),
                ToolExecutionContext(cwd=Path.cwd()),
            )
        )
        assert result.is_error
        assert "not found" in result.output.lower()

    def test_no_engine_available(self, tmp_path):
        audio = tmp_path / "test.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 100)

        tool = WhisperSTTTool()
        with patch("prometheus.tools.builtin.whisper_stt._detect_whisper_engine", return_value=None):
            result = asyncio.run(
                tool.execute(
                    WhisperSTTInput(audio_path=str(audio)),
                    ToolExecutionContext(cwd=Path.cwd()),
                )
            )
        assert result.is_error
        assert "no whisper engine" in result.output.lower()

    def test_detect_engine_none(self):
        with patch("shutil.which", return_value=None):
            assert _detect_whisper_engine() is None

    def test_detect_engine_whisper(self):
        def _which(name):
            return "/usr/bin/whisper" if name == "whisper" else None
        with patch("shutil.which", side_effect=_which):
            assert _detect_whisper_engine() == "whisper"

    def test_detect_engine_faster_whisper(self):
        def _which(name):
            return "/usr/bin/faster-whisper-xxl" if name == "faster-whisper-xxl" else None
        with patch("shutil.which", side_effect=_which):
            assert _detect_whisper_engine() == "faster-whisper"

    def test_is_read_only(self):
        tool = WhisperSTTTool()
        assert tool.is_read_only(WhisperSTTInput(audio_path="/x.ogg"))
