"""Tests for the WEAVE web tools — youtube_transcript and download_file.

Network-dependent tests are marked @pytest.mark.network so they can be
skipped offline with `pytest -m 'not network'`.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import httpx

from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin.download_file import (
    DownloadFileInput,
    DownloadFileTool,
    _filename_from_url,
    _format_size,
    _is_safe_url,
    _resolve_destination,
)
from prometheus.tools.builtin.web_fetch import (
    WebFetchInput,
    WebFetchTool,
    _html_to_text,
)
from prometheus.tools.builtin.youtube_transcript import (
    YouTubeTranscriptInput,
    YouTubeTranscriptTool,
    _classify_ytdlp_error,
    _normalize_url,
    _parse_srv3,
    _parse_vtt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(cwd: Path | None = None) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=cwd or Path.cwd())


def _has_network() -> bool:
    try:
        socket.gethostbyname("duckduckgo.com")
        return True
    except OSError:
        return False


# ===========================================================================
# YouTubeTranscriptTool — pure-function tests (no network)
# ===========================================================================


class TestYouTubeUrlNormalization:
    def test_full_watch_url(self):
        assert _normalize_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == \
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_short_youtu_be_url(self):
        assert _normalize_url("https://youtu.be/dQw4w9WgXcQ") == \
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_short_youtu_be_with_query(self):
        # youtu.be?t=10 — only the path matters
        assert _normalize_url("https://youtu.be/dQw4w9WgXcQ?t=42") == \
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_bare_video_id(self):
        assert _normalize_url("dQw4w9WgXcQ") == \
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_embed_url(self):
        assert _normalize_url("https://www.youtube.com/embed/dQw4w9WgXcQ") == \
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_shorts_url(self):
        assert _normalize_url("https://youtube.com/shorts/dQw4w9WgXcQ") == \
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_non_youtube_url_rejected(self):
        assert _normalize_url("https://vimeo.com/123") is None

    def test_garbage_input(self):
        assert _normalize_url("") is None
        assert _normalize_url("not a url") is None

    def test_invalid_video_id_length(self):
        # Too short
        assert _normalize_url("abc") is None
        # Wrong character set
        assert _normalize_url("!!!!!!!!!!!") is None


class TestVttParser:
    def test_strips_timing_lines(self):
        vtt = (
            "WEBVTT\n"
            "Kind: captions\n"
            "Language: en\n\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Hello world\n\n"
            "00:00:04.000 --> 00:00:06.000\n"
            "Second line\n"
        )
        out = _parse_vtt(vtt, include_timestamps=False)
        assert "Hello world" in out
        assert "Second line" in out
        assert "-->" not in out
        assert "WEBVTT" not in out

    def test_dedupes_consecutive_identical_lines(self):
        vtt = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Hello world\n\n"
            "00:00:03.000 --> 00:00:05.000\n"
            "Hello world\n\n"
            "00:00:05.000 --> 00:00:07.000\n"
            "Different line\n"
        )
        out = _parse_vtt(vtt, include_timestamps=False)
        assert out.count("Hello world") == 1
        assert "Different line" in out

    def test_includes_timestamps_when_requested(self):
        vtt = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:03.000\n"
            "Hello world\n"
        )
        out = _parse_vtt(vtt, include_timestamps=True)
        assert "[00:00:01]" in out


class TestSrv3Parser:
    def test_parses_simple_srv3(self):
        srv3 = (
            '<?xml version="1.0"?>'
            '<timedtext>'
            '<body>'
            '<p t="1000" d="2000">First line</p>'
            '<p t="3000" d="2000">Second line</p>'
            '</body>'
            '</timedtext>'
        )
        out = _parse_srv3(srv3, include_timestamps=False)
        assert "First line" in out
        assert "Second line" in out

    def test_srv3_with_timestamps(self):
        srv3 = (
            '<?xml version="1.0"?>'
            '<timedtext><body>'
            '<p t="3700000" d="1000">Hello</p>'
            '</body></timedtext>'
        )
        out = _parse_srv3(srv3, include_timestamps=True)
        assert "[01:01:40]" in out
        assert "Hello" in out

    def test_malformed_srv3_returns_empty(self):
        assert _parse_srv3("not xml at all", include_timestamps=False) == ""


class TestErrorClassification:
    def test_private_video(self):
        msg = _classify_ytdlp_error("ERROR: Private video. Sign in if you've been granted access", "en")
        assert "private" in msg.lower()

    def test_no_captions(self):
        msg = _classify_ytdlp_error("WARNING: No captions found for this video", "en")
        assert "transcript" in msg.lower()

    def test_unavailable(self):
        msg = _classify_ytdlp_error("ERROR: Video unavailable", "en")
        assert "unavailable" in msg.lower()

    def test_unknown_error_passthrough(self):
        msg = _classify_ytdlp_error("ERROR: Some weird new failure mode", "en")
        assert "yt-dlp failed" in msg


class TestYouTubeToolBehavior:
    @pytest.mark.asyncio
    async def test_invalid_url_returns_error(self):
        tool = YouTubeTranscriptTool()
        result = await tool.execute(
            YouTubeTranscriptInput(url="https://example.com/not-youtube"),
            _ctx(),
        )
        assert result.is_error
        assert "not a recognizable YouTube" in result.output

    @pytest.mark.asyncio
    async def test_missing_ytdlp_handled(self):
        tool = YouTubeTranscriptTool()
        with patch("prometheus.tools.builtin.youtube_transcript.shutil.which", return_value=None):
            result = await tool.execute(
                YouTubeTranscriptInput(url="https://youtu.be/dQw4w9WgXcQ"),
                _ctx(),
            )
        assert result.is_error
        assert "yt-dlp not installed" in result.output

    def test_is_read_only_unless_save_to_set(self):
        tool = YouTubeTranscriptTool()
        assert tool.is_read_only(YouTubeTranscriptInput(url="dQw4w9WgXcQ")) is True
        assert tool.is_read_only(
            YouTubeTranscriptInput(url="dQw4w9WgXcQ", save_to="/tmp/x.txt")
        ) is False

    def test_tool_registers_with_correct_metadata(self):
        tool = YouTubeTranscriptTool()
        assert tool.name == "youtube_transcript"
        assert tool.input_model is YouTubeTranscriptInput
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "youtube_transcript"


# ===========================================================================
# WebFetchTool — HTML extractor regression tests
# ===========================================================================


class TestWebFetchHtmlExtractor:
    def test_inline_tags_no_space_insertion(self):
        """Inline tags (code, span, a, em, ...) must NOT insert separators
        when stripped, or dotted names like ``asyncio.gather`` get split into
        ``asyncio . gather`` which the model can't keyword-match.
        """
        html = (
            "<html><body><p>"
            "Use <code>asyncio.gather</code> to run coroutines."
            "</p><p>"
            "Or write <span>asyncio</span><span>.</span><span>gather</span>"
            "(...) explicitly."
            "</p><p>"
            "See the <a href='/x'>asyncio.gather</a> reference."
            "</p></body></html>"
        )
        out = _html_to_text(html)
        # Three different inline-tag arrangements all produce the glued name.
        assert out.count("asyncio.gather") == 3
        # The buggy regex stripper would have produced "asyncio . gather".
        assert "asyncio . gather" not in out

    def test_article_tag_preferred_over_body(self):
        """When <article> exists its content wins; surrounding nav/footer
        in the body is dropped even though it would be visible to a naive
        full-body strip.
        """
        html = (
            "<html><body>"
            "<nav>Site nav: Home About Contact</nav>"
            "<header>Banner: SUBSCRIBE NOW</header>"
            "<article>"
            "<h1>Real article title</h1>"
            "<p>The actual article body explains how asyncio.gather works "
            "by scheduling coroutines concurrently.</p>"
            "</article>"
            "<footer>Copyright 2026 — all rights reserved boilerplate</footer>"
            "</body></html>"
        )
        out = _html_to_text(html)
        assert "Real article title" in out
        assert "actual article body" in out
        assert "asyncio.gather" in out
        # Nav/header/footer must be excluded.
        assert "Site nav" not in out
        assert "SUBSCRIBE NOW" not in out
        assert "Copyright 2026" not in out

    def test_div_role_main_keeps_capturing_through_inner_divs(self):
        """A target-tag match must track which open tag actually matched
        rather than decrementing on every same-tag close. Pages like the
        Python docs wrap <div role="main"> around hundreds of inner
        non-matching <div>s; before this fix the first inner </div> would
        drop us out of capture mode and we'd return ~500 chars of header
        instead of the actual content.
        """
        # Build a long page with a single <div role="main"> that wraps
        # many inner <div>s, plus a misleading inner <div role="main"> a
        # naive matcher could double-count.
        inner = "".join(
            f"<div class=section><p>Section {i} mentions asyncio.gather.</p></div>"
            for i in range(20)
        )
        html = (
            "<html><body>"
            "<nav>nav junk</nav>"
            f'<div class="body" role="main">{inner}</div>'
            "<footer>footer junk</footer>"
            "</body></html>"
        )
        out = _html_to_text(html)
        # All 20 inner sections should be captured — not just one.
        assert out.count("asyncio.gather") == 20, (
            f"Expected 20 'asyncio.gather' hits, got "
            f"{out.count('asyncio.gather')} — early exit from capture mode?"
        )
        # And the dotted name must stay glued.
        assert "asyncio . gather" not in out
        # Nav/footer remain excluded.
        assert "nav junk" not in out
        assert "footer junk" not in out


# ===========================================================================
# DownloadFileTool — pure-function tests
# ===========================================================================


class TestFilenameFromUrl:
    def test_simple_filename(self):
        assert _filename_from_url("https://example.com/foo.pdf") == "foo.pdf"

    def test_filename_with_query_string(self):
        assert _filename_from_url("https://example.com/foo.pdf?token=abc") == "foo.pdf"

    def test_url_decoded_filename(self):
        assert _filename_from_url("https://example.com/hello%20world.pdf") == "hello world.pdf"

    def test_no_filename_falls_back_to_host(self):
        name = _filename_from_url("https://example.com/")
        assert "example" in name


class TestPathTraversalGuard:
    def test_etc_path_blocked(self):
        with pytest.raises(ValueError):
            _resolve_destination("https://x.com/x", "/etc/passwd")

    def test_sys_path_blocked(self):
        with pytest.raises(ValueError):
            _resolve_destination("https://x.com/x", "/sys/test")

    def test_traversal_resolves_first(self):
        # Even if the literal string starts safely, ../ should be resolved
        # before the protected-path check.
        with pytest.raises(ValueError):
            _resolve_destination(
                "https://x.com/x",
                "/tmp/../etc/foo",
            )

    def test_default_destination_uses_downloads_dir(self):
        path = _resolve_destination("https://example.com/test.bin", None)
        assert ".prometheus/downloads" in str(path)
        assert path.name == "test.bin"

    def test_relative_path_resolves_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = _resolve_destination("https://x.com/x", "subdir/file.txt")
        assert path == (tmp_path / "subdir" / "file.txt").resolve()


class TestSafeUrl:
    def test_blocks_localhost(self):
        assert _is_safe_url("http://localhost/x") is False

    def test_blocks_127_0_0_1(self):
        assert _is_safe_url("http://127.0.0.1/x") is False

    def test_blocks_private_192(self):
        assert _is_safe_url("http://192.168.1.1/x") is False

    def test_blocks_private_10(self):
        assert _is_safe_url("http://10.0.0.1/x") is False

    def test_blocks_no_hostname(self):
        assert _is_safe_url("not a url") is False


class TestFormatSize:
    def test_bytes(self):
        assert _format_size(500) == "500B"

    def test_kilobytes(self):
        assert _format_size(2048) == "2.0KB"

    def test_megabytes(self):
        assert _format_size(5 * 1024 * 1024) == "5.0MB"


class TestDownloadToolBehavior:
    @pytest.mark.asyncio
    async def test_overwrite_blocked_when_file_exists(self, tmp_path):
        existing = tmp_path / "exists.bin"
        existing.write_text("hello")
        tool = DownloadFileTool()
        result = await tool.execute(
            DownloadFileInput(
                url="https://example.com/x",
                destination=str(existing),
                overwrite=False,
            ),
            _ctx(tmp_path),
        )
        assert result.is_error
        assert "already exists" in result.output

    @pytest.mark.asyncio
    async def test_blocked_url_fails_safely(self, tmp_path):
        tool = DownloadFileTool()
        result = await tool.execute(
            DownloadFileInput(
                url="http://127.0.0.1/secret",
                destination=str(tmp_path / "out.bin"),
            ),
            _ctx(tmp_path),
        )
        assert result.is_error
        assert "Blocked" in result.output

    @pytest.mark.asyncio
    async def test_protected_path_rejected(self, tmp_path):
        tool = DownloadFileTool()
        result = await tool.execute(
            DownloadFileInput(
                url="https://example.com/x.bin",
                destination="/etc/owned.bin",
            ),
            _ctx(tmp_path),
        )
        assert result.is_error
        assert "protected path" in result.output.lower()

    def test_is_not_read_only(self):
        tool = DownloadFileTool()
        assert tool.is_read_only(DownloadFileInput(url="https://example.com/x")) is False

    def test_tool_registers_with_correct_metadata(self):
        tool = DownloadFileTool()
        assert tool.name == "download_file"
        assert tool.input_model is DownloadFileInput
        schema = tool.to_openai_schema()
        assert schema["function"]["name"] == "download_file"

    def test_max_mb_constructor_arg(self):
        tool = DownloadFileTool(max_mb=5)
        assert tool._max_mb == 5
        assert tool._max_bytes == 5 * 1024 * 1024


# ===========================================================================
# Network-dependent tests
# ===========================================================================


@pytest.mark.network
class TestDownloadFileNetwork:
    @pytest.mark.asyncio
    async def test_downloads_small_file(self, tmp_path):
        if not _has_network():
            pytest.skip("no network")
        tool = DownloadFileTool()
        dest = tmp_path / "robots.txt"
        result = await tool.execute(
            DownloadFileInput(
                url="https://www.python.org/robots.txt",
                destination=str(dest),
            ),
            _ctx(tmp_path),
        )
        assert not result.is_error, result.output
        assert dest.exists()
        assert dest.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_404_returns_error(self, tmp_path):
        if not _has_network():
            pytest.skip("no network")
        tool = DownloadFileTool()
        result = await tool.execute(
            DownloadFileInput(
                url="https://www.python.org/this-does-not-exist-xyz",
                destination=str(tmp_path / "x.bin"),
            ),
            _ctx(tmp_path),
        )
        assert result.is_error
        assert "HTTP" in result.output or "fetching" in result.output

    @pytest.mark.asyncio
    async def test_size_limit_aborts(self, tmp_path):
        """Use a tiny max_mb to force the limit guard to trip."""
        if not _has_network():
            pytest.skip("no network")
        tool = DownloadFileTool(max_mb=0)  # 0MB => any download fails
        # Need the bytes-written check (not Content-Length) to trip in case
        # server doesn't return Content-Length. _max_bytes=0 makes any
        # written byte trip immediately.
        result = await tool.execute(
            DownloadFileInput(
                url="https://www.python.org/robots.txt",
                destination=str(tmp_path / "should_not_exist.bin"),
            ),
            _ctx(tmp_path),
        )
        assert result.is_error


@pytest.mark.network
class TestYouTubeTranscriptNetwork:
    @pytest.mark.asyncio
    async def test_invalid_video_id_returns_error(self, tmp_path):
        if not _has_network():
            pytest.skip("no network")
        if shutil.which("yt-dlp") is None:
            pytest.skip("yt-dlp not installed")
        tool = YouTubeTranscriptTool()
        result = await tool.execute(
            YouTubeTranscriptInput(url="https://www.youtube.com/watch?v=ZZZZZZZZZZZ"),
            _ctx(tmp_path),
        )
        assert result.is_error
