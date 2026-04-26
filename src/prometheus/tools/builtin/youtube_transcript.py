# Source: Original implementation for Prometheus (WEAVE Session A)
# License: MIT
# Purpose: Fetch YouTube transcripts via yt-dlp without downloading the video.

"""YouTube transcript tool — wraps yt-dlp to fetch subtitle text only."""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be")
_FETCH_TIMEOUT = 60


class YouTubeTranscriptInput(BaseModel):
    """Arguments for fetching a YouTube transcript."""

    url: str = Field(description="YouTube URL (full, short, or bare 11-char video ID)")
    language: str = Field(default="en", description="Subtitle language code (default 'en')")
    save_to: str | None = Field(
        default=None,
        description="Optional local path to save the transcript text",
    )
    include_timestamps: bool = Field(
        default=False,
        description="Include [HH:MM:SS] timestamps before each line",
    )


class YouTubeTranscriptTool(BaseTool):
    """Fetch the auto-generated or manual subtitle track from a YouTube video.

    Does not download video content — only the subtitle file. Returns plain
    transcript text. All errors are returned as ToolResult(is_error=True);
    no exceptions propagate.
    """

    name = "youtube_transcript"
    description = (
        "Fetch the transcript/subtitles from a YouTube video by URL or video ID. "
        "Returns the full text content without downloading the video."
    )
    input_model = YouTubeTranscriptInput
    example_call = {"url": "https://youtu.be/dQw4w9WgXcQ"}

    def is_read_only(self, arguments: YouTubeTranscriptInput) -> bool:
        # Saving to disk is the only side-effect — guard it.
        return arguments.save_to is None

    async def execute(
        self,
        arguments: YouTubeTranscriptInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if shutil.which("yt-dlp") is None:
            return ToolResult(
                output=(
                    "Error: yt-dlp not installed — install via "
                    "`pip install yt-dlp` or system package manager."
                ),
                is_error=True,
            )

        url = _normalize_url(arguments.url)
        if url is None:
            return ToolResult(
                output=f"Error: '{arguments.url}' is not a recognizable YouTube URL or video ID.",
                is_error=True,
            )

        with tempfile.TemporaryDirectory(prefix="prometheus-yt-") as tmp:
            tmp_path = Path(tmp)
            output_template = str(tmp_path / "%(id)s.%(ext)s")
            cmd = [
                "yt-dlp",
                "--skip-download",
                "--write-auto-sub",
                "--write-sub",
                "--sub-lang",
                arguments.language,
                "--sub-format",
                "vtt/srv3/best",
                "--no-warnings",
                "--quiet",
                "--output",
                output_template,
                url,
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=_FETCH_TIMEOUT
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    output=f"Error: yt-dlp timed out after {_FETCH_TIMEOUT}s",
                    is_error=True,
                )
            except FileNotFoundError:
                return ToolResult(
                    output="Error: yt-dlp not installed — install via `pip install yt-dlp`",
                    is_error=True,
                )

            if proc.returncode != 0:
                err_text = (stderr.decode("utf-8", errors="replace")
                            + stdout.decode("utf-8", errors="replace")).strip()
                return ToolResult(
                    output=_classify_ytdlp_error(err_text, arguments.language),
                    is_error=True,
                )

            sub_files = sorted(tmp_path.glob(f"*.{arguments.language}.*"))
            if not sub_files:
                # Fall back to whatever language was emitted (auto-generated may differ)
                sub_files = sorted(tmp_path.glob("*.vtt")) + sorted(tmp_path.glob("*.srv3"))
            if not sub_files:
                return ToolResult(
                    output=(
                        f"Error: No transcript available for this video "
                        f"in language '{arguments.language}'."
                    ),
                    is_error=True,
                )

            sub_path = sub_files[0]
            try:
                raw = sub_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return ToolResult(
                    output=f"Error: failed to read subtitle file: {exc}",
                    is_error=True,
                )

            if sub_path.suffix == ".vtt":
                transcript = _parse_vtt(raw, arguments.include_timestamps)
            elif sub_path.suffix == ".srv3":
                transcript = _parse_srv3(raw, arguments.include_timestamps)
            else:
                transcript = raw.strip()

        if not transcript.strip():
            return ToolResult(
                output="Error: subtitle file was empty after parsing.",
                is_error=True,
            )

        saved_note = ""
        if arguments.save_to:
            try:
                dest = _resolve_save_path(context.cwd, arguments.save_to)
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(transcript, encoding="utf-8")
                saved_note = f"\n\n[Saved to {dest}]"
            except OSError as exc:
                saved_note = f"\n\n[Warning: failed to save to {arguments.save_to}: {exc}]"

        return ToolResult(
            output=f"Source: {url}\nLanguage: {arguments.language}\n\n{transcript}{saved_note}",
            metadata={"video_url": url, "language": arguments.language},
        )


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _normalize_url(raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    if _VIDEO_ID_RE.match(raw):
        return f"https://www.youtube.com/watch?v={raw}"
    from urllib.parse import urlparse, parse_qs

    try:
        parsed = urlparse(raw)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if host not in _YT_HOSTS:
        return None
    if host == "youtu.be":
        vid = parsed.path.lstrip("/").split("/")[0]
        return f"https://www.youtube.com/watch?v={vid}" if _VIDEO_ID_RE.match(vid) else None
    if parsed.path.startswith(("/watch", "/embed/", "/shorts/", "/live/")):
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [""])[0]
        else:
            vid = parsed.path.split("/")[2] if len(parsed.path.split("/")) > 2 else ""
        return f"https://www.youtube.com/watch?v={vid}" if _VIDEO_ID_RE.match(vid) else None
    return None


def _resolve_save_path(base: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _classify_ytdlp_error(err_text: str, language: str) -> str:
    lowered = err_text.lower()
    if "private video" in lowered or "members-only" in lowered:
        return "Error: Video is private or members-only."
    if "video unavailable" in lowered or "not available" in lowered:
        return "Error: Video is unavailable (removed or region-blocked)."
    if "no such video" in lowered or "video does not exist" in lowered:
        return "Error: No such video at that URL."
    if "no captions" in lowered or "no subtitles" in lowered:
        return f"Error: No transcript available for this video in '{language}'."
    if "sign in" in lowered or "age" in lowered:
        return "Error: Video requires sign-in (age-restricted)."
    truncated = err_text.strip().splitlines()[-1] if err_text.strip() else "(no detail)"
    return f"Error: yt-dlp failed: {truncated}"


# ---------------------------------------------------------------------------
# Subtitle parsers — stdlib only
# ---------------------------------------------------------------------------

_VTT_TIMING_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}",
)


def _parse_vtt(content: str, include_timestamps: bool) -> str:
    lines = content.splitlines()
    out: list[str] = []
    last: str = ""
    current_ts = ""
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
            continue
        if stripped.isdigit():
            continue
        match = _VTT_TIMING_RE.match(stripped)
        if match:
            current_ts = stripped.split(" ")[0][:8]
            continue
        text = re.sub(r"<[^>]+>", "", stripped).strip()
        if not text or text == last:
            continue
        if include_timestamps and current_ts:
            out.append(f"[{current_ts}] {text}")
        else:
            out.append(text)
        last = text
    return "\n".join(out)


def _parse_srv3(content: str, include_timestamps: bool) -> str:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return ""
    out: list[str] = []
    last: str = ""
    for event in root.findall(".//p"):
        text_parts = [(event.text or "")]
        for child in event:
            text_parts.append(child.text or "")
            text_parts.append(child.tail or "")
        text = " ".join(p for p in text_parts if p).strip()
        text = re.sub(r"\s+", " ", text)
        if not text or text == last:
            continue
        if include_timestamps:
            t_ms = event.get("t")
            if t_ms is not None:
                seconds = int(t_ms) // 1000
                hh = seconds // 3600
                mm = (seconds % 3600) // 60
                ss = seconds % 60
                out.append(f"[{hh:02d}:{mm:02d}:{ss:02d}] {text}")
                last = text
                continue
        out.append(text)
        last = text
    return "\n".join(out)
