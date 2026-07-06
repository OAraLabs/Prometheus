# Provenance: Novel code for Prometheus
# License: MIT
# CLOUD EXPANSION (2026-07): Kling 3.0 video generation tool, modeled on
# image_generate's backend pattern (async submit → poll → download bytes to
# the media cache).

"""Generate videos via the Kling 3.0 API (text-to-video / image-to-video).

PAID, dormant-until-keyed: the tool ships fakes-tested — no Kling keys
existed on this box when it landed. Endpoint/auth/request shapes come from
the 2026-07-05 research pass and are pinned by tests/test_video_generate.py;
verify at first live use.

Auth: Kling uses an AccessKey + SecretKey pair. Each HTTP request carries a
self-signed JWT (HS256; payload ``iss``=AccessKey, ``exp``=now+1800,
``nbf``=now-5) as a Bearer token, re-minted per request. The JWT is built
with the STDLIB ONLY (hmac + hashlib + base64 + json) — deliberately no
PyJWT dependency; the encoding is pinned by a known-answer test.

Flow: POST /v1/videos/text2video (or /v1/videos/image2video when an
``image_path`` is given) → task_id → poll GET /v1/videos/<kind>/{task_id}
every ~10s until succeed/failed. Videos take MINUTES to render, so the
poll budget is generous (config ``video_generation.kling.poll_budget_seconds``,
default 600). The finished video is downloaded immediately to
``~/.prometheus/cache/videos/`` (result URLs are not assumed durable —
same download-now rule as the DashScope image backend).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults (config-overridable via video_generation.kling)
# ---------------------------------------------------------------------------

_KLING_DEFAULT_BASE = "https://api-singapore.klingai.com"
_KLING_DEFAULT_MODEL = "kling-v3"
_KLING_ACCESS_KEY_ENV = "KLING_ACCESS_KEY"
_KLING_SECRET_KEY_ENV = "KLING_SECRET_KEY"
_KLING_POLL_INTERVAL = 10.0
_KLING_POLL_BUDGET = 600.0
_KLING_HTTP_TIMEOUT = 60.0
_KLING_JWT_TTL = 1800          # exp = now + 1800 (Kling-documented value)
_KLING_JWT_NBF_SKEW = 5        # nbf = now - 5

# Kling terminal task states (normalized lowercase).
_KLING_TERMINAL_OK = ("succeed", "succeeded")
_KLING_TERMINAL_FAIL = ("failed",)


# ---------------------------------------------------------------------------
# Stdlib JWT (HS256) — no PyJWT, no new dependencies
# ---------------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    """base64url WITHOUT padding — the JWT wire encoding (RFC 7515 §2)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def mint_kling_jwt(access_key: str, secret_key: str, now: int | None = None) -> str:
    """Self-sign the short-lived HS256 JWT Kling expects as a Bearer token.

    Payload per Kling's API docs: ``iss`` = AccessKey, ``exp`` = now+1800,
    ``nbf`` = now-5. Re-minted per request (tokens are cheap; clock drift
    handling stays trivial). JSON is serialized with compact separators and
    insertion-ordered keys — the exact byte layout is pinned by the
    known-answer test in tests/test_video_generate.py.
    """
    if now is None:
        now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": access_key,
        "exp": now + _KLING_JWT_TTL,
        "nbf": now - _KLING_JWT_NBF_SKEW,
    }
    segments = [
        _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8")),
        _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
    ]
    signing_input = ".".join(segments).encode("ascii")
    signature = hmac.new(
        secret_key.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    segments.append(_b64url(signature))
    return ".".join(segments)


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------

class VideoGenerateInput(BaseModel):
    """Arguments for video generation."""

    prompt: str = Field(
        description=(
            "Text description of the video to generate. Be specific about "
            "subject, motion, camera movement, style. Example: 'A paper "
            "boat drifting down a rainy gutter stream, low tracking shot, "
            "cinematic'."
        ),
        min_length=1,
        max_length=2500,
    )
    image_path: str | None = Field(
        default=None,
        description=(
            "Optional path to a local image. When set, runs image-to-video "
            "(the image is the first frame) instead of text-to-video."
        ),
    )
    duration: Literal[5, 10] = Field(
        default=5,
        description="Video length in seconds: 5 or 10 (Kling 3.0 supports up to 15; 5/10 are the stable options).",
    )
    model_name: str = Field(
        default="",
        description=(
            "Kling model name. Empty uses the configured default "
            "(video_generation.kling.model_name, ships as 'kling-v3')."
        ),
    )
    resolution: Literal["720p", "1080p"] = Field(
        default="1080p",
        description=(
            "Output resolution. Maps to Kling's mode parameter "
            "(720p=std, 1080p=pro)."
        ),
    )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class KlingVideoTool(BaseTool):
    """Generate a video from a text prompt (or image) via Kling 3.0."""

    name = "video_generate"
    description = (
        "Generate a video from a text prompt via the Kling 3.0 API (PAID — "
        "requires KLING_ACCESS_KEY + KLING_SECRET_KEY). Pass image_path for "
        "image-to-video (the image becomes the first frame). Durations 5 or "
        "10 seconds. Generation takes MINUTES — the tool waits and polls. "
        "Returns the path to the saved .mp4 in ~/.prometheus/cache/videos/."
    )
    input_model = VideoGenerateInput

    def is_read_only(self, arguments: VideoGenerateInput) -> bool:
        # Writes only to the fixed media cache (never a model-chosen path),
        # matching image_generate's output_path-is-None case. There is no
        # output_path override on this tool, so it is always cache-confined.
        del arguments
        return True

    async def execute(
        self,
        arguments: VideoGenerateInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        cfg = _video_config(context)
        kling = cfg.get("kling", {}) or {}

        access_key, secret_key = _kling_keys(kling)
        if not (access_key and secret_key):
            ak_env = kling.get("access_key_env") or _KLING_ACCESS_KEY_ENV
            sk_env = kling.get("secret_key_env") or _KLING_SECRET_KEY_ENV
            return ToolResult(
                output=(
                    f"video_generate: Kling API keys not configured. This is "
                    f"a PAID service — get an AccessKey + SecretKey pair from "
                    f"the Kling AI console (https://app.klingai.com → API "
                    f"access), add {ak_env}=<access-key> and "
                    f"{sk_env}=<secret-key> to ~/.config/prometheus/env, and "
                    f"restart the daemon."
                ),
                is_error=True,
            )

        base = (kling.get("base_url") or _KLING_DEFAULT_BASE).rstrip("/")
        model_name = (
            arguments.model_name or kling.get("model_name") or _KLING_DEFAULT_MODEL
        )
        poll_budget = float(
            kling.get("poll_budget_seconds", _KLING_POLL_BUDGET)
        )

        # t2v vs i2v — presence of image_path picks the endpoint.
        payload: dict[str, Any] = {
            "model_name": model_name,
            "prompt": arguments.prompt,
            "duration": str(arguments.duration),
            # Kling "mode": std|pro. 720p→std, 1080p→pro — 2026-07 research
            # mapping, verify at first live use.
            "mode": "pro" if arguments.resolution == "1080p" else "std",
        }
        if arguments.image_path is not None:
            image_file = Path(arguments.image_path).expanduser()
            if not image_file.is_file():
                return ToolResult(
                    output=(
                        f"video_generate: image_path {arguments.image_path!r} "
                        f"does not exist or is not a file."
                    ),
                    is_error=True,
                )
            payload["image"] = base64.b64encode(image_file.read_bytes()).decode("ascii")
            kind = "image2video"
        else:
            kind = "text2video"

        logger.info(
            "video_generate: kind=%s model=%s duration=%ss resolution=%s prompt_len=%d",
            kind, model_name, arguments.duration, arguments.resolution,
            len(arguments.prompt),
        )

        try:
            async with httpx.AsyncClient(timeout=_KLING_HTTP_TIMEOUT) as client:
                # Submit — JWT minted fresh for this request.
                r = await client.post(
                    f"{base}/v1/videos/{kind}",
                    json=payload,
                    headers=_kling_headers(access_key, secret_key),
                )
                r.raise_for_status()
                body = r.json()
                if body.get("code") not in (0, "0", None):
                    return ToolResult(
                        output=(
                            f"video_generate: Kling rejected the task — "
                            f"code={body.get('code')} "
                            f"message={body.get('message') or '(none)'}"
                        ),
                        is_error=True,
                    )
                task_id = (body.get("data") or {}).get("task_id")
                if not task_id:
                    return ToolResult(
                        output=(
                            f"video_generate: submit returned no task_id. "
                            f"Body: {r.text[:400]}"
                        ),
                        is_error=True,
                    )

                outcome = await _poll_kling_task(
                    client, base, kind, task_id,
                    access_key=access_key, secret_key=secret_key,
                    budget=poll_budget,
                )
                if outcome is None:
                    return ToolResult(
                        output=(
                            f"video_generate: task {task_id} did not finish "
                            f"within {poll_budget:.0f}s. Kling renders take "
                            f"minutes and the task may still complete "
                            f"server-side — retry later or raise "
                            f"video_generation.kling.poll_budget_seconds."
                        ),
                        is_error=True,
                    )
                status, data = outcome
                if status not in _KLING_TERMINAL_OK:
                    detail = data.get("task_status_msg") or "(no detail)"
                    return ToolResult(
                        output=(
                            f"video_generate: task {task_id} ended "
                            f"{status} — {detail}"
                        ),
                        is_error=True,
                    )

                video_url, video_duration = _extract_kling_video(data)
                if not video_url:
                    return ToolResult(
                        output=(
                            f"video_generate: task {task_id} succeeded but "
                            f"no video URL in task_result."
                        ),
                        is_error=True,
                    )

                # Download NOW — result URLs are not assumed durable.
                v = await client.get(video_url)
                v.raise_for_status()
                video_bytes = v.content
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                output=(
                    f"video_generate: Kling HTTP {exc.response.status_code} — "
                    f"{exc.response.text[:400] if exc.response.text else '(empty)'}"
                ),
                is_error=True,
            )
        except httpx.HTTPError as exc:
            return ToolResult(
                output=f"video_generate: network error — {exc}",
                is_error=True,
            )

        if not video_bytes:
            return ToolResult(
                output="video_generate: downloaded empty video bytes.",
                is_error=True,
            )
        from prometheus.gateway.media_cache import cache_video_from_bytes
        saved_path = cache_video_from_bytes(video_bytes, ext=".mp4")
        size_mb = len(video_bytes) / (1024.0 * 1024.0)
        return ToolResult(
            output=(
                f"Saved video to {saved_path}\n"
                f"  backend: kling ({kind})\n"
                f"  model: {model_name}\n"
                f"  duration: {video_duration or arguments.duration}s\n"
                f"  resolution: {arguments.resolution}\n"
                f"  size: {size_mb:.1f} MB\n"
                f"  task: {task_id}"
            ),
        )


# ---------------------------------------------------------------------------
# Config + auth plumbing
# ---------------------------------------------------------------------------

def _video_config(context: ToolExecutionContext) -> dict[str, Any]:
    """Pull the ``video_generation`` config block from runtime context.

    Same fallback shape as image_generate._image_config: context metadata
    first, bare prometheus.yaml second (some tool-call entry points build
    contexts without the loaded config).
    """
    meta = getattr(context, "metadata", {}) or {}
    config = meta.get("config") if isinstance(meta, dict) else None
    if isinstance(config, dict):
        block = config.get("video_generation", {})
        if isinstance(block, dict):
            return block
    try:
        import yaml
        from prometheus.config.paths import get_config_dir
        for candidate in (
            Path(__file__).resolve().parents[4] / "config" / "prometheus.yaml",
            get_config_dir() / "prometheus.yaml",
        ):
            if candidate.is_file():
                cfg = yaml.safe_load(candidate.read_text()) or {}
                block = cfg.get("video_generation", {})
                if isinstance(block, dict):
                    return block
                break
    except Exception:
        pass
    return {}


def _kling_keys(kling_cfg: dict[str, Any]) -> tuple[str, str]:
    """Resolve the AccessKey/SecretKey pair: direct config values
    (env-override injected) first, then the configured env var names.
    Empty strings = not configured."""
    import os

    access = str(kling_cfg.get("access_key") or "")
    secret = str(kling_cfg.get("secret_key") or "")
    if not access:
        access = os.environ.get(
            kling_cfg.get("access_key_env") or _KLING_ACCESS_KEY_ENV, ""
        )
    if not secret:
        secret = os.environ.get(
            kling_cfg.get("secret_key_env") or _KLING_SECRET_KEY_ENV, ""
        )
    return access, secret


def _kling_headers(access_key: str, secret_key: str) -> dict[str, str]:
    """Bearer headers with a freshly minted JWT (re-mint per request)."""
    return {
        "Authorization": f"Bearer {mint_kling_jwt(access_key, secret_key)}",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Poll + result extraction
# ---------------------------------------------------------------------------

async def _poll_kling_task(
    client: httpx.AsyncClient,
    base: str,
    kind: str,
    task_id: str,
    *,
    access_key: str,
    secret_key: str,
    budget: float,
) -> tuple[str, dict[str, Any]] | None:
    """Poll ``GET {base}/v1/videos/{kind}/{task_id}`` until terminal.

    Returns ``(normalized_status, data_dict)`` on succeed/failed, or
    ``None`` on budget exhaustion. Each poll re-mints the JWT.
    """
    deadline = asyncio.get_event_loop().time() + budget
    while asyncio.get_event_loop().time() < deadline:
        try:
            r = await client.get(
                f"{base}/v1/videos/{kind}/{task_id}",
                headers=_kling_headers(access_key, secret_key),
            )
            r.raise_for_status()
            data = r.json().get("data") or {}
            status = str(data.get("task_status") or "").lower()
            if status in _KLING_TERMINAL_OK + _KLING_TERMINAL_FAIL:
                return status, data
        except httpx.HTTPError as exc:
            logger.debug("kling task poll error: %s", exc)
        await asyncio.sleep(_KLING_POLL_INTERVAL)
    return None


def _extract_kling_video(data: dict[str, Any]) -> tuple[str, str]:
    """Pull ``(url, duration)`` of the first video from a succeeded task."""
    videos = (data.get("task_result") or {}).get("videos") or []
    for video in videos:
        if isinstance(video, dict) and video.get("url"):
            return str(video["url"]), str(video.get("duration") or "")
    return "", ""
