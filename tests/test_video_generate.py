"""CLOUD EXPANSION (2026-07) — Kling 3.0 video_generate tool, fakes only.

Pins:
* the stdlib-only HS256 JWT mint against a KNOWN-ANSWER vector (the pinned
  token below was cross-verified byte-for-byte against PyJWT 2.x with the
  same header/payload/secret at authoring time — this repo deliberately
  does NOT depend on PyJWT),
* text2video and image2video task flows (request shapes included),
* poll-budget exhaustion and key-missing error surfaces,
* registration via the standard try_register path.

No network: httpx.AsyncClient is replaced by a scripted fake. No keys —
the tool ships dormant-until-keyed.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any
from unittest.mock import patch

import pytest

from prometheus.tools.base import ToolExecutionContext, ToolRegistry
from prometheus.tools.builtin import video_generate as vg
from prometheus.tools.builtin.video_generate import (
    KlingVideoTool,
    VideoGenerateInput,
    mint_kling_jwt,
)

_MP4 = b"\x00\x00\x00\x18ftypmp42fake-video-bytes"

_KLING_CFG: dict[str, Any] = {
    "kling": {
        "access_key_env": "KLING_ACCESS_KEY",
        "secret_key_env": "KLING_SECRET_KEY",
        "base_url": "https://api-singapore.klingai.com",
        "model_name": "kling-v3",
        "poll_budget_seconds": 600,
    },
}

_ENV = {"KLING_ACCESS_KEY": "test-ak", "KLING_SECRET_KEY": "test-sk"}


def _ctx(tmp_path) -> ToolExecutionContext:
    return ToolExecutionContext(
        cwd=tmp_path, metadata={"config": {"video_generation": _KLING_CFG}},
    )


# ---------------------------------------------------------------------------
# JWT — known-answer + structural pins
# ---------------------------------------------------------------------------


class TestKlingJwt:
    # Known-answer vector: fixed AK/SK/time → EXACT token. Cross-checked
    # against PyJWT (jwt.encode({"iss": ak, "exp": now+1800, "nbf": now-5},
    # sk, algorithm="HS256")) at authoring time — byte-identical.
    AK = "test-access-key"
    SK = "test-secret-key"
    NOW = 1_760_000_000
    EXPECTED = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJpc3MiOiJ0ZXN0LWFjY2Vzcy1rZXkiLCJleHAiOjE3NjAwMDE4MDAsIm5iZiI6MTc1OTk5OTk5NX0"
        ".98e7ErBPCGVVNlH5hx_9ZgG1Vdb06nbSY2y1r866wAk"
    )

    def test_known_answer_vector(self) -> None:
        assert mint_kling_jwt(self.AK, self.SK, now=self.NOW) == self.EXPECTED

    def test_segments_are_base64url_without_padding(self) -> None:
        token = mint_kling_jwt(self.AK, self.SK, now=self.NOW)
        assert "=" not in token
        assert "+" not in token
        assert "/" not in token
        assert token.count(".") == 2

    def test_payload_decodes_to_kling_claims(self) -> None:
        token = mint_kling_jwt(self.AK, self.SK, now=self.NOW)
        header_b64, payload_b64, _sig = token.split(".")

        def _decode(seg: str) -> dict:
            padded = seg + "=" * (-len(seg) % 4)
            return json.loads(base64.urlsafe_b64decode(padded))

        assert _decode(header_b64) == {"alg": "HS256", "typ": "JWT"}
        assert _decode(payload_b64) == {
            "iss": self.AK,
            "exp": self.NOW + 1800,
            "nbf": self.NOW - 5,
        }

    def test_default_now_uses_wall_clock(self) -> None:
        import time as _time

        before = int(_time.time())
        token = mint_kling_jwt(self.AK, self.SK)
        payload_b64 = token.split(".")[1]
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        assert before + 1800 <= payload["exp"] <= int(_time.time()) + 1800


# ---------------------------------------------------------------------------
# Scripted httpx fake (same shape as tests/test_image_dashscope.py)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self, json_data: dict | None = None, content: bytes = b"", text: str = "",
    ) -> None:
        self._json = json_data or {}
        self.content = content
        self.text = text

    def json(self) -> dict:
        return self._json

    def raise_for_status(self) -> None:
        return None


class _FakeAsyncClient:
    def __init__(self, script: "_Script") -> None:
        self._script = script

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self._script.calls.append(("POST", url, kwargs))
        return self._script.on_post(url, kwargs)

    async def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self._script.calls.append(("GET", url, kwargs))
        return self._script.on_get(url, kwargs)


class _Script:
    def __init__(self, on_post, on_get) -> None:
        self.on_post = on_post
        self.on_get = on_get
        self.calls: list[tuple[str, str, dict]] = []


def _patch_client(monkeypatch, script: _Script) -> None:
    monkeypatch.setattr(
        vg.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(script),
    )


def _patch_cache(monkeypatch, tmp_path) -> dict[str, Any]:
    saved: dict[str, Any] = {}

    def _fake_cache(data: bytes, ext: str = ".mp4") -> str:
        saved["data"], saved["ext"] = data, ext
        out = tmp_path / f"vid{ext}"
        out.write_bytes(data)
        return str(out)

    monkeypatch.setattr(
        "prometheus.gateway.media_cache.cache_video_from_bytes", _fake_cache,
    )
    return saved


# ---------------------------------------------------------------------------
# text2video flow
# ---------------------------------------------------------------------------


class TestText2Video:
    async def test_full_t2v_flow(self, monkeypatch, tmp_path) -> None:
        base = "https://api-singapore.klingai.com"
        video_url = "https://kling-result.example/v-9.mp4"
        poll_states = iter(["submitted", "processing", "succeed"])

        def on_post(url, kwargs):
            assert url == f"{base}/v1/videos/text2video"
            headers = kwargs["headers"]
            # Bearer <jwt> — three dot-separated base64url segments
            auth = headers["Authorization"]
            assert auth.startswith("Bearer ey")
            assert auth.removeprefix("Bearer ").count(".") == 2
            payload = kwargs["json"]
            assert payload["model_name"] == "kling-v3"
            assert payload["prompt"] == "a paper boat in a rain gutter"
            assert payload["duration"] == "5"        # string per Kling API
            assert payload["mode"] == "pro"          # 1080p default
            assert "image" not in payload            # t2v carries no image
            return _FakeResponse({"code": 0, "data": {"task_id": "task-v1"}})

        def on_get(url, kwargs):
            if url == f"{base}/v1/videos/text2video/task-v1":
                state = next(poll_states)
                data: dict[str, Any] = {"task_id": "task-v1", "task_status": state}
                if state == "succeed":
                    data["task_result"] = {
                        "videos": [{"id": "v-9", "url": video_url, "duration": "5"}],
                    }
                return _FakeResponse({"code": 0, "data": data})
            assert url == video_url
            return _FakeResponse(content=_MP4)

        script = _Script(on_post, on_get)
        _patch_client(monkeypatch, script)
        saved = _patch_cache(monkeypatch, tmp_path)
        monkeypatch.setattr(vg, "_KLING_POLL_INTERVAL", 0.0)

        tool = KlingVideoTool()
        with patch.dict(os.environ, _ENV):
            result = await tool.execute(
                VideoGenerateInput(prompt="a paper boat in a rain gutter"),
                _ctx(tmp_path),
            )

        assert result.is_error is False
        assert "backend: kling (text2video)" in result.output
        assert "task: task-v1" in result.output
        assert "duration: 5s" in result.output
        assert saved["data"] == _MP4
        assert saved["ext"] == ".mp4"
        # submit + 3 polls + download
        methods = [(m, u) for m, u, _ in script.calls]
        assert methods == [
            ("POST", f"{base}/v1/videos/text2video"),
            ("GET", f"{base}/v1/videos/text2video/task-v1"),
            ("GET", f"{base}/v1/videos/text2video/task-v1"),
            ("GET", f"{base}/v1/videos/text2video/task-v1"),
            ("GET", video_url),
        ]

    async def test_jwt_reminted_per_request(self, monkeypatch, tmp_path) -> None:
        """Every HTTP call carries its own freshly minted token."""
        minted: list[str] = []
        real_mint = vg.mint_kling_jwt

        def _counting_mint(ak, sk, now=None):
            token = real_mint(ak, sk, now=now)
            minted.append(token)
            return token

        monkeypatch.setattr(vg, "mint_kling_jwt", _counting_mint)

        def on_post(url, kwargs):
            return _FakeResponse({"code": 0, "data": {"task_id": "t"}})

        def on_get(url, kwargs):
            if url.endswith("/text2video/t"):
                return _FakeResponse({"code": 0, "data": {
                    "task_status": "succeed",
                    "task_result": {"videos": [{"url": "https://r/v.mp4"}]},
                }})
            return _FakeResponse(content=_MP4)

        _patch_client(monkeypatch, _Script(on_post, on_get))
        _patch_cache(monkeypatch, tmp_path)
        monkeypatch.setattr(vg, "_KLING_POLL_INTERVAL", 0.0)

        tool = KlingVideoTool()
        with patch.dict(os.environ, _ENV):
            result = await tool.execute(
                VideoGenerateInput(prompt="p"), _ctx(tmp_path),
            )
        assert result.is_error is False
        # One mint for the submit + one per poll (the raw download GET
        # carries no auth headers).
        assert len(minted) == 2

    async def test_720p_maps_to_std_mode(self, monkeypatch, tmp_path) -> None:
        seen: dict[str, Any] = {}

        def on_post(url, kwargs):
            seen["payload"] = kwargs["json"]
            return _FakeResponse({"code": 0, "data": {"task_id": "t"}})

        def on_get(url, kwargs):
            if url.endswith("/text2video/t"):
                return _FakeResponse({"code": 0, "data": {
                    "task_status": "succeed",
                    "task_result": {"videos": [{"url": "https://r/v.mp4"}]},
                }})
            return _FakeResponse(content=_MP4)

        _patch_client(monkeypatch, _Script(on_post, on_get))
        _patch_cache(monkeypatch, tmp_path)
        tool = KlingVideoTool()
        with patch.dict(os.environ, _ENV):
            result = await tool.execute(
                VideoGenerateInput(prompt="p", resolution="720p", duration=10),
                _ctx(tmp_path),
            )
        assert result.is_error is False
        assert seen["payload"]["mode"] == "std"
        assert seen["payload"]["duration"] == "10"


# ---------------------------------------------------------------------------
# image2video flow
# ---------------------------------------------------------------------------


class TestImage2Video:
    async def test_i2v_uses_image2video_endpoint_with_base64_image(
        self, monkeypatch, tmp_path,
    ) -> None:
        base = "https://api-singapore.klingai.com"
        first_frame = tmp_path / "frame.png"
        image_bytes = b"\x89PNGfake-first-frame"
        first_frame.write_bytes(image_bytes)
        seen: dict[str, Any] = {}

        def on_post(url, kwargs):
            seen["url"] = url
            seen["payload"] = kwargs["json"]
            return _FakeResponse({"code": 0, "data": {"task_id": "task-i2v"}})

        def on_get(url, kwargs):
            if url == f"{base}/v1/videos/image2video/task-i2v":
                return _FakeResponse({"code": 0, "data": {
                    "task_status": "succeed",
                    "task_result": {"videos": [{"url": "https://r/i.mp4", "duration": "5"}]},
                }})
            return _FakeResponse(content=_MP4)

        _patch_client(monkeypatch, _Script(on_post, on_get))
        _patch_cache(monkeypatch, tmp_path)
        tool = KlingVideoTool()
        with patch.dict(os.environ, _ENV):
            result = await tool.execute(
                VideoGenerateInput(
                    prompt="the boat sails away", image_path=str(first_frame),
                ),
                _ctx(tmp_path),
            )

        assert result.is_error is False
        assert seen["url"] == f"{base}/v1/videos/image2video"
        assert seen["payload"]["image"] == (
            base64.b64encode(image_bytes).decode("ascii")
        )
        assert "backend: kling (image2video)" in result.output

    async def test_missing_image_file_is_an_error_before_any_request(
        self, monkeypatch, tmp_path,
    ) -> None:
        script = _Script(
            lambda u, k: pytest.fail("no HTTP call for a missing image"),
            lambda u, k: pytest.fail("no HTTP call for a missing image"),
        )
        _patch_client(monkeypatch, script)
        tool = KlingVideoTool()
        with patch.dict(os.environ, _ENV):
            result = await tool.execute(
                VideoGenerateInput(
                    prompt="p", image_path=str(tmp_path / "nope.png"),
                ),
                _ctx(tmp_path),
            )
        assert result.is_error is True
        assert "does not exist" in result.output
        assert script.calls == []


# ---------------------------------------------------------------------------
# Error surfaces
# ---------------------------------------------------------------------------


class TestKlingErrors:
    async def test_missing_keys_actionable_and_no_request(
        self, monkeypatch, tmp_path,
    ) -> None:
        script = _Script(
            lambda u, k: pytest.fail("no HTTP call may happen without keys"),
            lambda u, k: pytest.fail("no HTTP call may happen without keys"),
        )
        _patch_client(monkeypatch, script)
        tool = KlingVideoTool()
        with patch.dict(os.environ, {}, clear=True):
            result = await tool.execute(
                VideoGenerateInput(prompt="p"), _ctx(tmp_path),
            )
        assert result.is_error is True
        assert "KLING_ACCESS_KEY" in result.output
        assert "KLING_SECRET_KEY" in result.output
        assert "app.klingai.com" in result.output
        assert script.calls == []

    async def test_one_key_of_the_pair_is_not_enough(
        self, monkeypatch, tmp_path,
    ) -> None:
        _patch_client(monkeypatch, _Script(
            lambda u, k: pytest.fail("no HTTP call with half a key pair"),
            lambda u, k: pytest.fail("no HTTP call with half a key pair"),
        ))
        tool = KlingVideoTool()
        with patch.dict(os.environ, {"KLING_ACCESS_KEY": "only-ak"}, clear=True):
            result = await tool.execute(
                VideoGenerateInput(prompt="p"), _ctx(tmp_path),
            )
        assert result.is_error is True
        assert "KLING_SECRET_KEY" in result.output

    async def test_failed_task_surfaces_status_msg(
        self, monkeypatch, tmp_path,
    ) -> None:
        def on_post(url, kwargs):
            return _FakeResponse({"code": 0, "data": {"task_id": "t-f"}})

        def on_get(url, kwargs):
            return _FakeResponse({"code": 0, "data": {
                "task_status": "failed",
                "task_status_msg": "content policy violation",
            }})

        _patch_client(monkeypatch, _Script(on_post, on_get))
        tool = KlingVideoTool()
        with patch.dict(os.environ, _ENV):
            result = await tool.execute(
                VideoGenerateInput(prompt="p"), _ctx(tmp_path),
            )
        assert result.is_error is True
        assert "failed" in result.output
        assert "content policy violation" in result.output

    async def test_poll_budget_exhaustion_is_honest(
        self, monkeypatch, tmp_path,
    ) -> None:
        def on_post(url, kwargs):
            return _FakeResponse({"code": 0, "data": {"task_id": "t-slow"}})

        def on_get(url, kwargs):
            return _FakeResponse({"code": 0, "data": {"task_status": "processing"}})

        _patch_client(monkeypatch, _Script(on_post, on_get))
        monkeypatch.setattr(vg, "_KLING_POLL_INTERVAL", 0.0)
        cfg = {"kling": dict(_KLING_CFG["kling"], poll_budget_seconds=0.05)}
        tool = KlingVideoTool()
        with patch.dict(os.environ, _ENV):
            result = await tool.execute(
                VideoGenerateInput(prompt="p"),
                ToolExecutionContext(
                    cwd=tmp_path, metadata={"config": {"video_generation": cfg}},
                ),
            )
        assert result.is_error is True
        assert "did not finish" in result.output
        assert "poll_budget_seconds" in result.output

    async def test_nonzero_submit_code_is_an_error(
        self, monkeypatch, tmp_path,
    ) -> None:
        def on_post(url, kwargs):
            return _FakeResponse({"code": 1102, "message": "insufficient balance"})

        _patch_client(monkeypatch, _Script(on_post, lambda u, k: _FakeResponse()))
        tool = KlingVideoTool()
        with patch.dict(os.environ, _ENV):
            result = await tool.execute(
                VideoGenerateInput(prompt="p"), _ctx(tmp_path),
            )
        assert result.is_error is True
        assert "insufficient balance" in result.output


# ---------------------------------------------------------------------------
# Registration + schema
# ---------------------------------------------------------------------------


class TestRegistrationAndSchema:
    def test_try_register_registers_video_generate(self) -> None:
        from prometheus.tools.registration import try_register

        registry = ToolRegistry()
        ok = try_register(
            registry, "KlingVideoTool",
            "prometheus.tools.builtin.video_generate", "KlingVideoTool",
        )
        assert ok is True
        assert "video_generate" in [t.name for t in registry.list_tools()]

    def test_is_read_only_cache_confined(self) -> None:
        # No output_path override exists — writes land only in the media
        # cache, mirroring image_generate's output_path-is-None case.
        tool = KlingVideoTool()
        assert tool.is_read_only(VideoGenerateInput(prompt="p")) is True

    def test_duration_restricted_to_5_or_10(self) -> None:
        with pytest.raises(Exception):
            VideoGenerateInput(prompt="p", duration=7)

    def test_execution_timeout_exceeds_poll_budget(self) -> None:
        """The agent loop kills tools at 300s by default; Kling's poll
        budget alone is 600s. The per-tool override must clear the default
        poll budget with margin, or every real render dies mid-poll."""
        tool = KlingVideoTool()
        assert tool.execution_timeout_seconds is not None
        assert tool.execution_timeout_seconds > vg._KLING_POLL_BUDGET
