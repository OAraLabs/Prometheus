"""CLOUD EXPANSION (2026-07) — WAN 2.5 (DashScope) image backend, fakes only.

Pins:
* the HARD RULE that ``auto`` backend resolution NEVER lands on the paid
  DashScope backend (even with DASHSCOPE_API_KEY set),
* the async submit → poll → download-bytes-now happy path (request shapes
  included — the X-DashScope-Async header, the size parameter, the task
  poll URL),
* FAILED-task and key-missing error surfaces.

No network: httpx.AsyncClient is replaced by a scripted fake.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin import image_generate as ig
from prometheus.tools.builtin.image_generate import (
    ImageGenerateInput,
    ImageGenerateTool,
    _resolve_backend,
)

_PNG = b"\x89PNG\r\n\x1a\nfake-wan-image-bytes"

_DS_CFG: dict[str, Any] = {
    "dashscope": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "model": "wan2.5-t2i-preview",
        "base_url": "https://dashscope-intl.aliyuncs.com/api/v1",
    },
}


# ---------------------------------------------------------------------------
# Scripted httpx fake
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(
        self,
        json_data: dict | None = None,
        content: bytes = b"",
        headers: dict | None = None,
        text: str = "",
    ) -> None:
        self._json = json_data or {}
        self.content = content
        self.headers = headers or {}
        self.text = text

    def json(self) -> dict:
        return self._json

    def raise_for_status(self) -> None:
        return None


class _FakeAsyncClient:
    """Route POST/GET by URL against scripted handlers, recording calls."""

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
        ig.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(script),
    )


# ---------------------------------------------------------------------------
# THE PIN: auto never selects the paid backend
# ---------------------------------------------------------------------------


class TestAutoNeverPicksDashscope:
    async def test_auto_with_key_set_and_no_comfyui_goes_pollinations(
        self, monkeypatch,
    ) -> None:
        """A present DASHSCOPE_API_KEY must not change what auto picks."""

        async def _unreachable(cfg):
            return False

        monkeypatch.setattr(ig, "_comfyui_reachable", _unreachable)
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "paid-key"}):
            assert await _resolve_backend("auto", _DS_CFG) == "pollinations"

    async def test_auto_with_comfyui_up_goes_comfyui(self, monkeypatch) -> None:
        async def _reachable(cfg):
            return True

        monkeypatch.setattr(ig, "_comfyui_reachable", _reachable)
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "paid-key"}):
            assert await _resolve_backend("auto", _DS_CFG) == "comfyui"

    async def test_explicit_argument_reaches_dashscope(self) -> None:
        assert await _resolve_backend("dashscope", _DS_CFG) == "dashscope"

    async def test_explicit_config_default_reaches_dashscope(self) -> None:
        cfg = dict(_DS_CFG, default_backend="dashscope")
        assert await _resolve_backend("auto", cfg) == "dashscope"


# ---------------------------------------------------------------------------
# Happy path: submit → poll → download → cache
# ---------------------------------------------------------------------------


class TestDashscopeHappyPath:
    async def test_full_generate_flow(self, monkeypatch, tmp_path) -> None:
        base = "https://dashscope-intl.aliyuncs.com/api/v1"
        image_url = "https://dashscope-result.oss/img-123.png"
        poll_states = iter(["PENDING", "SUCCEEDED"])

        def on_post(url, kwargs):
            assert url == f"{base}/services/aigc/image-generation/generation"
            headers = kwargs["headers"]
            assert headers["X-DashScope-Async"] == "enable"   # async-ONLY API
            assert headers["Authorization"] == "Bearer paid-key"
            payload = kwargs["json"]
            assert payload["model"] == "wan2.5-t2i-preview"
            assert payload["input"] == {"prompt": "a red lighthouse at dusk"}
            assert payload["parameters"]["size"] == "1024*1024"
            return _FakeResponse({"output": {"task_id": "task-abc"}})

        def on_get(url, kwargs):
            if url == f"{base}/tasks/task-abc":
                state = next(poll_states)
                out: dict[str, Any] = {"task_status": state}
                if state == "SUCCEEDED":
                    out["results"] = [{"url": image_url}]
                return _FakeResponse({"output": out})
            assert url == image_url  # 24h-expiring URL → downloaded NOW
            return _FakeResponse(
                content=_PNG, headers={"content-type": "image/png"},
            )

        script = _Script(on_post, on_get)
        _patch_client(monkeypatch, script)
        monkeypatch.setattr(ig, "_DASHSCOPE_POLL_INTERVAL", 0.0)
        saved: dict[str, Any] = {}

        def _fake_save(data, *, ext, override_path):
            saved["data"], saved["ext"] = data, ext
            return str(tmp_path / f"img{ext}")

        monkeypatch.setattr(ig, "_save_image_bytes", _fake_save)

        tool = ImageGenerateTool()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "paid-key"}):
            result = await tool.execute(
                ImageGenerateInput(
                    prompt="a red lighthouse at dusk", backend="dashscope",
                ),
                ToolExecutionContext(
                    cwd=tmp_path, metadata={"config": {"image_generation": _DS_CFG}},
                ),
            )

        assert result.is_error is False
        assert "backend: dashscope" in result.output
        assert saved["data"] == _PNG
        assert saved["ext"] == ".png"
        # One submit, two polls, one download
        methods = [(m, u) for m, u, _ in script.calls]
        assert methods == [
            ("POST", f"{base}/services/aigc/image-generation/generation"),
            ("GET", f"{base}/tasks/task-abc"),
            ("GET", f"{base}/tasks/task-abc"),
            ("GET", image_url),
        ]

    async def test_seed_forwarded_when_set(self, monkeypatch, tmp_path) -> None:
        base = "https://dashscope-intl.aliyuncs.com/api/v1"
        seen: dict[str, Any] = {}

        def on_post(url, kwargs):
            seen["payload"] = kwargs["json"]
            return _FakeResponse({"output": {"task_id": "t"}})

        def on_get(url, kwargs):
            if url.endswith("/tasks/t"):
                return _FakeResponse({"output": {
                    "task_status": "SUCCEEDED",
                    "results": [{"url": "https://r/x.png"}],
                }})
            return _FakeResponse(
                content=_PNG, headers={"content-type": "image/png"},
            )

        _patch_client(monkeypatch, _Script(on_post, on_get))
        monkeypatch.setattr(
            ig, "_save_image_bytes",
            lambda data, *, ext, override_path: str(tmp_path / "x.png"),
        )
        tool = ImageGenerateTool()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "k"}):
            result = await tool.execute(
                ImageGenerateInput(
                    prompt="p", backend="dashscope", seed=42,
                ),
                ToolExecutionContext(
                    cwd=tmp_path, metadata={"config": {"image_generation": _DS_CFG}},
                ),
            )
        assert result.is_error is False
        assert seen["payload"]["parameters"]["seed"] == 42


# ---------------------------------------------------------------------------
# Error surfaces
# ---------------------------------------------------------------------------


class TestDashscopeErrors:
    async def test_failed_task_is_honest(self, monkeypatch, tmp_path) -> None:
        def on_post(url, kwargs):
            return _FakeResponse({"output": {"task_id": "task-bad"}})

        def on_get(url, kwargs):
            return _FakeResponse({"output": {
                "task_status": "FAILED",
                "code": "InvalidParameter",
                "message": "prompt rejected by moderation",
            }})

        _patch_client(monkeypatch, _Script(on_post, on_get))
        tool = ImageGenerateTool()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "k"}):
            result = await tool.execute(
                ImageGenerateInput(prompt="p", backend="dashscope"),
                ToolExecutionContext(
                    cwd=tmp_path, metadata={"config": {"image_generation": _DS_CFG}},
                ),
            )
        assert result.is_error is True
        assert "FAILED" in result.output
        assert "prompt rejected by moderation" in result.output

    async def test_missing_key_is_actionable_and_makes_no_request(
        self, monkeypatch, tmp_path,
    ) -> None:
        script = _Script(
            lambda u, k: pytest.fail("no HTTP call may happen without a key"),
            lambda u, k: pytest.fail("no HTTP call may happen without a key"),
        )
        _patch_client(monkeypatch, script)
        tool = ImageGenerateTool()
        with patch.dict(os.environ, {}, clear=True):
            result = await tool.execute(
                ImageGenerateInput(prompt="p", backend="dashscope"),
                ToolExecutionContext(
                    cwd=tmp_path, metadata={"config": {"image_generation": _DS_CFG}},
                ),
            )
        assert result.is_error is True
        assert "DASHSCOPE_API_KEY" in result.output
        assert "PAID" in result.output
        assert "dashscope.console.aliyun.com" in result.output
        assert script.calls == []

    async def test_poll_budget_exhaustion_is_honest(
        self, monkeypatch, tmp_path,
    ) -> None:
        def on_post(url, kwargs):
            return _FakeResponse({"output": {"task_id": "task-slow"}})

        def on_get(url, kwargs):
            return _FakeResponse({"output": {"task_status": "RUNNING"}})

        _patch_client(monkeypatch, _Script(on_post, on_get))
        monkeypatch.setattr(ig, "_DASHSCOPE_POLL_INTERVAL", 0.0)
        monkeypatch.setattr(ig, "_DASHSCOPE_POLL_BUDGET", 0.05)
        tool = ImageGenerateTool()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "k"}):
            result = await tool.execute(
                ImageGenerateInput(prompt="p", backend="dashscope"),
                ToolExecutionContext(
                    cwd=tmp_path, metadata={"config": {"image_generation": _DS_CFG}},
                ),
            )
        assert result.is_error is True
        assert "did not finish" in result.output


class TestExecutionTimeoutOverride:
    def test_tool_timeout_exceeds_dashscope_poll_budget(self) -> None:
        """The agent loop kills tools at 300s by default — exactly the
        dashscope poll budget, leaving no room for submit + download. The
        per-tool override must clear the budget with margin so a slow WAN
        render surfaces the tool's own honest timeout message."""
        tool = ImageGenerateTool()
        assert tool.execution_timeout_seconds is not None
        assert tool.execution_timeout_seconds > ig._DASHSCOPE_POLL_BUDGET
