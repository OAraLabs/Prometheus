"""Tests for infrastructure self-awareness — AnatomyScanner, AnatomyWriter, ProjectConfigStore."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from prometheus.infra.anatomy import AnatomyScanner, AnatomyState
from prometheus.infra.anatomy_writer import AnatomyWriter
from prometheus.infra.project_configs import (
    ModelSlot,
    ProjectConfig,
    ProjectConfigStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_state(**overrides) -> AnatomyState:
    defaults = dict(
        hostname="test-gpu",
        platform="Linux",
        cpu="AMD Ryzen 9 7950X",
        ram_total_gb=64.0,
        ram_available_gb=48.2,
        gpu_name="NVIDIA RTX 4090",
        gpu_vram_total_mb=24576,
        gpu_vram_used_mb=18432,
        gpu_vram_free_mb=6144,
        model_name="gemma-4-26B-A4B-it-Q4_K_XL.gguf",
        model_file="gemma-4-26B-A4B-it-Q4_K_XL.gguf",
        model_quantization="Q4_K_XL",
        inference_engine="llama_cpp",
        inference_url="http://192.0.2.1:8080",
        inference_features=["streaming"],
        vision_enabled=True,
        whisper_model="base",
        tailscale_ip="192.0.2.1",
        tailscale_peers=[
            {"name": "test-brain", "ip": "192.0.2.2", "online": True},
            {"name": "test-gpu", "ip": "192.0.2.1", "online": True},
        ],
        disk_total_gb=500.0,
        disk_free_gb=220.0,
        prometheus_data_size_mb=42.5,
        scanned_at="2026-04-06T21:30:00Z",
    )
    defaults.update(overrides)
    return AnatomyState(**defaults)


# ===========================================================================
# AnatomyScanner
# ===========================================================================


class TestAnatomyScanner:
    def test_detects_hostname_and_platform(self) -> None:
        scanner = AnatomyScanner()
        state = AnatomyState()
        scanner._detect_platform(state)
        assert state.hostname  # non-empty
        assert state.platform in ("Linux", "Darwin", "Windows")

    def test_detects_ram(self) -> None:
        scanner = AnatomyScanner()
        state = AnatomyState()
        scanner._detect_ram(state)
        # Should detect at least some RAM on any system
        assert state.ram_total_gb > 0

    def test_detects_disk(self) -> None:
        scanner = AnatomyScanner()
        state = AnatomyState()
        scanner._detect_disk(state)
        assert state.disk_total_gb > 0
        assert state.disk_free_gb > 0

    def test_gpu_detection_handles_no_nvidia_smi(self) -> None:
        scanner = AnatomyScanner()
        state = AnatomyState()
        # Mock nvidia-smi to fail (most CI/test environments have no GPU)
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            asyncio.run(scanner._detect_gpu(state))
        assert state.gpu_name is None

    def test_gpu_detection_parses_nvidia_smi(self) -> None:
        scanner = AnatomyScanner()
        state = AnatomyState()

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(b"NVIDIA RTX 4090, 24576, 18432, 6144\n", b"")
        )
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            asyncio.run(scanner._detect_gpu(state))

        assert state.gpu_name == "NVIDIA RTX 4090"
        assert state.gpu_vram_total_mb == 24576
        assert state.gpu_vram_used_mb == 18432
        assert state.gpu_vram_free_mb == 6144

    def test_gpu_remote_fallback_via_ssh(self) -> None:
        scanner = AnatomyScanner(
            llama_cpp_url="http://192.0.2.1:8080",
            inference_engine="llama_cpp",
        )
        state = AnatomyState()

        # Local nvidia-smi fails
        local_proc = AsyncMock()
        local_proc.communicate = AsyncMock(return_value=(b"", b"not found"))
        local_proc.returncode = 1

        # Remote SSH nvidia-smi succeeds
        remote_proc = AsyncMock()
        remote_proc.communicate = AsyncMock(
            return_value=(b"NVIDIA RTX 4090, 24576, 18432, 6144\n", b"")
        )
        remote_proc.returncode = 0

        call_count = 0
        async def mock_exec(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return local_proc  # first call: local nvidia-smi
            return remote_proc  # second call: ssh nvidia-smi

        with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
            asyncio.run(scanner._detect_gpu(state))

        assert state.gpu_name == "NVIDIA RTX 4090"
        assert state.gpu_vram_total_mb == 24576

    def test_gpu_remote_skipped_for_localhost(self) -> None:
        scanner = AnatomyScanner(
            llama_cpp_url="http://127.0.0.1:8080",
            inference_engine="llama_cpp",
        )
        state = AnatomyState()

        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            asyncio.run(scanner._detect_gpu(state))

        # Should not attempt SSH for localhost — gpu stays None
        assert state.gpu_name is None

    def test_parse_nvidia_smi(self) -> None:
        state = AnatomyState()
        ok = AnatomyScanner._parse_nvidia_smi(state, "NVIDIA RTX 4090, 24576, 18432, 6144\n")
        assert ok is True
        assert state.gpu_name == "NVIDIA RTX 4090"
        assert state.gpu_vram_free_mb == 6144

        state2 = AnatomyState()
        ok2 = AnatomyScanner._parse_nvidia_smi(state2, "bad output")
        assert ok2 is False
        assert state2.gpu_name is None

    def test_model_detection_handles_server_down(self) -> None:
        scanner = AnatomyScanner(llama_cpp_url="http://127.0.0.1:99999")
        state = AnatomyState()
        asyncio.run(scanner._detect_model(state))
        assert state.model_name is None

    def test_model_detection_parses_llama_cpp(self) -> None:
        scanner = AnatomyScanner()
        state = AnatomyState()

        mock_data = {
            "/v1/models": {"data": [{"id": "gemma-4-Q4_K_XL.gguf"}]},
            "/props": {"total_slots": 1},
            "/slots": [{"has_vision": True}],
        }

        class FakeResponse:
            def __init__(self, data, code=200):
                self._data = data
                self.status_code = code
            def raise_for_status(self):
                pass
            def json(self):
                return self._data

        class FakeClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def get(self, url, **kw):
                for path, data in mock_data.items():
                    if path in url:
                        return FakeResponse(data)
                return FakeResponse({}, 404)

        with patch("prometheus.infra.anatomy.httpx.AsyncClient", return_value=FakeClient()):
            asyncio.run(scanner._detect_model_llama_cpp(state))

        assert state.model_name == "gemma-4-Q4_K_XL.gguf"
        assert state.model_quantization == "Q4_K_XL"
        assert state.vision_enabled is True

    def test_parse_model_id_extracts_quantization(self) -> None:
        state = AnatomyState()
        AnatomyScanner._parse_model_id(state, "gemma-4-26B-A4B-it-Q4_K_XL.gguf")
        assert state.model_quantization == "Q4_K_XL"

        state2 = AnatomyState()
        AnatomyScanner._parse_model_id(state2, "model-BF16.gguf")
        assert state2.model_quantization == "BF16"

        state3 = AnatomyState()
        AnatomyScanner._parse_model_id(state3, "no-quant-info")
        assert state3.model_quantization is None

    def test_tailscale_detection_handles_missing_binary(self) -> None:
        scanner = AnatomyScanner()
        state = AnatomyState()
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            asyncio.run(scanner._detect_tailscale(state))
        assert state.tailscale_ip is None

    def test_tailscale_detection_parses_peers(self) -> None:
        scanner = AnatomyScanner()
        state = AnatomyState()

        ts_json = json.dumps({
            "TailscaleIPs": ["192.0.2.2"],
            "Peer": {
                "node1": {
                    "HostName": "test-gpu",
                    "TailscaleIPs": ["192.0.2.1"],
                    "Online": True,
                },
                "node2": {
                    "HostName": "phone",
                    "TailscaleIPs": ["198.51.100.1"],
                    "Online": False,
                },
            },
        })

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(
            return_value=(ts_json.encode(), b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            asyncio.run(scanner._detect_tailscale(state))

        assert state.tailscale_ip == "192.0.2.2"
        assert len(state.tailscale_peers) == 2
        assert state.tailscale_peers[0]["name"] == "test-gpu"
        assert state.tailscale_peers[0]["ip"] == "192.0.2.1"
        assert state.tailscale_peers[0]["online"] is True
        assert state.tailscale_peers[1]["online"] is False

    def test_disk_detection(self) -> None:
        from collections import namedtuple
        DiskUsage = namedtuple("usage", ["total", "used", "free"])
        scanner = AnatomyScanner()
        state = AnatomyState()
        with patch("prometheus.infra.anatomy.shutil.disk_usage", return_value=DiskUsage(500 * 1024**3, 280 * 1024**3, 220 * 1024**3)):
            scanner._detect_disk(state)
        assert state.disk_total_gb == 500.0
        assert state.disk_free_gb == 220.0

    def test_full_scan_returns_state(self) -> None:
        scanner = AnatomyScanner(llama_cpp_url="http://127.0.0.1:99999")
        # Full scan should succeed even with everything unavailable
        state = asyncio.run(scanner.scan())
        assert state.scanned_at
        assert state.hostname
        assert state.platform

    def test_quick_scan_returns_state(self) -> None:
        scanner = AnatomyScanner(llama_cpp_url="http://127.0.0.1:99999")
        state = asyncio.run(scanner.quick_scan())
        assert state.scanned_at
        assert state.hostname


# ===========================================================================
# AnatomyWriter
# ===========================================================================


class TestAnatomyWriter:
    def test_write_creates_anatomy_md(self, tmp_path: Path) -> None:
        writer = AnatomyWriter(anatomy_path=tmp_path / "ANATOMY.md")
        state = _sample_state()
        content = writer.write(state)

        path = tmp_path / "ANATOMY.md"
        assert path.exists()
        text = path.read_text()
        assert "Active Configuration" in text
        assert "NVIDIA RTX 4090" in text
        assert "gemma-4-26B" in text
        assert "Last scanned:" in text

    def test_write_includes_mermaid(self, tmp_path: Path) -> None:
        writer = AnatomyWriter(anatomy_path=tmp_path / "ANATOMY.md")
        state = _sample_state()
        content = writer.write(state)
        assert "```mermaid" in content
        assert "graph LR" in content

    def test_render_mermaid(self) -> None:
        writer = AnatomyWriter()
        state = _sample_state()
        diagram = writer.render_mermaid(state)
        assert "graph LR" in diagram
        assert "4090" in diagram
        assert "Vision" in diagram  # vision is enabled

    def test_render_mermaid_no_gpu(self) -> None:
        writer = AnatomyWriter()
        state = _sample_state(gpu_name=None)
        diagram = writer.render_mermaid(state)
        assert "Local Model" in diagram  # fallback for no GPU

    def test_update_active_section(self, tmp_path: Path) -> None:
        writer = AnatomyWriter(anatomy_path=tmp_path / "ANATOMY.md")
        state1 = _sample_state(gpu_vram_used_mb=18000, scanned_at="2026-04-06T21:00:00Z")
        writer.write(state1)

        # Update with new VRAM stats
        state2 = _sample_state(gpu_vram_used_mb=20000, scanned_at="2026-04-06T21:05:00Z")
        writer.update_active_section(state2)

        text = (tmp_path / "ANATOMY.md").read_text()
        assert "20000MB" in text  # updated
        assert "18000MB" not in text  # old value replaced
        assert "Architecture" in text  # other sections preserved

    def test_render_summary_compact(self) -> None:
        writer = AnatomyWriter()
        state = _sample_state()
        summary = writer.render_summary(state, project_names=["default", "skillforge"])
        assert "## Infrastructure" in summary
        assert "RTX 4090" in summary
        assert "Vision enabled" in summary
        assert "default, skillforge" in summary
        # Should be compact
        assert len(summary) < 1500  # well under 300 tokens

    def test_write_with_project_summaries(self, tmp_path: Path) -> None:
        writer = AnatomyWriter(anatomy_path=tmp_path / "ANATOMY.md")
        state = _sample_state()
        summaries = [
            {"name": "default", "description": "Daily driver", "status": "active"},
            {"name": "eval", "description": "A/B testing"},
        ]
        content = writer.write(state, summaries)
        assert "Project Configurations" in content
        assert "Daily driver" in content


# ===========================================================================
# ProjectConfigStore
# ===========================================================================


class TestProjectConfigStore:
    def test_list_empty(self, tmp_path: Path) -> None:
        store = ProjectConfigStore(projects_dir=tmp_path / "projects")
        assert store.list_projects() == []

    def test_save_and_load(self, tmp_path: Path) -> None:
        store = ProjectConfigStore(projects_dir=tmp_path / "projects")
        cfg = ProjectConfig(
            name="test",
            description="Test config",
            models=[ModelSlot(name="TestModel", role="primary", vram_estimate_gb=8.0)],
            services=["daemon"],
            notes="testing",
            active=True,
        )
        store.save(cfg)

        loaded = store.get("test")
        assert loaded is not None
        assert loaded.name == "test"
        assert loaded.description == "Test config"
        assert len(loaded.models) == 1
        assert loaded.models[0].name == "TestModel"
        assert loaded.active is True

    def test_get_active(self, tmp_path: Path) -> None:
        store = ProjectConfigStore(projects_dir=tmp_path / "projects")
        store.save(ProjectConfig(name="a", active=False))
        store.save(ProjectConfig(name="b", active=True))

        active = store.get_active()
        assert active is not None
        assert active.name == "b"

    def test_activate(self, tmp_path: Path) -> None:
        store = ProjectConfigStore(projects_dir=tmp_path / "projects")
        store.save(ProjectConfig(name="a", active=True))
        store.save(ProjectConfig(name="b", active=False))

        store.activate("b")
        assert store.get("a").active is False
        assert store.get("b").active is True

    def test_summaries(self, tmp_path: Path) -> None:
        store = ProjectConfigStore(projects_dir=tmp_path / "projects")
        store.save(ProjectConfig(
            name="default",
            description="Daily driver",
            models=[ModelSlot(name="Gemma 4", role="primary")],
            active=True,
        ))
        summaries = store.summaries()
        assert len(summaries) == 1
        assert summaries[0]["name"] == "default"
        assert summaries[0]["status"] == "active"

    def test_get_nonexistent(self, tmp_path: Path) -> None:
        store = ProjectConfigStore(projects_dir=tmp_path / "projects")
        assert store.get("nonexistent") is None

    def test_loads_real_default_config(self) -> None:
        """Verify the seeded default.yaml loads correctly."""
        store = ProjectConfigStore()
        cfg = store.get("default")
        if cfg is not None:
            assert cfg.name == "default"
            assert len(cfg.models) >= 1
            assert cfg.models[0].engine == "llama_cpp"


# ===========================================================================
# AnatomyTool
# ===========================================================================


class TestAnatomyTool:
    def test_tool_schema(self) -> None:
        from prometheus.tools.builtin.anatomy import AnatomyTool
        tool = AnatomyTool()
        schema = tool.to_api_schema()
        assert schema["name"] == "anatomy"
        assert "infrastructure" in schema["description"].lower()

    def test_tool_not_initialized(self) -> None:
        from prometheus.tools.builtin.anatomy import AnatomyTool, AnatomyInput, _scanner
        from prometheus.tools.base import ToolExecutionContext

        # Ensure singletons are None
        import prometheus.tools.builtin.anatomy as mod
        old_scanner = mod._scanner
        mod._scanner = None

        tool = AnatomyTool()
        ctx = ToolExecutionContext(cwd=Path.cwd())
        result = asyncio.run(tool.execute(AnatomyInput(action="status"), ctx))
        assert result.is_error
        assert "not initialized" in result.output

        mod._scanner = old_scanner


# ===========================================================================
# System prompt integration
# ===========================================================================


class TestAnatomySummaryInPrompt:
    def test_anatomy_summary_loads_into_static(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Write a minimal ANATOMY.md
        anatomy = config_dir / "ANATOMY.md"
        anatomy.write_text(
            "# Anatomy\nLast scanned: 2026-04-06\n\n"
            "## Active Configuration\n"
            "### Model\n- **Loaded:** gemma-4-26B\n\n"
            "## Architecture\n```mermaid\ngraph LR\n```\n",
            encoding="utf-8",
        )

        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=config_dir):
            from prometheus.context.prompt_assembler import (
                _load_anatomy_summary,
                build_runtime_system_prompt,
            )
            from prometheus.context.system_prompt import SYSTEM_PROMPT_DYNAMIC_BOUNDARY

            summary = _load_anatomy_summary()
            assert summary is not None
            assert "gemma-4-26B" in summary
            # Should NOT include Architecture section
            assert "mermaid" not in summary

            prompt = build_runtime_system_prompt(cwd=str(tmp_path))
            static, _ = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
            assert "gemma-4-26B" in static

    def test_anatomy_disabled_by_config(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "ANATOMY.md").write_text(
            "# Anatomy\n\n## Active Configuration\nStuff\n",
            encoding="utf-8",
        )

        config = {"anatomy": {"include_in_system_prompt": False}}
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=config_dir):
            from prometheus.context.prompt_assembler import build_runtime_system_prompt
            prompt = build_runtime_system_prompt(cwd=str(tmp_path), config=config)
        assert "Stuff" not in prompt

    def test_missing_anatomy_graceful(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        # No ANATOMY.md
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=config_dir):
            from prometheus.context.prompt_assembler import build_runtime_system_prompt
            from prometheus.context.system_prompt import SYSTEM_PROMPT_DYNAMIC_BOUNDARY
            prompt = build_runtime_system_prompt(cwd=str(tmp_path))
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in prompt


# ===========================================================================
# /anatomy command
# ===========================================================================


class TestCmdAnatomy:
    async def test_cmd_anatomy_not_initialized(self) -> None:
        import prometheus.tools.builtin.anatomy as mod
        old = mod._scanner
        mod._scanner = None
        from prometheus.gateway.commands import cmd_anatomy
        text = await cmd_anatomy()
        assert "not initialized" in text.lower()
        mod._scanner = old

    async def test_cmd_anatomy_full_output(self) -> None:
        """Verify the formatted output includes all expected sections."""
        import prometheus.tools.builtin.anatomy as mod
        from prometheus.gateway.commands import cmd_anatomy

        old_s, old_w, old_p = mod._scanner, mod._writer, mod._project_store
        mock_scanner = MagicMock()

        state = _sample_state(
            hostname="test-brain",
            platform="Linux",
            inference_url="http://192.0.2.1:8080",
        )
        mock_scanner.scan = AsyncMock(return_value=state)
        mock_writer = MagicMock()
        mod._scanner = mock_scanner
        mod._writer = mock_writer
        mod._project_store = None

        try:
            text = await cmd_anatomy()
        finally:
            mod._scanner, mod._writer, mod._project_store = old_s, old_w, old_p

        assert "Prometheus Anatomy" in text
        assert "test-brain" in text
        assert "remote inference" in text
        assert "NVIDIA RTX 4090" in text
        assert "GPU:" in text
        assert "VRAM:" in text
        assert "Tailscale:" in text
        assert "test-gpu" in text or "test-brain" in text
        assert "Disk:" in text
        assert "Model:" in text

    async def test_cmd_anatomy_graceful_gpu_unavailable(self) -> None:
        """When GPU detection fails, should show 'remote (stats unavailable)'."""
        import prometheus.tools.builtin.anatomy as mod
        from prometheus.gateway.commands import cmd_anatomy

        old_s, old_w, old_p = mod._scanner, mod._writer, mod._project_store
        mock_scanner = MagicMock()

        state = _sample_state(
            gpu_name=None,
            gpu_vram_total_mb=None,
            gpu_vram_used_mb=None,
            gpu_vram_free_mb=None,
            inference_url="http://192.0.2.1:8080",
        )
        mock_scanner.scan = AsyncMock(return_value=state)
        mod._scanner = mock_scanner
        mod._writer = MagicMock()
        mod._project_store = None

        try:
            text = await cmd_anatomy()
        finally:
            mod._scanner, mod._writer, mod._project_store = old_s, old_w, old_p

        assert "remote (stats unavailable)" in text

    async def test_cmd_anatomy_no_tailscale(self) -> None:
        """When Tailscale is not available, skip that section."""
        import prometheus.tools.builtin.anatomy as mod
        from prometheus.gateway.commands import cmd_anatomy

        old_s, old_w, old_p = mod._scanner, mod._writer, mod._project_store
        mock_scanner = MagicMock()

        state = _sample_state(
            tailscale_ip=None,
            tailscale_peers=[],
            inference_url="http://127.0.0.1:8080",
        )
        mock_scanner.scan = AsyncMock(return_value=state)
        mod._scanner = mock_scanner
        mod._writer = MagicMock()
        mod._project_store = None

        try:
            text = await cmd_anatomy()
        finally:
            mod._scanner, mod._writer, mod._project_store = old_s, old_w, old_p

        assert "Tailscale" not in text


class TestUptime:
    def test_read_uptime(self, tmp_path: Path) -> None:
        import time
        from prometheus.gateway.commands import _read_uptime

        uptime_file = tmp_path / ".daemon_started"
        uptime_file.write_text(str(time.time() - 8100), encoding="utf-8")

        with patch("prometheus.config.paths.get_config_dir", return_value=tmp_path):
            result = _read_uptime()
        assert result == "2h 15m"

    def test_read_uptime_minutes_only(self, tmp_path: Path) -> None:
        import time
        from prometheus.gateway.commands import _read_uptime

        uptime_file = tmp_path / ".daemon_started"
        uptime_file.write_text(str(time.time() - 300), encoding="utf-8")

        with patch("prometheus.config.paths.get_config_dir", return_value=tmp_path):
            result = _read_uptime()
        assert result == "5m"

    def test_read_uptime_missing_file(self, tmp_path: Path) -> None:
        from prometheus.gateway.commands import _read_uptime

        with patch("prometheus.config.paths.get_config_dir", return_value=tmp_path):
            result = _read_uptime()
        assert result is None
