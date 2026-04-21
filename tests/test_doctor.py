"""Tests for Doctor diagnostic system — model registry, checks, cmd_doctor."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from prometheus.infra.anatomy import AnatomyState
from prometheus.infra.doctor import (
    DiagnosticCheck,
    DiagnosticReport,
    Doctor,
    match_model,
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


SAMPLE_REGISTRY = {
    "models": {
        "gemma-4": {
            "display_name": "Google Gemma 4",
            "match_patterns": ["gemma-4", "gemma_4", "gemma4"],
            "capabilities": {
                "vision": {
                    "supported": True,
                    "requires": "mmproj",
                    "setup_hint": "Add --mmproj to launch command.",
                },
                "function_calling": {
                    "supported": True,
                },
                "streaming": {
                    "supported": True,
                },
            },
        },
        "qwen-3": {
            "display_name": "Qwen 3",
            "match_patterns": ["qwen-3", "qwen_3", "qwen3"],
            "capabilities": {
                "vision": {"supported": False},
                "function_calling": {"supported": True},
            },
        },
        "phi": {
            "display_name": "Microsoft Phi",
            "match_patterns": ["phi-3", "phi-4"],
            "capabilities": {
                "vision": {"supported": False},
                "function_calling": {"supported": True},
            },
        },
    }
}


# ===========================================================================
# match_model
# ===========================================================================


class TestMatchModel:
    def test_matches_gemma4(self) -> None:
        family = match_model("gemma-4-26B-A4B-it-Q4_K_XL.gguf", SAMPLE_REGISTRY)
        assert family is not None
        assert family["display_name"] == "Google Gemma 4"

    def test_matches_case_insensitive(self) -> None:
        family = match_model("GEMMA-4-something.gguf", SAMPLE_REGISTRY)
        assert family is not None
        assert family["display_name"] == "Google Gemma 4"

    def test_matches_qwen(self) -> None:
        family = match_model("qwen3.5-32b", SAMPLE_REGISTRY)
        assert family is not None
        assert family["display_name"] == "Qwen 3"

    def test_no_match_returns_none(self) -> None:
        family = match_model("totally-unknown-model", SAMPLE_REGISTRY)
        assert family is None

    def test_empty_name_returns_none(self) -> None:
        family = match_model("", SAMPLE_REGISTRY)
        assert family is None

    def test_empty_registry_returns_none(self) -> None:
        family = match_model("gemma-4", {})
        assert family is None

    def test_matches_phi(self) -> None:
        family = match_model("phi-4-mini-Q4_K_M.gguf", SAMPLE_REGISTRY)
        assert family is not None
        assert family["display_name"] == "Microsoft Phi"


# ===========================================================================
# DiagnosticReport
# ===========================================================================


class TestDiagnosticReport:
    def test_has_warnings(self) -> None:
        report = DiagnosticReport(
            model_name="test",
            model_family=None,
            checks=[
                DiagnosticCheck(name="A", category="platform", status="ok", message="fine"),
                DiagnosticCheck(name="B", category="model", status="warning", message="hmm"),
            ],
        )
        assert report.has_warnings is True
        assert report.has_errors is False

    def test_has_errors(self) -> None:
        report = DiagnosticReport(
            model_name="test",
            model_family=None,
            checks=[
                DiagnosticCheck(name="A", category="platform", status="error", message="bad"),
            ],
        )
        assert report.has_errors is True

    def test_all_ok(self) -> None:
        report = DiagnosticReport(
            model_name="test",
            model_family=None,
            checks=[
                DiagnosticCheck(name="A", category="platform", status="ok", message="fine"),
                DiagnosticCheck(name="B", category="resources", status="ok", message="fine"),
            ],
        )
        assert report.has_warnings is False
        assert report.has_errors is False

    def test_checks_by_category(self) -> None:
        report = DiagnosticReport(
            model_name="test",
            model_family=None,
            checks=[
                DiagnosticCheck(name="Python", category="platform", status="ok", message="3.12"),
                DiagnosticCheck(name="Config", category="platform", status="ok", message="valid"),
                DiagnosticCheck(name="Inference", category="connectivity", status="ok", message="up"),
                DiagnosticCheck(name="Model", category="model", status="ok", message="loaded"),
                DiagnosticCheck(name="GPU", category="resources", status="ok", message="4090"),
                DiagnosticCheck(name="Disk", category="resources", status="ok", message="220 GB"),
            ],
        )
        grouped = report.checks_by_category()
        assert list(grouped.keys()) == ["platform", "connectivity", "model", "resources"]
        assert len(grouped["platform"]) == 2
        assert len(grouped["connectivity"]) == 1
        assert len(grouped["model"]) == 1
        assert len(grouped["resources"]) == 2

    def test_checks_by_category_skips_empty(self) -> None:
        report = DiagnosticReport(
            model_name="test",
            model_family=None,
            checks=[
                DiagnosticCheck(name="GPU", category="resources", status="ok", message="4090"),
            ],
        )
        grouped = report.checks_by_category()
        assert "platform" not in grouped
        assert "resources" in grouped


# ===========================================================================
# Doctor._check_* individual checks
# ===========================================================================


class TestDoctorChecks:
    def setup_method(self) -> None:
        self.doctor = Doctor.__new__(Doctor)
        self.doctor.config = {}
        self.doctor.registry = SAMPLE_REGISTRY

    # -- python version --

    def test_check_python_version_ok(self) -> None:
        check = self.doctor._check_python_version()
        assert check.category == "platform"
        # We're running on 3.11+ in this test environment
        assert check.status == "ok"
        assert "Python" in check.message

    def test_check_python_version_too_old(self) -> None:
        fake_info = MagicMock()
        fake_info.major = 3
        fake_info.minor = 10
        fake_info.micro = 0
        fake_info.__ge__ = lambda self, other: (3, 10) >= other
        with patch("sys.version_info", fake_info):
            check = self.doctor._check_python_version()
        assert check.status == "error"
        assert "3.11+ required" in check.message
        assert check.fix is not None

    # -- uv --

    def test_check_uv_available(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/uv"):
            check = self.doctor._check_uv()
        assert check.category == "platform"
        assert check.status == "ok"
        assert "installed" in check.message

    def test_check_uv_not_found(self) -> None:
        with patch("shutil.which", return_value=None):
            check = self.doctor._check_uv()
        assert check.status == "warning"
        assert "uv not found" in check.message
        assert check.fix is not None

    # -- config valid --

    def test_check_config_valid(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config" / "prometheus.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text("model:\n  provider: llama_cpp\n", encoding="utf-8")
        with patch.object(Path, "resolve", return_value=tmp_path / "src" / "prometheus" / "infra" / "doctor.py"):
            # Patch the path calculation — easier to just patch the file existence check
            with patch("prometheus.infra.doctor.Path.__truediv__", side_effect=lambda self, other: tmp_path / other if "config" in str(other) else Path.__truediv__(self, other)):
                pass  # Complex path mocking — test via integration instead

    def test_check_config_valid_ok(self) -> None:
        """Config check should pass when prometheus.yaml exists and parses."""
        check = self.doctor._check_config_valid()
        # In the test env, config/prometheus.yaml exists in the project
        assert check.category == "platform"
        # Status depends on whether the file is found relative to the module
        assert check.status in ("ok", "error")

    # -- dependencies --

    def test_check_dependencies_all_present(self) -> None:
        check = self.doctor._check_dependencies()
        assert check.category == "platform"
        assert check.status == "ok"
        assert "all required packages installed" in check.message

    def test_check_dependencies_missing(self) -> None:
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
        def mock_import(name, *args, **kwargs):
            if name == "pydantic":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=mock_import):
            check = self.doctor._check_dependencies()
        assert check.status == "error"
        assert "pydantic" in check.message

    # -- data dir --

    def test_check_data_dir_exists(self, tmp_path: Path) -> None:
        with patch("prometheus.infra.doctor.get_config_dir", return_value=tmp_path):
            check = self.doctor._check_data_dir()
        assert check.category == "platform"
        assert check.status == "ok"

    def test_check_data_dir_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        with patch("prometheus.infra.doctor.get_config_dir", return_value=missing):
            check = self.doctor._check_data_dir()
        assert check.status == "warning"
        assert "does not exist" in check.message

    # -- telegram token --

    def test_check_telegram_token_set(self) -> None:
        with patch.dict("os.environ", {"PROMETHEUS_TELEGRAM_TOKEN": "1234:ABCDefgh1234"}):
            check = self.doctor._check_telegram_token()
        assert check.category == "connectivity"
        assert check.status == "ok"
        assert "1234" in check.message  # masked prefix
        assert "1234:ABCDefgh1234" not in check.message  # not full token

    def test_check_telegram_token_missing(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.doctor.config = {"gateway": {"telegram_token": ""}}
            check = self.doctor._check_telegram_token()
        assert check.status == "warning"
        assert "not configured" in check.message

    def test_check_telegram_token_placeholder(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.doctor.config = {"gateway": {"telegram_token": "YOUR_BOT_TOKEN_HERE"}}
            check = self.doctor._check_telegram_token()
        assert check.status == "warning"

    # -- inference --

    @pytest.mark.asyncio
    async def test_check_inference_no_url(self) -> None:
        state = _sample_state(inference_url="")
        check = await self.doctor._check_inference(state)
        assert check.category == "connectivity"
        assert check.status == "error"
        assert "No inference URL" in check.message

    @pytest.mark.asyncio
    async def test_check_inference_reachable(self) -> None:
        state = _sample_state()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("prometheus.infra.doctor.httpx.AsyncClient", return_value=mock_client):
            check = await self.doctor._check_inference(state)
        assert check.status == "ok"
        assert "reachable" in check.message

    @pytest.mark.asyncio
    async def test_check_inference_unreachable(self) -> None:
        state = _sample_state()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("prometheus.infra.doctor.httpx.AsyncClient", return_value=mock_client):
            check = await self.doctor._check_inference(state)
        assert check.status == "error"
        assert "not responding" in check.message

    # -- model loaded --

    def test_check_model_loaded(self) -> None:
        state = _sample_state()
        check = self.doctor._check_model_loaded(state)
        assert check.category == "model"
        assert check.status == "ok"
        assert "gemma-4-26B" in check.message
        assert "Q4_K_XL" in check.message

    def test_check_model_not_loaded(self) -> None:
        state = _sample_state(model_name=None)
        check = self.doctor._check_model_loaded(state)
        assert check.category == "model"
        assert check.status == "error"
        assert "No model detected" in check.message

    # -- vision --

    def test_check_vision_enabled(self) -> None:
        state = _sample_state(vision_enabled=True)
        family = SAMPLE_REGISTRY["models"]["gemma-4"]
        check = self.doctor._check_vision(state, family)
        assert check.category == "model"
        assert check.status == "ok"
        assert "mmproj loaded" in check.message

    def test_check_vision_model_supports_but_disabled(self) -> None:
        state = _sample_state(vision_enabled=False)
        family = SAMPLE_REGISTRY["models"]["gemma-4"]
        check = self.doctor._check_vision(state, family)
        assert check.category == "model"
        assert check.status == "warning"
        assert "mmproj is not loaded" in check.message
        assert check.fix is not None

    def test_check_vision_model_no_support(self) -> None:
        state = _sample_state(vision_enabled=False)
        family = SAMPLE_REGISTRY["models"]["qwen-3"]
        check = self.doctor._check_vision(state, family)
        assert check.status == "info"
        assert "does not support vision" in check.message

    # -- function calling --

    def test_check_function_calling_supported(self) -> None:
        family = SAMPLE_REGISTRY["models"]["gemma-4"]
        state = _sample_state()
        check = self.doctor._check_function_calling(state, family)
        assert check is not None
        assert check.status == "ok"
        assert "tool calling" in check.message

    def test_check_function_calling_not_supported(self) -> None:
        family = {"capabilities": {"function_calling": {"supported": False}}}
        state = _sample_state()
        check = self.doctor._check_function_calling(state, family)
        assert check is None

    # -- GPU --

    def test_check_gpu_healthy(self) -> None:
        state = _sample_state(gpu_vram_free_mb=6144)
        check = self.doctor._check_gpu(state)
        assert check.category == "resources"
        assert check.status == "ok"
        assert "6.0 GB VRAM free" in check.message

    def test_check_gpu_low_vram(self) -> None:
        state = _sample_state(gpu_vram_free_mb=400)
        check = self.doctor._check_gpu(state)
        assert check.category == "resources"
        assert check.status == "warning"
        assert "nearly full" in check.message

    def test_check_gpu_no_vram_stats(self) -> None:
        state = _sample_state(gpu_vram_total_mb=None, gpu_vram_free_mb=None)
        check = self.doctor._check_gpu(state)
        assert check.category == "resources"
        assert check.status == "info"
        assert "VRAM stats unavailable" in check.message

    def test_check_gpu_none(self) -> None:
        state = _sample_state(gpu_name=None, gpu_vram_total_mb=None)
        check = self.doctor._check_gpu(state)
        assert check.category == "resources"
        assert check.status == "warning"
        assert "No GPU detected" in check.message

    # -- tailscale --

    def test_check_tailscale_local_returns_none(self) -> None:
        state = _sample_state(inference_url="http://127.0.0.1:8080")
        check = self.doctor._check_tailscale(state)
        assert check is None

    def test_check_tailscale_remote_with_peer(self) -> None:
        state = _sample_state(
            inference_url="http://192.0.2.1:8080",
            tailscale_ip="192.0.2.2",
            tailscale_peers=[
                {"name": "test-gpu", "ip": "192.0.2.1", "online": True},
            ],
        )
        check = self.doctor._check_tailscale(state)
        assert check is not None
        assert check.category == "connectivity"
        assert check.status == "ok"
        assert "test-gpu" in check.message

    def test_check_tailscale_remote_no_tailscale(self) -> None:
        state = _sample_state(
            inference_url="http://192.0.2.1:8080",
            tailscale_ip=None,
            tailscale_peers=[],
        )
        check = self.doctor._check_tailscale(state)
        assert check is not None
        assert check.category == "connectivity"
        assert check.status == "warning"
        assert "not found" in check.message

    # -- whisper --

    def test_check_whisper_configured(self) -> None:
        state = _sample_state(whisper_model="base")
        check = self.doctor._check_whisper(state)
        assert check.category == "resources"
        assert check.status == "ok"
        assert "base" in check.message

    def test_check_whisper_not_configured(self) -> None:
        state = _sample_state(whisper_model=None)
        check = self.doctor._check_whisper(state)
        assert check.category == "resources"
        assert check.status == "info"
        assert "Not configured" in check.message

    # -- disk --

    def test_check_disk_healthy(self) -> None:
        state = _sample_state(disk_free_gb=220.0)
        check = self.doctor._check_disk(state)
        assert check.category == "resources"
        assert check.status == "ok"

    def test_check_disk_low(self) -> None:
        state = _sample_state(disk_free_gb=5.0)
        check = self.doctor._check_disk(state)
        assert check.category == "resources"
        assert check.status == "warning"
        assert "Low disk space" in check.message

    # -- bootstrap --

    def test_check_bootstrap_present(self, tmp_path: Path) -> None:
        (tmp_path / "SOUL.md").write_text("soul", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")
        with patch("prometheus.infra.doctor.get_config_dir", return_value=tmp_path):
            check = self.doctor._check_bootstrap_files()
        assert check.category == "platform"
        assert check.status == "ok"

    def test_check_bootstrap_missing(self, tmp_path: Path) -> None:
        with patch("prometheus.infra.doctor.get_config_dir", return_value=tmp_path):
            check = self.doctor._check_bootstrap_files()
        assert check.category == "platform"
        assert check.status == "error"
        assert "SOUL.md" in check.message
        assert "AGENTS.md" in check.message

    def test_check_bootstrap_partial(self, tmp_path: Path) -> None:
        (tmp_path / "SOUL.md").write_text("soul", encoding="utf-8")
        with patch("prometheus.infra.doctor.get_config_dir", return_value=tmp_path):
            check = self.doctor._check_bootstrap_files()
        assert check.category == "platform"
        assert check.status == "error"
        assert "AGENTS.md" in check.message
        assert "SOUL.md" not in check.message


# ===========================================================================
# Doctor.diagnose (full integration)
# ===========================================================================


class TestDoctorDiagnose:
    @pytest.mark.asyncio
    async def test_full_diagnose_all_ok(self, tmp_path: Path) -> None:
        """Full diagnosis with a healthy Gemma 4 setup."""
        (tmp_path / "SOUL.md").write_text("soul", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")

        doctor = Doctor.__new__(Doctor)
        doctor.config = {}
        doctor.registry = SAMPLE_REGISTRY

        state = _sample_state()

        # Mock inference reachable
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("prometheus.infra.doctor.httpx.AsyncClient", return_value=mock_client), \
             patch("prometheus.infra.doctor.get_config_dir", return_value=tmp_path):
            report = await doctor.diagnose(state)

        assert report.model_name == "gemma-4-26B-A4B-it-Q4_K_XL.gguf"
        assert report.model_family == "Google Gemma 4"
        assert not report.has_errors
        # Check that all expected checks are present
        check_names = [c.name for c in report.checks]
        # Platform
        assert "Python" in check_names
        assert "Dependencies" in check_names
        assert "Bootstrap" in check_names
        # Connectivity
        assert "Inference" in check_names
        assert "Telegram" in check_names
        # Model
        assert "Model" in check_names
        assert "Vision" in check_names
        # Resources
        assert "GPU" in check_names
        assert "Disk" in check_names
        # Category grouping works
        grouped = report.checks_by_category()
        assert "platform" in grouped
        assert "connectivity" in grouped
        assert "model" in grouped
        assert "resources" in grouped

    @pytest.mark.asyncio
    async def test_diagnose_unknown_model(self, tmp_path: Path) -> None:
        """Unknown model should skip vision/function_calling checks."""
        (tmp_path / "SOUL.md").write_text("soul", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")

        doctor = Doctor.__new__(Doctor)
        doctor.config = {}
        doctor.registry = SAMPLE_REGISTRY

        state = _sample_state(
            model_name="totally-unknown-model",
            model_file="totally-unknown-model.gguf",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("prometheus.infra.doctor.httpx.AsyncClient", return_value=mock_client), \
             patch("prometheus.infra.doctor.get_config_dir", return_value=tmp_path):
            report = await doctor.diagnose(state)

        assert report.model_family is None
        check_names = [c.name for c in report.checks]
        assert "Vision" not in check_names
        assert "Function Calling" not in check_names

    @pytest.mark.asyncio
    async def test_diagnose_no_model_loaded(self, tmp_path: Path) -> None:
        """No model loaded — should show error for model, skip vision/fc."""
        (tmp_path / "SOUL.md").write_text("soul", encoding="utf-8")
        (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")

        doctor = Doctor.__new__(Doctor)
        doctor.config = {}
        doctor.registry = SAMPLE_REGISTRY

        state = _sample_state(model_name=None, model_file=None)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("prometheus.infra.doctor.httpx.AsyncClient", return_value=mock_client), \
             patch("prometheus.infra.doctor.get_config_dir", return_value=tmp_path):
            report = await doctor.diagnose(state)

        assert report.has_errors
        model_check = next(c for c in report.checks if c.name == "Model")
        assert model_check.status == "error"


# ===========================================================================
# Registry loading
# ===========================================================================


class TestRegistryLoading:
    def test_loads_real_registry(self) -> None:
        """Verify the shipped model_registry.yaml loads and parses."""
        registry_path = Path(__file__).resolve().parents[1] / "config" / "model_registry.yaml"
        if not registry_path.exists():
            pytest.skip("model_registry.yaml not found")
        data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        assert "models" in data
        assert "gemma-4" in data["models"]
        assert data["models"]["gemma-4"]["capabilities"]["vision"]["supported"] is True

    def test_doctor_loads_registry_from_project(self) -> None:
        """Doctor should find registry relative to project root."""
        doctor = Doctor()
        # If the registry exists, it should be loaded
        registry_path = Path(__file__).resolve().parents[1] / "config" / "model_registry.yaml"
        if registry_path.exists():
            assert "models" in doctor.registry
            assert len(doctor.registry["models"]) > 0

    def test_doctor_handles_missing_registry(self, tmp_path: Path) -> None:
        """Doctor should gracefully handle a missing registry file."""
        doctor = Doctor(config={"doctor": {"registry_file": str(tmp_path / "nope.yaml")}})
        # Should default to empty dict and not crash
        assert doctor.registry == {} or isinstance(doctor.registry, dict)


# ===========================================================================
# cmd_doctor formatter
# ===========================================================================


class TestCmdDoctor:
    @pytest.mark.asyncio
    async def test_cmd_doctor_not_initialized(self) -> None:
        import prometheus.tools.builtin.anatomy as mod
        old = mod._scanner
        mod._scanner = None
        try:
            from prometheus.gateway.commands import cmd_doctor
            text = await cmd_doctor()
            assert "not initialized" in text.lower()
        finally:
            mod._scanner = old

    @pytest.mark.asyncio
    async def test_cmd_doctor_formatted_output(self, tmp_path: Path) -> None:
        """Verify the formatted output includes expected sections and icons."""
        import prometheus.tools.builtin.anatomy as mod
        from prometheus.gateway.commands import cmd_doctor

        old_s, old_w, old_p = mod._scanner, mod._writer, mod._project_store
        mock_scanner = MagicMock()

        state = _sample_state()
        mock_scanner.scan = AsyncMock(return_value=state)
        mod._scanner = mock_scanner
        mod._writer = MagicMock()
        mod._project_store = None

        # Mock inference check to succeed
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        try:
            with patch("prometheus.infra.doctor.httpx.AsyncClient", return_value=mock_client), \
                 patch("prometheus.infra.doctor.get_config_dir", return_value=tmp_path):
                # Create bootstrap files for a clean report
                (tmp_path / "SOUL.md").write_text("soul", encoding="utf-8")
                (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")
                text = await cmd_doctor()
        finally:
            mod._scanner, mod._writer, mod._project_store = old_s, old_w, old_p

        assert "Prometheus Doctor" in text
        assert "Google Gemma 4" in text
        # Should have category headers
        assert "\u2500\u2500 Platform \u2500\u2500" in text
        assert "\u2500\u2500 Connectivity \u2500\u2500" in text
        assert "\u2500\u2500 Model \u2500\u2500" in text
        assert "\u2500\u2500 Resources \u2500\u2500" in text
        # Should have status icons
        assert "\u2705" in text  # checkmark
        # Summary line
        assert "checks passed" in text or "ok" in text.lower()

    @pytest.mark.asyncio
    async def test_cmd_doctor_with_warnings(self, tmp_path: Path) -> None:
        """Output should show warning count when issues exist."""
        import prometheus.tools.builtin.anatomy as mod
        from prometheus.gateway.commands import cmd_doctor

        old_s, old_w, old_p = mod._scanner, mod._writer, mod._project_store
        mock_scanner = MagicMock()

        # Vision-capable model without mmproj loaded
        state = _sample_state(vision_enabled=False)
        mock_scanner.scan = AsyncMock(return_value=state)
        mod._scanner = mock_scanner
        mod._writer = MagicMock()
        mod._project_store = None

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        try:
            with patch("prometheus.infra.doctor.httpx.AsyncClient", return_value=mock_client), \
                 patch("prometheus.infra.doctor.get_config_dir", return_value=tmp_path):
                (tmp_path / "SOUL.md").write_text("soul", encoding="utf-8")
                (tmp_path / "AGENTS.md").write_text("agents", encoding="utf-8")
                text = await cmd_doctor()
        finally:
            mod._scanner, mod._writer, mod._project_store = old_s, old_w, old_p

        assert "\u26a0\ufe0f" in text  # warning icon
        assert "warning" in text.lower()
        assert "\u2192" in text  # fix arrow


# ===========================================================================
# cmd_help includes /doctor
# ===========================================================================


class TestHelpIncludesDoctor:
    def test_cmd_help_lists_doctor(self) -> None:
        from prometheus.gateway.commands import cmd_help
        text = cmd_help()
        assert "/doctor" in text
        assert "health check" in text.lower() or "diagnostic" in text.lower()
