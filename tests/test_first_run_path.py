"""The first-run path — audit item 4.

Five defects, each of which broke `oara setup` (or the boot that follows it)
for someone who had never run Prometheus before, and none of which any test
could see:

a. ``templates/`` did not ship in the wheel, so identity generation raised
   ``FileNotFoundError`` on every pip install;
b. the setup detector wrote ``provider: lm_studio`` / ``vllm``, names the
   provider registry did not know — a detected LM Studio produced a config
   whose first boot died;
c. the wizard offered "leave blank to allow all users" for the Telegram
   allowlist, an outcome the daemon stopped delivering when it began
   REFUSING an empty allowlist;
d. ``oara doctor`` read cloud keys from ``os.environ`` only, not the env file
   the daemon loads — so a correct install reported a missing key;
e. ``_detect_gpu`` caught only ``FileNotFoundError``; a second GPU made
   ``int()`` raise and aborted setup.

The through-line (§2d): every one of these was invisible because the tests
asserted the developer's container — this checkout, this shell — rather than
what a stranger receives.
"""

from __future__ import annotations

import subprocess

import pytest

from prometheus.cli import init as init_mod
from prometheus.cli import generate_identity as gi
from prometheus.providers.registry import ProviderRegistry


# ── (b) every provider setup can write must be constructible ────────

class TestSetupWritesAConstructibleConfig:
    """The contract test_setup_deadends.py states — "no setup path may write
    a config that is known to be broken" — applied to the ONE field that
    decides whether the daemon boots at all."""

    def test_every_detected_provider_is_known_to_the_registry(self):
        detected = sorted({s["provider"] for s in init_mod.KNOWN_LOCAL_SERVERS})
        unknown = [p for p in detected
                   if p not in ProviderRegistry.list_providers()]
        assert not unknown, (
            f"`oara setup` detects {unknown} and writes the name straight into "
            f"model.provider, but ProviderRegistry.create() raises on it. "
            f"Setup reports success and the first boot dies."
        )

    @pytest.mark.parametrize("provider,url", [
        ("llama_cpp", "http://localhost:8080"),
        ("ollama", "http://localhost:11434"),
        ("lm_studio", "http://localhost:1234"),
        ("vllm", "http://localhost:8000"),
    ])
    def test_the_written_config_constructs_a_provider(self, provider, url):
        """End-to-end on the real writer, not a hand-built dict."""
        server = init_mod.DetectedServer(
            name=provider, url=url, provider=provider, models=["a-model"],
        )
        model_cfg = init_mod._default_config(server, None)["model"]
        assert ProviderRegistry.create(model_cfg) is not None


class TestLocalOpenAICompatProviders:
    def test_lm_studio_needs_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        p = ProviderRegistry.create({"provider": "lm_studio"})
        assert p.provider_name == "lm_studio"
        assert p._base_url == "http://localhost:1234/v1"

    def test_vllm_needs_no_api_key(self):
        p = ProviderRegistry.create({"provider": "vllm"})
        assert p.provider_name == "vllm"
        assert p._base_url == "http://localhost:8000/v1"

    def test_a_placeholder_bearer_is_sent_not_an_empty_one(self):
        """OpenAICompatProvider REFUSES an empty bearer, so "no key" cannot
        mean "no string" — it means a value both servers ignore."""
        p = ProviderRegistry.create({"provider": "lm_studio"})
        assert p._resolve_bearer()

    def test_neither_is_counted_as_cloud(self):
        """is_cloud drives cost accounting and teacher/student corpus
        labelling. A localhost server filed as cloud poisons both."""
        assert not ProviderRegistry.is_cloud("lm_studio")
        assert not ProviderRegistry.is_cloud("vllm")

    def test_a_configured_key_still_wins(self, monkeypatch):
        monkeypatch.setenv("VLLM_KEY", "secret-vllm")
        p = ProviderRegistry.create(
            {"provider": "vllm", "api_key_env": "VLLM_KEY"})
        assert p._resolve_bearer() == "secret-vllm"

    def test_a_configured_key_that_is_unset_still_raises(self, monkeypatch):
        """Asking for a variable and getting silence is a mistake worth
        hearing about — the fallback covers UNCONFIGURED, not misconfigured."""
        monkeypatch.delenv("VLLM_KEY", raising=False)
        with pytest.raises(ValueError, match="VLLM_KEY"):
            ProviderRegistry.create(
                {"provider": "vllm", "api_key_env": "VLLM_KEY"})

    def test_base_url_env_override(self, monkeypatch):
        monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://gpu.local:1234/v1")
        p = ProviderRegistry.create({"provider": "lm_studio"})
        assert p._base_url == "http://gpu.local:1234/v1"

    def test_the_local_endpoints_are_not_in_the_cloud_catalogue(self):
        """web/server.py publishes CLOUD_DEFAULTS as /api/models' cloud list.
        A localhost entry there would advertise a provider nobody can use."""
        from prometheus.providers.registry import CLOUD_DEFAULTS

        assert "lm_studio" not in CLOUD_DEFAULTS
        assert "vllm" not in CLOUD_DEFAULTS


# ── (e) GPU detection cannot abort an install ───────────────────────

class _FakeCompleted:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


class TestGpuDetectionNeverRaises:
    def test_two_gpus_do_not_raise(self, monkeypatch):
        """The original defect: one line per GPU, split on ',' across the
        whole blob, int() on 'NVIDIA GeForce RTX 4090\\n24564'."""
        out = "NVIDIA GeForce RTX 3090 Ti, 24564\nNVIDIA GeForce RTX 4090, 24564\n"
        monkeypatch.setattr(gi.subprocess, "run",
                            lambda *a, **k: _FakeCompleted(out))
        assert gi._detect_gpu() == "NVIDIA GeForce RTX 3090 Ti (23GB)"

    def test_one_gpu_still_reports_vram(self, monkeypatch):
        monkeypatch.setattr(
            gi.subprocess, "run",
            lambda *a, **k: _FakeCompleted("NVIDIA GeForce RTX 3090 Ti, 24564\n"))
        assert gi._detect_gpu() == "NVIDIA GeForce RTX 3090 Ti (23GB)"

    def test_unusable_memory_field_degrades_to_the_name(self, monkeypatch):
        monkeypatch.setattr(
            gi.subprocess, "run",
            lambda *a, **k: _FakeCompleted("NVIDIA T4, [N/A]\n"))
        assert gi._detect_gpu() == "NVIDIA T4"

    def test_a_hung_driver_does_not_abort_setup(self, monkeypatch):
        def _boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)
        monkeypatch.setattr(gi.subprocess, "run", _boom)
        monkeypatch.setattr(gi.platform, "system", lambda: "Linux")
        assert gi._detect_gpu() is None

    def test_an_unexecutable_nvidia_smi_does_not_abort_setup(self, monkeypatch):
        def _boom(*a, **k):
            raise PermissionError(13, "Permission denied")
        monkeypatch.setattr(gi.subprocess, "run", _boom)
        monkeypatch.setattr(gi.platform, "system", lambda: "Linux")
        assert gi._detect_gpu() is None

    def test_detect_hardware_survives_a_broken_nvidia_smi(self, monkeypatch):
        """The caller `oara setup` actually runs."""
        def _boom(*a, **k):
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)
        monkeypatch.setattr(gi.subprocess, "run", _boom)
        hw = gi.detect_hardware()
        assert hw["has_gpu"] is False


# ── (c) the Telegram allowlist prompt tells the truth ───────────────

class TestTelegramAllowlistPrompt:
    def _wizard(self, monkeypatch, answers):
        from prometheus import setup_wizard as wizard_mod

        it = iter(answers)
        monkeypatch.setattr(wizard_mod, "_input",
                            lambda label, default="": next(it, ""))
        monkeypatch.setattr(
            wizard_mod.SetupWizard, "_test_telegram_token",
            lambda self, token: "testbot")
        w = wizard_mod.SetupWizard()
        return w

    def test_a_chat_id_is_recorded(self, monkeypatch):
        w = self._wizard(monkeypatch, ["1234:token", "987654321"])
        w._setup_telegram()
        assert w._telegram_chat_ids == [987654321]

    def test_blank_is_not_described_as_allow_all_users(self, monkeypatch, capsys):
        """The daemon REFUSES an empty allowlist. A prompt that offers it as
        "allow all users" describes an outcome that has not existed since."""
        w = self._wizard(monkeypatch, ["1234:token", "", ""])
        w._setup_telegram()
        out = capsys.readouterr().out
        assert "allow all users" not in out.lower()
        assert w._telegram_chat_ids == []
        assert "REFUSE" in out or "will not start" in out, (
            "a blank allowlist must say the gateway will not start — "
            "otherwise setup reports Telegram configured and boot disagrees"
        )

    def test_a_non_numeric_id_is_reprompted_not_silently_dropped(
        self, monkeypatch, capsys,
    ):
        w = self._wizard(monkeypatch, ["1234:token", "@myhandle", "555"])
        w._setup_telegram()
        assert w._telegram_chat_ids == [555]

    def test_the_daemon_agrees_with_what_the_prompt_promises(self):
        """Bind the prompt's claim to the code that decides. If the daemon
        ever starts accepting an empty allowlist again, this fails and the
        wizard text gets revisited rather than drifting a second time."""
        from prometheus.daemon import telegram_gateway_decision

        start, reason = telegram_gateway_decision(
            {"telegram_enabled": True, "allowed_chat_ids": []}, "1234:token")
        assert start is False
        assert reason and "allowed_chat_ids" in reason


# ── (d) doctor answers the same question the daemon answers ─────────

class TestDoctorReadsTheEnvFile:
    """`oara doctor` is the eternal support answer, so a doctor that
    disagrees with the daemon costs more than no doctor at all.

    The daemon gets its cloud key from the env file — systemd loads it via
    ``EnvironmentFile=`` and ``oara daemon`` calls ``load_env_file()``. Doctor
    read ``os.environ`` alone, so running it from a plain shell reported a
    correctly-installed key as missing, at exactly the first-run moment the
    command exists for.
    """

    def _check(self, monkeypatch, tmp_path, env_text, model_cfg):
        from prometheus.cli import doctor as doctor_mod

        env_path = tmp_path / "env"
        env_path.write_text(env_text)
        monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(env_path))
        reach, _model = doctor_mod.check_inference({"model": model_cfg})
        return reach

    def test_key_only_in_the_env_file_is_found(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        reach = self._check(
            monkeypatch, tmp_path, "ANTHROPIC_API_KEY=sk-ant-from-file\n",
            {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY",
             "model": "claude-haiku-4-5-20251001"},
        )
        assert reach.status == "ok", reach.message
        assert "env file" in reach.message

    def test_api_key_env_may_be_absent_the_registry_has_a_default(
        self, monkeypatch, tmp_path,
    ):
        """`_cloud_default_config` writes api_key_env, but a hand-written
        config need not: the registry falls back to the provider's
        default_env. Doctor demanded the key and reported
        '<api_key_env unset>' on configs the daemon started fine."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        reach = self._check(
            monkeypatch, tmp_path, "ANTHROPIC_API_KEY=sk-ant-default-var\n",
            {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        )
        assert reach.status == "ok", reach.message
        assert "<api_key_env unset>" not in reach.message

    def test_the_environment_still_wins_and_is_named(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
        reach = self._check(
            monkeypatch, tmp_path, "", 
            {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY",
             "model": "claude-haiku-4-5-20251001"},
        )
        assert reach.status == "ok"
        assert "environment" in reach.message

    def test_a_key_that_is_nowhere_is_still_an_error(self, monkeypatch, tmp_path):
        """The check must not become unfalsifiable in the process."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        reach = self._check(
            monkeypatch, tmp_path, "# nothing here\n",
            {"provider": "anthropic", "api_key_env": "ANTHROPIC_API_KEY",
             "model": "claude-haiku-4-5-20251001"},
        )
        assert reach.status == "error"

    def test_doctor_and_the_registry_agree_on_where_the_key_is(
        self, monkeypatch, tmp_path,
    ):
        """The claim that matters: doctor says OK exactly when the daemon
        can actually build the provider."""
        from prometheus.config.env_file import load_env_file

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        model_cfg = {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}
        reach = self._check(
            monkeypatch, tmp_path, "ANTHROPIC_API_KEY=sk-ant-agreement\n", model_cfg)
        assert reach.status == "ok"

        # The daemon's own step: load the env file, then build.
        load_env_file()
        assert ProviderRegistry.create(model_cfg) is not None
