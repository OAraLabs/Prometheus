"""``oara setup`` with no prompts can land on a cloud provider.

Installability survey, 2026-09-01: every cloud choice lived behind an
interactive menu, so ``setup --noninteractive`` on a box with no local
server wrote nothing and exited 2 — exactly the cloud-only stranger's
case. These tests pin the two new ways in (``--provider``, and one cloud
key in the environment) and keep the dead-end rule: a cloud config is
never written without its key.
"""

from __future__ import annotations

import argparse

import pytest
import yaml

from prometheus.cli import init as init_mod
from prometheus.cli.init import run_init
from prometheus.cli.setup import add_setup_arguments, run_setup

_NO_SERVERS = [{
    "name": "ghost", "url": "http://127.0.0.1:9",  # discard port — closed
    "models_path": "/v1/models", "provider": "test",
}]


@pytest.fixture(autouse=True)
def no_cloud_keys(monkeypatch):
    """Start every test with no cloud key in the environment."""
    for key_env, _model, _limit in init_mod._CLOUD_FAST_PROVIDERS.values():
        monkeypatch.delenv(key_env, raising=False)


def _written(tmp_path):
    return yaml.safe_load((tmp_path / "prometheus.yaml").read_text())


class TestExplicitProvider:
    def test_provider_with_key_in_environment_writes_cloud_config(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key-for-tests")
        result = run_init(
            noninteractive=True, target_dir=tmp_path, timeout=0.1,
            candidates=_NO_SERVERS, provider="openai",
        )
        assert result is not None
        cfg = _written(tmp_path)
        assert cfg["model"]["provider"] == "openai"
        assert cfg["model"]["api_key_env"] == "OPENAI_API_KEY"
        assert cfg["model"]["model"] == init_mod._CLOUD_FAST_PROVIDERS["openai"][1]
        assert "base_url" not in cfg["model"]
        assert cfg["web"]["enabled"] is True
        # The daemon does not inherit this shell: the key is copied to the
        # env file, and never into the yaml.
        assert "OPENAI_API_KEY=dummy-openai-key-for-tests" in (tmp_path / "env").read_text()
        assert "sk-test-openai" not in (tmp_path / "prometheus.yaml").read_text()
        assert "copied to" in capsys.readouterr().out

    def test_provider_with_key_only_in_env_file_is_enough(self, tmp_path):
        (tmp_path / "env").write_text("ANTHROPIC_API_KEY=dummy-anthropic-key-from-file\n")
        result = run_init(
            noninteractive=True, target_dir=tmp_path, timeout=0.1,
            candidates=_NO_SERVERS, provider="anthropic",
        )
        assert result is not None
        assert _written(tmp_path)["model"]["provider"] == "anthropic"

    def test_provider_without_a_key_anywhere_writes_nothing(self, tmp_path, capsys):
        result = run_init(
            noninteractive=True, target_dir=tmp_path, timeout=0.1,
            candidates=_NO_SERVERS, provider="openai",
        )
        assert result is None
        assert not (tmp_path / "prometheus.yaml").exists()
        out = capsys.readouterr().out
        assert "OPENAI_API_KEY" in out
        assert "No config was written" in out

    def test_api_key_env_and_model_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_DEEPSEEK", "dummy-deepseek-key")
        result = run_init(
            noninteractive=True, target_dir=tmp_path, timeout=0.1,
            candidates=_NO_SERVERS, provider="deepseek",
            api_key_env="MY_DEEPSEEK", model="deepseek-reasoner",
        )
        assert result is not None
        cfg = _written(tmp_path)
        assert cfg["model"] == {
            "provider": "deepseek", "model": "deepseek-reasoner",
            "api_key_env": "MY_DEEPSEEK",
            "max_tool_iterations": cfg["model"]["max_tool_iterations"],
        }

    def test_explicit_provider_wins_over_a_detected_local_server(self, tmp_path, monkeypatch, capsys):
        """A detected server is a default; --provider is a decision."""
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key-for-tests")
        fake = init_mod.DetectedServer(
            name="llama.cpp", url="http://127.0.0.1:8080", provider="llama_cpp",
            models=["local-model"],
        )
        monkeypatch.setattr(init_mod, "detect_local_servers", lambda **kw: [fake])
        result = run_init(
            noninteractive=True, target_dir=tmp_path, timeout=0.1, provider="openai",
        )
        assert result is not None
        assert _written(tmp_path)["model"]["provider"] == "openai"
        assert "not the detected local server" in capsys.readouterr().out


class TestAutodetectFromEnvironment:
    def test_one_cloud_key_and_no_server_degrades_to_that_provider(self, tmp_path, monkeypatch, capsys):
        """The cloud-only stranger: no GPU, a key exported, one command."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
        result = run_init(
            noninteractive=True, target_dir=tmp_path, timeout=0.1, candidates=_NO_SERVERS,
        )
        assert result is not None
        assert _written(tmp_path)["model"]["provider"] == "anthropic"
        assert "ANTHROPIC_API_KEY is set" in capsys.readouterr().out

    def test_several_keys_is_ambiguous_and_refused_with_a_hint(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
        result = run_init(
            noninteractive=True, target_dir=tmp_path, timeout=0.1, candidates=_NO_SERVERS,
        )
        assert result is None
        assert not (tmp_path / "prometheus.yaml").exists()
        out = capsys.readouterr().out
        assert "--provider" in out and "anthropic" in out and "openai" in out

    def test_no_key_keeps_the_install_instructions_dead_end(self, tmp_path, capsys):
        result = run_init(
            noninteractive=True, target_dir=tmp_path, timeout=0.1, candidates=_NO_SERVERS,
        )
        assert result is None
        assert "ollama.com/install.sh" in capsys.readouterr().out

    def test_a_detected_local_server_still_wins_without_provider(self, tmp_path, monkeypatch):
        """Local-first is unchanged: a key in the environment does not
        override a server that is actually there."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
        fake = init_mod.DetectedServer(
            name="llama.cpp", url="http://127.0.0.1:8080", provider="llama_cpp",
            models=["local-model"],
        )
        monkeypatch.setattr(init_mod, "detect_local_servers", lambda **kw: [fake])
        result = run_init(noninteractive=True, target_dir=tmp_path, timeout=0.1)
        assert result is not None
        assert _written(tmp_path)["model"]["provider"] == "llama_cpp"


class TestCommandSurface:
    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        add_setup_arguments(parser)
        return parser.parse_args(argv)

    def test_provider_flag_implies_the_fast_path_and_exit_codes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key-for-tests")
        args = self._parse(["--provider", "openai", "--target-dir", str(tmp_path), "--timeout", "0.1"])
        monkeypatch.setattr(init_mod, "detect_local_servers", lambda **kw: [])
        assert run_setup(args) == 0
        assert _written(tmp_path)["model"]["provider"] == "openai"

    def test_missing_key_is_exit_2(self, tmp_path, monkeypatch):
        args = self._parse(["--provider", "openai", "--target-dir", str(tmp_path), "--timeout", "0.1"])
        monkeypatch.setattr(init_mod, "detect_local_servers", lambda **kw: [])
        assert run_setup(args) == 2
        assert not (tmp_path / "prometheus.yaml").exists()

    def test_unknown_provider_is_rejected_by_argparse(self):
        with pytest.raises(SystemExit):
            self._parse(["--provider", "does_not_exist"])

    def test_flags_do_not_collide_with_the_global_model_and_provider(self):
        args = self._parse(["--provider", "qwen", "--model", "qwen3.7-max", "--api-key-env", "K"])
        assert args.setup_provider == "qwen"
        assert args.setup_model == "qwen3.7-max"
        assert args.api_key_env == "K"
