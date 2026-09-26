"""Coding mode builds its adapter for the model the server actually serves (WP-X.28, PR 2c).

THE DEFECT
----------
`oara code` (the subprocess the daemon launches per coding task) built its
adapter from the config's model HINT alone. That hint is blank on the shipped
template and on the deployed daemon, so `_get_adapter_tier(provider, "")`
matched no registry family and every coding run ran at tier `full` — its
tools written into the prompt and its replies read as text — while a chat
turn on the same server ran at `light` with native tools. The `coding_run`
golden recorded it: every coding request carried `model: ""`.

The daemon (`daemon.py`, boot) and the interactive CLI (`__main__`, before
`create_adapter`) already ask llama-server which model it serves. The coding
path now asks the same way, and the name reaches the adapter, the session and
the telemetry rows. On main the adapter is built for `""`; these tests fail
there.
"""

from __future__ import annotations

import pytest

from prometheus import __main__ as m

SERVED = "/models-root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"


class _StubSandbox:
    def __init__(self, tmp_path):
        self.root = tmp_path / "clone"


def _args(tmp_path, cfg_text: str):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(cfg_text, encoding="utf-8")

    class _Args:
        config = str(cfg)
        repo = str(tmp_path / "r")
        task_description = "x"
        acceptance_command = "true"
        task_id = "served"
        max_rounds = 5
        max_wall_seconds = 60
        sandbox_parent = str(tmp_path / "sb")
        suppress_thinking = False
        control_dir = None

    return _Args()


def _drive(monkeypatch, tmp_path, cfg_text: str, detected: str | None):
    """Run `oara code` up to the session, capturing what the adapter and the
    session were built for. The session raises at once, as the crash test
    beside this one does, so no model, git or clone is needed."""
    seen: dict = {}

    def _create_provider(cfg):
        seen["provider_cfg_model"] = cfg.get("model", "")
        return object(), cfg.get("model", "")

    def _detect(base_url, config_model):
        seen["asked_base_url"] = base_url
        return detected if detected is not None else config_model

    def _create_adapter(model_cfg, adapter_cfg=None, **kwargs):
        seen["adapter_model"] = model_cfg.get("model", "")
        return None

    monkeypatch.setattr(m, "create_provider", _create_provider)
    monkeypatch.setattr(m, "_detect_model_or_fallback", _detect)
    monkeypatch.setattr(m, "create_adapter", _create_adapter)
    monkeypatch.setattr("prometheus.coding.sandbox.clone_repo_for_sandbox",
                        lambda *a, **k: _StubSandbox(tmp_path))

    from prometheus.coding.session import CodingSession

    real_init = CodingSession.__init__

    def _init(self, *a, **k):
        seen["session_model"] = k.get("model")
        real_init(self, *a, **k)

    def _boom(self):
        raise RuntimeError("stop here")

    monkeypatch.setattr(CodingSession, "__init__", _init)
    monkeypatch.setattr(CodingSession, "run", _boom)
    rc = m.run_coding_task(_args(tmp_path, cfg_text))
    assert rc == 1  # the simulated crash; the run got as far as the session
    return seen


def test_llama_cpp_coding_run_is_built_for_the_served_model(monkeypatch, tmp_path):
    """A blank hint, as shipped and as deployed: the served name decides."""
    seen = _drive(monkeypatch, tmp_path,
                  "coding:\n  enabled: true\nmodel:\n  provider: llama_cpp\n"
                  "  base_url: http://box:8080\n  model: ''\n", detected=SERVED)
    assert seen["asked_base_url"] == "http://box:8080"
    assert seen["adapter_model"] == SERVED, "the adapter was built for the hint, not the served model"
    assert seen["session_model"] == SERVED, "the session (and its requests) carry the hint, not the served model"


def test_the_hint_stands_when_the_server_cannot_be_asked(monkeypatch, tmp_path):
    seen = _drive(monkeypatch, tmp_path,
                  "coding:\n  enabled: true\nmodel:\n  provider: llama_cpp\n"
                  "  model: my-hint\n", detected=None)
    assert seen["adapter_model"] == "my-hint" and seen["session_model"] == "my-hint"


def test_ollama_serves_the_name_it_is_asked_for(monkeypatch, tmp_path):
    seen = _drive(monkeypatch, tmp_path,
                  "coding:\n  enabled: true\nmodel:\n  provider: ollama\n"
                  "  base_url: http://mini:11434\n  model: qwen2.5:7b-instruct\n", detected="ignored")
    assert "asked_base_url" not in seen, "nothing to detect on ollama"
    assert seen["adapter_model"] == "qwen2.5:7b-instruct"
    assert seen["session_model"] == "qwen2.5:7b-instruct"


def test_the_real_registry_resolves_the_served_name_to_light():
    """The point: with the served name, the same chooser that gives a chat
    turn tier light gives the coding run tier light."""
    assert m._get_adapter_tier("llama_cpp", "") == "full"
    assert m._get_adapter_tier("llama_cpp", SERVED) == "light"


@pytest.mark.parametrize("name", ["", "   "])
def test_a_blank_hint_alone_still_means_full(name):
    """Documented, not fixed here: with no server to ask, a blank hint is
    still no registry family."""
    assert m._get_adapter_tier("llama_cpp", name) == "full"
