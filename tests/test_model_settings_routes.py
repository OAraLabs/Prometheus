"""Change the provider, model and API key AFTER setup, without a terminal (Beacon v0.4, P8 item 7).

A person who picked wrong during setup (the wrong vendor, a mistyped key, an account with no credit) must be able to
fix it without reinstalling. The routes:

    GET  /api/model              what is running, what is saved, whether they differ, which keys are held
    PUT  /api/model              {"provider", "model"?, "api_key"?, "check"?}  save a change
    POST /api/model/check-key    {"provider", "api_key"?}  does this key (or the one held) work?

Shape, on purpose the one ``PUT /api/network`` already has: it SAVES, and the daemon applies it when it restarts
(``POST /api/restart``), saying so in ``applied``. The config file is edited as text, comments and all, and the edit is
verified before anything is written. Operator only (the master token or an owner device); a scoped device is a 403.

What is pinned, because each is a way for "fix my setup" to make it worse:

* a switch of PROVIDER must not carry the old vendor's endpoint or inline key to the new one (a stale ``base_url`` or
  ``api_key`` in ``model:`` outranks everything, and the old key would be sent to the new vendor);
* the key is checked first (the same check ``/api/setup/check-key`` runs): one the provider refuses is not saved, one
  that merely cannot be confirmed is, and the answer says which;
* a value an environment variable pins (``PROMETHEUS_MODEL``, ``PROMETHEUS_PROVIDER``) is a refusal, not a save that
  changes nothing;
* the key lives in the env file only: never the yaml, the answer or a log line;
* what is saved is what the daemon's own provider registry then builds (the consumed layer).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from pathlib import Path

import httpx
import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.cli.init import _CLOUD_FAST_PROVIDERS  # noqa: E402
from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.config.env_file import parse_env_file  # noqa: E402
from prometheus.providers import key_check  # noqa: E402
from prometheus.web.network import PersistError  # noqa: E402
from prometheus.providers.registry import ProviderRegistry  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from tests.test_setup_check_key import (  # noqa: E402
    ANTH,
    OK_CHAT,
    OK_KEY,
    OR_CHAT,
    OR_KEY,
    Upstream,
    reply,
)

TOKEN = "model-test-" + secrets.token_hex(8)
OR_NEW = "sk-or-v1-" + "9f8e7d6c" * 8
ANTH_NEW = "sk-ant-api03-" + "Mn7p" * 12
OLD_INLINE = "sk-OLDVENDOR-inline-key-that-must-not-travel-0000"

CLOUD_FILE = """\
# Prometheus config. Comments are the reasoning; they must survive an edit.
model:
  provider: "openrouter"
  model: "anthropic/claude-sonnet-5"   # what setup chose
  api_key_env: "OPENROUTER_API_KEY"
  max_tool_iterations: 50

context:
  effective_limit: 100000   # setup's budget for this provider

web:
  enabled: true
  bind: "127.0.0.1"
"""

LOCAL_FILE = f"""\
# Prometheus config, hand-edited. Comments are the reasoning; they must survive an edit.
model:
  # Env var holding this provider's API key (cloud providers only).
  api_key_env: ""
  api_key: "{OLD_INLINE}"   # discouraged, and left over from before
  provider: "llama_cpp"             # llama_cpp, ollama, lm_studio, vllm, openai, anthropic
  base_url: "http://localhost:8080" # Inference server URL
  model: "qwen3.8"                  # a hint

context:
  effective_limit: 24000   # tuned for the local window
  compression_trigger: 0.75

web:
  enabled: true
"""


def comment_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("#"))


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    for env, _model, _limit in _CLOUD_FAST_PROVIDERS.values():
        monkeypatch.delenv(env, raising=False)
    for var in ("PROMETHEUS_MODEL", "PROMETHEUS_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(tmp_path / "env"))
    yield
    for env, _model, _limit in _CLOUD_FAST_PROVIDERS.values():
        os.environ.pop(env, None)  # the PUT route mirrors a saved key into this process


class Daemon:
    """A configured daemon: a config file on disk, the app that booted from it, and an upstream stand-in."""

    def __init__(self, tmp_path: Path, monkeypatch, text: str = CLOUD_FILE) -> None:
        self.path = tmp_path / "prometheus.yaml"
        self.path.write_text(text, encoding="utf-8")
        self.env_file = tmp_path / "env"
        monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(self.env_file))
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.upstream = Upstream({OR_KEY: OK_KEY, OR_CHAT: OK_CHAT, ANTH: httpx.Response(200, json={"data": []})})
        monkeypatch.setattr(key_check, "_make_client", lambda: self.upstream.client())
        self.client = self.boot()

    def boot(self) -> TestClient:
        """(Re)start: the app reads its running config from the file, like the daemon does."""
        config = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        config.setdefault("web", {})["api_token"] = TOKEN
        app = create_app(config, device_store=self.devices)
        app.state.config_path = str(self.path)
        self.client = TestClient(app)
        return self.client

    def call(self, method: str, url: str, token: str = TOKEN, **kw):
        return self.client.request(method, url, headers={"Authorization": f"Bearer {token}"}, **kw)

    def put(self, **body):
        return self.call("PUT", "/api/model", json=body)

    @property
    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    @property
    def model_section(self) -> dict:
        return yaml.safe_load(self.text)["model"]

    def env(self) -> dict:
        return parse_env_file(self.env_file) if self.env_file.exists() else {}


@pytest.fixture
def daemon(tmp_path, monkeypatch) -> Daemon:
    return Daemon(tmp_path, monkeypatch)


@pytest.fixture
def local_daemon(tmp_path, monkeypatch) -> Daemon:
    return Daemon(tmp_path, monkeypatch, LOCAL_FILE)


# --------------------------------------------------------------------------- #
# GET /api/model
# --------------------------------------------------------------------------- #


class TestRead:
    def test_it_reports_what_is_running_and_that_nothing_is_waiting(self, daemon):
        body = daemon.call("GET", "/api/model").json()
        assert body["running"] == {"provider": "openrouter", "model": "anthropic/claude-sonnet-5",
                                   "api_key_env": "OPENROUTER_API_KEY", "kind": "cloud"}
        assert body["saved"] == body["running"]
        assert body["pending"] is False and body["applied"] == "live"
        assert body["can_change"] is True and body["warnings"] == []

    def test_a_local_provider_is_described_as_local(self, local_daemon):
        running = local_daemon.call("GET", "/api/model").json()["running"]
        assert running["provider"] == "llama_cpp" and running["kind"] == "local"

    def test_it_lists_the_choosable_providers_with_whether_a_key_is_held(self, daemon):
        daemon.env_file.write_text("ANTHROPIC_API_KEY=" + ANTH_NEW + "\n", encoding="utf-8")
        rows = {r["provider"]: r for r in daemon.call("GET", "/api/model").json()["providers"]}
        assert {"openrouter", "anthropic", "openai", "deepseek"} <= set(rows)
        assert rows["openrouter"] == {"provider": "openrouter", "api_key_env": "OPENROUTER_API_KEY",
                                      "default_model": "anthropic/claude-sonnet-5", "key_present": False,
                                      "checkable": True}
        assert rows["anthropic"]["key_present"] is True and rows["deepseek"]["checkable"] is False

    def test_a_key_is_reported_as_present_never_as_a_value(self, daemon):
        daemon.env_file.write_text("OPENROUTER_API_KEY=" + OR_NEW + "\n", encoding="utf-8")
        resp = daemon.call("GET", "/api/model")
        assert resp.json()["key"] == {"env": "OPENROUTER_API_KEY", "present": True}
        assert OR_NEW not in resp.text

    def test_a_saved_change_waiting_for_a_restart_is_visible(self, daemon):
        daemon.put(provider="anthropic", api_key=ANTH_NEW, check=False)
        body = daemon.call("GET", "/api/model").json()
        assert body["running"]["provider"] == "openrouter" and body["saved"]["provider"] == "anthropic"
        assert body["pending"] is True and body["applied"] == "on_restart"
        assert [w["code"] for w in body["warnings"]] == ["restart_required"]

    def test_after_the_restart_the_change_is_what_is_running(self, daemon):
        daemon.put(provider="anthropic", api_key=ANTH_NEW, check=False)
        daemon.boot()
        body = daemon.call("GET", "/api/model").json()
        assert body["running"]["provider"] == "anthropic" and body["pending"] is False


class TestWhoMayUseIt:
    def test_a_scoped_device_is_refused_all_three(self, daemon):
        scoped = daemon.devices.mint("approved phone", "ios")["token"]
        for method, url, kw in (("GET", "/api/model", {}),
                                ("PUT", "/api/model", {"json": {"provider": "anthropic"}}),
                                ("POST", "/api/model/check-key", {"json": {"provider": "openrouter"}})):
            r = daemon.call(method, url, token=scoped, **kw)
            assert r.status_code == 403 and r.json()["error"] == "operator_only", (method, url)
        assert daemon.model_section["provider"] == "openrouter", "a refused request must change nothing"
        assert daemon.call("GET", "/api/model").status_code == 200, "the same routes answer the operator"

    def test_each_handler_refuses_a_scoped_device_on_its_own(self, daemon, monkeypatch):
        """The gate in front is one layer; with it opened the handlers still say no (the network routes do the same)."""
        monkeypatch.setattr("prometheus.web.server.is_scoped_allowed", lambda *a, **k: True)
        scoped = daemon.devices.mint("approved phone", "ios")["token"]
        for method, url, kw in (("GET", "/api/model", {}),
                                ("PUT", "/api/model", {"json": {"provider": "anthropic", "api_key": ANTH_NEW}}),
                                ("POST", "/api/model/check-key", {"json": {"provider": "openrouter"}})):
            r = daemon.call(method, url, token=scoped, **kw)
            assert r.status_code == 403 and r.json()["error"] == "operator_only", (method, url)
        assert daemon.model_section["provider"] == "openrouter" and daemon.env() == {}
        assert daemon.upstream.requests == []

    def test_an_owner_device_may(self, daemon):
        owner = daemon.devices.mint_owner("Beacon", "macos", by="same-mac-pairing")["token"]
        assert daemon.call("GET", "/api/model", token=owner).status_code == 200

    def test_the_token_is_required(self, daemon):
        assert daemon.client.get("/api/model").status_code == 401
        assert daemon.client.put("/api/model", json={"provider": "anthropic"}).status_code == 401

    def test_with_the_token_off_everyone_is_the_owner(self, tmp_path, monkeypatch):
        d = Daemon(tmp_path, monkeypatch)
        app = create_app(yaml.safe_load(d.text), device_store=d.devices)  # no web.api_token
        app.state.config_path = str(d.path)
        assert TestClient(app).get("/api/model").json()["can_change"] is True


# --------------------------------------------------------------------------- #
# PUT /api/model — saving a change
# --------------------------------------------------------------------------- #


class TestSave:
    def test_a_cloud_to_cloud_switch_saves_provider_model_and_key_variable(self, daemon):
        resp = daemon.put(provider="anthropic", api_key=ANTH_NEW)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["changed"] is True and body["applied"] == "on_restart" and body["pending"] is True
        assert body["saved"] == {"provider": "anthropic", "model": _CLOUD_FAST_PROVIDERS["anthropic"][1],
                                 "api_key_env": "ANTHROPIC_API_KEY", "kind": "cloud"}
        assert body["running"]["provider"] == "openrouter", "nothing live changed"
        section = daemon.model_section
        assert (section["provider"], section["model"], section["api_key_env"]) == (
            "anthropic", _CLOUD_FAST_PROVIDERS["anthropic"][1], "ANTHROPIC_API_KEY")
        assert section["max_tool_iterations"] == 50, "keys the edit does not own are untouched"

    def test_comments_and_the_rest_of_the_file_survive_byte_for_byte(self, daemon):
        before = daemon.text
        assert daemon.put(provider="anthropic", api_key=ANTH_NEW).status_code == 200
        after = daemon.text
        assert after != before and comment_count(after) == comment_count(before)
        assert "# setup's budget for this provider" in after and "# Prometheus config." in after
        assert after.count("\n") == before.count("\n")
        assert yaml.safe_load(after)["web"] == {"enabled": True, "bind": "127.0.0.1"}

    def test_the_key_goes_to_the_env_file_and_nowhere_else(self, daemon, caplog):
        caplog.set_level(logging.DEBUG)
        resp = daemon.put(provider="anthropic", api_key=ANTH_NEW)
        assert daemon.env()["ANTHROPIC_API_KEY"] == ANTH_NEW
        assert ANTH_NEW not in daemon.text and ANTH_NEW not in resp.text
        assert ANTH_NEW not in "\n".join(r.getMessage() for r in caplog.records)

    def test_a_model_may_be_chosen_and_otherwise_the_providers_default_is_used(self, daemon):
        daemon.put(provider="anthropic", api_key=ANTH_NEW, model="claude-opus-5")
        assert daemon.model_section["model"] == "claude-opus-5"
        daemon.put(provider="anthropic")
        assert daemon.model_section["model"] == "claude-opus-5", "same provider, no model asked: keep the model"
        daemon.put(provider="openrouter", api_key=OR_NEW)
        assert daemon.model_section["model"] == _CLOUD_FAST_PROVIDERS["openrouter"][1], "new provider: its default"

    def test_changing_only_the_model_leaves_a_custom_endpoint_and_the_budget_alone(self, tmp_path, monkeypatch):
        text = CLOUD_FILE.replace('  max_tool_iterations: 50\n',
                                  '  max_tool_iterations: 50\n  base_url: "https://workspace.example/v1"\n')
        d = Daemon(tmp_path, monkeypatch, text)
        d.env_file.write_text("OPENROUTER_API_KEY=" + OR_NEW + "\n", encoding="utf-8")
        assert d.put(provider="openrouter", model="openai/gpt-5.6").status_code == 200
        assert d.model_section["base_url"] == "https://workspace.example/v1", "the owner's own endpoint stays"
        assert yaml.safe_load(d.text)["context"]["effective_limit"] == 100000

    def test_a_new_key_for_the_same_provider_replaces_the_old_one(self, daemon):
        daemon.env_file.write_text("OPENROUTER_API_KEY=sk-or-v1-OLDKEY\nOTHER=keep\n", encoding="utf-8")
        resp = daemon.put(provider="openrouter", api_key=OR_NEW)
        body = resp.json()
        assert resp.status_code == 200 and body["changed"] is False and body["key_saved"] is True
        assert daemon.env() == {"OPENROUTER_API_KEY": OR_NEW, "OTHER": "keep"}
        # the file reads the same, but the running provider still holds the old key: only a restart swaps it
        assert body["pending"] is True and body["applied"] == "on_restart"
        assert [w["code"] for w in body["warnings"]][:1] == ["restart_required"]
        assert "key" in body["warnings"][0]["message"]

    def test_saving_what_is_already_saved_changes_nothing(self, daemon):
        daemon.env_file.write_text("OPENROUTER_API_KEY=" + OR_NEW + "\n", encoding="utf-8")
        before = daemon.text
        body = daemon.put(provider="openrouter", model="anthropic/claude-sonnet-5").json()
        assert body["changed"] is False and body["applied"] == "live" and body["pending"] is False
        assert daemon.text == before

    def test_the_file_keeps_its_permissions(self, daemon):
        daemon.path.chmod(0o640)
        before = daemon.text
        assert daemon.put(provider="anthropic", api_key=ANTH_NEW).status_code == 200
        assert daemon.text != before and daemon.path.stat().st_mode & 0o777 == 0o640


class TestSwitchingProviderDoesNotCarryTheOldOnesSecrets:
    def test_the_old_vendors_endpoint_and_inline_key_do_not_survive_a_switch(self, local_daemon):
        d = local_daemon
        resp = d.put(provider="anthropic", api_key=ANTH_NEW)
        assert resp.status_code == 200, resp.text
        section = d.model_section
        assert not section.get("base_url"), "a leftover localhost URL would hijack the cloud provider"
        assert not section.get("api_key"), "a leftover inline key outranks the env var and would be sent to the new vendor"
        assert OLD_INLINE not in d.text
        assert comment_count(d.text) == comment_count(LOCAL_FILE) and "# Inference server URL" in d.text

    def test_what_is_saved_is_what_the_daemons_registry_builds(self, local_daemon):
        d = local_daemon
        d.put(provider="anthropic", api_key=ANTH_NEW)
        saved = yaml.safe_load(d.text)["model"]
        os.environ["ANTHROPIC_API_KEY"] = d.env()["ANTHROPIC_API_KEY"]
        try:
            provider = ProviderRegistry.create(saved)
        finally:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        assert type(provider).__name__ == "AnthropicProvider"

    def test_a_new_key_for_the_same_provider_also_clears_a_stale_inline_key(self, tmp_path, monkeypatch):
        text = CLOUD_FILE.replace('  api_key_env: "OPENROUTER_API_KEY"\n',
                                  f'  api_key_env: "OPENROUTER_API_KEY"\n  api_key: "{OLD_INLINE}"\n')
        d = Daemon(tmp_path, monkeypatch, text)
        d.put(provider="openrouter", api_key=OR_NEW)
        assert not d.model_section.get("api_key") and d.env()["OPENROUTER_API_KEY"] == OR_NEW

    def test_the_context_budget_follows_the_provider_only_when_it_was_not_tuned(self, tmp_path, monkeypatch):
        def budget(text: str, provider: str, subdir: str) -> int:
            (tmp_path / subdir).mkdir()
            d = Daemon(tmp_path / subdir, monkeypatch, text)
            assert d.put(provider=provider, api_key="dk-" + "x" * 30, check=False).status_code == 200
            return yaml.safe_load(d.text)["context"]["effective_limit"]

        deepseek = _CLOUD_FAST_PROVIDERS["deepseek"][2]
        # local -> cloud: a local window is not a cloud budget
        assert budget(LOCAL_FILE, "deepseek", "local") == deepseek
        # cloud -> cloud, setup's untouched default: follows the new provider
        assert budget(CLOUD_FILE, "deepseek", "untouched") == deepseek
        # cloud -> cloud, a value the owner chose: left alone
        assert budget(CLOUD_FILE.replace("100000", "250000"), "deepseek", "tuned") == 250000

    def test_the_old_vendors_endpoint_variable_does_not_survive_a_switch_either(self, tmp_path, monkeypatch):
        text = CLOUD_FILE.replace('  max_tool_iterations: 50\n',
                                  '  max_tool_iterations: 50\n  base_url_env: "OLDVENDOR_BASE_URL"\n')
        d = Daemon(tmp_path, monkeypatch, text)
        assert d.put(provider="anthropic", api_key=ANTH_NEW).status_code == 200
        assert not d.model_section.get("base_url_env"), "it would read the old vendor's endpoint as the new one's"


# --------------------------------------------------------------------------- #
# The key check (the same one setup uses)
# --------------------------------------------------------------------------- #


class TestTheKeyIsCheckedFirst:
    def test_a_key_the_provider_refuses_is_not_saved(self, daemon):
        daemon.upstream.answers[OR_KEY] = reply("openrouter_key_unknown")
        before, env_before = daemon.text, daemon.env()
        resp = daemon.put(provider="openrouter", api_key=OR_NEW)
        assert resp.status_code == 400 and resp.json()["error"] == "key_rejected"
        assert resp.json()["key_check"]["result"] == "bad_key"
        assert daemon.text == before and daemon.env() == env_before, "nothing is written for a refused key"

    def test_a_valid_key_on_an_empty_account_is_saved_and_the_answer_says_so(self, daemon):
        daemon.upstream.answers[OR_CHAT] = reply("openrouter_chat_no_credit")
        resp = daemon.put(provider="openrouter", api_key=OR_NEW, model="openai/gpt-5.6")
        assert resp.status_code == 200
        assert resp.json()["key_check"]["result"] == "no_credit"
        assert "key_no_credit" in [w["code"] for w in resp.json()["warnings"]]
        assert daemon.env()["OPENROUTER_API_KEY"] == OR_NEW

    def test_a_key_that_cannot_be_confirmed_is_saved_with_a_warning(self, daemon):
        daemon.upstream.answers[OR_KEY] = httpx.ConnectError("offline")
        resp = daemon.put(provider="openrouter", api_key=OR_NEW)
        assert resp.status_code == 200 and resp.json()["key_check"]["result"] == "unreachable"
        assert "key_unchecked" in [w["code"] for w in resp.json()["warnings"]]

    def test_the_held_key_is_checked_when_none_is_sent(self, daemon):
        daemon.env_file.write_text("OPENROUTER_API_KEY=" + OR_NEW + "\n", encoding="utf-8")
        daemon.put(provider="openrouter", model="openai/gpt-5.6")
        sent = [r.headers["authorization"] for r in daemon.upstream.requests]
        assert sent and all(h == f"Bearer {OR_NEW}" for h in sent)

    def test_check_false_skips_the_network_entirely(self, daemon):
        resp = daemon.put(provider="anthropic", api_key=ANTH_NEW, check=False)
        assert resp.status_code == 200 and resp.json()["key_check"] is None
        assert daemon.upstream.requests == []

    def test_a_key_for_an_address_of_the_owners_own_is_not_checked_against_the_vendors(self, tmp_path, monkeypatch):
        """A valid key for the owner's proxy would be 'refused' by the vendor's public address: wrongly, and fatally."""
        text = CLOUD_FILE.replace('  max_tool_iterations: 50\n',
                                  '  max_tool_iterations: 50\n  base_url: "https://gateway.example/v1"\n')
        d = Daemon(tmp_path, monkeypatch, text)
        d.upstream.answers[OR_KEY] = reply("openrouter_key_unknown")           # what the vendor would say
        resp = d.put(provider="openrouter", api_key=OR_NEW)
        assert resp.status_code == 200 and resp.json()["key_check"] is None and d.upstream.requests == []
        assert "key_unchecked" in [w["code"] for w in resp.json()["warnings"]]
        assert d.env()["OPENROUTER_API_KEY"] == OR_NEW

    def test_a_provider_with_no_check_is_saved_unchecked(self, daemon):
        resp = daemon.put(provider="deepseek", api_key="dk-" + "y" * 30)
        assert resp.status_code == 200 and resp.json()["key_check"] is None
        assert "key_unchecked" in [w["code"] for w in resp.json()["warnings"]], "said, not implied"
        assert daemon.upstream.requests == []


class TestCheckKeyRoute:
    def test_it_checks_a_key_it_is_given(self, daemon):
        resp = daemon.call("POST", "/api/model/check-key", json={"provider": "openrouter", "api_key": OR_NEW})
        assert resp.status_code == 200 and resp.json()["result"] == "ok"

    def test_it_checks_the_key_the_daemon_already_holds(self, daemon):
        daemon.env_file.write_text("ANTHROPIC_API_KEY=" + ANTH_NEW + "\n", encoding="utf-8")
        resp = daemon.call("POST", "/api/model/check-key", json={"provider": "anthropic"})
        assert resp.json()["result"] == "ok"
        assert daemon.upstream.requests[0].headers["x-api-key"] == ANTH_NEW

    def test_with_no_key_anywhere_it_says_so(self, daemon):
        resp = daemon.call("POST", "/api/model/check-key", json={"provider": "anthropic"})
        assert resp.status_code == 400 and resp.json()["error"] == "cloud_key_missing"
        assert daemon.upstream.requests == []

    @pytest.mark.parametrize("body", [{}, {"provider": "deepseek"}, {"provider": "evil", "api_key": OR_NEW},
                                      {"provider": "openrouter", "api_key": OR_NEW, "base_url": "https://x"},
                                      {"provider": "openrouter", "api_key": 5}])
    def test_a_request_that_cannot_be_checked_is_a_400_and_calls_nobody(self, daemon, body):
        resp = daemon.call("POST", "/api/model/check-key", json=body)
        assert resp.status_code == 400 and daemon.upstream.requests == []

    def test_the_key_is_never_in_the_answer_or_the_logs(self, daemon, caplog):
        caplog.set_level(logging.DEBUG)
        resp = daemon.call("POST", "/api/model/check-key", json={"provider": "openrouter", "api_key": OR_NEW})
        assert resp.status_code == 200 and daemon.upstream.requests, "the key was really checked"
        assert OR_NEW not in resp.text and OR_NEW not in "\n".join(r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# What it refuses, and why nothing is written when it does
# --------------------------------------------------------------------------- #


class TestRefusals:
    @pytest.mark.parametrize("body,error", [
        ({"provider": "llama_cpp"}, "unsupported_provider"),
        ({"provider": "evil"}, "unsupported_provider"),
        ({"provider": "anthropic", "base_url": "https://evil.example/v1"}, "unexpected_base_url"),
        ({"provider": "anthropic", "mode": "x"}, "invalid_request"),
        ({"model": "x"}, "invalid_request"),
        ({"provider": ["anthropic"]}, "invalid_request"),
        ({"provider": "anthropic", "api_key": 7}, "invalid_request"),
        ({"provider": "anthropic", "check": "yes"}, "invalid_request"),
        ({"provider": "anthropic", "model": ""}, "bad_model"),
        ({"provider": "anthropic", "model": "has space"}, "bad_model"),
        ({"provider": "anthropic", "model": "x" * 300}, "bad_model"),
        ({"provider": "anthropic", "model": "a\nb: c"}, "bad_model"),
        ({"provider": "anthropic", "api_key": "  "}, "invalid_key"),
        ({"provider": "anthropic", "api_key": "line\nINJECT=1"}, "invalid_key"),
        ({"provider": "anthropic", "api_key": "k" * 5000}, "invalid_key"),
    ])
    def test_a_bad_request_is_a_coded_400_and_nothing_is_written(self, daemon, body, error):
        before, env_before = daemon.text, daemon.env()
        resp = daemon.call("PUT", "/api/model", json=body)
        assert resp.status_code == 400 and resp.json()["error"] == error, resp.text
        assert daemon.text == before and daemon.env() == env_before and daemon.upstream.requests == []

    def test_a_body_that_is_not_json_is_a_400(self, daemon):
        resp = daemon.client.put("/api/model", content=b"nope",
                                 headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
        assert resp.status_code == 400 and resp.json()["error"] == "invalid_request"

    def test_a_switch_with_no_key_anywhere_is_refused(self, daemon):
        before = daemon.text
        resp = daemon.put(provider="anthropic")
        assert resp.status_code == 400 and resp.json()["error"] == "cloud_key_missing"
        assert daemon.text == before, "a config that cannot start is not written"

    def test_a_switch_works_with_a_key_already_held(self, daemon):
        daemon.env_file.write_text("ANTHROPIC_API_KEY=" + ANTH_NEW + "\n", encoding="utf-8")
        assert daemon.put(provider="anthropic").status_code == 200

    def test_no_config_file_is_a_409(self, daemon):
        daemon.client.app.state.config_path = None
        resp = daemon.put(provider="anthropic", api_key=ANTH_NEW)
        assert resp.status_code == 409 and resp.json()["error"] == "no_config_file"
        assert daemon.env() == {}, "the key is not stored for a switch that cannot be saved"

    @pytest.mark.parametrize("var,value,asked,source", [
        ("PROMETHEUS_PROVIDER", "openrouter", {"provider": "anthropic"}, "PROMETHEUS_PROVIDER"),
        ("PROMETHEUS_MODEL", "anthropic/claude-sonnet-5", {"provider": "openrouter", "model": "openai/gpt-5.6"},
         "PROMETHEUS_MODEL"),
    ])
    def test_a_value_an_environment_variable_pins_cannot_be_changed_here(self, daemon, monkeypatch, var, value,
                                                                         asked, source):
        monkeypatch.setenv(var, value)
        before = daemon.text
        resp = daemon.call("PUT", "/api/model", json={**asked, "api_key": ANTH_NEW, "check": False})
        assert resp.status_code == 409 and resp.json()["error"] == "model_overridden"
        assert resp.json()["source"] == source and daemon.text == before

    def test_a_pinned_value_may_still_be_saved_as_what_it_already_is(self, daemon, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_PROVIDER", "openrouter")
        daemon.env_file.write_text("OPENROUTER_API_KEY=" + OR_NEW + "\n", encoding="utf-8")
        assert daemon.put(provider="openrouter").status_code == 200

    def test_a_file_the_edit_cannot_be_verified_against_is_left_untouched(self, tmp_path, monkeypatch):
        flow = "model: {provider: openrouter, model: anthropic/claude-sonnet-5, api_key_env: OPENROUTER_API_KEY}\nweb:\n  enabled: true\n"
        d = Daemon(tmp_path, monkeypatch, flow)
        resp = d.put(provider="anthropic", api_key=ANTH_NEW, check=False)
        assert resp.status_code == 500 and resp.json()["error"] == "persist_failed"
        assert d.text == flow
        assert d.env() == {}, "nothing is half-saved"

    def test_a_config_that_is_not_yaml_is_not_touched(self, tmp_path, monkeypatch):
        d = Daemon(tmp_path, monkeypatch)
        d.path.write_text("model: [unclosed\n", encoding="utf-8")
        resp = d.put(provider="anthropic", api_key=ANTH_NEW, check=False)
        assert resp.status_code == 500 and d.text == "model: [unclosed\n"


class TestWhenTheWriteGoesWrong:
    def test_a_key_that_cannot_be_stored_leaves_the_config_alone(self, daemon, monkeypatch):
        def refuse(*_a, **_k):
            raise PermissionError("env file is read-only")

        monkeypatch.setattr("prometheus.config.env_file.set_env_value", refuse)
        before = daemon.text
        resp = daemon.put(provider="anthropic", api_key=ANTH_NEW, check=False)
        assert resp.status_code == 500 and resp.json()["error"] == "persist_failed"
        assert daemon.text == before and ANTH_NEW not in resp.text

    def test_a_config_that_cannot_be_replaced_says_so(self, daemon, monkeypatch):
        def refuse(*_a, **_k):
            raise PersistError("disk full")

        monkeypatch.setattr("prometheus.web.model_routes.replace_config", refuse)
        before = daemon.text
        resp = daemon.put(provider="anthropic", api_key=ANTH_NEW, check=False)
        assert resp.status_code == 500 and resp.json()["error"] == "persist_failed"
        assert daemon.text == before, "the file is as it was"
        assert daemon.env() == {"ANTHROPIC_API_KEY": ANTH_NEW}, "a stray key is the safe half-save, and it is stated"


class HookedUpstream(Upstream):
    """An upstream that runs *hook* when a request arrives, i.e. while the daemon is waiting on the key check."""

    def __init__(self, answers, hook) -> None:
        super().__init__(answers)
        self.hook = hook

    def __call__(self, request):
        self.hook()
        return super().__call__(request)


class TestAnotherWriterDuringTheKeyCheck:
    def test_a_change_made_while_the_key_was_being_checked_is_not_written_over(self, daemon):
        """The check takes seconds. PUT /api/network (or a hand edit) can change the file meanwhile; the edit is
        rebuilt from the file as it is then, so that change survives."""
        def another_writer():
            daemon.path.write_text(daemon.text.replace('bind: "127.0.0.1"', 'bind: "0.0.0.0"'), encoding="utf-8")

        daemon.upstream = HookedUpstream(daemon.upstream.answers, another_writer)
        assert daemon.put(provider="anthropic", api_key=ANTH_NEW).status_code == 200
        saved = yaml.safe_load(daemon.text)
        assert saved["model"]["provider"] == "anthropic" and saved["web"]["bind"] == "0.0.0.0"


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_the_three_routes_are_classified_operator_only():
    from prometheus.web.route_access import OPERATOR_ONLY, SCOPED_ALLOWED

    for route in (("GET", "/api/model"), ("PUT", "/api/model"), ("POST", "/api/model/check-key")):
        assert route in OPERATOR_ONLY and route not in SCOPED_ALLOWED


def test_the_answer_is_json_a_client_can_decode(daemon):
    resp = daemon.put(provider="anthropic", api_key=ANTH_NEW, check=False)
    assert resp.status_code == 200 and resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert json.loads(json.dumps(body)) == body and body["changed"] is True
