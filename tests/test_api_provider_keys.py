"""Provider-keys REST (MODELS KEYS UI sprint) — /api/providers/keys.

The contract under test:
- the catalog DERIVES from OVERRIDE_PRESETS + the media tool constants
  (adding a preset shows up with zero endpoint changes),
- PUT persists to the (per-test tmp) env file AND flips os.environ so
  /api/models availability changes in the SAME process — no restart,
- wrong env-var names 400 with nothing written; empty string = explicit clear,
- key VALUES never appear in any response or log line (booleans only),
- the routes sit behind the same bearer middleware as the rest of /api/*.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.env_file import parse_env_file  # noqa: E402
from prometheus.providers.key_catalog import provider_key_services  # noqa: E402
from prometheus.router.model_router import (  # noqa: E402
    OVERRIDE_PRESETS,
    ModelRouter,
    RouterConfig,
)
from prometheus.web.server import create_app  # noqa: E402

FAKE_KEY = "sk-test-deepseek-4242-NEVER-LOGGED"

_VARS_UNDER_TEST = (
    "DEEPSEEK_API_KEY",
    "MOONSHOT_API_KEY",
    "KLING_ACCESS_KEY",
    "KLING_SECRET_KEY",
)


@pytest.fixture(autouse=True)
def _clean_key_vars(monkeypatch):
    """Guarantee the key vars start absent AND are restored after the test.

    The PUT endpoint mutates os.environ directly (that is the feature), so a
    plain delenv(raising=False) on a missing var would record nothing and leak
    the endpoint's writes past the test. setenv-then-delenv records both
    states; monkeypatch teardown unwinds to the pre-test environment.
    """
    for var in _VARS_UNDER_TEST:
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)


def _env_file() -> Path:
    # The autouse conftest fixture points this at a per-test tmp path.
    return Path(os.environ["PROMETHEUS_ENV_FILE"])


def _client(*, token: str = "", with_router: bool = False) -> TestClient:
    cfg: dict = {"model": {"model": "qwen3.5-32b", "provider": "local"}}
    if token:
        cfg["web"] = {"api_token": token}
    router = None
    if with_router:
        router = ModelRouter(
            RouterConfig(),
            primary_provider=object(),
            primary_adapter=object(),
            primary_model="qwen3.5-32b",
        )
    return TestClient(create_app(cfg, model_router=router))


# ── Catalog (GET) ────────────────────────────────────────────────────


def test_catalog_covers_presets_plus_media_and_is_grouped():
    c = _client()
    r = c.get("/api/providers/keys")
    assert r.status_code == 200
    services = r.json()["services"]
    by_id = {s["id"]: s for s in services}

    # Every LLM preset is present, derived — provider keys, not hand-typed ids.
    for key, preset in OVERRIDE_PRESETS.items():
        svc = by_id[key]
        assert svc["kind"] == "llm"
        assert [v["name"] for v in svc["env_vars"]] == [preset["api_key_env"]]
        assert svc["default_model"] == preset["model"]
        assert svc["label"] and svc["docs_url"].startswith("https://")

    # The media services, incl. the one two-var service (Kling pair).
    assert by_id["wan-image"]["kind"] == "image"
    assert [v["name"] for v in by_id["wan-image"]["env_vars"]] == ["DASHSCOPE_API_KEY"]
    assert by_id["kling-video"]["kind"] == "video"
    assert [v["name"] for v in by_id["kling-video"]["env_vars"]] == [
        "KLING_ACCESS_KEY", "KLING_SECRET_KEY",
    ]

    # Nothing set in this test env → every boolean is False; no value-shaped
    # fields exist at all (set is a bool, never a value/prefix/length).
    for svc in services:
        for var in svc["env_vars"]:
            assert var["set"] is False
            assert set(var.keys()) == {"name", "set"}


def test_catalog_derives_from_presets_adding_one_shows_up(monkeypatch):
    # The proof of ONE-catalog: a brand-new preset appears in the endpoint
    # without touching key_catalog or server code.
    monkeypatch.setitem(
        OVERRIDE_PRESETS,
        "newcloud",
        {"provider": "newcloud", "api_key_env": "NEWCLOUD_API_KEY",
         "model": "newcloud-1"},
    )
    services = {s["id"]: s for s in
                _client().get("/api/providers/keys").json()["services"]}
    svc = services["newcloud"]
    assert svc["kind"] == "llm"
    assert [v["name"] for v in svc["env_vars"]] == ["NEWCLOUD_API_KEY"]
    assert svc["default_model"] == "newcloud-1"
    assert svc["label"] == "Newcloud"  # metadata fallback, not a KeyError


def test_catalog_respects_user_slash_commands_config():
    # Same resolution path as /claude and /api/models: user config merges over
    # the preset, so the card shows the model the daemon would actually run.
    cfg = {
        "model": {"model": "qwen3.5-32b", "provider": "local"},
        "slash_commands": {"claude": {"model": "claude-sonnet-4-5"}},
    }
    c = TestClient(create_app(cfg))
    services = {s["id"]: s for s in
                c.get("/api/providers/keys").json()["services"]}
    assert services["claude"]["default_model"] == "claude-sonnet-4-5"


# ── PUT: persist + immediate liveness ────────────────────────────────


def test_put_persists_0600_flips_environ_and_models_availability(caplog):
    caplog.set_level(logging.DEBUG)
    c = _client(with_router=True)

    # Before: deepseek not available (no key anywhere).
    models = {m["key"]: m for m in c.get("/api/models").json()["models"]}
    assert models["deepseek"]["available"] is False

    r = c.put("/api/providers/keys/deepseek",
              json={"values": {"DEEPSEEK_API_KEY": FAKE_KEY}})
    assert r.status_code == 200
    svc = next(s for s in r.json()["services"] if s["id"] == "deepseek")
    assert svc["env_vars"][0] == {"name": "DEEPSEEK_API_KEY", "set": True}

    # Persisted to the env file, which was created 0600 (it holds secrets).
    path = _env_file()
    assert parse_env_file(path)["DEEPSEEK_API_KEY"] == FAKE_KEY
    assert (path.stat().st_mode & 0o777) == 0o600

    # Live IMMEDIATELY in this process: os.environ + /api/models flips true
    # with no restart (registry/cmd_provider_override read os.environ at use
    # time; _model_catalog reads it at request time).
    assert os.environ["DEEPSEEK_API_KEY"] == FAKE_KEY
    models = {m["key"]: m for m in c.get("/api/models").json()["models"]}
    assert models["deepseek"]["available"] is True

    # Grep-proof: the value appears in NO response body and NO log record.
    assert FAKE_KEY not in caplog.text
    assert FAKE_KEY not in r.text
    for path_ in ("/api/providers/keys", "/api/models"):
        assert FAKE_KEY not in c.get(path_).text
    # ...but the write WAS logged (names only).
    assert "DEEPSEEK_API_KEY" in caplog.text


def test_put_two_var_kling_pair():
    c = _client()
    r = c.put("/api/providers/keys/kling-video", json={"values": {
        "KLING_ACCESS_KEY": "ak-test-1", "KLING_SECRET_KEY": "sk-test-2"}})
    assert r.status_code == 200
    svc = next(s for s in r.json()["services"] if s["id"] == "kling-video")
    assert all(v["set"] is True for v in svc["env_vars"])
    values = parse_env_file(_env_file())
    assert values["KLING_ACCESS_KEY"] == "ak-test-1"
    assert values["KLING_SECRET_KEY"] == "sk-test-2"
    # Partial update is fine too — one var of the pair, the other untouched.
    r = c.put("/api/providers/keys/kling-video",
              json={"values": {"KLING_SECRET_KEY": "sk-test-3"}})
    assert parse_env_file(_env_file())["KLING_ACCESS_KEY"] == "ak-test-1"
    assert os.environ["KLING_SECRET_KEY"] == "sk-test-3"


def test_put_strips_pasted_whitespace():
    c = _client()
    c.put("/api/providers/keys/kimi",
          json={"values": {"MOONSHOT_API_KEY": "  mk-test-7\n"}})
    assert parse_env_file(_env_file())["MOONSHOT_API_KEY"] == "mk-test-7"
    assert os.environ["MOONSHOT_API_KEY"] == "mk-test-7"


def test_put_empty_string_is_explicit_clear():
    c = _client()
    c.put("/api/providers/keys/deepseek",
          json={"values": {"DEEPSEEK_API_KEY": FAKE_KEY}})
    assert os.environ.get("DEEPSEEK_API_KEY") == FAKE_KEY

    r = c.put("/api/providers/keys/deepseek",
              json={"values": {"DEEPSEEK_API_KEY": ""}})
    assert r.status_code == 200
    svc = next(s for s in r.json()["services"] if s["id"] == "deepseek")
    assert svc["env_vars"][0]["set"] is False
    # Gone from the live process env; a deliberate BLANK assignment stays in
    # the env file (api_token.py convention for "deliberately blank").
    assert "DEEPSEEK_API_KEY" not in os.environ
    assert parse_env_file(_env_file())["DEEPSEEK_API_KEY"] == ""


# ── PUT: validation — nothing written on a bad request ───────────────


def test_wrong_var_name_is_400_and_writes_nothing():
    c = _client()
    r = c.put("/api/providers/keys/deepseek", json={"values": {
        "DEEPSEEK_API_KEY": FAKE_KEY,          # valid...
        "OPENAI_API_KEY": "sk-smuggled",       # ...but this one isn't deepseek's
    }})
    assert r.status_code == 400
    assert "OPENAI_API_KEY" in r.json()["error"]
    # All-or-nothing: the valid var was NOT written either.
    assert not _env_file().exists() or parse_env_file(_env_file()) == {}
    assert "DEEPSEEK_API_KEY" not in os.environ


@pytest.mark.parametrize("body", [
    {},                                   # no values at all
    {"values": {}},                       # empty values dict
    {"values": {"DEEPSEEK_API_KEY": 7}},  # non-string value
])
def test_malformed_body_is_400(body):
    assert _client().put("/api/providers/keys/deepseek", json=body).status_code == 400


def test_control_characters_in_value_are_400():
    # The env file is line-based — a \n in a value could inject a second
    # VAR= assignment. Refused outright.
    r = _client().put("/api/providers/keys/deepseek", json={"values": {
        "DEEPSEEK_API_KEY": "abc\nPROMETHEUS_API_TOKEN="}})
    assert r.status_code == 400
    assert not _env_file().exists() or parse_env_file(_env_file()) == {}


def test_unknown_service_is_404():
    r = _client().put("/api/providers/keys/nope",
                      json={"values": {"X": "y"}})
    assert r.status_code == 404


# ── Auth parity with the rest of /api/* ──────────────────────────────


def test_routes_require_bearer_when_token_set():
    c = _client(token="secret")
    assert c.get("/api/providers/keys").status_code == 401
    assert c.put("/api/providers/keys/deepseek",
                 json={"values": {"DEEPSEEK_API_KEY": "x"}}).status_code == 401
    ok = c.get("/api/providers/keys",
               headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200


# ── Catalog module unit: derivation, not duplication ─────────────────


def test_key_catalog_module_mirrors_presets():
    services = {s.id: s for s in provider_key_services()}
    for key, preset in OVERRIDE_PRESETS.items():
        assert services[key].env_vars == (preset["api_key_env"],)
        assert services[key].default_model == preset["model"]
    assert services["kling-video"].env_vars == (
        "KLING_ACCESS_KEY", "KLING_SECRET_KEY")
