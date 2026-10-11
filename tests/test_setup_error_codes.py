"""Every way setup can fail has a stable code, and no failure is anything else (Beacon v0.4, P8 item 7).

Beacon shows a person a plain-English message when setup fails: "that key was not accepted", "that address did not
answer". It writes the words itself, from a CODE, so the code is the contract: it must be there on every failure,
must not change when the daemon's own English does, and must say WHICH input it is about, so the right box can be
highlighted. Before this, the codes existed for most failures but not all, and nothing stopped the next one arriving
as bare text:

* a provider name nobody supports, at an address that answers, wrote a config naming a provider the daemon cannot
  build (reported, when it failed at all, as "backend unreachable");
* a pasted key or token with a line break went into the env file raw, and a line break there starts a second
  assignment;
* a disk that cannot be written, or any other surprise, was a plain-text 500 that is no code at all;
* a response could say "bad_request" without saying about what.

What is pinned:

* each failure below is the status, the code and the fields written here, as literals, not read from the catalog
  that is meant to hold them to it;
* the catalog (``web/setup_errors.py``) holds every code the setup app can emit, with one status each, and
  ``docs/contracts/setup-errors.md`` lists exactly those: a code cannot ship undocumented, or be documented and gone;
* nothing in ``setup_server.py`` builds an error response by hand, so a new failure has to be given a code;
* hostile bodies on every route never produce an answer that is not coded JSON, and never a stack trace;
* existing codes keep their names and statuses (the table's first rows are what Beacon already handles).
"""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.cli.init import _CLOUD_FAST_PROVIDERS  # noqa: E402
from prometheus.config.env_file import parse_env_file  # noqa: E402
from prometheus.web.setup_server import PairingState, SetupModeState, create_setup_app  # noqa: E402
from tests.test_setup_api_phase2 import (  # noqa: E402
    CODE,
    _FakeLlamaCppHandler,
    _free_port,
    _serve,
    config_dir,  # noqa: F401 — fixtures re-exported
    env_file,  # noqa: F401
    make_client,
    pair,
)

REPO = Path(__file__).resolve().parents[1]
SETUP_SERVER = REPO / "src" / "prometheus" / "web" / "setup_server.py"
CONTRACT = REPO / "docs" / "contracts" / "setup-errors.md"

KEY = "sk-test-" + "ab12" * 8


@pytest.fixture(autouse=True)
def no_cloud_keys_in_environment(monkeypatch):
    for key_env, _model, _limit in _CLOUD_FAST_PROVIDERS.values():
        monkeypatch.delenv(key_env, raising=False)


# --------------------------------------------------------------------------- #
# The failures, as the app answers them
# --------------------------------------------------------------------------- #


_UNSET = object()


class World:
    """A paired setup-mode app and the places it writes."""

    def __init__(self, config_dir, env_file, *, pairing: PairingState | None = None) -> None:
        self.config_dir, self.env_file = config_dir, env_file
        self.state = SetupModeState()
        app = create_setup_app(pairing or PairingState(code=CODE), api_port=8123, ws_port=8124, state=self.state)
        self.client = TestClient(app, raise_server_exceptions=False)   # a surprise must be an ANSWER, not a raise
        self.headers: dict[str, str] = {}

    def pair(self) -> World:
        self.headers = pair(self.client)
        return self

    def post(self, path: str, body=_UNSET, *, content: bytes | None = None):
        """POST *body* as JSON (any JSON value, ``None`` included), *content* verbatim, or nothing."""
        if body is not _UNSET:
            content = json.dumps(body).encode()
        headers = {**self.headers, "content-type": "application/json"} if content is not None else self.headers
        return self.client.post(path, content=content, headers=headers)

    def configure(self, body):
        return self.post("/api/setup/configure", body)

    @property
    def wrote_config(self) -> bool:
        return (self.config_dir / "prometheus.yaml").exists()

    @property
    def env(self) -> dict:
        return parse_env_file(self.env_file) if self.env_file.exists() else {}


@pytest.fixture
def world(config_dir, env_file) -> World:  # noqa: F811
    return World(config_dir, env_file).pair()


def configure_error(world: World, body, status: int, code: str, fields: list[str] | None) -> dict:
    resp = world.configure(body)
    assert resp.status_code == status, resp.text
    out = resp.json()
    assert out["error"] == code, out
    assert isinstance(out["detail"], str) and out["detail"], "the English stays, for logs and clients with no copy"
    assert out.get("fields") == fields, out
    assert not world.wrote_config, "a refused configure writes no config"
    return out


class TestConfigureRefusals:
    """The codes Beacon already handles keep their names and statuses; each now says which input it is about."""

    def test_a_body_that_is_not_json_is_a_bad_request_about_no_particular_field(self, world):
        resp = world.post("/api/setup/configure", content=b"not json")
        assert (resp.status_code, resp.json()["error"]) == (400, "bad_request")
        assert "fields" not in resp.json()

    def test_a_local_provider_with_nothing_says_exactly_what_is_missing(self, world):
        configure_error(world, {}, 400, "bad_request", ["provider", "base_url", "model"])
        configure_error(world, {"provider": "llama_cpp", "model": "m"}, 400, "bad_request", ["base_url"])

    def test_an_address_that_is_not_http_is_a_bad_base_url(self, world):
        configure_error(world, {"provider": "llama_cpp", "base_url": "ftp://x", "model": "m"},
                        400, "bad_base_url", ["base_url"])

    def test_a_cloud_provider_takes_no_address(self, world):
        configure_error(world, {"provider": "anthropic", "base_url": "http://x", "api_key": KEY},
                        400, "unexpected_base_url", ["base_url"])

    def test_a_cloud_provider_with_no_key_anywhere(self, world):
        configure_error(world, {"provider": "anthropic"}, 400, "cloud_key_missing", ["api_key"])

    def test_half_a_slack_pair(self, world):
        body = {"provider": "anthropic", "api_key": KEY, "slack_bot_token": "xoxb-1"}
        configure_error(world, body, 400, "slack_token_pair_incomplete", ["slack_bot_token", "slack_app_token"])

    def test_guild_ids_that_are_not_numbers(self, world):
        body = {"provider": "anthropic", "api_key": KEY, "discord_token": "t", "discord_guild_ids": ["abc"]}
        configure_error(world, body, 400, "bad_discord_guild_ids", ["discord_guild_ids"])

    def test_an_address_that_does_not_answer(self, world):
        body = {"provider": "llama_cpp", "base_url": f"http://127.0.0.1:{_free_port()}", "model": "m"}
        configure_error(world, body, 400, "backend_unreachable", ["base_url"])


class TestWhatUsedToSlipThrough:
    def test_a_provider_nobody_supports_is_refused_even_where_its_address_answers(self, world):
        """It used to be probed as if it were one, then written into the config for the daemon to fail on."""
        with _serve(_FakeLlamaCppHandler) as url:
            configure_error(world, {"provider": "mystery", "base_url": url, "model": "m"},
                            400, "unsupported_provider", ["provider"])

    def test_a_cloud_name_that_is_not_in_the_table_is_not_mistaken_for_a_local_server(self, world):
        out = configure_error(world, {"provider": "nonesuch", "api_key": KEY}, 400, "unsupported_provider",
                              ["provider"])
        assert "anthropic" in out["detail"] and "llama_cpp" in out["detail"], "it says what IS supported"

    @pytest.mark.parametrize("field,code", [
        ("api_key", "invalid_key"), ("telegram_token", "invalid_token"), ("slack_bot_token", "invalid_token"),
        ("slack_app_token", "invalid_token"), ("discord_token", "invalid_token"),
    ])
    def test_a_secret_with_a_line_break_would_start_a_second_assignment_in_the_env_file(self, world, field, code):
        """The env file is line-based; ``KEY=value\\nPROMETHEUS_API_TOKEN=attacker`` is two assignments."""
        body = {"provider": "anthropic", "api_key": KEY, "slack_bot_token": "xoxb-1", "slack_app_token": "xapp-1"}
        body[field] = "good-start\nPROMETHEUS_API_TOKEN=attacker"
        before = world.env
        configure_error(world, body, 400, code, [field])
        assert world.env == before
        assert not world.env_file.exists() or "attacker" not in world.env_file.read_text(encoding="utf-8")

    @pytest.mark.parametrize("field", ["provider", "base_url", "model", "api_key", "agent_name", "persona",
                                       "telegram_token", "slack_bot_token", "slack_app_token", "discord_token"])
    def test_a_value_of_the_wrong_type_is_refused_not_turned_into_text(self, world, field):
        """``str({"a": 1})`` used to be saved as the API key: ``ANTHROPIC_API_KEY={'a': 1}``."""
        body = {"provider": "anthropic", "api_key": KEY, "slack_bot_token": "xoxb-1", "slack_app_token": "xapp-1"}
        body[field] = {"a": 1}
        before = world.env
        configure_error(world, body, 400, "bad_request", [field])
        assert world.env == before

    def test_a_secret_that_is_absurdly_long_is_refused_the_same_way(self, world):
        configure_error(world, {"provider": "anthropic", "api_key": "k" * 5000}, 400, "invalid_key", ["api_key"])

    def test_a_model_with_a_control_character(self, world):
        configure_error(world, {"provider": "anthropic", "api_key": KEY, "model": "a\x00b"},
                        400, "bad_model", ["model"])

    def test_nothing_is_written_when_any_one_input_is_refused(self, world):
        """Validation first, then writes: a bad token must not leave the good key behind."""
        before = world.env
        configure_error(world, {"provider": "anthropic", "api_key": KEY, "telegram_token": "a\nb"},
                        400, "invalid_token", ["telegram_token"])
        assert world.env == before and "ANTHROPIC_API_KEY" not in world.env

    def test_a_disk_that_cannot_be_written_is_a_coded_500(self, world, monkeypatch):
        def refuse(*_a, **_k):
            raise PermissionError(13, "Permission denied", "/secret/path/prometheus.yaml")

        monkeypatch.setattr("prometheus.cli.init.write_config", refuse)
        resp = world.configure({"provider": "anthropic", "api_key": KEY})
        assert resp.status_code == 500 and resp.json()["error"] == "config_write_failed", resp.text
        assert world.state.configured is False and KEY not in resp.text

    def test_any_other_surprise_is_a_coded_500_that_leaks_nothing(self, world, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("secret-detail-sk-live-1234")

        monkeypatch.setattr("prometheus.cli.init._cloud_default_config", boom)
        resp = world.configure({"provider": "anthropic", "api_key": KEY})
        assert resp.status_code == 500 and resp.headers["content-type"].startswith("application/json")
        assert resp.json()["error"] == "internal_error"
        assert "secret-detail" not in resp.text and "Traceback" not in resp.text


class TestTheOtherRoutes:
    def test_check_key_says_which_input(self, world):
        def refused(body, code, fields):
            resp = world.post("/api/setup/check-key", body)
            assert (resp.status_code, resp.json()["error"], resp.json().get("fields")) == (400, code, fields), resp.text

        refused({}, "bad_request", ["provider"])
        refused({"provider": "anthropic"}, "bad_request", ["api_key"])
        refused({"provider": "anthropic", "api_key": KEY, "base_url": "http://x"}, "unexpected_base_url", ["base_url"])
        refused({"provider": "llama_cpp", "api_key": KEY}, "unsupported_provider", ["provider"])

    def test_detect_with_a_bad_address(self, world):
        resp = world.client.get("/api/setup/detect", params={"base_url": "ftp://x"}, headers=world.headers)
        assert (resp.status_code, resp.json()["error"], resp.json().get("fields")) == (400, "bad_base_url", ["base_url"])

    def test_complete_before_configure(self, world):
        resp = world.post("/api/setup/complete")
        assert (resp.status_code, resp.json()["error"]) == (409, "not_configured")

    def test_before_pairing_and_with_the_wrong_token(self, config_dir, env_file):
        unpaired = World(config_dir, env_file)
        resp = unpaired.post("/api/setup/configure", {})
        assert (resp.status_code, resp.json()["error"]) == (401, "not_paired")
        unpaired.pair()
        resp = unpaired.client.post("/api/setup/configure", json={}, headers={"Authorization": "Bearer wrong"})
        assert (resp.status_code, resp.json()["error"]) == (401, "unauthorized")

    def test_anything_else_is_the_honest_refusal(self, world):
        resp = world.client.get("/api/sessions")
        assert (resp.status_code, resp.json()["error"]) == (403, "setup_mode")
        resp = world.client.post("/api/pair/requests", json={})
        assert (resp.status_code, resp.json()["error"]) == (403, "pairing_unavailable")


class TestPairing:
    def test_a_missing_code(self, config_dir, env_file):
        w = World(config_dir, env_file)
        resp = w.post("/api/setup/pair", {})
        assert (resp.status_code, resp.json()["error"], resp.json().get("fields")) == (400, "bad_request", ["code"])

    def test_a_wrong_code_keeps_counting(self, config_dir, env_file):
        resp = World(config_dir, env_file).post("/api/setup/pair", {"code": "000000"})
        assert (resp.status_code, resp.json()["error"]) == (401, "invalid_code")
        assert resp.json()["attempts_remaining"] == 4

    def test_a_code_that_was_used(self, config_dir, env_file):
        w = World(config_dir, env_file).pair()
        resp = w.post("/api/setup/pair", {"code": CODE})
        assert (resp.status_code, resp.json()["error"]) == (403, "pairing_used")

    def test_a_code_that_expired(self, config_dir, env_file):
        w = World(config_dir, env_file, pairing=PairingState(code=CODE, created_at=time.time() - 10_000))
        resp = w.post("/api/setup/pair", {"code": CODE})
        assert (resp.status_code, resp.json()["error"]) == (403, "pairing_expired")

    def test_a_window_that_locked(self, config_dir, env_file):
        w = World(config_dir, env_file)
        for _ in range(5):
            w.post("/api/setup/pair", {"code": "000000"})
        resp = w.post("/api/setup/pair", {"code": CODE})
        assert (resp.status_code, resp.json()["error"]) == (403, "pairing_locked")


# --------------------------------------------------------------------------- #
# Hostile bodies: whatever is sent, the answer is coded JSON, never a trace
# --------------------------------------------------------------------------- #

HOSTILE = [
    [], "x", 7, None, True, 1.5, {"provider": {"a": 1}}, {"provider": ["anthropic"]}, {"provider": 5},
    {"provider": "anthropic", "api_key": {"a": 1}}, {"provider": "anthropic", "api_key": ["k"]},
    {"provider": "anthropic", "api_key": None}, {"provider": "anthropic", "model": {"a": 1}},
    {"provider": "anthropic", "api_key": KEY, "agent_name": {"a": 1}},
    {"provider": "anthropic", "api_key": KEY, "telegram_token": ["t"], "telegram_chat_ids": {"a": 1}},
    {"provider": "anthropic", "api_key": KEY, "slack_channels": 5, "slack_bot_token": 5},
    {"provider": "anthropic", "api_key": KEY, "discord_token": "t", "discord_guild_ids": {"a": 1}},
    {"provider": "llama_cpp", "base_url": ["http://x"], "model": "m"}, {"provider": "llama_cpp", "base_url": 5},
    {"provider": "", "base_url": "", "model": ""}, {"provider": "anthropic\x00", "api_key": KEY},
    {"provider": "A" * 100_000}, {"provider": "anthropic", "api_key": "k" * 100_000},
    {"": 1, "provider": "anthropic", "api_key": KEY, "extra": {"deep": [1, 2, {"x": None}]}},
]
ROUTES = ("/api/setup/pair", "/api/setup/check-key", "/api/setup/configure", "/api/setup/complete")


class TestNoUncodedAnswer:
    @pytest.mark.parametrize("path", ROUTES)
    @pytest.mark.parametrize("body", HOSTILE, ids=lambda b: repr(b)[:40])
    def test_every_failure_is_coded_json(self, config_dir, env_file, path, body):
        w = World(config_dir, env_file).pair()
        # a fresh window per request: the pair above used the code, which is not what is being tried here
        resp = w.post(path, body)
        if resp.status_code < 400:
            return
        assert resp.headers["content-type"].startswith("application/json"), (resp.status_code, resp.text[:200])
        out = resp.json()
        assert isinstance(out.get("error"), str) and out["error"] in catalog_codes(), out
        assert "Traceback" not in resp.text

    def test_the_wire_is_never_a_framework_validation_error(self, world):
        resp = world.client.get("/api/setup/detect", params={"base_url": ["a", "b"]}, headers=world.headers)
        assert resp.status_code < 400 or resp.json().get("error") in catalog_codes()


# --------------------------------------------------------------------------- #
# The catalog, its document, and the guard against hand-built errors
# --------------------------------------------------------------------------- #

#: Codes the setup app can emit that are produced and tested elsewhere (shared middleware, the same-Mac route).
ELSEWHERE = {"browser_not_allowed", "host_not_allowed", "rate_limited", "not_loopback"}
#: What this file drives, by code. The catalog may not hold a code nothing here (or ELSEWHERE) exercises.
DRIVEN_HERE = {
    "bad_request", "bad_base_url", "unexpected_base_url", "cloud_key_missing", "slack_token_pair_incomplete",
    "bad_discord_guild_ids", "backend_unreachable", "unsupported_provider", "invalid_key", "invalid_token",
    "bad_model", "config_write_failed", "internal_error", "not_configured", "not_paired", "unauthorized",
    "setup_mode", "pairing_unavailable", "invalid_code", "pairing_used", "pairing_expired", "pairing_locked",
}


def catalog():
    from prometheus.web import setup_errors

    return setup_errors


def catalog_codes() -> set[str]:
    return set(catalog().CATALOG)


class TestTheCatalog:
    def test_it_holds_exactly_the_codes_the_tests_drive_or_name_as_elsewhere(self):
        assert catalog_codes() == DRIVEN_HERE | ELSEWHERE

    def test_every_code_has_one_status_a_sentence_and_a_sensible_shape(self):
        for code, entry in catalog().CATALOG.items():
            assert code == code.lower() and code.replace("_", "").isalnum(), code
            assert 400 <= entry.status <= 599, code
            assert entry.meaning.endswith(".") and len(entry.meaning) > 20, code

    def test_the_statuses_existing_clients_depend_on_did_not_move(self):
        expected = {"bad_request": 400, "bad_base_url": 400, "unexpected_base_url": 400, "cloud_key_missing": 400,
                    "backend_unreachable": 400, "not_paired": 401, "unauthorized": 401, "invalid_code": 401,
                    "pairing_used": 403, "pairing_expired": 403, "pairing_locked": 403, "setup_mode": 403,
                    "not_configured": 409, "pairing_unavailable": 403, "not_loopback": 403}
        assert {code: catalog().CATALOG[code].status for code in expected} == expected

    def test_refuse_builds_the_one_shape(self):
        resp = catalog().refuse("bad_model", "that is not a model id", fields=["model"])
        assert resp.status_code == 400
        assert json.loads(resp.body) == {"error": "bad_model", "detail": "that is not a model id", "fields": ["model"]}
        assert "fields" not in json.loads(catalog().refuse("bad_request", "x").body)
        assert json.loads(catalog().refuse("invalid_code", "x", attempts_remaining=3).body)["attempts_remaining"] == 3

    def test_refuse_does_not_invent_a_code(self):
        with pytest.raises(KeyError):
            catalog().refuse("made_up_code", "x")

    def test_the_contract_document_lists_exactly_the_catalog(self):
        text = CONTRACT.read_text(encoding="utf-8")

        documented = set(re.findall(r"^\| `([a-z_]+)` \|", text, flags=re.MULTILINE))
        assert documented == catalog_codes()
        for code, entry in catalog().CATALOG.items():
            row = next(line for line in text.splitlines() if line.startswith(f"| `{code}` |"))
            assert f"| {entry.status} |" in row, f"{code}: the document and the catalog disagree on the status"


class TestNothingBuildsAnErrorByHand:
    def test_setup_server_has_no_raw_error_response(self):
        """A failure answered with ``JSONResponse(status_code=4xx/5xx, ...)`` is a code nobody catalogued."""
        tree = ast.parse(SETUP_SERVER.read_text(encoding="utf-8"))
        raw = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "JSONResponse":
                for kw in node.keywords:
                    if kw.arg == "status_code" and not (isinstance(kw.value, ast.Constant) and kw.value.value < 400):
                        raw.append(node.lineno)
        assert raw == [], f"hand-built error responses at setup_server.py lines {raw}: use setup_errors.refuse"
