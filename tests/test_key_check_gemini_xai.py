"""Gemini and xAI keys can be checked, in setup and after it (Beacon v0.4, P8 item 7).

``POST /api/setup/check-key`` and ``POST /api/model/check-key`` knew OpenRouter, Anthropic and OpenAI. A pasted Gemini
or xAI key was refused as ``unsupported_provider``, so Beacon had to skip the check and save the key unconfirmed.

What is different about these two, and the reason this is not just two more rows in the table: **both answer an
invalid key with HTTP 400, not 401.** ``providers/key_check.py`` classifies by status alone, which for a 400 says
"could not tell" (``unreachable``). That would report a mistyped Gemini key as a network problem. Recorded from the
live services on 2026-10-11 with deliberately invalid keys (``tests/fixtures/provider_key_checks``):

* Gemini: ``400`` with ``error.details[].reason == "API_KEY_INVALID"``, a ``google.rpc.ErrorInfo`` enum. (Its
  OpenAI-compatible endpoint, which the daemon calls, answers a bare ``400 INVALID_ARGUMENT`` with no reason, so the
  check uses the native ``/v1beta/models`` with the key in ``x-goog-api-key``: same key, same service.)
* xAI: ``400`` with ``code == "invalid-argument"``; a missing or empty credential is ``401 unauthenticated:*``.

So for these two, and only for a **400**, a machine-readable code names the key as the problem; any other 400, and
anything unrecognised, is still ``unreachable``, never ``ok`` and never ``bad_key``. The English message is never
read, in either direction. A valid key was not recorded (no real credential is ever used here): the ``200`` answers
are synthetic and labelled so.

Pinned: the two recorded 400s; message-independence both ways; a 400 that does not name the key stays
``unreachable``; the key travels only in the auth header, never the URL or a log; both routes and ``PUT /api/model``.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

pytest.importorskip("fastapi")

from prometheus.providers import key_check  # noqa: E402
from tests.test_model_settings_routes import Daemon, clean_environment, daemon  # noqa: E402,F401
from tests.test_setup_api_phase2 import config_dir, env_file, make_client, pair  # noqa: E402,F401
from tests.test_setup_check_key import Upstream, check, reply  # noqa: E402

GEMINI_KEY = "AIza" + "Sy" * 17 + "x"
XAI_KEY = "xai-" + "Qw3r" * 20
GEM = ("GET", "generativelanguage.googleapis.com/v1beta/models")
XAI = ("GET", "api.x.ai/v1/models")

#: SYNTHETIC answers (no valid key was ever used to record them): the documented success shapes.
GEMINI_OK = httpx.Response(200, json={"models": [{"name": "models/gemini-2.5-flash"}]})
XAI_OK = httpx.Response(200, json={"object": "list", "data": [{"id": "grok-4.5", "object": "model"}]})

PROVIDERS = [("gemini", GEM, GEMINI_KEY, GEMINI_OK, "gemini_models_bad"),
             ("xai", XAI, XAI_KEY, XAI_OK, "xai_models_bad")]


def recorded(name: str) -> dict:
    return json.loads((key_check_fixture_dir() / f"{name}.json").read_text(encoding="utf-8"))


def key_check_fixture_dir():
    from pathlib import Path

    return Path(__file__).parent / "fixtures" / "provider_key_checks"


class TestTheRecordedAnswers:
    def test_the_fixtures_say_where_they_came_from(self):
        for name in ("gemini_models_bad", "xai_models_bad"):
            doc = recorded(name)
            assert doc["status"] == 400, "both services answer an invalid key with 400"
            assert "RECORDED from the live service" in doc["provenance"] and "deliberately invalid" in doc["provenance"]

    @pytest.mark.parametrize("provider,call,key,_ok,fixture", PROVIDERS)
    def test_the_recorded_400_is_bad_key_not_unreachable(self, provider, call, key, _ok, fixture):
        res = check(provider, key, Upstream({call: reply(fixture)}))
        assert (res.result, res.status) == ("bad_key", 400)

    def test_what_the_machine_readable_codes_are(self):
        gemini = recorded("gemini_models_bad")["body"]["error"]
        assert [d["reason"] for d in gemini["details"] if "reason" in d] == ["API_KEY_INVALID"]
        assert recorded("xai_models_bad")["body"]["code"] == "invalid-argument"


@pytest.mark.parametrize("provider,call,key,ok,fixture", PROVIDERS)
class TestClassification:
    def test_a_success_is_ok(self, provider, call, key, ok, fixture):
        assert check(provider, key, Upstream({call: ok})).result == "ok"

    @pytest.mark.parametrize("status,expected", [(401, "bad_key"), (403, "bad_key"), (404, "unreachable"),
                                                  (429, "unreachable"), (500, "unreachable"), (503, "unreachable")])
    def test_other_statuses_are_classified_by_status_alone(self, provider, call, key, ok, fixture, status, expected):
        up = Upstream({call: httpx.Response(status, json={"error": {"message": "words that may change"}})})
        assert check(provider, key, up).result == expected

    def test_the_message_is_never_read(self, provider, call, key, ok, fixture):
        """A reworded message must not change a verdict, so the code decides, not the English."""
        doc = recorded(fixture)
        body = json.loads(json.dumps(doc["body"]).replace("not valid", "unacceptable").replace("Incorrect", "Wrong"))
        if provider == "gemini":
            body["error"]["message"] = "completely different words"
            for detail in body["error"]["details"]:
                if "message" in detail:
                    detail["message"] = "completely different words"
        else:
            body["error"] = "completely different words"
        assert check(provider, key, Upstream({call: httpx.Response(400, json=body)})).result == "bad_key"

    def test_a_400_that_does_not_name_the_key_is_not_a_bad_key(self, provider, call, key, ok, fixture):
        """The words can match while the code does not: still 'could not tell', never 'your key is wrong'."""
        if provider == "gemini":
            body = {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                              "status": "INVALID_ARGUMENT",
                              "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "OTHER"}]}}
        else:
            body = {"code": "failed-precondition", "error": "Incorrect API key provided."}
        assert check(provider, key, Upstream({call: httpx.Response(400, json=body)})).result == "unreachable"

    @pytest.mark.parametrize("body", [None, [], "text", 7, {}, {"error": None}, {"error": {"details": "x"}},
                                      {"error": {"details": [None, 3, "x", {"reason": 5}]}}, {"code": None},
                                      {"code": ["invalid-argument"]}])
    def test_a_400_that_is_not_the_expected_shape_is_unreachable(self, provider, call, key, ok, fixture, body):
        resp = httpx.Response(400, content=b"<html>not json</html>") if body is None else httpx.Response(400, json=body)
        assert check(provider, key, Upstream({call: resp})).result == "unreachable"

    def test_a_network_failure_is_unreachable(self, provider, call, key, ok, fixture):
        assert check(provider, key, Upstream({call: httpx.ConnectError("down")})).result == "unreachable"

    def test_nothing_the_provider_says_is_relayed(self, provider, call, key, ok, fixture):
        res = check(provider, key, Upstream({call: reply(fixture)}))
        blob = json.dumps(res.as_dict())
        assert "API key" not in blob and "console.x.ai" not in blob and "INVALID" not in blob and key not in blob


class TestTheRequest:
    def test_gemini_sends_the_key_in_its_own_header_and_nowhere_else(self):
        up = Upstream({GEM: GEMINI_OK})
        check("gemini", GEMINI_KEY, up)
        req = up.requests[0]
        assert req.headers["x-goog-api-key"] == GEMINI_KEY and "authorization" not in req.headers
        assert GEMINI_KEY not in str(req.url) and "key=" not in str(req.url)
        assert req.url.host == "generativelanguage.googleapis.com" and req.url.path == "/v1beta/models"
        assert dict(req.url.params) == {"pageSize": "1"}, "one model, not the whole catalogue"
        assert req.method == "GET" and req.content == b""

    def test_xai_sends_a_bearer_and_nothing_else(self):
        up = Upstream({XAI: XAI_OK})
        check("xai", XAI_KEY, up)
        req = up.requests[0]
        assert req.headers["authorization"] == f"Bearer {XAI_KEY}" and XAI_KEY not in str(req.url)
        assert (req.url.host, req.url.path) == ("api.x.ai", "/v1/models") and req.method == "GET" and req.content == b""

    @pytest.mark.parametrize("provider,call,key,ok,fixture", PROVIDERS)
    def test_one_free_call_and_nothing_spent(self, provider, call, key, ok, fixture):
        up = Upstream({call: ok})
        check(provider, key, up)
        assert len(up.requests) == 1 and up.requests[0].method == "GET"

    @pytest.mark.parametrize("provider,call,key,ok,fixture", PROVIDERS)
    def test_a_key_that_cannot_be_a_header_value_calls_nobody(self, provider, call, key, ok, fixture):
        up = Upstream({})
        assert check(provider, "bad key\nwith newline 12345", up).result == "bad_key" and up.requests == []

    @pytest.mark.parametrize("provider,call,key,ok,fixture", PROVIDERS)
    def test_nothing_is_logged_with_the_key_or_the_body(self, provider, call, key, ok, fixture, caplog):
        caplog.set_level(logging.DEBUG)
        check(provider, key, Upstream({call: reply(fixture)}))
        boom = httpx.ConnectError(f"failed; Authorization: Bearer {key}; x-goog-api-key: {key}")
        check(provider, key, Upstream({call: boom}))
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert key not in joined and "Authorization" not in joined and "x-goog-api-key" not in joined
        assert "Incorrect API key" not in joined and "API_KEY_INVALID" not in joined

    def test_both_are_known_to_the_check_and_nothing_else_was_added(self):
        assert sorted(key_check.CHECKABLE_PROVIDERS) == ["anthropic", "gemini", "openai", "openrouter", "xai"]


@pytest.fixture
def routed(env_file, config_dir, monkeypatch):  # noqa: F811
    holder = {"up": Upstream({GEM: GEMINI_OK, XAI: XAI_OK})}
    monkeypatch.setattr(key_check, "_make_client", lambda: holder["up"].client())
    client, _state = make_client()
    return client, pair(client), holder


@pytest.mark.parametrize("provider,call,key,ok,fixture", PROVIDERS)
class TestSetupRoute:
    def test_each_result_comes_through(self, routed, provider, call, key, ok, fixture):
        client, auth, holder = routed
        resp = client.post("/api/setup/check-key", json={"provider": provider, "api_key": key}, headers=auth)
        assert resp.status_code == 200 and resp.json()["result"] == "ok"
        holder["up"] = Upstream({call: reply(fixture)})
        resp = client.post("/api/setup/check-key", json={"provider": provider, "api_key": key}, headers=auth)
        assert resp.status_code == 200 and resp.json()["result"] == "bad_key" and resp.json()["status"] == 400
        assert key not in resp.text

    def test_it_still_takes_no_address(self, routed, provider, call, key, ok, fixture):
        client, auth, holder = routed
        resp = client.post("/api/setup/check-key", headers=auth,
                           json={"provider": provider, "api_key": key, "base_url": "https://evil.example"})
        assert resp.status_code == 400 and holder["up"].requests == []


@pytest.mark.parametrize("provider,call,key,ok,fixture", PROVIDERS)
class TestAfterSetup:
    def test_the_model_route_checks_a_key_it_is_given_and_the_one_it_holds(self, daemon, provider, call, key, ok, fixture):
        daemon.upstream = Upstream({call: reply(fixture)})
        resp = daemon.call("POST", "/api/model/check-key", json={"provider": provider, "api_key": key})
        assert resp.status_code == 200 and resp.json()["result"] == "bad_key"
        env = {"gemini": "GEMINI_API_KEY", "xai": "XAI_API_KEY"}[provider]
        daemon.env_file.write_text(f"{env}={key}\n", encoding="utf-8")
        daemon.upstream = Upstream({call: ok})
        assert daemon.call("POST", "/api/model/check-key", json={"provider": provider}).json()["result"] == "ok"

    def test_a_refused_key_is_not_saved_by_the_put_route(self, daemon, provider, call, key, ok, fixture):
        daemon.upstream = Upstream({call: reply(fixture)})
        before = daemon.text
        resp = daemon.put(provider=provider, api_key=key)
        assert resp.status_code == 400 and resp.json()["error"] == "key_rejected"
        assert resp.json()["key_check"]["result"] == "bad_key" and daemon.text == before and daemon.env() == {}

    def test_a_good_key_is_saved_and_the_answer_says_it_was_confirmed(self, daemon, provider, call, key, ok, fixture):
        daemon.upstream = Upstream({call: ok})
        resp = daemon.put(provider=provider, api_key=key)
        assert resp.status_code == 200 and resp.json()["key_check"]["result"] == "ok"
        assert "key_unchecked" not in [w["code"] for w in resp.json()["warnings"]], "it was checked now"

    def test_the_provider_list_marks_them_checkable(self, daemon, provider, call, key, ok, fixture):
        rows = {r["provider"]: r for r in daemon.call("GET", "/api/model").json()["providers"]}
        assert rows[provider]["checkable"] is True and rows["deepseek"]["checkable"] is False
