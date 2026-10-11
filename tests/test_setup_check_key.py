"""POST /api/setup/check-key — Beacon asks "does this key work?" before it configures anything.

Contract (Beacon's O9 spec, section 6): the master token, ``{"provider", "api_key"}`` →
``200 {"result": "ok" | "bad_key" | "no_credit" | "unreachable"}``. Beacon runs the OpenRouter sign-in itself and
hands the key to this check, then to ``/api/setup/configure``; a pasted Anthropic or OpenAI key takes the same road.

What is pinned, and why each matters:

* the four results, classified from what each service REALLY answers — the 401 bodies below are recorded from the
  live services (tests/fixtures/provider_key_checks), the 402 is documented and labelled so;
* classification is by HTTP STATUS, never by a provider's wording: OpenAI's 401 body echoes a masked copy of the key,
  and a service rewording its message must not change a verdict;
* OpenRouter's "no credit" is the whole point of its check, and only a request that SPENDS can show it, so that one
  runs after the free ``/key`` call and never when the key is already refused;
* the key travels only in the auth header, is never in the answer, the URL or the logs, and no provider body is
  relayed (it can echo the key);
* only the three known providers are called, at fixed URLs (this route must not become an authenticated proxy), and
  nothing follows a redirect;
* anything unexpected is ``unreachable``, never ``ok``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

pytest.importorskip("fastapi")

from tests.test_setup_api_phase2 import (  # noqa: E402
    config_dir,  # noqa: F401 — fixtures re-exported
    env_file,  # noqa: F401
    make_client,
    pair,
)

FIXTURES = Path(__file__).parent / "fixtures" / "provider_key_checks"
KEY = "sk-or-v1-" + "a1b2c3d4" * 8
ANTHROPIC_KEY = "sk-ant-api03-" + "Zy9x" * 12
OPENAI_KEY = "sk-proj-" + "Qw3r" * 12


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def reply(name: str) -> httpx.Response:
    """A recorded response, as httpx would hand it back."""
    rec = fixture(name)
    return httpx.Response(rec["status"], json=rec["body"], headers={"content-type": rec["content_type"]})


class Upstream:
    """A stand-in for the three providers: records every request, answers from a table."""

    def __init__(self, answers: dict[tuple[str, str], httpx.Response | Exception]) -> None:
        self.answers = answers
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.get((request.method, f"{request.url.host}{request.url.path}"))
        if answer is None:
            raise AssertionError(f"unexpected upstream call {request.method} {request.url}")
        if isinstance(answer, Exception):
            raise answer
        return answer

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self), follow_redirects=False, timeout=5.0)

    @property
    def paths(self) -> list[str]:
        return [f"{r.method} {r.url.host}{r.url.path}" for r in self.requests]


OR_KEY = ("GET", "openrouter.ai/api/v1/key")
OR_CHAT = ("POST", "openrouter.ai/api/v1/chat/completions")
ANTH = ("GET", "api.anthropic.com/v1/models")
OAI = ("GET", "api.openai.com/v1/models")
OK_KEY = httpx.Response(200, json={"data": {"label": "sk-or-v1-abc...xyz", "limit": None, "is_free_tier": False}})
OK_CHAT = httpx.Response(200, json={"id": "gen-1", "choices": [{"message": {"role": "assistant", "content": "h"}}]})


def check(provider: str, key: str, up: Upstream):
    from prometheus.providers import key_check

    return key_check.check_provider_key(provider, key, client=up.client())


# --------------------------------------------------------------------------- #
# OpenRouter: free /key first, a one-token request only when the key is accepted
# --------------------------------------------------------------------------- #


class TestOpenRouter:
    def test_a_working_key_with_credit_is_ok(self):
        up = Upstream({OR_KEY: OK_KEY, OR_CHAT: OK_CHAT})
        assert check("openrouter", KEY, up).result == "ok"
        assert up.paths == ["GET openrouter.ai/api/v1/key", "POST openrouter.ai/api/v1/chat/completions"]

    def test_the_recorded_unknown_key_is_bad_key_and_nothing_is_spent(self):
        up = Upstream({OR_KEY: reply("openrouter_key_unknown")})
        assert check("openrouter", KEY, up).result == "bad_key"
        assert up.paths == ["GET openrouter.ai/api/v1/key"], "a refused key must never reach the request that spends"

    def test_the_recorded_missing_header_answer_is_bad_key(self):
        up = Upstream({OR_KEY: reply("openrouter_key_bad")})
        assert check("openrouter", "not-an-openrouter-key", up).result == "bad_key"

    def test_a_key_that_is_refused_on_the_spending_request_is_bad_key(self):
        up = Upstream({OR_KEY: OK_KEY, OR_CHAT: reply("openrouter_chat_unknown")})
        assert check("openrouter", KEY, up).result == "bad_key"

    def test_no_credit_is_the_402(self):
        up = Upstream({OR_KEY: OK_KEY, OR_CHAT: reply("openrouter_chat_no_credit")})
        result = check("openrouter", KEY, up)
        assert result.result == "no_credit" and result.status == 402

    def test_forbidden_is_bad_key(self):
        up = Upstream({OR_KEY: OK_KEY, OR_CHAT: httpx.Response(403, json={"error": {"code": 403, "message": "x"}})})
        assert check("openrouter", KEY, up).result == "bad_key"

    @pytest.mark.parametrize("status", [400, 404, 408, 429, 500, 502, 503])
    def test_every_other_answer_is_unreachable_never_ok(self, status):
        up = Upstream({OR_KEY: OK_KEY, OR_CHAT: httpx.Response(status, json={"error": {"code": status}})})
        result = check("openrouter", KEY, up)
        assert result.result == "unreachable" and result.status == status

    def test_a_redirect_is_not_followed_and_is_unreachable(self):
        up = Upstream({OR_KEY: httpx.Response(302, headers={"location": "https://evil.example/steal"})})
        assert check("openrouter", KEY, up).result == "unreachable"
        assert up.paths == ["GET openrouter.ai/api/v1/key"], "a redirect must never be followed"

    @pytest.mark.parametrize("exc", [
        httpx.ConnectError("no route"), httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"),
        httpx.RemoteProtocolError("garbled"),
    ])
    def test_a_network_failure_is_unreachable(self, exc):
        assert check("openrouter", KEY, Upstream({OR_KEY: exc})).result == "unreachable"

    def test_the_spending_request_is_one_token_on_a_paid_model(self):
        from prometheus.providers.registry import CLOUD_DEFAULTS

        up = Upstream({OR_KEY: OK_KEY, OR_CHAT: OK_CHAT})
        check("openrouter", KEY, up)
        body = json.loads(up.requests[1].content)
        assert body["max_tokens"] == 1
        assert body["model"] == CLOUD_DEFAULTS["openrouter"]["model"]
        assert not body["model"].endswith(":free"), "a free model would pass the check on an account with no credit"


# --------------------------------------------------------------------------- #
# Anthropic and OpenAI: a free, model-independent call
# --------------------------------------------------------------------------- #


class TestPastedKeys:
    def test_anthropic_ok_and_its_header_shape(self):
        up = Upstream({ANTH: httpx.Response(200, json={"data": []})})
        assert check("anthropic", ANTHROPIC_KEY, up).result == "ok"
        req = up.requests[0]
        assert req.headers["x-api-key"] == ANTHROPIC_KEY and req.headers["anthropic-version"]
        assert "authorization" not in req.headers

    def test_the_recorded_anthropic_401_is_bad_key(self):
        assert check("anthropic", ANTHROPIC_KEY, Upstream({ANTH: reply("anthropic_models_bad")})).result == "bad_key"

    def test_openai_ok_and_its_header_shape(self):
        up = Upstream({OAI: httpx.Response(200, json={"data": []})})
        assert check("openai", OPENAI_KEY, up).result == "ok"
        assert up.requests[0].headers["authorization"] == f"Bearer {OPENAI_KEY}"

    def test_the_recorded_openai_401_is_bad_key(self):
        assert check("openai", OPENAI_KEY, Upstream({OAI: reply("openai_models_bad")})).result == "bad_key"

    @pytest.mark.parametrize("provider,call,key", [("anthropic", ANTH, ANTHROPIC_KEY), ("openai", OAI, OPENAI_KEY)])
    @pytest.mark.parametrize("status,expected", [
        (401, "bad_key"), (403, "bad_key"), (429, "unreachable"), (500, "unreachable"), (503, "unreachable"),
    ])
    def test_classification_is_by_status_alone(self, provider, call, key, status, expected):
        up = Upstream({call: httpx.Response(status, json={"error": {"message": "words that may change"}})})
        assert check(provider, key, up).result == expected

    @pytest.mark.parametrize("provider,call,key", [("anthropic", ANTH, ANTHROPIC_KEY), ("openai", OAI, OPENAI_KEY)])
    def test_a_network_failure_is_unreachable(self, provider, call, key):
        assert check(provider, key, Upstream({call: httpx.ConnectError("down")})).result == "unreachable"


# --------------------------------------------------------------------------- #
# The key is a secret, and the caller is not trusted with a proxy
# --------------------------------------------------------------------------- #


class TestTheKeyStaysPut:
    def test_the_key_is_only_ever_in_the_auth_header(self):
        up = Upstream({OR_KEY: OK_KEY, OR_CHAT: OK_CHAT})
        check("openrouter", KEY, up)
        for req in up.requests:
            assert KEY not in str(req.url) and KEY.encode() not in req.content
            assert req.headers["authorization"] == f"Bearer {KEY}"

    def test_no_provider_text_is_relayed(self):
        """OpenAI's 401 body echoes a masked copy of the key; what we say about it is ours."""
        res = check("openai", OPENAI_KEY, Upstream({OAI: reply("openai_models_bad")}))
        blob = json.dumps(res.as_dict())
        assert "sk-inval" not in blob and "platform.openai.com" not in blob and OPENAI_KEY not in blob

    def test_nothing_is_logged_with_the_key_or_the_provider_body(self, caplog):
        caplog.set_level(logging.DEBUG)
        check("openai", OPENAI_KEY, Upstream({OAI: reply("openai_models_bad")}))
        check("openrouter", KEY, Upstream({OR_KEY: OK_KEY, OR_CHAT: reply("openrouter_chat_no_credit")}))
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert OPENAI_KEY not in joined and KEY not in joined and "sk-inval" not in joined

    @pytest.mark.parametrize("key", ["", " ", "short", "has space in it 1234567890", "new\nline-1234567890",
                                     "tab\there-1234567890", "k" * 600, "ключ-1234567890123456"])
    def test_a_key_that_cannot_be_an_http_header_value_is_bad_key_and_calls_nobody(self, key):
        up = Upstream({})
        assert check("openrouter", key, up).result == "bad_key"
        assert up.requests == []

    def test_only_the_three_providers_are_known(self):
        from prometheus.providers import key_check

        assert sorted(key_check.CHECKABLE_PROVIDERS) == ["anthropic", "openai", "openrouter"]
        with pytest.raises(ValueError):
            key_check.check_provider_key("evil", KEY, client=Upstream({}).client())


# --------------------------------------------------------------------------- #
# The route
# --------------------------------------------------------------------------- #


@pytest.fixture
def routed(env_file, config_dir, monkeypatch):  # noqa: F811
    """A setup-mode app whose provider calls go to a stand-in, not the internet."""
    from prometheus.providers import key_check

    holder: dict[str, Upstream] = {"up": Upstream({OR_KEY: OK_KEY, OR_CHAT: OK_CHAT})}
    monkeypatch.setattr(key_check, "_make_client", lambda: holder["up"].client())
    client, _state = make_client()
    return client, pair(client), holder


class TestCheckKeyRoute:
    def test_it_answers_the_contract_shape(self, routed):
        client, auth, _ = routed
        resp = client.post("/api/setup/check-key", json={"provider": "openrouter", "api_key": KEY}, headers=auth)
        assert resp.status_code == 200, resp.text
        assert resp.json()["result"] == "ok"
        assert set(resp.json()) <= {"result", "status", "detail"}

    @pytest.mark.parametrize("answers,expected", [
        ({OR_KEY: reply("openrouter_key_unknown")}, "bad_key"),
        ({OR_KEY: OK_KEY, OR_CHAT: reply("openrouter_chat_no_credit")}, "no_credit"),
        ({OR_KEY: httpx.ConnectError("down")}, "unreachable"),
    ])
    def test_each_result_comes_through(self, routed, answers, expected):
        client, auth, holder = routed
        holder["up"] = Upstream(answers)
        resp = client.post("/api/setup/check-key", json={"provider": "openrouter", "api_key": KEY}, headers=auth)
        assert resp.status_code == 200 and resp.json()["result"] == expected

    def test_the_master_token_is_required(self, routed):
        client, _auth, holder = routed
        body = {"provider": "openrouter", "api_key": KEY}
        assert client.post("/api/setup/check-key", json=body).status_code == 401
        assert client.post("/api/setup/check-key", json=body,
                           headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert holder["up"].requests == [], "an unauthenticated call must not reach a provider"

    def test_it_is_not_reachable_before_pairing(self, env_file, config_dir, monkeypatch):  # noqa: F811
        from prometheus.providers import key_check

        monkeypatch.setattr(key_check, "_make_client", lambda: Upstream({}).client())
        client, _ = make_client()
        resp = client.post("/api/setup/check-key", json={"provider": "openrouter", "api_key": KEY})
        assert resp.status_code == 401 and resp.json()["error"] == "not_paired"

    @pytest.mark.parametrize("body", [
        {}, {"api_key": KEY}, {"provider": "openrouter"}, {"provider": "openrouter", "api_key": ""},
        {"provider": "evil", "api_key": KEY}, {"provider": "gemini", "api_key": KEY},
        {"provider": ["openrouter"], "api_key": KEY}, {"provider": "openrouter", "api_key": 7},
        {"provider": "openrouter", "api_key": KEY, "base_url": "https://evil.example"},
    ])
    def test_a_malformed_request_is_a_400_and_calls_nobody(self, routed, body):
        client, auth, holder = routed
        resp = client.post("/api/setup/check-key", json=body, headers=auth)
        assert resp.status_code == 400, resp.text
        assert resp.json()["error"] in {"bad_request", "unsupported_provider", "unexpected_base_url"}
        assert holder["up"].requests == []

    def test_a_body_that_is_not_json_is_a_400(self, routed):
        client, auth, _ = routed
        resp = client.post("/api/setup/check-key", content=b"nope",
                           headers={**auth, "Content-Type": "application/json"})
        assert resp.status_code == 400

    def test_the_key_is_not_in_the_answer_or_the_logs(self, routed, caplog):
        client, auth, _ = routed
        caplog.set_level(logging.DEBUG)
        resp = client.post("/api/setup/check-key", json={"provider": "openrouter", "api_key": KEY}, headers=auth)
        assert KEY not in resp.text
        assert KEY not in "\n".join(r.getMessage() for r in caplog.records)

    def test_checking_a_key_writes_nothing(self, routed, env_file, config_dir):
        client, auth, _ = routed
        client.post("/api/setup/check-key", json={"provider": "openrouter", "api_key": KEY}, headers=auth)
        assert not config_dir.exists() or not any(config_dir.iterdir())
        assert KEY not in (env_file.read_text() if env_file.exists() else "")

    def test_a_pasted_key_goes_the_same_road(self, routed):
        client, auth, holder = routed
        holder["up"] = Upstream({ANTH: httpx.Response(200, json={"data": []}), OAI: reply("openai_models_bad")})
        a = client.post("/api/setup/check-key", json={"provider": "anthropic", "api_key": ANTHROPIC_KEY}, headers=auth)
        o = client.post("/api/setup/check-key", json={"provider": "openai", "api_key": OPENAI_KEY}, headers=auth)
        assert (a.json()["result"], o.json()["result"]) == ("ok", "bad_key")
