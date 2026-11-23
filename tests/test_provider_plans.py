"""Subscription plans (providers/plans.py) and their REST surface on /api/providers/keys.

The contract under test:
- plans come from a fixed list; the default plan is the provider's own default host,
- PUT saves a plan only TOGETHER with its key, writes the endpoint only from the list (or the one workspace pattern
  with a checked id and region), and writes nothing on any refusal,
- reading never reveals a workspace host or a hand-set endpoint,
- a saved plan takes effect without a restart, for sessions already on the provider too,
- an interactive-only plan on an unattended route is a warning, never a refusal.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.env_file import parse_env_file  # noqa: E402
from prometheus.providers import plans  # noqa: E402
from prometheus.providers.registry import CLOUD_DEFAULTS, ProviderRegistry  # noqa: E402
from prometheus.router.model_router import (  # noqa: E402
    OVERRIDE_PRESETS,
    ModelRouter,
    RouterConfig,
    resolve_model_choices,
    resolve_model_target,
)
from prometheus.web.server import create_app  # noqa: E402

QWEN_KEY = "sk-sp-test-coding-4242-NEVER-LOGGED"
KIMI_KEY = "sk-kimi-test-7777-NEVER-LOGGED"
WS_ID = "ws-a1b2c3d4e5"

_VARS = (
    "QWEN_API_KEY", "QWEN_BASE_URL", "MOONSHOT_API_KEY", "MOONSHOT_BASE_URL",
    "ZAI_API_KEY", "ZAI_BASE_URL", "DEEPSEEK_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_vars(monkeypatch):
    # setenv-then-delenv records the var so the route's own writes to os.environ are unwound after the test.
    for var in _VARS:
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)


def _env_file() -> Path:
    return Path(os.environ["PROMETHEUS_ENV_FILE"])


def _written() -> dict[str, str]:
    return parse_env_file(_env_file()) if _env_file().exists() else {}


def _router() -> ModelRouter:
    return ModelRouter(
        RouterConfig(), primary_provider=object(), primary_adapter=object(), primary_model="local-model",
    )


def _client(cfg: dict | None = None, router: ModelRouter | None = None) -> TestClient:
    base: dict = {"model": {"model": "local-model", "provider": "llama_cpp"}}
    base.update(cfg or {})
    return TestClient(create_app(base, model_router=router))


def _svc(resp, service_id: str) -> dict:
    return next(s for s in resp.json()["services"] if s["id"] == service_id)


# ── The list itself ──────────────────────────────────────────────────


def test_every_plan_is_https_on_its_providers_documented_domain():
    domains = {"qwen": "aliyuncs.com", "glm": "z.ai", "kimi": ("moonshot.ai", "kimi.ai")}
    for service_id, service_plans in plans.PROVIDER_PLANS.items():
        assert service_id in OVERRIDE_PRESETS
        ids = [p.id for p in service_plans]
        assert len(ids) == len(set(ids))
        for plan in service_plans:
            assert plan.terms in (plans.TERMS_METERED, plans.TERMS_INTERACTIVE)
            assert plan.terms_note
            if plan.id == plans.WORKSPACE_PLAN:
                assert plan.endpoint == ""
                continue
            parts = urlsplit(plan.endpoint)
            assert parts.scheme == "https" and not parts.query and not parts.username
            assert parts.hostname.endswith(domains[service_id])


def test_the_default_plan_is_the_providers_own_default_host():
    for service_id, service_plans in plans.PROVIDER_PLANS.items():
        provider = OVERRIDE_PRESETS[service_id]["provider"]
        assert service_plans[0].endpoint == CLOUD_DEFAULTS[provider]["base_url"]
        assert service_plans[0].terms == plans.TERMS_METERED
        assert plans.endpoint_env_for(service_id) == CLOUD_DEFAULTS[provider]["default_base_url_env"]
    assert plans.endpoint_env_for("deepseek") == ""


def test_documented_plan_hosts():
    by = {(s, p.id): p.endpoint for s, ps in plans.PROVIDER_PLANS.items() for p in ps}
    assert by[("glm", "coding-plan")] == "https://api.z.ai/api/coding/paas/v4"
    assert by[("kimi", "kimi-code")] == "https://api.kimi.ai/coding/v1"
    assert by[("qwen", "coding-plan")] == "https://coding-intl.dashscope.aliyuncs.com/v1"
    assert by[("qwen", "token-plan")] == "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"


@pytest.mark.parametrize("ws_id,region", [
    ("evil.example.com/x", "ap-southeast-1"),
    ("ws-1.attacker", "ap-southeast-1"),
    ("WS-UPPER", "ap-southeast-1"),
    ("", "ap-southeast-1"),
    ("-lead", "ap-southeast-1"),
    ("a" * 61, "ap-southeast-1"),
    ("ws-1@x", "ap-southeast-1"),
    ("token-plan", "ap-southeast-1"),
    ("trial", "cn-beijing"),
    ("ws-ok", "evil.example.com"),
    ("ws-ok", "ap-southeast-1.attacker"),
    (None, "ap-southeast-1"),
    ("ws-ok", None),
])
def test_workspace_endpoint_refuses_anything_off_the_pattern(ws_id, region):
    with pytest.raises(plans.PlanError):
        plans.workspace_endpoint(ws_id, region)


def test_workspace_endpoint_builds_the_documented_host():
    assert plans.workspace_endpoint(WS_ID, "us-east-1") == (
        f"https://{WS_ID}.us-east-1.maas.aliyuncs.com/compatible-mode/v1")


def test_plan_for_endpoint():
    assert plans.plan_for_endpoint("qwen", "") == "payg-intl"
    assert plans.plan_for_endpoint("qwen", "https://coding-intl.dashscope.aliyuncs.com/v1/") == "coding-plan"
    assert plans.plan_for_endpoint("qwen", plans.workspace_endpoint(WS_ID, "ap-southeast-1")) == "workspace"
    assert plans.plan_for_endpoint("qwen", f"https://{WS_ID}.mars-1.maas.aliyuncs.com/compatible-mode/v1") == "custom"
    assert plans.plan_for_endpoint("qwen", "https://proxy.example/v1") == "custom"
    assert plans.plan_for_endpoint("deepseek", "") is None


# ── GET ──────────────────────────────────────────────────────────────


def test_get_lists_plans_and_the_one_in_use():
    r = _client().get("/api/providers/keys")
    assert r.status_code == 200
    assert r.json()["capabilities"] == {"plans": 1}
    qwen = _svc(r, "qwen")
    assert [p["id"] for p in qwen["plans"]] == ["payg-intl", "payg-us", "token-plan", "coding-plan", "workspace"]
    assert qwen["plan"] == "payg-intl"
    assert qwen["endpoint_set"] is False
    assert qwen["plan_pinned_in_config"] is False
    by_id = {p["id"]: p for p in qwen["plans"]}
    assert [p["default"] for p in qwen["plans"]] == [True, False, False, False, False]
    assert by_id["workspace"]["custom"] is True and by_id["coding-plan"]["custom"] is False
    coding = by_id["coding-plan"]
    assert coding["subscription"] is True and "interactive" in coding["terms"]
    assert coding["models"][0] == "qwen3.7-plus"
    assert by_id["payg-intl"]["subscription"] is False and by_id["payg-intl"]["terms"] is None
    # No host anywhere in the answer: plans are picked by id.
    for service_plans in plans.PROVIDER_PLANS.values():
        for plan in service_plans:
            if plan.endpoint:
                assert urlsplit(plan.endpoint).hostname not in r.text
    for no_plans in ("deepseek", "gemini", "claude", "wan-image"):
        svc = _svc(r, no_plans)
        assert svc["plans"] == [] and svc["plan"] is None and svc["endpoint_set"] is False


def test_get_never_reveals_a_workspace_or_hand_set_host(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", plans.workspace_endpoint(WS_ID, "eu-central-1"))
    monkeypatch.setenv("MOONSHOT_BASE_URL", "https://private-proxy.internal.example/v1")
    r = _client().get("/api/providers/keys")
    assert _svc(r, "qwen")["plan"] == "workspace"
    assert _svc(r, "kimi")["plan"] == "custom"
    assert WS_ID not in r.text and "private-proxy" not in r.text
    assert WS_ID not in _client().get("/api/models").text


def test_get_reads_a_yaml_pin_and_says_so():
    cfg = {"slash_commands": {"glm": {"base_url": "https://api.z.ai/api/coding/paas/v4"}}}
    glm = _svc(_client(cfg).get("/api/providers/keys"), "glm")
    assert glm["plan"] == "coding-plan" and glm["plan_pinned_in_config"] is True


# ── PUT ──────────────────────────────────────────────────────────────


def test_put_saves_key_and_plan_together(caplog):
    caplog.set_level(logging.DEBUG)
    c = _client()
    r = c.put("/api/providers/keys/qwen",
              json={"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "coding-plan"})
    assert r.status_code == 200, r.text
    assert _svc(r, "qwen")["plan"] == "coding-plan"
    assert _svc(r, "qwen")["default_model"] == "qwen3.7-plus"
    assert _written()["QWEN_API_KEY"] == QWEN_KEY
    assert _written()["QWEN_BASE_URL"] == "https://coding-intl.dashscope.aliyuncs.com/v1"
    assert os.environ["QWEN_BASE_URL"] == "https://coding-intl.dashscope.aliyuncs.com/v1"
    assert QWEN_KEY not in caplog.text and QWEN_KEY not in r.text
    assert "coding-plan" in caplog.text


def test_choosing_the_default_plan_clears_the_endpoint():
    c = _client()
    c.put("/api/providers/keys/qwen", json={"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "token-plan"})
    r = c.put("/api/providers/keys/qwen", json={"values": {"QWEN_API_KEY": "sk-payg"}, "plan": "payg-intl"})
    assert _svc(r, "qwen")["plan"] == "payg-intl"
    assert "QWEN_BASE_URL" not in os.environ
    assert _written()["QWEN_BASE_URL"] == ""


def test_put_workspace_plan_writes_the_built_host_and_reads_back_no_host(caplog):
    caplog.set_level(logging.DEBUG)
    c = _client()
    r = c.put("/api/providers/keys/qwen", json={
        "values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "workspace",
        "workspace": {"id": WS_ID, "region": "ap-southeast-1"}})
    assert r.status_code == 200, r.text
    assert _written()["QWEN_BASE_URL"] == f"https://{WS_ID}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    assert _svc(r, "qwen")["plan"] == "workspace"
    assert WS_ID not in r.text and WS_ID not in caplog.text


def test_a_key_alone_keeps_todays_behaviour_and_the_plan():
    c = _client()
    c.put("/api/providers/keys/qwen", json={"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "coding-plan"})
    r = c.put("/api/providers/keys/qwen", json={"values": {"QWEN_API_KEY": "sk-sp-rotated"}})
    assert r.status_code == 200
    assert _svc(r, "qwen")["plan"] == "coding-plan"


@pytest.mark.parametrize("service,body,status", [
    # a plan without its key
    ("qwen", {"values": {"QWEN_API_KEY": ""}, "plan": "coding-plan"}, 400),
    # a plan off the list, and a plan that is not a string
    ("qwen", {"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "https://evil.example/v1"}, 400),
    ("qwen", {"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": ["coding-plan"]}, 400),
    ("qwen", {"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": None}, 400),
    # a plan for a service that has none
    ("deepseek", {"values": {"DEEPSEEK_API_KEY": "sk-x"}, "plan": "payg"}, 400),
    # the endpoint variable sent as a value: not the service's, refused like any other name
    ("qwen", {"values": {"QWEN_API_KEY": QWEN_KEY, "QWEN_BASE_URL": "https://evil.example/v1"}}, 400),
    # workspace refusals
    ("qwen", {"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "workspace"}, 400),
    ("qwen", {"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "workspace",
              "workspace": {"id": "evil.example.com/x", "region": "ap-southeast-1"}}, 400),
    ("qwen", {"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "workspace",
              "workspace": {"id": WS_ID, "region": "moon-1"}}, 400),
    ("qwen", {"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "coding-plan",
              "workspace": {"id": WS_ID, "region": "ap-southeast-1"}}, 400),
    ("qwen", {"values": {"QWEN_API_KEY": QWEN_KEY}, "workspace": {"id": WS_ID, "region": "ap-southeast-1"}}, 400),
])
def test_refusals_write_nothing(service, body, status):
    r = _client().put(f"/api/providers/keys/{service}", json=body)
    assert r.status_code == status, r.text
    assert _written() == {}
    for var in _VARS:
        assert var not in os.environ


def test_replacing_a_hand_set_endpoint_needs_replace_custom(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "https://my-gateway.example/v1")
    c = _client()
    body = {"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "coding-plan"}
    r = c.put("/api/providers/keys/qwen", json=body)
    assert r.status_code == 409 and "my-gateway" not in r.text
    assert _written() == {} and os.environ["QWEN_BASE_URL"] == "https://my-gateway.example/v1"
    r = c.put("/api/providers/keys/qwen", json={**body, "replace_custom": True})
    assert r.status_code == 200 and _svc(r, "qwen")["plan"] == "coding-plan"
    assert _svc(r, "qwen")["endpoint_set"] is True


def test_a_yaml_pin_refuses_a_plan_and_writes_nothing():
    cfg = {"slash_commands": {"qwen": {"base_url": "https://dashscope-us.aliyuncs.com/compatible-mode/v1"}}}
    r = _client(cfg).put("/api/providers/keys/qwen",
                         json={"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "coding-plan"})
    assert r.status_code == 409
    assert _written() == {}


def test_plan_routes_are_the_operators():
    c = TestClient(create_app({"model": {"provider": "llama_cpp"}, "web": {"api_token": "secret"}}))
    r = c.put("/api/providers/keys/qwen", json={"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "coding-plan"})
    assert r.status_code == 401
    assert _written() == {}


# ── No restart ───────────────────────────────────────────────────────


def test_a_saved_plan_takes_effect_without_a_restart():
    router = _router()
    c = _client(router=router)
    c.put("/api/providers/keys/kimi", json={"values": {"MOONSHOT_API_KEY": "sk-payg-old"}})

    # A session already on Kimi, its provider built against pay as you go.
    router.set_override("s1", resolve_model_target("kimi", {}), key="kimi", persist=False)
    entry = router._overrides["s1"]
    entry.provider = ProviderRegistry.create(entry.provider_config)
    assert entry.provider._base_url == "https://api.moonshot.ai/v1"

    r = c.put("/api/providers/keys/kimi",
              json={"values": {"MOONSHOT_API_KEY": KIMI_KEY}, "plan": "kimi-code"})
    assert r.status_code == 200, r.text

    # The built provider is dropped; the next turn builds against the plan, its key and its model.
    assert entry.provider is None
    assert entry.provider_config["model"] == "kimi-for-coding"
    rebuilt = ProviderRegistry.create(entry.provider_config)
    assert rebuilt._base_url == "https://api.kimi.ai/coding/v1"
    assert rebuilt._api_key == KIMI_KEY

    # A new choice resolves the same way, and the picker offers the plan's models.
    assert ProviderRegistry.create(resolve_model_target("kimi", {}))._base_url == "https://api.kimi.ai/coding/v1"
    assert resolve_model_choices("kimi", {})[0] == "kimi-for-coding"
    models = {m["key"] for m in c.get("/api/models").json()["models"]}
    assert "kimi:kimi-for-coding-highspeed" in models
    assert not any(k.startswith("kimi:kimi-k2.6") for k in models)


def test_a_model_the_new_plan_does_not_serve_falls_back_to_its_default():
    router = _router()
    c = _client(router=router)
    router.set_override("s1", resolve_model_target("qwen:qwen3.8-flash", {}), key="qwen:qwen3.8-flash",
                        persist=False)
    c.put("/api/providers/keys/qwen", json={"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "coding-plan"})
    assert router._overrides["s1"].provider_config["model"] == "qwen3.7-plus"


def test_a_model_pinned_in_yaml_still_wins_over_the_plans_default(monkeypatch):
    monkeypatch.setenv("MOONSHOT_BASE_URL", "https://api.kimi.ai/coding/v1")
    cfg = {"slash_commands": {"kimi": {"model": "k3"}}}
    assert resolve_model_target("kimi", cfg)["model"] == "k3"


# ── Terms: a warning, never a refusal ────────────────────────────────


def test_an_interactive_plan_on_an_unattended_route_warns(monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", "https://coding-intl.dashscope.aliyuncs.com/v1")
    cfg = {
        "model": {"provider": "qwen"},
        "router": {"fallback": [{"provider": "glm", "base_url": "https://api.z.ai/api/coding/paas/v4"}]},
    }
    warnings = plans.unattended_warnings(cfg)
    assert len(warnings["qwen"]) == 1 and "Coding Plan" in warnings["qwen"][0]
    assert "router.fallback[0]" in warnings["glm"][0]
    assert plans.plan_use_warning("qwen", cfg)
    r = TestClient(create_app(cfg)).get("/api/providers/keys")
    assert _svc(r, "qwen")["warnings"] == warnings["qwen"]
    # ...and the key still saves: the terms are a label.
    ok = TestClient(create_app(cfg)).put(
        "/api/providers/keys/qwen", json={"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "token-plan"})
    assert ok.status_code == 200


def test_metered_plans_and_other_providers_do_not_warn(monkeypatch):
    cfg = {"model": {"provider": "qwen"}, "router": {"fallback": [{"provider": "deepseek"}]}}
    assert plans.unattended_warnings(cfg) == {}
    assert plans.plan_use_warning("qwen", cfg) is None
    assert plans.plan_use_warning("deepseek", cfg) is None


# ── The contract fixture Beacon records against ──────────────────────


def test_the_get_fixture_is_what_the_route_serves():
    import json

    fixture = json.loads((Path(__file__).parent / "fixtures" / "provider_plans" / "get_keys.json").read_text())
    live = _client().get("/api/providers/keys").json()
    wanted = {s["id"] for s in fixture["services"]}
    live["services"] = [s for s in live["services"] if s["id"] in wanted]
    assert live == fixture, "regenerate tests/fixtures/provider_plans/get_keys.json from the route"


def test_the_bridge_advertises_plans():
    from prometheus.web.ws_server import WebSocketBridge

    assert "plans" in WebSocketBridge(loop_context=object(), config={})._server_capabilities()


def test_a_plan_for_a_provider_swapped_in_yaml_is_refused():
    cfg = {"slash_commands": {"qwen": {"provider": "openai_compatible"}}}
    r = _client(cfg).put("/api/providers/keys/qwen",
                         json={"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "coding-plan"})
    assert r.status_code == 409
    assert _written() == {}


def test_key_and_endpoint_are_one_write(monkeypatch):
    import prometheus.config.env_file as env_file

    def boom(values, path=None):
        raise OSError("disk full")

    monkeypatch.setattr(env_file, "set_env_values", boom)
    with pytest.raises(OSError):
        _client().put("/api/providers/keys/qwen",
                      json={"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "coding-plan"})
    assert "QWEN_API_KEY" not in os.environ and "QWEN_BASE_URL" not in os.environ


def test_a_fallen_back_choice_says_so():
    router = _router()
    stored: list = []
    router.persist_override = lambda sid, key: stored.append((sid, key))
    c = _client(router=router)
    router.set_override("s1", resolve_model_target("qwen:qwen3.8-flash", {}), key="qwen:qwen3.8-flash")
    c.put("/api/providers/keys/qwen", json={"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "coding-plan"})
    assert router._overrides["s1"].key == "qwen"
    assert stored[-1] == ("s1", "qwen")


def test_the_routers_own_built_providers_of_that_provider_are_dropped():
    cfg = RouterConfig(
        fallback_chain=[{"provider": "qwen", "model": "qwen3.8-max"}, {"provider": "deepseek"}],
        simple_provider={"provider": "qwen", "model": "qwen3.8-flash"},
        escalation_provider={"provider": "deepseek"},
    )
    router = ModelRouter(cfg, primary_provider=object(), primary_adapter=object(), primary_model="m")
    sentinel = object()
    router._fallback_cache = [(c, sentinel) for c, _ in router._fallback_cache]
    router._simple_provider = sentinel
    router._escalation_provider = sentinel
    router._task_rule_providers = {(("provider", repr("qwen")),): sentinel, (("provider", repr("glm")),): sentinel}
    router.refresh_preset("qwen", {})
    assert [cached for _, cached in router._fallback_cache] == [None, sentinel]
    assert router._simple_provider is None and router._escalation_provider is sentinel
    assert list(router._task_rule_providers) == [(("provider", repr("glm")),)]


def test_every_subscription_plan_is_billed_as_a_subscription():
    from prometheus.telemetry.cost import SUBSCRIPTION_HOST_MARKERS

    for service_plans in plans.PROVIDER_PLANS.values():
        for plan in service_plans:
            if not plan.endpoint:
                continue
            is_sub = any(m in plan.endpoint for m in SUBSCRIPTION_HOST_MARKERS)
            assert is_sub == (plan.terms == plans.TERMS_INTERACTIVE), plan.id


def test_workspace_plan_carries_its_regions_and_every_plan_says_separate_keys():
    qwen = _svc(_client().get("/api/providers/keys"), "qwen")
    by_id = {p["id"]: p for p in qwen["plans"]}
    assert by_id["workspace"]["regions"] == list(plans.WORKSPACE_REGIONS)
    assert all(p["regions"] == [] for p in qwen["plans"] if p["id"] != "workspace")
    assert all(p["separate_keys"] is True for p in qwen["plans"])


def test_the_longest_workspace_id_still_builds():
    assert plans.workspace_endpoint("a" * 60, "us-east-1").startswith("https://" + "a" * 60 + ".us-east-1.")


def test_error_bodies_never_quote_a_key_or_a_built_host(monkeypatch):
    c = _client()
    bodies = [
        {"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "workspace", "workspace": {"id": WS_ID, "region": "moon-1"}},
        {"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "nope"},
        {"values": {"QWEN_API_KEY": QWEN_KEY + "\nX=1"}, "plan": "coding-plan"},
        {"values": {"QWEN_API_KEY": QWEN_KEY, "OPENAI_API_KEY": QWEN_KEY}},
    ]
    for body in bodies:
        r = c.put("/api/providers/keys/qwen", json=body)
        assert r.status_code >= 400
        assert QWEN_KEY not in r.text and WS_ID not in r.text and "maas.aliyuncs" not in r.text
    monkeypatch.setenv("QWEN_BASE_URL", plans.workspace_endpoint(WS_ID, "us-east-1"))
    r = _client({"slash_commands": {"qwen": {"base_url": "https://pinned.example/v1"}}}).put(
        "/api/providers/keys/qwen", json={"values": {"QWEN_API_KEY": QWEN_KEY}, "plan": "payg-us"})
    assert r.status_code == 409 and "pinned.example" not in r.text and WS_ID not in r.text
