# Provenance: Novel code for Prometheus
# License: MIT
"""Subscription plans: the fixed list of endpoints a chat provider's key may be pointed at.

Several providers sell the same models on more than one host: pay as you go, a coding subscription, a region, a
workspace of your own. Each host takes its own key, and a key only works against the host it was issued for. A plan
here is one of those hosts, named, with a one-line note on its terms.

**The list is the boundary.** ``PUT /api/providers/keys/{id}`` writes a provider's endpoint variable only from this
list (or, for a workspace, from the one host pattern the provider documents, with an id and a region checked against
it), so a client can never send a key to a host this file does not name. There is no free-form endpoint.

**Terms are a label, not a gate.** A plan meant for interactive coding tools says so (``terms == "interactive"``), and
anything that would run it unattended gets a warning (:func:`unattended_warnings`). Nothing is refused for it.

**Reading never reveals a workspace host.** :func:`active_plan` answers with a plan id: ``"workspace"`` for a
workspace host and ``"custom"`` for an endpoint set by hand that matches no plan, never the host itself.

Every endpoint below was read from the provider's own documentation on 2026-11-23:
- Alibaba Model Studio: alibabacloud.com/help/en/model-studio/base-url and …/coding-plan-faq
- Z.ai: docs.z.ai/devpack/tool/others
- Kimi Code: kimi.com/code/docs/en/
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: A plan's terms: ``"metered"`` is ordinary pay-as-you-go use with no limit on what calls it; ``"interactive"`` is a
#: subscription its provider limits to a person working in a coding tool or agent, not unattended or backend use.
TERMS_METERED = "metered"
TERMS_INTERACTIVE = "interactive"

#: The plan id :func:`active_plan` answers for an endpoint set by hand (yaml ``base_url`` or the variable) that is on
#: no list. Never the endpoint itself.
CUSTOM_PLAN = "custom"

#: The plan id of a workspace-scoped host (Alibaba). Its host names the account, so it is never read back.
WORKSPACE_PLAN = "workspace"


@dataclass(frozen=True)
class Plan:
    """One host a provider's key may be pointed at."""

    id: str
    label: str
    #: The base URL, exactly as the provider documents it. Empty for the workspace plan, whose host is built by
    #: :func:`workspace_endpoint` from a checked id and region.
    endpoint: str
    terms: str
    #: One line on what the plan is for, shown beside it.
    terms_note: str
    #: The models this plan serves, default first. Empty means the provider's own list (``PRESET_MODEL_CHOICES``).
    models: tuple[str, ...] = ()
    #: True when this plan's keys are issued separately from the provider's other plans, so a key saved for one
    #: never works on another. The PUT route requires a newly sent key with EVERY plan change regardless.
    separate_keys: bool = True


_INTERACTIVE_NOTE_ALIBABA = (
    "For interactive coding tools and agents only, not automated scripts, backends or scheduled jobs."
)

#: Per catalog key (``OVERRIDE_PRESETS``), the plans in UI order. The FIRST is the default: its endpoint is the
#: provider's ``CLOUD_DEFAULTS`` base URL, and choosing it clears the endpoint variable. A provider with no entry here
#: has one host and no plan choice.
PROVIDER_PLANS: dict[str, tuple[Plan, ...]] = {
    "qwen": (
        Plan(
            id="payg-intl",
            label="Pay as you go (international)",
            endpoint="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            terms=TERMS_METERED,
            terms_note="Ordinary metered API use.",
        ),
        Plan(
            id="payg-us",
            label="Pay as you go (US)",
            endpoint="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            terms=TERMS_METERED,
            terms_note="Ordinary metered API use, served from the US region.",
        ),
        Plan(
            id="token-plan",
            # Singapore is the only region Alibaba offers the Token Plan in (base-url page, read 2026-11-23); there
            # is no US Token Plan host.
            label="Token Plan (Singapore)",
            endpoint="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
            terms=TERMS_INTERACTIVE,
            terms_note=_INTERACTIVE_NOTE_ALIBABA,
        ),
        Plan(
            id="coding-plan",
            label="Coding Plan",
            endpoint="https://coding-intl.dashscope.aliyuncs.com/v1",
            terms=TERMS_INTERACTIVE,
            terms_note=_INTERACTIVE_NOTE_ALIBABA,
            # The Qwen models the Coding Plan FAQ names (its context-window table).
            models=("qwen3.7-plus", "qwen3.6-plus", "qwen3.5-plus"),
        ),
        Plan(
            id=WORKSPACE_PLAN,
            label="Workspace",
            endpoint="",
            terms=TERMS_METERED,
            terms_note="Your own workspace's host; it takes that workspace's key.",
        ),
    ),
    "glm": (
        Plan(
            id="payg",
            label="Pay as you go",
            endpoint="https://api.z.ai/api/paas/v4",
            terms=TERMS_METERED,
            terms_note="Ordinary metered API use.",
        ),
        Plan(
            id="coding-plan",
            label="GLM Coding Plan",
            endpoint="https://api.z.ai/api/coding/paas/v4",
            terms=TERMS_INTERACTIVE,
            terms_note="For the coding tools the plan supports; use elsewhere is outside its terms.",
            # The models the Coding Plan FAQ lists as callable.
            models=("glm-5.3", "glm-5.3-flash"),
        ),
    ),
    "kimi": (
        Plan(
            id="payg",
            label="Pay as you go",
            endpoint="https://api.moonshot.ai/v1",
            terms=TERMS_METERED,
            terms_note="Ordinary metered API use.",
        ),
        Plan(
            id="kimi-code",
            label="Kimi Code",
            # The overseas OpenAI-compatible host; the China host (api.kimi.com) takes China-issued memberships.
            endpoint="https://api.kimi.ai/coding/v1",
            terms=TERMS_INTERACTIVE,
            terms_note=(
                "A membership for programming tools; for your own product or unattended work, use pay as you go."
            ),
            # The ids the Kimi Code docs list for both protocols, minus k3.1-pro (it rejects requests with thinking
            # off, which this daemon sends for side calls).
            models=("kimi-for-coding", "kimi-for-coding-highspeed", "k3", "k3-256k"),
        ),
    ),
}

#: The regions Alibaba documents a workspace host in (base-url page, 2026-11-23).
WORKSPACE_REGIONS: tuple[str, ...] = (
    "ap-southeast-1", "us-east-1", "eu-central-1", "ap-northeast-1", "cn-hongkong", "cn-beijing",
)

#: A workspace id: one DNS label, lowercase letters, digits and hyphens. No dot, so it can never reach outside
#: ``<id>.<region>.maas.aliyuncs.com``.
_WORKSPACE_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,58}[a-z0-9])?")
_WORKSPACE_DOMAIN = ".maas.aliyuncs.com"
_WORKSPACE_HOST = re.compile(
    r"https://(?P<id>[a-z0-9](?:[a-z0-9-]{0,58}[a-z0-9])?)\.(?P<region>[a-z0-9-]+)"
    r"\.maas\.aliyuncs\.com/compatible-mode/v1"
)
#: Hosts under the same domain that are plans of their own, never a workspace.
_NOT_WORKSPACE_IDS = frozenset({"token-plan", "trial"})


class PlanError(ValueError):
    """A plan or workspace that is not on the list. The message is safe to show: it never echoes a host."""


def plans_for(service_id: str) -> tuple[Plan, ...]:
    return PROVIDER_PLANS.get(service_id, ())


def plan_by_id(service_id: str, plan_id: str) -> Plan | None:
    return next((p for p in plans_for(service_id) if p.id == plan_id), None)


def endpoint_env_for(service_id: str) -> str:
    """The variable a service's endpoint is read from (``QWEN_BASE_URL``), or ``""`` for one without plans."""
    if not plans_for(service_id):
        return ""
    from prometheus.providers.registry import CLOUD_DEFAULTS
    from prometheus.router.model_router import OVERRIDE_PRESETS

    provider = OVERRIDE_PRESETS.get(service_id, {}).get("provider", "")
    return str(CLOUD_DEFAULTS.get(provider, {}).get("default_base_url_env", ""))


def workspace_endpoint(workspace_id: str, region: str) -> str:
    """The workspace host for a checked id and region. :class:`PlanError` for anything else."""
    if not isinstance(workspace_id, str) or not _WORKSPACE_ID.fullmatch(workspace_id):
        raise PlanError("workspace id must be one label of lowercase letters, digits and hyphens")
    if workspace_id in _NOT_WORKSPACE_IDS:
        raise PlanError("that id names a plan host, not a workspace; choose the plan instead")
    if region not in WORKSPACE_REGIONS:
        raise PlanError(f"workspace region must be one of: {', '.join(WORKSPACE_REGIONS)}")
    endpoint = f"https://{workspace_id}.{region}{_WORKSPACE_DOMAIN}/compatible-mode/v1"
    # Belt and braces over the two checks above: the built URL must parse to a host in the provider's domain.
    from urllib.parse import urlsplit

    parts = urlsplit(endpoint)
    host = parts.hostname or ""
    if (
        parts.scheme != "https" or parts.port is not None or parts.username or parts.query or parts.fragment
        or host != f"{workspace_id}.{region}{_WORKSPACE_DOMAIN}" or not host.endswith(_WORKSPACE_DOMAIN)
    ):
        raise PlanError("workspace id and region do not make a host in the provider's domain")
    return endpoint


def _is_workspace_host(endpoint: str) -> bool:
    match = _WORKSPACE_HOST.fullmatch(endpoint)
    return bool(
        match and match.group("region") in WORKSPACE_REGIONS and match.group("id") not in _NOT_WORKSPACE_IDS
    )


def plan_for_endpoint(service_id: str, endpoint: str) -> str | None:
    """The plan id an endpoint is, ``"custom"`` when it is on no list, or None for a service without plans."""
    plans = plans_for(service_id)
    if not plans:
        return None
    normalized = endpoint.strip().rstrip("/")
    if not normalized:
        return plans[0].id
    for plan in plans:
        if plan.endpoint and plan.endpoint.rstrip("/") == normalized:
            return plan.id
    if plan_by_id(service_id, WORKSPACE_PLAN) is not None and _is_workspace_host(normalized):
        return WORKSPACE_PLAN
    return CUSTOM_PLAN


def active_plan(
    service_id: str,
    config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Which plan a service is on now: the one its provider would be built against on the next turn.

    A ``slash_commands.<id>.base_url`` in yaml wins over the variable (as it does for the provider), so a pinned
    endpoint reads as whatever it matches. A ``provider`` swapped in yaml takes the service off its plans: None, the
    same as a service that has none.
    """
    plans = plans_for(service_id)
    if not plans:
        return None
    user_entry = ((config or {}).get("slash_commands") or {}).get(service_id) or {}
    from prometheus.router.model_router import OVERRIDE_PRESETS

    if user_entry.get("provider") and user_entry["provider"] != OVERRIDE_PRESETS[service_id]["provider"]:
        return None
    pinned = user_entry.get("base_url")
    if pinned:
        return plan_for_endpoint(service_id, str(pinned))
    environ = os.environ if env is None else env
    return plan_for_endpoint(service_id, environ.get(endpoint_env_for(service_id), ""))


def active_plan_models(service_id: str, config: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """The model list of the plan a service is on, or ``()`` when that plan has none of its own."""
    plan_id = active_plan(service_id, config)
    plan = plan_by_id(service_id, plan_id) if plan_id else None
    return plan.models if plan else ()


def base_url_pinned_in_config(service_id: str, config: Mapping[str, Any] | None) -> bool:
    """True when yaml pins ``slash_commands.<id>.base_url``: a saved plan would not take effect over it."""
    user_entry = ((config or {}).get("slash_commands") or {}).get(service_id) or {}
    return bool(user_entry.get("base_url"))


def plan_is_interactive(service_id: str, config: Mapping[str, Any] | None = None) -> bool:
    plan_id = active_plan(service_id, config)
    plan = plan_by_id(service_id, plan_id) if plan_id else None
    return bool(plan and plan.terms == TERMS_INTERACTIVE)


def service_for_provider(provider: str) -> str | None:
    """The catalog key with plans whose preset uses *provider* (``"qwen"`` → ``"qwen"``), or None."""
    from prometheus.router.model_router import OVERRIDE_PRESETS

    for service_id in PROVIDER_PLANS:
        if OVERRIDE_PRESETS.get(service_id, {}).get("provider") == provider:
            return service_id
    return None


def _unattended_routes(config: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """(what runs unattended, provider name, its own base_url or "") for each configured route that serves work
    with no person waiting.

    The router section is read through the router's own parser, so an entry it would refuse is not warned about.
    """
    routes: list[tuple[str, str, str]] = []

    def add(what: str, entry: Any) -> None:
        if isinstance(entry, Mapping) and entry.get("provider"):
            routes.append((what, str(entry["provider"]), str(entry.get("base_url") or "")))

    add("the primary model (the daily briefing and background work run on it)", config.get("model"))
    from prometheus.router.model_router import read_router_config

    rc = read_router_config(dict(config))
    if rc.smart_routing_enabled:
        add("router.smart_routing.simple_provider", rc.simple_provider)
    if rc.escalation_enabled:
        add("router.escalation.provider", rc.escalation_provider)
    for i, entry in enumerate(rc.fallback_chain):
        add(f"router.fallback[{i}]", entry)
    for i, rule in enumerate(rc.task_rules):
        add(f"router.rules[{i}]", {"provider": rule.provider, "base_url": rule.base_url})
    return routes


def unattended_warnings(config: Mapping[str, Any] | None) -> dict[str, list[str]]:
    """Per service id, a warning for each route that would run an interactive-only plan unattended.

    A warning, never a refusal: the plan's terms are the provider's to enforce, and a person may well be at the
    keyboard. Empty when nothing configured is on such a plan.
    """
    out: dict[str, list[str]] = {}
    for what, provider, base_url in _unattended_routes(config or {}):
        service_id = service_for_provider(provider)
        if service_id is None:
            continue
        if base_url:
            plan_id = plan_for_endpoint(service_id, base_url)
        else:
            # A route of its own reads the endpoint variable directly; slash_commands pins do not apply to it.
            plan_id = plan_for_endpoint(service_id, os.environ.get(endpoint_env_for(service_id), ""))
        plan = plan_by_id(service_id, plan_id or "")
        if plan is None or plan.terms != TERMS_INTERACTIVE:
            continue
        out.setdefault(service_id, []).append(
            f"{what} uses {plan.label}, a plan for interactive use; unattended calls are outside its terms"
        )
    return out


def plan_use_warning(service_id: str, config: Mapping[str, Any] | None = None) -> str | None:
    """One line for a job or routine about to run on *service_id* with no person waiting, or None.

    The hook for anything scheduled that picks a model: show it, never refuse on it.
    """
    if not plan_is_interactive(service_id, config):
        return None
    plan = plan_by_id(service_id, active_plan(service_id, config) or "")
    label = plan.label if plan else service_id
    return f"{label} is for interactive use; a scheduled or unattended run is outside its terms"
