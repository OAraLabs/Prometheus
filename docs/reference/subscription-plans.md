# Subscription plans: the contract

How a client (the desktop and phone apps) lets the owner choose a chat provider's subscription plan together with its
key. Pinned to the code: `src/prometheus/providers/plans.py` (the list), `src/prometheus/web/server.py` (the routes),
`tests/test_provider_plans.py` (the behaviour), and `tests/fixtures/provider_plans/get_keys.json` (a `GET` answer
recorded from the route, held current by a test; record your client fixtures from it).

## What a plan is

Some providers sell the same models on more than one host: pay as you go, a coding subscription, a region, your own
workspace. Each host takes its own key, and a key works only against the host it was issued for. A plan is one of those
hosts, from a **fixed list in the daemon**. There is no free-form endpoint anywhere in this API: a client can never
send a key to a host the daemon does not name.

| Provider (`id`) | Plans (`plans[].id`, default first) |
|---|---|
| Qwen (`qwen`) | `payg-intl`, `payg-us`, `token-plan`, `coding-plan`, `workspace` |
| GLM (`glm`) | `payg`, `coding-plan` |
| Kimi (`kimi`) | `payg`, `kimi-code` |

Every other service has `plans: []`: one host, no choice.

## Capability

The bridge's `subscribed` ack lists `"plans"` in its capabilities, and `GET /api/providers/keys` carries
`"plans_version": 1`. With an older daemon (neither present), show no plan choice and save keys exactly as before.

## `GET /api/providers/keys`

Owner only, as before. New fields, at the top level:

- `plans_version`: `1`.
- `workspace_regions`: the regions a workspace host may be in, for the workspace sheet's picker.

And per service:

- `plans`: each `{id, label, endpoint, terms, terms_note, models, workspace}`.
  - `endpoint`: the documented public host, for display only. `null` for the workspace plan. Never send it back.
  - `terms`: `"metered"` (ordinary pay as you go) or `"interactive"` (the provider limits it to a person working in a
    coding tool or agent). Show `terms_note` beside an interactive plan.
  - `models`: the models this plan serves, default first; empty means the provider's usual list.
  - `workspace`: true for the one plan that needs a workspace id and region.
- `plan`: the plan in use, as an id. `"workspace"` for a workspace host and `"custom"` for an endpoint set by hand that
  is on no list. **The host itself is never returned.** `null` for a service without plans (or one whose provider the
  owner swapped in yaml).
- `endpoint_env`: the name of the variable the plan is stored in (`QWEN_BASE_URL`), or `null`. A name only.
- `plan_pinned_in_config`: true when `prometheus.yaml` pins `slash_commands.<id>.base_url`. That wins over a saved
  plan, so a `PUT` with a plan is refused (409); say so instead of offering the choice.
- `warnings`: lines naming each route that runs this provider unattended (the primary model, a router fallback, a
  rule, smart routing, escalation) while it is on an interactive plan. Show them; nothing is blocked.
- `default_model` now follows the plan: on Kimi Code it reads `kimi-for-coding`.

## `PUT /api/providers/keys/{id}`

Owner only, as before. The body may add a plan:

```json
{"values": {"QWEN_API_KEY": "sk-…"}, "plan": "coding-plan"}
{"values": {"QWEN_API_KEY": "sk-…"}, "plan": "workspace", "workspace": {"id": "ws-…", "region": "ap-southeast-1"}}
```

- A plan is saved **only together with its key**: `values` must carry the service's key, non-empty. A plan alone is
  400.
- `plan` must be an id from that service's `plans`. Anything else (a URL, a non-string, a service with no plans) is 400.
- `workspace` is required for, and only allowed with, the workspace plan. The id is one DNS label (lowercase letters,
  digits, hyphens, no dot); the region is one of `workspace_regions`. The daemon builds the host itself.
- The endpoint variable can never be sent in `values`: it is not the service's, so it is 400 like any other name.
- Every refusal writes nothing.
- Choosing the default plan clears the variable, so the provider's default host applies.
- A body with no `plan` behaves exactly as before: the key is saved and the plan in use is unchanged.
- The answer is the same as `GET`.

## Takes effect without a restart

The key and endpoint are written to the env file (0600) and the daemon's own environment. The next model choice of
that provider resolves against them, and conversations already on it rebuild against them on their next turn. A model
the new plan does not serve falls back to the plan's default. The daemon's own primary model (`/api/model`) still
applies on restart, as today.

## Terms

A plan marked `interactive` is a label, not a gate. The daemon warns at boot (`plans: …` in the log) and in `warnings`
when such a plan sits on a route that runs with no person waiting. `providers.plans.plan_use_warning(service_id)` is the
one line any future scheduled job or routine that picks a model shows before it runs; it never refuses.

## Where each host comes from

Read from each provider's own documentation on 2026-11-23: Alibaba Model Studio's base-URL and Coding Plan FAQ pages,
Z.ai's coding-tool page, and the Kimi Code docs. Each plan uses the provider's OpenAI-compatible host, the wire format
these providers already speak here. Their Anthropic-compatible hosts are not offered.
