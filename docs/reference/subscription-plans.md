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

`GET /api/providers/keys` carries `"capabilities": {"plans": 1}`, and the bridge's `subscribed` ack lists `"plans"`.
With an older daemon (neither present), show no plan choice and save keys exactly as before.

## `GET /api/providers/keys`

Owner only, as before. New at the top level:

- `capabilities`: `{"plans": 1}`.
- `workspace_regions`: the regions a workspace host may be in, for the workspace sheet's picker.

And per service:

- `plans`: each `{id, label, subscription, terms, default, custom, regions, separate_keys, models}`.
  - `subscription`: true for a plan its provider limits to a person working in a coding tool or agent.
  - `terms`: the one-line terms label for such a plan, else `null`. A label: nothing is blocked on it.
  - `default`: true for the provider's default host (the first plan).
  - `custom`: true for the one plan that takes input of its own (a workspace id and region).
  - `regions`: for a custom plan, the region slugs it accepts (the same list as `workspace_regions`); empty otherwise.
  - `separate_keys`: true when the plan's keys are issued separately. Every plan change needs a key in the same `PUT`
    regardless, so the daemon enforces this for every plan.
  - `models`: the models this plan serves, default first; empty means the provider's usual list.
- `plan`: the plan in use, as an id. When nothing is set, the default plan's id. `"workspace"` for a workspace host,
  `"custom"` for an endpoint set by hand that is on no list. `null` for a service without plans (or one whose provider
  the owner swapped in yaml).
- `endpoint_set`: whether the endpoint variable is set at all (false on the default plan).
- `plan_pinned_in_config`: true when `prometheus.yaml` pins `slash_commands.<id>.base_url`. That wins over a saved
  plan, so a `PUT` with a plan is refused (409). Say so instead of offering the choice.
- `warnings`: lines naming each route that runs this provider unattended (the primary model, a router fallback, a
  rule, smart routing, escalation) while it is on a subscription plan. Show them. Nothing is blocked.
- `default_model` now follows the plan: on Kimi Code it reads `kimi-for-coding`.

**No host or base URL appears anywhere in the answer.** A client picks plans by id.

## `PUT /api/providers/keys/{id}`

Owner only, as before. The body may add a plan:

```json
{"values": {"QWEN_API_KEY": "sk-…"}, "plan": "coding-plan"}
{"values": {"QWEN_API_KEY": "sk-…"}, "plan": "workspace", "workspace": {"id": "ws-…", "region": "ap-southeast-1"}}
```

- A plan is saved **only together with its key**: `values` must carry the service's key, non-empty. A plan alone is
  400. (A plan's host takes only that plan's key, so moving an old key to a new host is never what was meant.)
- `plan` must be an id from that service's `plans`. Anything else (a URL, a non-string, a service with no plans) is 400.
- `workspace` is required for, and only allowed with, the workspace plan. The id matches
  `^[a-z0-9](?:[a-z0-9-]{0,58}[a-z0-9])?$`, and the region is one of `regions`. The daemon builds the host itself, then
  checks that it parses to a host in the provider's domain.
- Error bodies never quote a key value or a host, whether built or set by hand.
- Replacing an endpoint set by hand (`plan: "custom"`) needs `"replace_custom": true`, else 409.
- The endpoint variable can never be sent in `values`: it is not the service's, so it is 400 like any other name.
- A service whose provider the owner swapped in yaml (`slash_commands.<id>.provider`) refuses a plan with 409.
- Every refusal writes nothing.
- Choosing the default plan clears the variable, so the provider's default host applies.
- A body with no `plan` behaves exactly as before: the key is saved and the plan in use is unchanged.
- The answer is the same as `GET`.

## Takes effect without a restart

The key and endpoint are written to the env file (0600) in one rewrite, then to the daemon's own environment. The next
model choice of that provider resolves against them, and conversations already on it rebuild against them on their
next turn. So do the router's own routes on that provider (fallback entries, rules, smart routing, escalation). A
model the new plan does not serve falls back to the plan's default, and the conversation's stored choice says so. A
model pinned in `slash_commands.<id>.model` still wins over the plan's default. The daemon's own primary model
(`/api/model`) still applies on restart, as today.

## Terms

A plan marked `interactive` is a label, not a gate. The daemon warns at boot (`plans: …` in the log) and in `warnings`
when such a plan sits on a route that runs with no person waiting. `providers.plans.plan_use_warning(service_id)` is the
one line any future scheduled job or routine that picks a model shows before it runs; it never refuses.

## Where each host comes from

Read from each provider's own documentation on 2026-11-23: Alibaba Model Studio's base-URL and Coding Plan FAQ pages,
Z.ai's coding-tool page, and the Kimi Code docs. Each plan uses the provider's OpenAI-compatible host, the wire format
these providers already speak here. Their Anthropic-compatible hosts are not offered.
