# WP-X.28 — Adapter tier for native tool-calling models the registry does not know

Design report, written read-only from a detached worktree at `origin/main` (`bfe8674`,
2026-09-25) and from the ladder branch `feat/model-ladder-suite` (`3bb6703`, not on
main). Nothing ran on the mini or the 4090 for it. Reviewed the same day; the decisions
are recorded at the end ("Decisions") and the PR plan below carries them. PR 1 was built
from `07921f1` (0.9.4).

## 0. The problem, restated from the code

The tier is chosen once, by name:

- `__main__._get_adapter_tier(provider_name, model_name)` (`src/prometheus/__main__.py:448`):
  `off` for `anthropic` or any `ProviderRegistry.is_cloud` name; `light` if
  `_has_native_tool_calling(model_name)` (`:403`) finds a `config/model_registry.yaml`
  entry whose `match_patterns` is a case-insensitive substring of the name and whose
  `function_calling.supported` is true; else `full`.
- `create_adapter(model_cfg, adapter_cfg)` (`:466`) builds the `ModelAdapter` for that tier:
  `light` = formatter by name (Gemma/Qwen), strictness NONE, 1 retry, adaptive on;
  `full` = QwenFormatter, MEDIUM, 3 retries. Only `light` logs a line (`:509`); `full`
  is silent unless the registry file is missing or unreadable (`:420`, `:439`).
- Callers: the daemon (`daemon.py:926`), the CLI (`__main__.py:1680`), coding mode
  (`__main__.py:851`), the gym (`gym/runner.py:71`), and six scripts.

A second chooser exists and ignores the registry entirely: the router builds adapters for
backend overrides, the fallback chain, task rules and smart routing through
`router/model_router.py::_build_adapter_for(provider_name)` (`:1190`, callers `:812`,
`:948`, `:994`, `:1028`, `:1090`, `:1143`). Cloud names get `off`; every local provider
gets `ModelAdapter(QwenFormatter(), strictness="MEDIUM")`, whose default tier is `full`.
So `/alt` to `qwen2.5:7b-instruct`, a registered model, runs at `full` today, and the
`repaired_tool_call` golden depends on exactly that (`scripts/parity/scenarios.py:281`).

What `full` does with a Qwen-family model:

1. `ModelAdapter.format_request` (`adapter/__init__.py:124`) appends the QwenFormatter
   instruction (`adapter/formatter.py:92`: "respond ONLY with a JSON object") to the
   system prompt and rewrites the tool schemas.
2. The loop withholds the tools from the payload at `full` (`engine/agent_loop.py:1436`),
   so `LlamaCppProvider._build_request_payload` (`providers/llama_cpp.py:634`) sends the
   GBNF grammar instead. The grammar is `root ::= tool-call | prose`, `prose ::= [^{]
   anychar*` (`adapter/enforcer.py`): a reply beginning `<tool_call>` is legal prose and
   is generated unconstrained.
3. With no `tools` in the request, llama-server's `--jinja` template renders no tool block
   and parses nothing; the model, trained on its template's format, still writes it.
4. The reply's text goes to `ModelAdapter.extract_tool_calls` →
   `StructuredOutputEnforcer.extract_tool_calls` (`adapter/enforcer.py:37`), four JSON
   strategies and nothing else. `QwenFormatter.parse_tool_calls` (`formatter.py:140`) has
   no caller in `src/` (tests only); the enforcer is the reader that matters.
5. `strip_tool_call_markup` then deletes every `<tool_call>…</tool_call>` span
   (`agent_loop.py:1766`). A reply that was only XML strips to nothing → "PARSE
   DISAGREEMENT" → `_strip_disagreement_feedback` (`:2677`) asks for JSON-in-tag and
   retries under the circuit breaker. XML after prose is deleted silently and the prose
   becomes the answer.

That is the 21 `parse_disagreement` halts in the Bonsai sweep
(`gym/results/ladder/tier-sweep-r27b-pq2-20260925.md`, ladder branch): 611 runs, `off`
188/203, `light` 186/204, `full` 135/204; paired `full − light` −0.250 (−0.338 to −0.172,
19 tasks flipped); `full` stopped by `parse_disagreement` 21, `empty_response` 7,
`circuit_breaker` 2. `full` also sends far smaller prompts (6.2k tokens in per run against
24.7k at `light`) because the schemas never reach the model.

`light` sends the tools natively (`llama_cpp.py:607`, `ollama.py:110`) and no grammar;
llama-server parses the reply into structured `tool_calls`. Escalation at `light`
(`adapter/__init__.py:268`, `_bump_tool_strictness`) raises strictness per tool; it never
withholds native tools. The circuit breaker's one-shot bump (`agent_loop.py:331`,
`_TIER_BUMP_LADDER`) does go `light → full` on a per-run copy of the adapter.

### What the servers expose (evidence from the committed goldens, not a live call)

`tests/fixtures/parity/tool_calls.trace.json` records the 4090's `GET /props`
(build `b1-9d57ce456`, 2026-09-25). Top-level keys:

```
bos_token build_info chat_template chat_template_caps cors_proxy_enabled
default_generation_settings endpoint_metrics endpoint_props endpoint_slots eos_token
is_sleeping media_marker modalities model_alias model_ftype model_path total_slots ui ui_settings
```

`chat_template_caps` is llama.cpp's own analysis of the template it will render:

```json
{"supports_object_arguments": true, "supports_parallel_tool_calls": true,
 "supports_preserve_reasoning": true, "supports_reasoning_effort": true,
 "supports_string_content": true, "supports_system_role": true,
 "supports_tool_calls": true, "supports_tools": true, "supports_typed_content": false}
```

`chat_template` (9,993 chars) renders tools as `<tools>…</tools>` in a system turn and
instructs, verbatim: "If you choose to call a function ONLY reply in the following format
with NO suffix: `<tool_call>` `<function=example_function_name>`
`<parameter=example_parameter_1>` `value_1` `</parameter>` … `</function>` `</tool_call>`".
The assistant side renders history calls the same way. This is the production 27B's
format, and Bonsai 2 is the same checkpoint (`gym/ladder/rungs.yaml`, `r27b-pq2`).
`scripts/run_model_ab_eval.py:384` already reads `chat_template_caps.supports_tool_calls`
and treats `false` as "is `--jinja` set?".

Every golden also records the alt backend's `POST /api/show {"model": "qwen2.5:7b-instruct"}`
(the backend registry's probe, `providers/backends.py:604`). The response has
`capabilities: ["completion", "tools"]` and a Go `template` whose assistant turn renders
`<tool_call>` `{"name": …, "arguments": …}` `</tool_call>` (JSON inside the tag: the
Qwen2.5/Hermes shape). `_probe_ollama` already keeps `capabilities` in
`BackendStatus.extra` (`backends.py:614`); `_probe_llama_cpp` (`:571`) reads `/props` but
not `chat_template_caps`.

### Where the tier is visible today

- Boot log: one INFO line for `light` only.
- `/doctor` (`gateway/commands.py:797` → `infra/doctor.py`): matches a registry family by
  name and prints "supports tool calling" from the registry (`doctor.py:445`); no tier.
  `oara doctor` (`cli/doctor.py:151`) checks reachability and a model name; no tier.
- `/api/status` (`web/server.py:843`): `model`, `provider`, a `context` block with a
  `source` field (`_context_block`), `backends`; no tier. `/api/models` rows (`:4570`)
  carry detected `vision`; no tier.
- Telemetry: `circuit_breaker_diagnostics.adapter_tier` only. The ladder records
  `adapter_tier`, `adapter_tier_start/end`, `tier_sweep` in its own rows (ladder branch,
  `gym/ladder/runner.py:766`).

### Parity facts the plan relies on

- Trigger: `.github/workflows/parity.yml:21-57` replays on changes to `engine/`, `hooks/`,
  `router/`, `adapter/`, `providers/`, `context/`, `daemon.py`, `__main__.py`, `config/**`,
  `config/model_registry.yaml`, `web/**`, `tools/**` and more. `src/prometheus/gym/**` is
  not listed.
- Probes (every GET, and POSTs that are not completions) are answered from the latest
  recording and never consumed (`scripts/parity/model_server.py:167`); an extra `/props`
  read or a second `/api/show` with the same body changes no trace. A completion whose
  normalized body changes is a diff.
- `*.expected.json` dumps every store with its table schemas
  (`circuit_breaker_diagnostics` columns appear in all 11). A new telemetry column changes
  all 11 goldens. This design adds none.
- The seam rule: `adapter/`, `engine/`, `hooks/`, `router/` changes need
  `parity_harness.py bench --baseline` on the mini (baseline p50 ≈ 17 ms per round,
  `tests/fixtures/parity/bench_baseline.json`, Python 3.11.15).
- The scenario's `config` is stored in the trace; changing a scenario's config is a
  re-record even when every exchange stays identical.

---

## 1. Auto-detect native tool calling from the chat template

### Design

One pure resolver, one provenance vocabulary, one decision record, and the existing
probes feeding it.

**New module `src/prometheus/adapter/tier.py`** (the adapter package owns the tier; it is
already on the parity path list, and `__main__` keeps only the wiring):

```python
TIER_SOURCES = ("override", "provider_class", "registry", "template", "fallback")

@dataclass(frozen=True)
class ToolTemplate:            # what a backend's template says; None = could not be read
    native: bool | None        # True/False = the template says; None = unreadable
    call_format: str | None    # "qwen-xml" | "qwen-json" | "unknown"
    evidence: str              # one human sentence, e.g. "llama.cpp /props chat_template_caps"

@dataclass(frozen=True)
class TierDecision:
    tier: str                  # off | light | full
    source: str                # one of TIER_SOURCES
    detail: str                # the sentence /doctor, the log and /api/status print
    call_format: str | None    # from the template when it decided or agreed
    decided_for: str           # the model name the decision was made for

def classify_template(*, chat_template: str | None, caps: dict | None,
                      ollama_capabilities: list | None, ollama_template: str | None) -> ToolTemplate
def resolve_tier(*, provider_name, model_name, template: ToolTemplate | None,
                 registry_entry: dict | None, override: str | None) -> TierDecision
```

`classify_template` rules, in order, each naming its evidence:

1. llama.cpp `chat_template_caps`: `supports_tools and supports_tool_calls` → native;
   both present and either false → not native; only one present → treated as
   unreadable (the caps are then not the shape this code was verified against).
2. Else llama.cpp `chat_template` text: native when a Jinja conditional references
   `tools` and the assistant branch renders `tool_calls`; the format from markers:
   `<function=` with `<parameter=` → `qwen-xml`; `<tool_call>` with a JSON object →
   `qwen-json`; else `unknown`.
3. Ollama: `"tools" in capabilities` → native; the Go `template` gives the format the same
   way (`.ToolCalls` plus the markers above).
4. Nothing → `ToolTemplate(native=None, …)`.

Only formats seen in committed evidence are named (the 4090's Qwen3.8 template and the
mini's qwen2.5 Ollama template). Gemma 4's markers are not asserted here; a registered
Gemma keeps today's name-based formatter choice.

`resolve_tier` precedence, first hit wins:

| source | rule | today's outcome kept? |
|---|---|---|
| `override` | item 2's config value, when not `auto` | new |
| `provider_class` | `anthropic` or `is_cloud` → `off` | yes (`_get_adapter_tier` tier 1) |
| `registry` | a `model_registry.yaml` entry matches → `light` if `function_calling.supported` and `requires` is null, else `full` | yes (tier 2), for every listed model |
| `template` | `ToolTemplate.native` True → `light`; False → `full` | new, unlisted models only |
| `fallback` | the shipped default for an unknown model with an unreadable template | `full` today; see the questions |

The registry precedes the template on purpose: "the registry stays as an override" and
"today's registered models must not change" both fall out of it. When they disagree
(a listed model whose template reports no tools, e.g. a server started without `--jinja`)
the decision is the registry's and one WARNING says the template disagreed. Template-first
is the alternative, listed under the questions.

**Feeding it.** Two probes, both landing on objects that already exist:

- `LlamaCppProvider.detect_tool_template()` and `OllamaProvider.detect_tool_template()`
  (`providers/llama_cpp.py`, `providers/ollama.py`): async, `GET /props` / `POST
  /api/show {"model": name}`, cached on the instance as `provider.tool_template`, the
  `supports_vision` / `server_context_size` pattern (`llama_cpp.py:103-108`). The daemon
  calls it beside `detect_vision` (`daemon.py:807`) before `create_adapter` (`:926`).
  This is one more `/props` GET per boot (the three `detect_*` calls each fetch it; folding
  them into one fetch is a separate cleanup).
- `_probe_llama_cpp` / `_probe_ollama` (`backends.py:571`, `:590`) store
  `st.extra["tool_template"] = classify_template(...)` from the payloads they already
  fetch. Zero new requests. This is what `/doctor`, `/api/models` and the future override
  path read for non-primary backends.

**Wiring.** `create_adapter(model_cfg, adapter_cfg, *, template=None, override=None)`
calls `resolve_tier` and stores the decision on the adapter (`adapter.tier_decision`; the
existing `adapter.tier` stays the tier string every consumer reads today).
`_get_adapter_tier(provider_name, model_name)` remains as a thin wrapper returning
`resolve_tier(..., template=None).tier`, so `tests/test_adapter_tiers.py:58-78`,
`tests/test_model_registry_packaging.py:110`, `:146` and `tests/test_wiring.py:2790-2840`
keep passing unchanged. The CLI (`__main__.py:1665`) is synchronous; it gets a sync helper
beside `_detect_model_or_fallback` (`:84`) that fetches the same payload and passes the
`ToolTemplate` in. Coding mode (`:851`) and the gym (`gym/runner.py:71`) pass nothing and
get the registry-or-fallback answer until the ladder PR (item 5) probes for them.

For a template-detected `light`, the formatter follows `call_format` (`qwen-xml` and
`qwen-json` → `QwenFormatter`); registered models keep the name rule (`gemma` in the
name → `GemmaFormatter`).

Not in scope, flagged: the adapter is built once at boot and the tier follows the boot
model; a served-model swap is observed by the identity probe (`daemon.py:515`) and the
backend registry's change detection but rebuilds nothing. `/doctor` (item 3) recomputes
from the latest probe, so a swap shows up there as "boot chose X for model A; the backend
now serves B, whose template says …".

### Behaviour for today's registered models

Unchanged. Every model with a registry entry resolves through `registry` exactly as
`_get_adapter_tier` does now; the template is consulted after it. Cloud providers stay
`off`. The router's override path is untouched in this item (see PR 6).

### Default when the template cannot be read

`full`, with a WARNING naming both misses: "adapter tier full for X by fallback: no
registry entry, and the template could not be read (reason)". It is today's behaviour, and
it is the safe direction for a model with no tool training (at `light` the tools would ride
in a `tools` field the template ignores, and the model would never see them). Whether the
default should be `light` is Will's call (question 1).

### Tests

New, in `tests/test_adapter_tier_resolver.py` (the `/props` and `/api/show` fixtures are
lifted from the committed goldens, so they are the real payloads):

- `classify_template` on the recorded 4090 `/props` → `native=True, call_format="qwen-xml"`;
  on the recorded `/api/show` → `native=True, "qwen-json"`; on `{}` → `native=None`.
- `resolve_tier("llama_cpp", "Ternary-Bonsai-2-27B-PQ2_0.gguf", template=<4090>)` → `light`,
  source `template`. **Fails on main** (`_get_adapter_tier` gives `full`).
- Same for an Ollama name with `capabilities: ["tools"]`. **Fails on main.**
- Every `match_patterns` entry in `config/model_registry.yaml` resolves to the tier
  `_get_adapter_tier` gives today, with and without a template that disagrees (guard).
- A listed model plus a template that says not native → `light`, source `registry`, and one
  WARNING mentions the disagreement.
- Unlisted, template unreadable → `full`, source `fallback`, WARNING. (Passes on main for
  the tier; the source and the warning are new.)
- `create_adapter` with `template=<4090>` for the Bonsai name → tier `light`,
  `QwenFormatter`, strictness NONE, 1 retry. **Fails on main.**
- Daemon wiring: with `detect_tool_template` monkeypatched to return the 4090 facts, the
  adapter the daemon builds for the Bonsai name is `light` (`tests/test_wiring.py` style).
  **Fails on main.**
- `_probe_llama_cpp` / `_probe_ollama` put a `ToolTemplate` in `extra` from the recorded
  payloads.
- `_get_adapter_tier` unchanged for its five existing cases (runs the existing tests).

### Goldens

None change. The primary is `Qwen3.8-27B-UD-Q4_K_XL.gguf` (registered → `light`, same
request shape); the alt is built by `_build_adapter_for`, untouched; the added boot probe
is a GET answered from the recording and never consumed. The parity job runs (providers,
daemon, `__main__`, adapter paths) and must pass unchanged; the seam rule applies (new
`adapter/` module → bench).

### Speed

Boot: one HTTP round trip, bounded by the provider's 10 s timeout; classification is a
regex pass over ≈10 KB, microseconds. Per round: nothing. Bench delta expected below
measurement noise (0.5 ms between sessions per the mini notes).

---

## 2. Per-model manual override in config

### Design

One key, one owner, registry-style matching:

```yaml
adapter:
  # Per-model adapter tier. A key is a case-insensitive substring of the served model
  # name (the way config/model_registry.yaml matches); the longest match wins.
  # Absent, or `auto`: provider class → registry → chat template → fallback.
  model_tiers: {}
  #   "ternary-bonsai-2": light
  #   "ornith": light
  #   "my-finetune": full
```

- `config/prometheus.yaml.default:250` (the `adapter:` section) documents it; it is an open
  map, which `tests/test_config_defaults_equality.py` handles through `open_maps`, and
  `docs/reference/config-keys.md` is regenerated (`scripts/gen_reference.py`, pinned by
  `tests/test_generated_reference.py`).
- Read in `create_adapter` (`__main__.py`) and passed to `resolve_tier` as `override`; the
  same lookup is available to `_build_adapter_for` when PR 6 routes it through the
  resolver, and to the ladder (`gym/ladder/runner.py::ladder_config` builds its own config,
  so a sweep of an unrung model can carry the override under test).
- Values: `auto | off | light | full`. `off` on a local backend is accepted and WARNS
  ("off disables text extraction; tool calls the server does not parse are lost"). Any
  other value is a config error at boot, refused the way `backends:` entries are
  (`backends.py`: recorded, logged, the rest loads).
- A key that matches no model the daemon builds an adapter for is logged once at boot,
  so a typo is not silent (the dark-config rule).

Why a per-model map and not `model.adapter_tier` / `backends.<name>.adapter_tier`: the
tier is a property of the served model, not of the box (an Ollama backend serves several
`models`; llama-server swaps its GGUF under a running daemon), and `context.model_overrides`
(`context/budget.py:37`) is the precedent for keying an override by model. The spec-level
alternative is question 4.

### Behaviour for today's registered models

Unchanged when the map is empty (the shipped default). An entry wins over the registry and
the template; that is the point.

### Default when the template cannot be read

The override does not need the template. Absent an entry, item 1's fallback applies.

### Tests

- `resolve_tier(... override="light")` beats a registry `full` and a template `full`;
  `override="auto"` is absence. **Fails on main** (no such parameter).
- Longest-substring wins; matching is case-insensitive; an unknown value is refused with
  the key named.
- `create_adapter` reads `adapter.model_tiers` from `adapter_cfg`. **Fails on main.**
- The defaults-equality suite: `adapter.model_tiers` is in the template (the ratchet fails
  the PR if the code reads a key the template lacks).
- Boot logs the unmatched-key line.

### Goldens

None change: the harness config (`scripts/parity/runner.py:36`) sets no override. The
parity job runs (`config/**`, `__main__.py`).

### Speed

None per round; one dict lookup at adapter construction.

---

## 3. Show the chosen tier and why

### Design

The decision record from item 1 is the single source; every surface prints
`TierDecision.tier`, `.source` and `.detail`. Source text, one table, the
`_LIMIT_SOURCE_TEXT` pattern (`gateway/commands.py:257`):

| source | reads to a human as |
|---|---|
| `override` | config — `adapter.model_tiers` names this model |
| `provider_class` | the API enforces structure (cloud provider) |
| `registry` | `config/model_registry.yaml` lists this model family |
| `template` | the backend's chat template reports native tool calls (`/props` caps, or Ollama `capabilities`) |
| `fallback` | nothing decided it: no registry entry and the template could not be read |

Surfaces:

1. **Startup.** `create_adapter` logs one INFO line for every tier (today only `light`):
   `Adapter tier: light — template (llama.cpp /props chat_template_caps; calls as
   qwen-xml) for Ternary-Bonsai-2-27B-PQ2_0.gguf`. `fallback` logs at WARNING. The
   interactive header (`__main__.py:678`) prints the same line under `Model:`. (That
   header currently prints the literal text `type(provider)`; fixing it is a one-line
   bystander change.)
2. **`/doctor`** (`gateway/commands.py::cmd_doctor` → `infra/doctor.py::Doctor.diagnose`):
   a new `model` check "Adapter tier". `AnatomyScanner` (`infra/anatomy.py:540`) copies the
   primary backend's `extra["tool_template"]` onto `AnatomyState`; `Doctor` calls
   `resolve_tier` with the config's override and the registry match it already makes
   (`doctor.py:156`), so `/doctor` and boot decide from the same function. `ok` for
   `override`/`registry`/`template`; `warning` for `fallback` with a `fix` ("add the model
   to `adapter.model_tiers`, or a `config/model_registry.yaml` entry; on llama-server check
   `--jinja`"); `warning` when the template and the registry disagree. A served-model
   change since boot is printed as such. `oara doctor` (`cli/doctor.py`) gets the same check
   from its own `/props` read, next to its `/v1/models` probe.
3. **API.** `/api/status` gains an `adapter` block read from the live adapter through the
   bridge (`bridge.loop_context.adapter`), the `_primary_supports_vision` pattern
   (`web/server.py`), so the panel and the turn cannot disagree:

   ```json
   "adapter": {"tier": "light", "source": "template",
               "detail": "llama.cpp /props chat_template_caps: supports_tools and supports_tool_calls; calls as qwen-xml",
               "call_format": "qwen-xml",
               "decided_for": "Ternary-Bonsai-2-27B-PQ2_0.gguf"}
   ```

   `/api/models` rows gain `adapter_tier` and `adapter_tier_source` (the primary from the
   live adapter; backend rows from `resolve_tier` over the registry snapshot's
   `tool_template`, no network). `/api/backends` exposes `tool_template` from `extra`.
   Beacon can show `tier` next to the model name and `source` on hover; the Beacon side is
   Beacon's work and is not linked here.

No telemetry column is added (all 11 expected dumps would change). The circuit breaker's
per-run tier copy (`agent_loop.py:496`) is not the shared adapter, so `/api/status` keeps
reporting the boot decision, and the breaker's own diagnostic already prints the bumped
tier. No `engine/` change.

### Behaviour for today's registered models

Unchanged in behaviour; they now say `registry` where before they said nothing.

### Default when the template cannot be read

Prints `fallback` and the reason; `/doctor` warns.

### Tests

- Boot logs one line for `light`, `full` and `off`, containing the source. **Fails on main**
  for `full` and `off` (nothing logged).
- `/doctor` report contains an "Adapter tier" check; `fallback` → `warning` with a `fix`;
  disagreement → `warning`. **Fails on main** (no such check).
- `/api/status["adapter"]` present with the five keys, from a wired bridge; absent bridge →
  `{"wired": false}` (the `_iteration_ceilings` shape). **Fails on main.**
- `/api/models` primary row carries `adapter_tier`. **Fails on main.**
- `oara doctor` prints the check.

### Goldens

None change: status and doctor are not scenario steps. The parity job runs (`web/**`,
`infra` is not on the list; `gateway` is not either) and must pass.

### Speed

None per round. `/doctor` already probes; `/api/status` reads attributes.

---

## 4. Teach the full-tier reader Qwen's XML

### Design

The reader that runs is the enforcer, so that is where the format goes;
`QwenFormatter.parse_tool_calls` is taught through the same helper so the two cannot
disagree (the #77 family, cited at `agent_loop.py:1806`).

`adapter/enforcer.py`:

- New helper `_parse_xml_tool_calls(text) -> list[ToolUseBlock]`: for each
  `<tool_call>…</tool_call>` span (a missing closing tag is tolerated, the
  `_repair_truncated_json` posture, because the prose grammar branch never constrains the
  XML and `max_tokens` can cut it), read every `<function=NAME>…</function>` block and its
  `<parameter=K>V</parameter>` children; `V` is the raw text between the tags with one
  leading and one trailing newline stripped (the template's own rendering); duplicate `K`
  keeps the last. Values stay strings: the XML carries no types, and at `full` the MEDIUM
  validator's coercion (`adapter/validator.py:330`) already turns `"42"` into an int where
  the schema says so. Bare `<function=…>` outside a `<tool_call>` is also read (the model
  drops the outer tag sometimes; the counter in the ladder already looks for both).
- `extract_tool_calls`: **Strategy 0**, before the JSON strategies: if
  `_parse_xml_tool_calls` finds any call, return those (registry-filtered as today) and
  stop. If it finds none, the four JSON strategies run exactly as now, so
  `<tool_call>{json}</tool_call>` replies (Qwen2.5, and the four such replies in
  `coding_run.trace.json`) take the same path and produce the same blocks as today.
- `QwenFormatter.parse_tool_calls` calls the same helper first.

`strip_tool_call_markup` and `ToolCallMarkupFilter` need no change: they strip the
envelope, XML included, and the extractor now runs before the stripper deletes the evidence.
The GBNF grammar needs no change (the XML is already the prose branch). The QwenFormatter
system-prompt instruction stays JSON; teaching the model the template's format at `full` is
question 5 because it changes a golden's requests.

`engine/agent_loop.py:2677` `_strip_disagreement_feedback` still tells the model to use
JSON-in-tag; after this change a disagreement can only come from a malformed envelope, and
the text can stay. Changing it would be an `engine/` touch; it is optional and flagged as
PR 6c.

### Behaviour for today's registered models

Unchanged at `light` (the server parses; the extractor sees text only when the server
returned no structured call, and JSON-in-tag text takes the same path as today). At
`full`, a Qwen-XML reply that was dropped now executes; no registered model runs at `full`
on the primary path, and the alt path's qwen2.5 writes JSON-in-tag.

### Default when the template cannot be read

Not applicable; the reader is format-driven, not tier-driven.

### Tests

`tests/test_enforcer_xml.py` (each **fails on main**: the enforcer returns `[]`):

- The template's own example, verbatim from the recorded `/props`: one call, two
  parameters, the multi-line value preserved.
- Two calls in one reply; prose before the call; a call with no parameters; a truncated
  envelope (no `</tool_call>`); nested `<parameter=` text containing `<` and `>`.
- `<function=…>` without `<tool_call>`.
- Registry filtering still applies (an unknown name is dropped).
- Regression: `<tool_call>{"name": …}</tool_call>` and the four fenced/bare JSON shapes
  produce the same blocks as before (passes on main; guards the "Strategy 0 changes
  nothing else" claim).
- `QwenFormatter.parse_tool_calls` reads the same XML.
- Loop test (`tests/test_loop_hardening_3.py` style, stub provider): a `full`-tier reply
  that is only the XML executes the tool and no `_strip_disagreement` breaker entry is
  recorded. **Fails on main** (parse disagreement).

### Goldens

**One changes: `coding_run` recorded the defect.** The report first said none would,
from a grep of the raw completion bodies for `<function=` — which found nothing because
the tokenizer splits the marker across SSE deltas (`<function` / `=code_run>`). The
mini's replay of PR 1 found it: reconstructed from their deltas, four `coding_run` replies
carry Qwen XML. The coding path sends no tools and no grammar and injects the JSON
instruction, so the production 27B wrote its JSON call and its XML call in the same reply
(completions 4 and 7: both read as the same call, either way), a JSON call with a stray
`</function>` (completion 5), and, at completion 6, the XML alone — which main dropped as a
parse disagreement and retried with feedback. PR 1 executes that call, so the recorded
run's sixth round disappears and the request stream after it differs. The other ten
goldens replay at PARITY. `coding_run` must be re-recorded with the primary live on the
4090 (the new turn has no recorded answer; a stand-in cannot supply one), which makes PR 1
a trace-changing PR — decision pending (see Decisions). The seam rule applies: replay plus
`bench --baseline` on the mini, with an `origin/main` control in the same session.

### Speed

One regex pass over the reply text per round at `full`, and at `light` only on text-only
replies; replies are a few KB, so tens of microseconds against a 17 ms per-round baseline.
Moving Bonsai to `light` (item 1) is the real cost: the sweep measured 24.7k prompt tokens
per run at `light` against 6.2k at `full` (10.4 s against 6.8 s per run) for +25 points.

---

## 5. A one-command tier sweep

### Today (ladder branch)

Three invocations of `scripts/ladder_run.py --rung R --force-adapter-tier T --run-label
L-T` plus `--tier-report L-off,L-light,L-full`. A sweep is "a sweep OF a rung"
(`gym/ladder/runner.py:654`): it needs a `rungs.yaml` entry with `match`, `quantization`,
`adapter_tier` and the `hf` pin, and it refuses when the daemon's pick differs from the
rung's `adapter_tier` (`:726`). For a model nobody has rung yet, that is the wrong first
step: the sweep is how you find out what the rung should say.
`forced_adapter_factory` (`gym/ladder/tiers.py:37`) monkeypatches `daemon._get_adapter_tier`
for one call.

### Design

```bash
uv run python scripts/ladder_run.py --tier-sweep \
    --provider llama_cpp --base-url "$LADDER_BASE_URL" \
    [--model Ornith-9B-Q4_K_M.gguf] [--tiers off,light,full] [--smoke | --classes ...] \
    [--runs-per-task 3] [--no-judge | --judge-base-url ...] [--telemetry-db ...]
```

- `scripts/ladder_run.py`: `--tier-sweep` (flag) and `--tiers` (default `off,light,full`).
  Without `--rung` it is allowed: the model comes from `--model` or the endpoint's identity
  probe; the quantization from the probe; no rung checks. With `--rung` it keeps today's
  checks. `--force-adapter-tier` stays for one arm by hand.
- `gym/ladder/runner.py`: `run_ladder` gains `tier_sweep: list[str] | None`. Preflight
  once (endpoint, identity, quant, thinking probe, judge pin, sandbox), then one arm per
  tier; across `--runs-per-task` repetitions the tier order rotates (the doc's advice,
  now done by the harness); each arm builds its adapters through the forced factory and
  gets its own run label `<slug>-sweep-<tier>-<date>`; rows carry
  `tier_sweep: {"of": rung_or_None, "model": name, "forced_tier": T, "daemon_tier": D,
  "daemon_tier_source": S}` (the source from item 1's `TierDecision`) and the sandbox is
  reset between arms as between runs.
- `gym/ladder/tiers.py`: `forced_adapter_factory` forces at the resolver seam
  (`adapter.tier.resolve_tier` returns `TierDecision(tier, source="forced")`) instead of
  `_get_adapter_tier`, so the forced adapter is still "the daemon's own adapter for that
  tier" after item 1. The ladder's `build_pipeline` probe passes the endpoint's
  `ToolTemplate` in, so "the daemon picks" is the daemon's real pick for that endpoint.
- `gym/ladder/record.py`: `render_tier_sweep` titles by model when there is no rung and
  adds one line under the tables: "the daemon picks `full` by `fallback`; `light − full` =
  +0.25 (95% −0.34 to −0.17); to run at light add `adapter.model_tiers` or a registry
  entry" — the same numbers the report already computes, plus the decision's source.
  `--compare` keeps refusing sweep labels.
- One command writes `gym/results/ladder/tier-sweep-<slug>-<date>.md` and exits 0 only
  when every arm ran and the empty-field check passed.

A sweep is measurement: it never changes config, and its rows never enter the rung table.

### Behaviour for today's registered models

None; `--rung` runs are unchanged, and `forced_adapter_factory` forcing the daemon's own
pick still builds the identical adapter (`tests/test_ladder.py:1411` stays green).

### Default when the template cannot be read

The sweep runs all three tiers regardless; the report says the daemon would pick
`full` by `fallback`.

### Tests (ladder branch, `tests/test_ladder.py`, the existing `_sweep_ladder` harness with
`build_pipeline` monkeypatched)

- `--tier-sweep` without `--rung` runs three arms, rotates the order across repetitions,
  writes one report, and files rows with `rung: None` and the model name. **Fails on the
  branch today** (`--tier-sweep` unknown; `force_adapter_tier` without a rung refused).
- `--tiers light,full` runs two arms; an unknown tier is refused before any run.
- Rows carry `daemon_tier_source`.
- `forced_adapter_factory` after item 1: forcing at the resolver seam restores the
  resolver; the existing fingerprint test is adapted from `_get_adapter_tier` to
  `resolve_tier`.
- The rendered report contains the recommendation line with the source.

### Goldens

None; `gym/` is not on the parity path list and the ladder writes its own DB.

### Speed

Nothing in the daemon; a sweep of the 12 smoke tasks × 3 tiers is 36 runs of a few
seconds each on the mini's 3090 Ti.

---

## PR plan, in order

Each PR is one behaviour; one trace-changing PR at a time; every `engine/agent_loop.py`
touch is named.

| # | PR | touches | traces | seam rule | engine/agent_loop.py |
|---|---|---|---|---|---|
| 1 | XML reader in the enforcer and formatter (item 4) | `adapter/enforcer.py`, `adapter/formatter.py`, `scripts/parity/scenarios.py` (the coding_run rule), tests | **one trace: `coding_run` recorded the defect and is re-recorded live**; the other ten replay at PARITY | bench on the mini (adapter/) | no |
| 2 | Tier resolver, template probes, boot line (items 1 and the log half of 3) | new `adapter/tier.py`; `__main__.py` (`create_adapter`, `_get_adapter_tier` wrapper, CLI helper); `daemon.py` (one probe call); `providers/llama_cpp.py`, `providers/ollama.py` (`detect_tool_template`); `providers/backends.py` (`extra["tool_template"]`); tests | none change; replay must pass | bench (adapter/) | no |
| 3 | Config override `adapter.model_tiers` (item 2) | `__main__.py`, `config/prometheus.yaml.default`, `docs/reference/config-keys.md` (regenerated), tests | none | no | no |
| 4 | Surfaces: `/doctor`, `oara doctor`, `/api/status.adapter`, `/api/models` tier fields, `/api/backends.tool_template` (item 3) | `infra/doctor.py`, `infra/anatomy.py`, `gateway/commands.py`, `cli/doctor.py`, `web/server.py`, `docs/guide/providers.md` (the "Adapter strictness" section), tests | none; replay runs on `web/**` | no | no |
| 5 | `--tier-sweep` (item 5), a follow-up after the ladder branch merges (the branch itself is not touched) | `scripts/ladder_run.py`, `gym/ladder/runner.py`, `tiers.py`, `record.py`, `docs/MODEL-LADDER.md`, tests | none | no | no |
| 6a | `repaired_tool_call` pins `alt` to `full` through `adapter.model_tiers` in its scenario config | `scripts/parity/scenarios.py`; re-record `repaired_tool_call` (exchanges identical; the stored config differs) | **one trace** | replay | no |
| 6b | Backend overrides, fallback and task rules resolve through `resolve_tier` (registry, template, override) instead of `_build_adapter_for`'s constant `full` | `router/model_router.py`; re-record `model_switch` (the alt turn goes `light`: native tools, no grammar, no injected instruction) | **one trace**; `model_switch` must be recorded with the alt live (the session-title race) | bench (router/) | no |
| 6c | NOT APPROVED (2026-09-25): `_strip_disagreement_feedback` mentioning the XML format | `engine/agent_loop.py:2677` | none (no golden has a disagreement) | bench (engine/) | **yes** |

PR 6a before 6b so the repair golden keeps its subject (at `light` the NONE validator
does no coercion, so an int path would never be repaired and `_req_repair` would refuse
the recording). Both are approved (decision 3) and go last, after the other trace-changing
work queued ahead of them, one trace each. 6b changes what `/alt` does for a registered
model, which today is the accidental `full`.

Docs that move with the PRs: `docs/guide/providers.md` "Adapter strictness" (which
today describes STRICT/MEDIUM/NONE and not the tiers at all), `docs/reference/config-keys.md`
(generated), `docs/MODEL-LADDER.md` (the "Tier sweep" bullet), `config/model_registry.yaml`
header (the registry is now an override, not the only source).

---

## Questions only Will can answer

1. **The default for an unknown model whose template cannot be read.** This design keeps
   `full` (today's behaviour, safe for a model with no tool training, loud about it).
   `light` would favour the Qwen-family case the sweep measured, and lose every call on a
   model whose template has no tool block.
2. **Precedence when the registry and the template disagree.** Registry-first here, to
   keep registered models byte-identical. Template-first would fix a registered model on a
   server without `--jinja` (caps say no tools) at the cost of changing that case.
3. **Should backend overrides, the fallback chain and task rules resolve the tier (PR 6b)?**
   Today they are always `full` for a local provider, registry or not. Resolving them
   re-records `model_switch` and changes `/alt` on the mini's qwen2.5 from `full` to `light`.
4. **The override's shape.** `adapter.model_tiers` keyed by model substring (this design),
   or `model.adapter_tier` plus `backends.<name>.adapter_tier` per spec.
5. **At `full`, should the injected instruction teach the template's format** (Qwen XML)
   instead of JSON when the template reports one? It changes the alt requests in
   `repaired_tool_call` and every `full`-tier prompt.
6. **Is `off` on a local backend an operator choice or measurement-only?** The sweep shows
   `off` ≈ `light` on Bonsai, and `off` also drops text extraction and validation.
7. **Where PR 5 lands**: on `feat/model-ladder-suite` before it merges, or as a follow-up
   after.
8. **One read-only `/props` on the 4090 during his window**, to confirm that
   `chat_template_caps` reflects the rendering path in use (`--jinja` on) on build
   `9d57ce456`; nothing here touched the 4090.
9. **Should a served-model swap rebuild the adapter?** Today the tier follows the boot
   model; the identity probe only warns.
10. **Ornith 9B's file name** (not in the repo), so the resolver tests and a `model_tiers`
    example can name it exactly. Answered: `Ornith-1.5-9B-Q4_K_M.gguf`; Bonsai is
    `Ternary-Bonsai-2-27B-PQ2_0.gguf`, exactly as the ladder recorded it.

---

## Decisions (Will, 2026-09-25)

The key claims above were verified on main (`bfe8674`) by the reviewer. Numbered as the
questions:

1. The fallback for an unknown model whose template cannot be read stays `full`, with a loud
   WARNING.
2. Registry first; a WARNING when the template disagrees.
3. PR 6a and 6b are approved, but go last, one trace each, after the other trace-changing
   work queued ahead of them.
4. `adapter.model_tiers`, keyed by model substring, as designed.
5. No: the injected instruction at `full` stays JSON for now.
6. `off` is allowed on a local backend, with a WARNING.
7. PR 5 lands after the ladder PR merges, as a follow-up; `feat/model-ladder-suite` is not
   touched.
8. No live 4090 read is needed; the goldens' recorded `/props` is the evidence.
9. Not now; `/doctor` showing the mismatch is enough.
10. Ornith's file is `Ornith-1.5-9B-Q4_K_M.gguf`; Bonsai's is the ladder's exact
    `Ternary-Bonsai-2-27B-PQ2_0.gguf`.

Ruled 2026-09-25 (evening), after the mini's replay of PR 1 found that `coding_run`
recorded the defect PR 1 fixes: PR 1 is the trace-changing PR (#580 has landed, none is in
flight). `coding_run` is re-recorded on the mini the way #580 was — the primary live on the
4090's running production server (approved for this PR only, outside 06:30–10:00, no
restart, endpoint never printed), the alt from a strict stand-in over the committed
exchanges. The new golden must show the XML call executing (a committed scenario rule now
requires an executed call read from a reply carrying `<function=`, and no parse-disagreement
feedback); the PR names the exchange and the tool call before and after. Why the coding run
sat at tier `full`: the `code` subcommand builds its adapter from the config's model HINT
(`model.model`, blank in the harness config and in the deployed one), and a blank name matches
no registry family. PR 2 does not change that tier (a blank name with no template read is
the fallback, `full`), so PR 2 stays trace-neutral.

Build order: PR 1 (the XML reader in the enforcer, taught to `QwenFormatter` through the
same helper), then PR 2 (resolver, template probes, boot line, and the `type(provider)`
header fix), one PR at a time, each from main. Every PR: tests failing on main where it is a
fix, the full suite green, replay 11/11 with no trace change, and the speed gate on the mini
(`bench --baseline` plus a main control in the same session). `engine/agent_loop.py` stays
untouched; 6c is not approved.

---

## Bystanders noticed, not in scope

- `__main__.py:678` prints the literal text `type(provider)` in the interactive header.
- `create_adapter` logs nothing for `full` and `off`; the registry-missing WARNING is the
  only `full` signal today (fixed by PR 2).
- At `light`, `format_request` still injects the QwenFormatter JSON instruction into the
  system prompt beside the template's XML block (two formats at once; noted in
  `docs/MODEL-LADDER.md` on the branch). Left alone: changing it changes every primary
  request in all 11 goldens.
- `_build_adapter_for` ignores the registry (PR 6b).
- `detect_context_size`, `detect_kv_cache_types` and `detect_vision` each fetch `/props`;
  the backend registry fetches it again (four GETs per boot in every golden). One fetch,
  three readers is a small cleanup.
- `docs/guide/providers.md` "Adapter strictness" documents strictness levels and never
  mentions the tiers or the registry.
