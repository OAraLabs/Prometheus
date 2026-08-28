# SPRINT: Loud Degrade on Terminal Provider Failure

**Branch:** `feat/provider-fallback`
**Status:** Spec. Not started. Independent of the Token Plan renewal — see Origin.
**Origin:** Review of #288 turned into a routing audit on 2026-08-28. Three findings, in
order of how much they should shape the design:

1. **No failover exists anywhere.** `failover`, `switch_provider`, `next_provider` return
   zero hits across `src/prometheus/`. When a cloud provider returns a terminal error the
   turn dies. `api/turn_errors.py` classifies the failure well and that is *reporting, not
   recovery*.
2. **A terminal auth failure does not degrade — it stops.** `RETRYABLE_STATUS_CODES =
   {500, 429, 502, 503}` (`providers/stub.py`, imported by `openai_compat.py`; `anthropic.py`
   keeps its own `_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 529}`). **401 and 403 are in
   neither set**, so an expired plan or revoked key raises on the FIRST response with no
   retries. Nothing watching for retry-storm symptoms will see it.
3. **The exposure is concentrated.** `qwen3.8-max` carried **82.2% of all tokens in the last
   7 days** (41.1M of 50M), 67.8% over 14d, 63.0% over 30d — rising, not historical.

The Token Plan expiry (2026-09-14) is the occasion, **not the scope**. A revoked key, a
billing lapse, or an account suspension on xAI or Anthropic produces the identical failure
tomorrow. Build for the class.

---

## The hazard this sprint exists to avoid

The naive fallback is worse than no fallback. `qwen3.8-max` has an order-of-magnitude larger
window than the local model (`compactor.py` comments reference capping a 1M-token model).
The local target, measured live at `http://100.110.140.39:8080/props` on 2026-08-28:

```
n_ctx: 32768     total_slots: 1     Qwen3.8-27B-UD-Q4_K_XL.gguf
```

**A turn that fits comfortably on one will not fit on the other.** Silently re-routing means
a session you believed ran with a 1M window quietly ran with 32k, and the only symptom is
that the output got worse for no stated reason. `total_slots: 1` adds a second edge: two
concurrent sessions degrading at once queue behind each other.

So the deliverable is not "retry elsewhere". It is: **degrade loudly, state the new
constraint, and refuse rather than truncate when the turn cannot fit.**

---

## Phase 0 — Survey (read-only, mandatory halts)

`git fetch origin && git rev-parse HEAD` — cite SHA; HALT on dirty tree / behind origin /
wrong branch.

Cite file:line:

1. Every call site that invokes a provider's `stream_message` on the chat path, and where an
   exception from it is currently caught. Name the layer that owns "this turn failed".
2. `api/turn_errors.py` — the full `KIND_*` set and where `classify_turn_error()` is called today. This
   is the intended trigger surface: the fallback must branch on an already-classified `kind`,
   NOT re-parse exceptions at a second site. If classification happens too late in the stack
   to act on, say so — that is a real finding and changes the design.
3. `context/compactor.py` `limit_for()` / `_detected_limit` / `_model_overrides` — how a
   per-model window is resolved per call, and whether a caller can ask "what window would
   model X get?" WITHOUT running a turn. The pre-flight check in Phase 2 depends on it.
4. `providers/registry.py` and `router/model_router.py` — how a model key resolves to a
   provider instance, and whether anything can construct the local provider on demand.
5. `run_coding_task` (`__main__.py:~764`) pins its provider deliberately: *"no router/model
   fallback mid-run — that would confound the acceptance metric."* Confirm the opt-out point
   so this sprint does not silently invalidate the coding acceptance metric.
6. Whether any streaming call site can have **already yielded tokens** before the error
   surfaces. Cite the code path. This decides Phase 2's hardest rule.

**HALT CHECKPOINT 1** — findings + a one-paragraph plan. If `turn_errors.classify_turn_error()` runs
below the layer that could switch providers, HALT: that is a restructuring question, not an
implementation detail.

---

## Phase 1 — Decide, don't retry

A terminal failure is a *different event* from a retryable one and must not reuse the retry
path.

- Add a predicate over the existing kinds — `KIND_AUTH`, `KIND_BILLING`, and an explicit
  account-state kind if `classify_turn_error()` distinguishes suspension. Terminal means **"trying again
  with this credential cannot help"**.
- 401/403 stay OUT of `RETRYABLE_STATUS_CODES`. Do not "fix" that set. Retrying an expired
  key is exactly the wrong behaviour; this sprint is about what happens *after* not retrying.
- `KIND_RATE_LIMIT` is NOT terminal — it already retries with backoff. Falling back on a 429
  would move traffic off a provider that was about to succeed.
- Two `RETRYABLE_STATUS_CODES` sets exist (`stub.py`, `anthropic.py`) and disagree on `529`.
  Do NOT unify them in this sprint. Note it, cite both, leave them.

**Test that must exist:** a 401 triggers the fallback path and a 429 does not, asserted
separately. Mutation: make the predicate return True for `KIND_RATE_LIMIT` — the 429 test
must go red. If it does not, the test is not testing the branch.

---

## Phase 2 — The context cliff

Before routing a turn to the fallback model, compute whether it fits.

1. Resolve the fallback's real window (detected `n_ctx`, not a configured constant — a config
   number that outlived a model swap is how a 32768-token server came to be budgeted at
   72000; `compactor.py:~170` records that incident).
2. Estimate the turn against that window using the compactor's existing estimator.
3. **If it does not fit, do not truncate and do not compact harder to force it.** Fail the
   turn with a message that names the numbers:

   > `qwen3.8-max` is unavailable (auth). This turn needs ~118k tokens of context; the local
   > fallback `Qwen3.8-27B` has a 32k window, so it cannot serve this conversation. Retry
   > after restoring the provider, or start a narrower session.

   A refusal that says *why* and *by how much* is the deliverable. "Something went wrong" is
   the current behaviour and is what this sprint replaces.
4. If it does fit, serve it — and go to Phase 3.

**Never fall back from the local model to itself.** Guard the loop explicitly.

**If tokens were already streamed to the client, do not fall back.** A second model resuming
mid-reply produces text that contradicts what the user already read. Fail the turn instead,
and say the reply was cut off. Phase 0 item 6 establishes whether this can happen.

---

## Phase 3 — Loud, in the turn itself

Three surfaces, all required. A degrade the user has to go looking for is a silent degrade.

- **The reply names its server.** The turn carries the model that ACTUALLY served it and the
  reason it was not the requested one. Not a log line, not a tooltip — attached to the reply.
- **A frame on the wire.** Emit over the SignalBus → WS path (the one #280 established for
  approvals) with a stable kind clients branch on. Beacon renders it inline; anything else
  connected gets the same fact. Reuse the `KIND_*` vocabulary — do not invent a parallel one.
- **Telemetry records both.** `requested_model` and `served_model` as separate fields, plus
  the reason. Today a fallback would be indistinguishable from the user having chosen the
  local model, which corrupts exactly the usage analysis that produced this sprint. This is
  also the field that makes "how often are we degrading?" answerable at all.

**The degrade is per-turn, not sticky.** Do not silently pin the session to the fallback —
the next turn tries the real provider again. If that is too chatty in practice, make
stickiness explicit and visible, never implicit.

---

## Phase 4 — Configuration

- Fallback target is configured, not hardcoded. Default: the local provider already serving
  as `is_default` in `GET /api/models`.
- A way to turn it OFF. Coding mode (Phase 0 item 5) must be able to keep its pinned-provider
  guarantee, and some callers genuinely prefer a hard failure.
- No new credential paths. The fallback uses a provider already configured; this sprint must
  not become a place where keys get read from somewhere new.

---

## Out of scope, deliberately

- Unifying the two `RETRYABLE_STATUS_CODES` sets.
- Multi-hop fallback chains (cloud → other cloud → local). One hop, to a configured target.
- Changing what `/qwen` routes to, or the Token Plan renewal itself.
- Making the local model bigger. The 32k window is a fact this sprint reports, not one it
  fixes.

## Verification

Live, not only unit — the standing rule in this repo, and the reason the routing picture
above is trustworthy at all. At minimum: force a 401 against a cloud provider with a
deliberately bad key and confirm (a) the turn degrades, (b) the reply names the model and the
reason, (c) telemetry shows `requested_model != served_model`, and (d) an oversized
conversation refuses with the numbers rather than truncating.

A green mutation sweep on this sprint proves only that the guards written are load-bearing.
Enumerate the call sites from Phase 0 item 1 and confirm each is covered — a path with no
fallback wired produces no failing mutation, because there is nothing there to break.
