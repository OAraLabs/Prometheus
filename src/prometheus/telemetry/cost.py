"""CostTracker — per-model token cost tracking for cloud API providers.

Tracks input/output tokens and calculates costs based on per-model
pricing tables. Reports session and cumulative costs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

# Pricing per million tokens (input, output) — USD.
#
# Standard tier (non-batch, non-cached, non-fast-mode). For batch discounts
# (50% off both sides on Anthropic) and prompt-caching multipliers (0.1×–2×
# on Anthropic), see the provider's pricing page. The values below are the
# "default" rate a per-call CostTracker.record() should multiply tokens by.
#
# Anthropic entries verified 2026-05-25 against
# docs.anthropic.com/en/docs/about-claude/pricing. Pre-4.6 aliases resolve
# server-side to the dated snapshot; both forms are listed for clarity so
# whichever string the caller passes through resolves to the same price.
#
# Gemini 2.5 Pro: $1.25 / $10 below 200k context, $2.50 / $15 above. The
# entry here is the ≤200k tier; long-context calls will under-bill by 2×
# until tiered pricing is wired. Verified 2026-05-25 against
# ai.google.dev/gemini-api/docs/pricing.
#
# OpenAI and xAI entries pre-date this verification pass. The OpenAI
# (openai.com/api/pricing) and xAI (docs.x.ai/docs/models, x.ai/api)
# pricing pages were unreachable (HTTP 403) or no longer list grok-3,
# so existing values are left intact rather than re-asserted with
# guesswork. See PR #19's "Drive-by findings" for follow-up.
PRICING: dict[str, tuple[float, float]] = {
    # ── OpenAI — verified 2026-09-21 from developers.openai.com/api/docs/pricing
    "gpt-6-astra": (10.00, 50.00),                  # current flagship
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),                   # /gpt default — cheap + fast, current generation
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5-nano": (0.05, 0.40),                     # cheapest on the table
    "gpt-4o": (2.50, 10.00),                        # still listed; was the /gpt default until 2026-09
    "gpt-4o-mini": (0.15, 0.60),
    "o3-mini": (1.10, 4.40),                        # not on the 2026-09 pricing page; kept for history
    # ── Anthropic — verified 2026-09-21 from
    #    platform.claude.com/docs/en/about-claude/pricing
    "claude-opus-5": (5.00, 25.00),                 # #533: was MISSING while PRESET_MODEL_CHOICES offered it → billed $0
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),               # PR #19: was (15.00, 75.00) — pricing dropped to match Opus 4.5+
    "claude-opus-4-5": (5.00, 25.00),               # added PR #19
    "claude-opus-4-1-20250805": (15.00, 75.00),     # retired except on Bedrock/Google Cloud; kept for history
    # Sonnet 5 is CHEAPER than Sonnet 4.6, which is why it cannot inherit a
    # prefix match from one: the launch "introductory" $2/$10 became the
    # standard price (the scheduled 2026-09-01 rise to $3/$15 did not happen).
    "claude-sonnet-5": (2.00, 10.00),               # #533: was MISSING while PRESET_MODEL_CHOICES offered it → billed $0
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),             # alias — added PR #19 (was missing despite being the user's /claude target)
    "claude-sonnet-4-5-20250929": (3.00, 15.00),    # dated snapshot of Sonnet 4.5 — added PR #19
    "claude-sonnet-4-20250514": (3.00, 15.00),      # legacy Sonnet 4 (deprecated June 2026)
    "claude-haiku-4-5": (1.00, 5.00),               # alias — added PR #19
    "claude-haiku-4-5-20251001": (1.00, 5.00),      # PR #19: was (0.80, 4.00) — pricing increased on GA
    # Gemini — verified 2026-09-21 from ai.google.dev/gemini-api/docs/pricing.
    # Flat approximation of a tiered schedule (service tier / input modality /
    # prompt length): these are the standard tier, text input.
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),                # ≤200k prompt
    # xAI (not re-verified — docs.x.ai no longer lists grok-3)
    "grok-3": (3.00, 15.00),
    "grok-3-mini": (0.30, 0.50),
    # grok-4.x — carried over from the grok-3 flagship rate as a PLACEHOLDER
    # (docs.x.ai pricing unreachable from this box; under the SuperGrok
    # subscription the marginal cost is $0 anyway — this only affects the
    # telemetry cost column for API-key usage). Verify at first keyed use.
    "grok-4.3": (3.00, 15.00),
    "grok-4.5": (3.00, 15.00),
    # ── DeepSeek — verified 2026-09-21 from api-docs.deepseek.com/quick_start/pricing
    #
    # DeepSeek prices on two axes this table has no column for: cache hit/miss,
    # and a time-of-day off-peak discount. These are the PEAK, CACHE-MISS rates
    # — the undiscounted list price, matching this table's stated "standard
    # tier" convention. Off-peak (UTC 16:30–00:30) is roughly half.
    #
    # The previous values, (0.14, 0.28) and (0.435, 0.87), were 2026-07 research
    # and were wrong by 4x on output. They under-billed for two months.
    "deepseek-flash": (0.30, 1.20),                 # current name
    "deepseek-v4-flash": (0.30, 1.20),              # legacy alias, still accepted upstream — must price identically
    "deepseek-v4-pro": (1.32, 3.96),                # reasoning flagship
    # ── Moonshot / Kimi — 2026-07 research, not re-verified in #533
    "kimi-k2.6": (0.95, 4.00),
    # ── Z.ai / GLM — verified 2026-09-21 from docs.z.ai/guides/overview/pricing
    "glm-5.3": (1.40, 4.40),                        # current flagship — same price as the 5.2 it replaces
    "glm-5.3-flash": (0.15, 0.50),
    "glm-5.3-flashx": (0.37, 1.25),
    "glm-5.2": (1.40, 4.40),                        # superseded, still listed and still priced — kept for history
    # ── Xiaomi MiMo — 2026-07 research, not re-verified in #533
    "mimo-v2.5-pro": (0.435, 0.87),
    # ── Alibaba Qwen — the international (Singapore) pay-as-you-go endpoint,
    #    which is what CLOUD_DEFAULTS["qwen"] points at.
    #
    # help.aliyun.com/en/model-studio/model-pricing lists Singapore in CNY
    # (qwen3.8-max: 14.988 in / 44.965 out; qwen3.8-flash: 1.094 / 3.427). These
    # USD figures are Alibaba's separately-published USD list price, which is
    # what a USD-billed account is charged; the CNY page corroborates them at the
    # prevailing rate rather than defining them. Verify at first live use.
    #
    # NOT a contradiction of billing_for()'s "qwen3.8-max has no per-million
    # price": that is true of the TOKEN PLAN, a flat subscription on a different
    # host. A row here prices pay-as-you-go usage; SUBSCRIPTION_HOST_MARKERS
    # still overrides it to `subscription` when the box is pointed at a plan.
    "qwen3.8-max": (2.00, 6.00),
    "qwen3.8-flash": (0.15, 0.48),
    "qwen3.7-max": (2.50, 7.50),                    # superseded by 3.8-max, which is cheaper
}


log = logging.getLogger(__name__)

# Models already warned about, so a busy loop does not print the same line thousands of times.
# Deliberately process-lifetime and unbounded: the set is one string per DISTINCT model, and a
# daemon that saw enough distinct unknown models to matter has a much louder problem.
_unpriced_seen: set[str] = set()


# ── Billing mode ─────────────────────────────────────────────────────────────────────────────
# A zero cost has FOUR different meanings, and a dashboard that renders them identically lies by
# omission. The distinction exists because of a concrete case: `qwen3.8-max` carried 89M tokens —
# 57% of everything this box ever spent — through an Alibaba Token Plan, a flat subscription with
# a credit allowance. It has no per-million price and never will, so "unpriced" was never the
# right label for it either.
#
#   local        — our own hardware; there is no bill and never was
#   subscription — a cloud model on a flat plan; the TOKENS are real, the marginal cost is not
#                  per-token, so a dollar column is the wrong question for it
#   metered      — priced per token; $0.00 here means genuinely no usage
#   unknown      — we do not know, which is a gap to close rather than a zero to display
#
# The rule for the UI follows from this: only `metered` may render a dollar figure. The others
# render their reason.

BillingMode = str  # 'local' | 'subscription' | 'metered' | 'unknown'

# Hosts that bill by subscription rather than per token. Matched against the provider's RESOLVED
# base_url, so this reflects how the box is actually configured rather than a guess from a name.
# (Alibaba's Token Plan and Coding Plan; both documented in providers/registry.py.)
SUBSCRIPTION_HOST_MARKERS: tuple[str, ...] = ("token-plan.", "coding-intl.")


def price_for(model: str) -> tuple[float, float] | None:
    """The (input, output) per-Mtok price for *model*, or None if unpriced.

    THE one price lookup. It existed in three hand-copied forms — the tracker's
    record loop, ``billing_for``'s membership test, and ``_price_for`` inside
    the ``/api/usage`` route — which meant "is this model priced?" could be
    answered three ways. Anything that asserts pricing coverage has to call
    THIS, not re-implement the prefix rule, or the assertion and the code can
    agree with each other while both being wrong.

    Exact match first, then longest-prefix (so ``gpt-4o-2024-05-13`` finds
    ``gpt-4o``). Longest wins because the table now holds overlapping families:
    a plain ``startswith`` scan over an unordered dict would let ``gpt-5.6``
    match a request for ``gpt-5.6-luna`` only by luck of iteration order.
    """
    exact = PRICING.get(model)
    if exact is not None:
        return exact
    best = ""
    for key in PRICING:
        if model.startswith(key) and len(key) > len(best):
            best = key
    return PRICING[best] if best else None


def billing_for(model: str, base_url: str | None = None) -> tuple[BillingMode, str]:
    """Classify how a model's tokens are paid for. Returns (mode, human reason).

    Structural before nominal: a filesystem path is local no matter what it is called, because
    that fact cannot be wrong. A name-based guess can be, and mislabelling local traffic as
    metered would invent a bill that does not exist.
    """
    name = (model or "").strip()
    if not name:
        return "unknown", "no model recorded on the row"
    if name.endswith(".gguf") or name.startswith("/") or "/" in name:
        return "local", "served from a local model file"
    if base_url and any(marker in base_url for marker in SUBSCRIPTION_HOST_MARKERS):
        # #474 / #468: name the MATCHED MARKER, not the host. The reason string
        # is persisted output — it is returned as `billing_reason` from
        # /api/usage to any authenticated client, and printed by
        # scripts/backfill_billing_mode.py to a terminal, shell history and any
        # transcript the output is pasted into. The resolved base_url is a real
        # infrastructure identifier (account-scoped, region-scoped); the marker
        # (`token-plan.`, `coding-intl.`) is a codebase constant already
        # committed in this file that carries the entire meaning. Redact to it.
        marker = next(m for m in SUBSCRIPTION_HOST_MARKERS if m in base_url)
        return "subscription", f"flat plan (resolved host matches {marker!r})"
    if price_for(name) is not None:
        return "metered", "priced per token"
    return "unknown", "no pricing entry and no configured plan — classify it in PRICING or BILLING"


def billing_host_of(provider: object) -> str | None:
    """The bare HOST a provider talks to, or None if it does not expose one.

    TRANSIENT. The return value is classified into a billing mode and then
    dropped — it is never stored on a row, never returned by ``/api/usage``,
    and must not become either. The conventions do not allow a real
    infrastructure identifier in anything that persists, and a telemetry
    database is copied between machines like any other file.

    Host only — no scheme, no path, no query string, so that a value which
    reaches a log line on the way past still cannot carry a credential
    (Gemini's ``?key=``); ``api/turn_errors.py`` keeps ``_URL_QUERY_RE`` for
    the same reason on the error path.

    Reads the private ``_base_url`` because that is where every provider in this
    repo actually keeps it (``OpenAICompatProvider``, ``LlamaCppProvider``); the
    public spelling is checked first so a future provider can expose one.
    Returns None rather than guessing — an unknown host is a recordable fact and
    a fabricated one is not.
    """
    raw = getattr(provider, "base_url", None) or getattr(provider, "_base_url", None)
    if not isinstance(raw, str) or not raw.strip():
        return None
    rest = raw.split("//", 1)[-1]
    host = rest.split("/", 1)[0].split("?", 1)[0].strip()
    return host or None


def matched_subscription_marker(base_url: str | None) -> str | None:
    """Which SUBSCRIPTION_HOST_MARKERS entry the URL matched, or None.

    #468: this is the persistable HALF of the billing evidence. The marker is
    a codebase constant committed in this file — nothing can be connected to
    using one — while the host it matched inside is an account-scoped
    infrastructure identifier that must not persist. A row stamped
    (``subscription``, ``token-plan.``) can be re-evaluated if that marker's
    meaning is ever revised, which a bare mode cannot.

    What it does NOT recover: a marker ADDED later cannot re-classify rows
    from the host it newly matches — those rows were stamped ``unknown`` at
    the time and the thing that would identify them is precisely the
    identifier we are declining to keep. That is the residue, and it is the
    right residue to accept.
    """
    if not base_url:
        return None
    return next((m for m in SUBSCRIPTION_HOST_MARKERS if m in base_url), None)


def billing_stamp_full(
    model: str | None, provider: object
) -> tuple[str | None, str | None]:
    """``(billing_mode, billing_marker)`` for a call ABOUT TO BE RECORDED.

    The marker is the matched ``SUBSCRIPTION_HOST_MARKERS`` entry (or None
    when the verdict did not come from a host marker — local/metered/unknown
    rows carry the mode alone). Never raises, same contract as
    :func:`billing_stamp`: a row that cannot be labelled is written
    unlabelled rather than lost.
    """
    try:
        host = billing_host_of(provider)
        url = f"//{host}" if host else None
        mode, _reason_withheld = billing_for(model or "", url)
        marker = matched_subscription_marker(url) if mode == "subscription" else None
        return mode, marker
    except Exception:  # noqa: BLE001 — never break a call to label it
        return None, None


def billing_stamp(model: str | None, provider: object) -> str | None:
    """The ``billing_mode`` for a call ABOUT TO BE RECORDED.

    The write-time counterpart of :func:`billing_for`, which classifies at READ
    time from whatever the configuration happens to say now. That read-time
    answer is correct only for as long as the configuration does not move, and
    it moved: a flat-plan host swapped for a metered one reclassifies every
    historical row that model ever wrote.

    Never raises — a telemetry row that cannot be stamped is written unstamped,
    which readers report as inferred. Losing the row would be worse than losing
    the label.

    The host that produced the answer is deliberately NOT returned. See
    :func:`billing_host_of` — it exists only long enough to be classified.
    """
    # Delegates to billing_stamp_full so the mode and the marker can never
    # be computed by two drifting copies of the classification.
    mode, _marker = billing_stamp_full(model, provider)
    return mode


@dataclass
class UsageRecord:
    """A single token usage entry.

    ``billing_mode`` is why ``cost_usd`` is what it is, carried on the row
    rather than re-derived by each reader. A bare ``cost_usd`` of 0.0 is
    ambiguous across three of the four modes — local (no bill exists),
    subscription (the bill is not per-token) and unknown (we could not price
    it) all land on the same float, and only one of them is "free". Readers
    that see the rows get the distinction here; readers that see only the
    rollup get it from :meth:`CostTracker.to_dict`.
    """

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float
    billing_mode: BillingMode = "unknown"


class CostTracker:
    """Track token usage and costs across a session."""

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []
        self._total_cost: float = 0.0
        self._total_input: int = 0
        self._total_output: int = 0
        # Tokens this tracker could NOT price, kept apart from the priced totals.
        # Folding them in would be the bug: the sum would look like a complete
        # answer while covering an unstated fraction of the traffic.
        self._unpriced_input: int = 0
        self._unpriced_output: int = 0
        self._unpriced_models: set[str] = set()

    @property
    def records(self) -> tuple[UsageRecord, ...]:
        """The recorded rows, each carrying its own ``billing_mode``."""
        return tuple(self._records)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record a usage event. Returns the cost in USD.

        The return value stays a float, and stays 0.0 for anything not metered,
        because callers and stored history read it as one. The fact that a 0.0
        can mean three different things is carried alongside it —
        ``UsageRecord.billing_mode`` per row, and the ``unpriced_*`` /
        ``cost_is_complete`` keys on :meth:`to_dict` for the rollup — rather
        than by changing this signature underneath every existing reader.
        """
        pricing = price_for(model)

        # ONE classification, driving both the number and the label. Computing
        # "is this free?" separately from "what did it cost?" is how the two
        # drift: billing_for() already knows that a .gguf path bills nothing and
        # an unrecognised name bills unknown, and it is the same function
        # /api/usage classifies unstamped history with.
        mode, _reason = billing_for(model)

        if pricing is None:
            # LOUD, once per model. Returning 0.0 quietly is its own defect, separate from the
            # missing price row: `qwen3.8-max` carried 57% of every token this box ever spent and
            # priced at $0.00 for months without a single line of output saying so. Whatever the
            # right answer for a given model is — a price, a subscription, or "local" — silence
            # is never it, and without this the NEXT model added repeats the same disappearance.
            #
            # Only `unknown` is a gap. A local model has no bill and a flat-plan
            # model has no per-token price; warning about either would train the
            # reader to ignore the line, which costs us the one case that matters.
            if mode == "unknown":
                self._unpriced_input += input_tokens
                self._unpriced_output += output_tokens
                self._unpriced_models.add(model)
                if model not in _unpriced_seen:
                    _unpriced_seen.add(model)
                    log.warning(
                        "cost: no pricing entry for model %r — %d input + %d output tokens are "
                        "being counted at $0.00. Add it to PRICING, or classify it in "
                        "SUBSCRIPTION_HOST_MARKERS so the zero is a stated fact rather than a gap.",
                        model,
                        input_tokens,
                        output_tokens,
                    )
            cost = 0.0
        else:
            input_price, output_price = pricing
            cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000

        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            timestamp=time.time(),
            billing_mode=mode,
        )
        self._records.append(record)
        self._total_cost += cost
        self._total_input += input_tokens
        self._total_output += output_tokens
        return cost

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        return self._total_input

    @property
    def total_output_tokens(self) -> int:
        return self._total_output

    @property
    def total_tokens(self) -> int:
        return self._total_input + self._total_output

    @property
    def unpriced_tokens(self) -> int:
        """Tokens spent on models with no PRICING row — NOT included in the cost."""
        return self._unpriced_input + self._unpriced_output

    @property
    def cost_is_complete(self) -> bool:
        """Does ``total_cost`` actually account for every token recorded?"""
        return not self._unpriced_models

    def report(self) -> str:
        """Human-readable cost report for the /status command.

        This string is the ONLY cost surface most users ever see (Telegram,
        Slack and Discord all render it through gateway/commands.py). A dollar
        figure here that silently excludes part of the traffic is the same lie
        /api/usage was fixed to stop telling, so the unpriced share is stated
        inline rather than left for a log line nobody is tailing.
        """
        if not self._records:
            return "Cost: $0.00 (no cloud API usage)"

        line = (
            f"Session cost: ${self._total_cost:.4f} "
            f"({self._total_input:,} input + {self._total_output:,} output tokens)"
        )
        if self._unpriced_models:
            names = ", ".join(sorted(self._unpriced_models))
            line += (
                f"\n  ⚠ excludes {self.unpriced_tokens:,} tokens with no price on file "
                f"({names}) — the total above is NOT the whole bill"
            )
        return line

    def to_dict(self) -> dict[str, Any]:
        """Structured cost data.

        ``cost_is_complete`` is the key that makes ``total_cost_usd`` safe to
        render: false means the figure covers only part of the traffic, and the
        ``unpriced_*`` keys say how much and which models. A client that ignores
        them is exactly as correct as it was before; one that reads them can
        state coverage instead of implying it.
        """
        return {
            "total_cost_usd": round(self._total_cost, 6),
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "records": len(self._records),
            "unpriced_input_tokens": self._unpriced_input,
            "unpriced_output_tokens": self._unpriced_output,
            "unpriced_models": sorted(self._unpriced_models),
            "cost_is_complete": self.cost_is_complete,
        }


# ── module-level handle (audit H/T3: make cost accounting actually accumulate) ──
# The daemon registers a CostTracker here ONLY for cloud providers. The agent
# loop's per-LLM-call usage rows all flow through one seam —
# ``ToolCallTelemetry.record_subsystem_run`` (written by ``LLMCallEnvelope``) —
# which feeds this handle, so cost covers every entry point (telegram / web /
# autonomous), not one gateway. None on the local box, where the feed is a
# no-op and ``report()`` honestly says "$0.00 (no cloud API usage)".
_cost_tracker_handle: CostTracker | None = None


def set_cost_tracker_handle(tracker: CostTracker | None) -> None:
    """Register (or clear) the process-wide cost tracker the telemetry seam feeds."""
    global _cost_tracker_handle
    _cost_tracker_handle = tracker


def get_cost_tracker_handle() -> CostTracker | None:
    return _cost_tracker_handle
