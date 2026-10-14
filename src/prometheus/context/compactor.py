"""ContextCompactor — single-layer, assembly-time context compaction.

When the estimated assembled prompt (system prompt + history + tool schemas)
approaches the model's effective context, summarize the oldest contiguous
span of turns via the SAME local model and substitute the summary into the
RENDER VIEW handed to the provider. The loop's message list, the session,
and lcm.db are never mutated — compaction is a prompt-assembly concern,
never a storage mutation. The LCM DAG stays the untouched source of truth.

SINGLE-LAYER BY CONSTRUCTION (the spec's hard constraint): the synthetic
summary message carries ``provenance="compactor"``, spans contain only
span-transparent messages (``_SPAN_TRANSPARENT``), and any other provenance
is a span BARRIER — so a summary can never be ingested into a later span.
Very long sessions accumulate *sibling* summaries, never summaries of
summaries.

Ships behind ``compaction.enabled`` (default false): with the section absent
``from_config`` returns ``None`` and nothing changes anywhere.

Config (``compaction:`` section of prometheus.yaml — ABSENT by default)::

    compaction:
      enabled: true
      threshold_pct: 0.75       # of (effective_limit - reserve_tokens)
      reserve_tokens: 4096      # output headroom subtracted from n_ctx
      protect_recent_turns: 8   # recent user turns never compacted
      max_summary_tokens: 512   # hard output cap for the ONE summary call

n_ctx comes from ``context.effective_limit`` with
``context.model_overrides.<model>.effective_limit`` applied — the same
resolution as ``context/budget.py``.

Fail-loud policy: a failed summarization call (the envelope already wrote
``silent_failures``) or an insane result (empty, or not smaller than the
span it replaces) logs at ERROR, emits a ``context_compaction_failed``
signal event (durable in ``telemetry.signal_events`` → /events, Beacon),
and falls back EXPLICITLY to the pre-existing overflow behavior: the
unmodified prompt is sent and the provider/server reports the overflow.
Never a silent fallback. Every successful compaction records a
``context_compaction`` event with span size, tokens before/after, duration,
and cache state.

Inspired by the context-compactor design in the Odysseus project (MIT);
clean-room, design knowledge only. No source copied.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from prometheus.config.ephemeral import recorded_session_id
from prometheus.context.budget import (
    LEGACY_FALLBACK_LIMIT,
    resolve_effective_limit,
)
from prometheus.context.token_estimation import (
    estimate_message_tokens,
    estimate_tokens,
)

if TYPE_CHECKING:
    from prometheus.engine.messages import ConversationMessage
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

# Provenances a compactable span may contain. "user" covers real user turns AND
# the loop's own assistant/tool-result messages, which default to it.
#
# "file_mutation_verifier" is here because that summary is turn-local debris —
# a claimed-vs-actual disk audit for ONE turn — and it lands at the end of
# every file-touching turn. Treating it as a barrier (the rule for every other
# non-user provenance) would pin the compactable prefix at the first file write
# of the session and stop compaction dead on exactly the long coding sessions
# it exists for. It is machinery-authored and trusted, so summarizing it away
# loses nothing: the model already acted on it, turns ago.
#
# The single-layer guarantee is untouched — it rests on "compactor" being a
# barrier, and compactor summaries are substituted at assembly time and never
# stored, so one can never re-enter a span regardless.
_SPAN_TRANSPARENT = frozenset({"user", "file_mutation_verifier"})

# #773: compact at 80% of the serving model's window. Per-model thresholds
# (``context.model_overrides.<model>.compaction_threshold_pct`` /
# ``compaction_threshold_tokens``) are set from evidence; none ship.
DEFAULT_THRESHOLD_PCT = 0.80
# Output room the projected-overflow rule assumes when a request's own
# ``max_tokens`` is unknown. No longer subtracted from the threshold (#773).
DEFAULT_RESERVE_TOKENS = 4096
DEFAULT_PROTECT_RECENT_TURNS = 8
DEFAULT_MAX_SUMMARY_TOKENS = 512

# Budget for a session routed to a cloud provider when no per-model override
# exists. Cloud APIs do not publish context length — Alibaba's /v1/models
# returns only id/object/created/owned_by, and the other OpenAI-compatible
# providers are the same — so this is a configured assumption, not a
# detection.
#
# Set GENEROUS on purpose, and the asymmetry is the reason:
#
#   Over-budget on cloud fails LOUD. The provider returns a 400 naming its
#   real limit, which is diagnosable in one read. Nothing like the local
#   failure mode, where llama.cpp silently truncates and the model returns
#   empty content with no indication why.
#
#   Under-budget on cloud fails SILENT. Compaction fires early and summarises
#   context that never needed summarising — quality quietly degrades and
#   nothing anywhere says so.
#
# A conservative floor therefore buys protection against the mild failure at
# the cost of the invisible one, which is backwards. Models with genuinely
# smaller windows are named in context.model_overrides instead, where being
# wrong is a one-line fix rather than a silent tax on every session.
DEFAULT_CLOUD_LIMIT = 1_000_000

# A span smaller than this many messages isn't worth a model call.
MIN_SPAN_MESSAGES = 4

# §6 (#773): a pass aims WELL below the line, not just under it. Landing a few
# tokens under meant the next round's growth crossed it again and the next
# pass re-summarised a slightly longer prefix — every round, for hours.
TARGET_FRACTION = 0.70

# The projected-overflow rule looks at the largest tool result in this many
# most recent tool rounds: the next round can bring back one like it.
RECENT_TOOL_ROUNDS = 5

# Labels for where a threshold came from (reported on /context, /api/status,
# the startup line and every context_compaction event).
SOURCE_GLOBAL_PCT = "global pct"
SOURCE_MODEL_PCT = "model pct"
SOURCE_MODEL_TOKENS = "model tokens"


def compaction_threshold(
    effective_limit: int,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> int:
    """The assembled-request token total at which compaction fires, by percentage.

    Module-level and pure so the surfaces that REPORT this number use the
    same arithmetic as the one that acts on it. #773: a plain percentage of the
    window; output room is the projected-overflow rule's job, using the
    request's real ``max_tokens``, not a fixed reserve subtracted here.
    """
    return int(max(1, int(effective_limit)) * float(threshold_pct))


def resolve_threshold(
    effective_limit: int,
    *,
    global_pct: float = DEFAULT_THRESHOLD_PCT,
    model_entry: Mapping[str, Any] | None = None,
) -> tuple[int, str]:
    """The threshold for one model and where it came from (#773).

    ``model_entry`` is the model's ``context.model_overrides`` entry. Its
    ``compaction_threshold_pct`` replaces the global percentage for that model;
    ``compaction_threshold_tokens`` is an absolute count. Both set: the smaller
    wins. Neither: the global percentage. Never touches the window itself.
    """
    candidates: list[tuple[int, str]] = []
    entry = model_entry or {}
    pct = entry.get("compaction_threshold_pct")
    if isinstance(pct, (int, float)) and not isinstance(pct, bool) and 0 < pct <= 1:
        candidates.append((compaction_threshold(effective_limit, float(pct)), SOURCE_MODEL_PCT))
    tokens = entry.get("compaction_threshold_tokens")
    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0:
        candidates.append((int(tokens), SOURCE_MODEL_TOKENS))
    if not candidates:
        return compaction_threshold(effective_limit, global_pct), SOURCE_GLOBAL_PCT
    return min(candidates, key=lambda c: c[0])


def compaction_threshold_from_config(
    effective_limit: int, config: dict | None, model: str | None = None,
) -> tuple[int, str]:
    """:func:`resolve_threshold` from a loaded prometheus.yaml, for *model*."""
    cfg = config or {}
    comp = cfg.get("compaction") or {}
    overrides = (cfg.get("context") or {}).get("model_overrides") or {}
    entry = overrides.get(model) if model else None
    return resolve_threshold(
        effective_limit,
        global_pct=comp.get("threshold_pct", DEFAULT_THRESHOLD_PCT),
        model_entry=entry if isinstance(entry, Mapping) else None,
    )

# Idempotence cache (FIFO-bounded): span content hash -> summary text.
_CACHE_MAX_ENTRIES = 16

# #773: the span is CHUNKED to fit the summarizer's own window, never cut.
# Before, one prompt was capped at 18,000 characters and every message past
# that was left out: 309 of 3,334 live compactions dropped input, the worst
# replacing 1,685 messages (919,676 tokens) with a summary of 46 of them.
#
# Share of the summarizer's window held back as estimation margin.
_WINDOW_MARGIN = 0.10
# The final summary's budget grows with the span it replaces (1 token per this
# many), from max_summary_tokens up to this multiple of it.
_FINAL_SUMMARY_RATIO = 20
_FINAL_SUMMARY_MAX_MULTIPLE = 4

# Prompt follows the repo's existing summarization voice
# (memory/lcm_summarize.py::_MESSAGE_SUMMARY_PROMPT) with the spec's
# preservation list.
_SUMMARY_PROMPT = (
    "You are a conversation compactor. Summarize the following conversation "
    "span into one concise factual digest. Preserve:\n"
    "  - The user's goals and any decisions made\n"
    "  - File paths, identifiers, commands, and technical details\n"
    "  - Unresolved questions and pending work\n\n"
    "Be terse and factual. Do NOT add commentary.\n\n"
    "Conversation:\n{conversation}"
)

_COMBINE_PROMPT = (
    "You are a conversation compactor. Below are summaries of consecutive "
    "parts of ONE conversation, in order. Merge them into one concise factual "
    "digest. Preserve:\n"
    "  - The user's goals and any decisions made\n"
    "  - File paths, identifiers, commands, and technical details\n"
    "  - Unresolved questions and pending work\n\n"
    "Be terse and factual. Do NOT add commentary.\n\n"
    "Part summaries:\n{parts}"
)

SUMMARY_MARKER_PREFIX = "[Compacted summary of turns"

#: The marker a tool result trimmed by §6 carries in the render view.
TRIMMED_RESULT_MARKER = "[tool result trimmed by the compactor to fit the context window: ~{n} tokens]"


class ContextCompactor:
    """Single-layer assembly-time compactor. Construct via :meth:`from_config`."""

    def __init__(
        self,
        *,
        provider: "ModelProvider",
        model: str,
        effective_limit: int,
        threshold_pct: float = DEFAULT_THRESHOLD_PCT,
        reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
        protect_recent_turns: int = DEFAULT_PROTECT_RECENT_TURNS,
        max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
        telemetry: Any | None = None,
        signal_bus: Any | None = None,
        model_overrides: dict[str, int] | None = None,
        detected_limit: int | None = None,
        cloud_default_limit: int = DEFAULT_CLOUD_LIMIT,
        windows: Callable[[], Mapping[str, Any]] | None = None,
        backend_hints: Mapping[str, int] | None = None,
        threshold_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        from prometheus.learning.llm_envelope import LLMCallEnvelope

        self._provider = provider
        self._model = model
        self._effective_limit = int(effective_limit)
        # Per-model limits, resolved PER CALL rather than frozen at
        # construction. The compactor is built once at daemon start against
        # the LOCAL model, but a session can be routed to a cloud provider
        # (/claude, /qwen, …) whose window is an order of magnitude larger —
        # budgeting those as if they were the local GGUF wasted almost the
        # whole context. See limit_for().
        self._model_overrides: dict[str, int] = dict(model_overrides or {})
        # What the local inference server actually reported (llama.cpp /props
        # n_ctx). Ground truth, and it beats the configured global — a config
        # number that outlived a model swap is exactly how a 32768-token
        # server came to be budgeted at 72000, leaving no room to generate.
        self._detected_limit = int(detected_limit) if detected_limit else None
        # Per-backend windows, read LIVE from the registry on every call (a
        # callable, not a snapshot): a box that restarted onto another model is
        # re-sized on the next turn, not the next daemon restart.
        self._windows = windows
        self._backend_hints: dict[str, int] = dict(backend_hints or {})
        # Fallback for a model we have no entry for AND that is not the local
        # one. Never inherit the local GGUF's window here: cloud models are
        # far larger, and silently capping a 1M-token model at 28k wastes
        # almost all of it.
        self._cloud_default_limit = int(cloud_default_limit)
        self._threshold_pct = float(threshold_pct)
        self._reserve_tokens = int(reserve_tokens)
        self._protect_recent_turns = int(protect_recent_turns)
        self._max_summary_tokens = int(max_summary_tokens)
        self._telemetry = telemetry
        self._signal_bus = signal_bus
        # span-content-hash -> summary text (FIFO-bounded)
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        # session_id -> anchored span end index. Once a span is summarized we
        # keep substituting THAT span (cache hit, zero calls) until the
        # compacted estimate re-crosses the threshold; only then the span
        # extends and one new call is paid. This is the spec's idempotence:
        # the cache invalidates only when the span actually has to change.
        self._session_spans: dict[str, int] = {}
        # #773: per-model compaction thresholds (compaction_threshold_pct /
        # compaction_threshold_tokens), separate from the window entries above.
        self._threshold_overrides: dict[str, dict[str, Any]] = {
            str(k): dict(v) for k, v in (threshold_overrides or {}).items()
        }
        # §6: session_id -> key of the protected floor a pass could NOT get
        # under. While the floor is unchanged, no new summary is paid for —
        # re-summarising cannot help, and every attempt loses more history.
        self._stuck: dict[str, str] = {}
        # return_none: failures are already loud (silent_failures + failed
        # subsystem_runs); apply() then falls back explicitly. Not redacted:
        # the summary replaces turns of the live conversation, and a token
        # the user gave the agent must survive into it (llm_envelope, item 4).
        self._envelope = LLMCallEnvelope(
            subsystem="context_compactor",
            telemetry=telemetry,
            on_failure="return_none",
            redact_prompts=False,
        )

    # -- construction -----------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        provider: "ModelProvider",
        model: str,
        telemetry: Any | None = None,
        signal_bus: Any | None = None,
        detected_limit: int | None = None,
        registry: Any | None = None,
    ) -> "ContextCompactor | None":
        """Build from the loaded prometheus.yaml dict.

        Returns ``None`` unless ``compaction.enabled`` is truthy — the
        default-off contract: absent section = zero behavior change.

        *detected_limit* is the context size the inference server actually
        reported (llama.cpp ``/props`` n_ctx). Pass it and the local model
        needs no config entry at all — which is the point, because an
        exact-filename override silently stops matching the moment the GGUF
        is swapped, and the failure is a total inability to answer.
        """
        cfg = config or {}
        comp = cfg.get("compaction") or {}
        if not comp.get("enabled", False):
            return None

        ctx = cfg.get("context") or {}

        # REACHABLE WITH AN UNRESOLVED CONFIG — the answer, not the question,
        # because this shape has now shipped as a bug twice (daemon.py's
        # 72000-over-32768, and /api/lcm's literal 24000).
        #
        # This is an ENFORCEMENT path: the value below becomes
        # ``_effective_limit``, which limit_for() returns whenever
        # ``_detected_limit`` is absent, and _threshold_tokens() compacts
        # against. A config with ``compaction.enabled: true`` but no
        # ``context.effective_limit``, on a boot where the backend was
        # unreachable, previously produced a hard 24000 that no part of the
        # system had agreed to — silently, and at the point where prompts get
        # cut. That is reachable today: neither key is required, and detection
        # returns None whenever the model server is down at daemon start.
        #
        # So it resolves through the same resolver as every other surface.
        # Behaviour is unchanged where anything IS resolvable: limit_for()
        # already prefers _detected_limit for the local model, and the
        # per-model override branch is what the resolver returns first. The
        # only changed case is the unresolvable one, which is now LOUD and
        # named instead of silent and literal.
        resolved, source = resolve_effective_limit(
            ctx,
            model=model,
            # from_config always builds for the model this daemon serves
            # locally; limit_for() handles cloud-routed sessions per call.
            local_model=model,
            detected_limit=detected_limit,
        )
        if resolved is None:
            effective_limit = LEGACY_FALLBACK_LIMIT
            log.warning(
                "COMPACTION IS ENABLED WITH NO RESOLVABLE CONTEXT WINDOW: "
                "the server reported nothing and context.effective_limit is "
                "absent/invalid, so compaction will enforce a placeholder "
                "%d tokens for model %r. Set context.effective_limit, or "
                "bring the inference server up before the daemon, or this "
                "number is arbitrary.",
                LEGACY_FALLBACK_LIMIT, model,
            )
        else:
            effective_limit = int(resolved)
            log.debug(
                "Compactor budget for %s: %d (source=%s)",
                model, effective_limit, source,
            )
        overrides = ctx.get("model_overrides") or {}

        # Flatten every per-model entry so limit_for() can resolve a model
        # this compactor was NOT built with (a cloud override mid-session).
        flat_overrides: dict[str, int] = {}
        threshold_overrides: dict[str, dict[str, Any]] = {}
        for name, entry in overrides.items():
            if isinstance(entry, dict) and "effective_limit" in entry:
                flat_overrides[str(name)] = int(entry["effective_limit"])
            if isinstance(entry, dict):
                picked = {
                    k: entry[k] for k in
                    ("compaction_threshold_pct", "compaction_threshold_tokens")
                    if k in entry
                }
                if picked:
                    threshold_overrides[str(name)] = picked

        return cls(
            provider=provider,
            model=model,
            effective_limit=effective_limit,
            model_overrides=flat_overrides,
            threshold_overrides=threshold_overrides,
            detected_limit=detected_limit,
            # The registry's per-backend windows (live) and operator hints — None
            # when the daemon booted without a registry (tests, web-only).
            windows=registry.detected_windows if registry is not None else None,
            backend_hints={
                spec.name: spec.context_limit
                for spec in (registry.specs() if registry is not None else ())
                if getattr(spec, "context_limit", None)
            },
            cloud_default_limit=int(
                ctx.get("cloud_default_limit", DEFAULT_CLOUD_LIMIT)
            ),
            threshold_pct=comp.get("threshold_pct", DEFAULT_THRESHOLD_PCT),
            reserve_tokens=comp.get("reserve_tokens", DEFAULT_RESERVE_TOKENS),
            protect_recent_turns=comp.get(
                "protect_recent_turns", DEFAULT_PROTECT_RECENT_TURNS),
            max_summary_tokens=comp.get(
                "max_summary_tokens", DEFAULT_MAX_SUMMARY_TOKENS),
            telemetry=telemetry,
            signal_bus=signal_bus,
        )

    @property
    def signal_bus(self) -> Any | None:
        return self._signal_bus

    @signal_bus.setter
    def signal_bus(self, bus: Any) -> None:
        """Late wiring — the daemon constructs SignalBus after the loop."""
        self._signal_bus = bus

    # -- estimation --------------------------------------------------------

    @staticmethod
    def _message_tokens(msg: "ConversationMessage") -> int:
        """Estimated tokens for one message.

        Delegates to the shared estimator so the number /context REPORTS and
        the number this class ACTS ON are produced by one function. They were
        two before, and the reporting one omitted the conversation entirely.
        """
        return estimate_message_tokens(msg)

    def estimate_total(
        self,
        system_prompt: str,
        messages: list,
        tools_chars: int = 0,
    ) -> int:
        """Estimated tokens of the assembled request."""
        total = estimate_tokens(system_prompt or "")
        total += sum(self._message_tokens(m) for m in messages)
        total += max(0, int(tools_chars)) // 4  # same chars/4 heuristic
        return total

    def limit_for(self, model: str | None = None, backend: str | None = None) -> int:
        """Context budget for whichever model is serving THIS call.

        Precedence, most specific first:

        1. ``context.model_overrides.<model>`` — an operator said so
           explicitly; nothing should second-guess that.
        2. The size the local server REPORTED, when this is the local model.
           llama.cpp publishes ``n_ctx`` at ``/props``, so for local
           inference the right number is knowable and never needs configuring.
        3. ``cloud_default_limit`` for any other model — i.e. a session
           routed to a cloud provider. Cloud APIs do NOT publish context
           length (Alibaba's ``/v1/models`` returns only id/object/created/
           owned_by, and the others are the same), so this is a configured
           floor, not a detection. Override per model when you know better.
        4. The configured global, as a last resort.

        The failure this ordering exists to prevent runs BOTH ways: budgeting
        a 32k local server at 72k leaves no room to generate (empty
        responses), and budgeting a 1M cloud model at the local GGUF's 28k
        throws away 97% of the window.
        """
        return self._resolve_limit(model, backend)[0]

    def is_measured(self, model: str | None = None, backend: str | None = None) -> bool:
        """True when the budget for *model* came from a server's REPORT (n_ctx).

        What the #356 pre-flight is gated on: refusing against a measured
        window is honest, refusing against a configured floor is a guess. It
        used to be inferred from the fallback target or a named backend, so a
        local box with no fallback configured — the CLI's spawned workers
        among them (#704) — had a measured window and no pre-flight.
        """
        return self._resolve_limit(model, backend)[1].startswith("detected")

    def _resolve_limit(self, model: str | None, backend: str | None) -> tuple[int, str]:
        name = model or self._model
        if not name:
            return (int(self._detected_limit or self._effective_limit),
                    "detected" if self._detected_limit else "config")
        # One resolver (context/budget.py) for the reported number AND the
        # enforced one — this method used to restate the precedence by hand,
        # which is how the two could drift. Multi-backend: `backend` selects
        # that box's REPORTED window from the registry, live.
        ctx = {
            "model_overrides": {
                m: {"effective_limit": v} for m, v in self._model_overrides.items()
            },
            "effective_limit": self._effective_limit,
            "cloud_default_limit": self._cloud_default_limit,
        }
        resolved, source = resolve_effective_limit(
            ctx,
            model=name,
            local_model=self._model,
            detected_limit=self._detected_limit,
            backend=backend,
            detected=self._windows() if (backend and self._windows) else None,
            backend_hint=self._backend_hints.get(backend) if backend else None,
        )
        if resolved is None:
            return (int(self._detected_limit or self._effective_limit),
                    "detected" if self._detected_limit else "config")
        return int(resolved), source

    def threshold_for(
        self, model: str | None = None, backend: str | None = None,
    ) -> tuple[int, str]:
        """``(tokens, source)``: when *model*'s context is compacted (#773).

        80% of its window by default; a per-model entry overrides it. The one
        place the threshold is resolved, for acting AND for reporting.
        """
        name = model or self._model
        return resolve_threshold(
            self.limit_for(model, backend),
            global_pct=self._threshold_pct,
            model_entry=self._threshold_overrides.get(name) if name else None,
        )

    def _threshold_tokens(self, model: str | None = None) -> int:
        return self.threshold_for(model)[0]

    # -- span selection ----------------------------------------------------

    def _boundary_for(self, messages: list, keep_turns: int) -> int:
        """First protected index when *keep_turns* user turns are protected.

        0 when the conversation holds fewer than that many user turns — the
        whole thing is protected and nothing is compactable.
        """
        seen = 0
        for i in range(len(messages) - 1, -1, -1):
            if getattr(messages[i], "role", "") == "user":
                seen += 1
                if seen >= keep_turns:
                    return self._align(messages, i)
        return 0

    def _align(self, messages: list, boundary: int) -> int:
        """Move *boundary* back so the tail never STARTS with tool results.

        Tool results are user-role messages, so a boundary counted in user
        messages can land on one — and then the call it answers is summarised
        away while its result stays, which every provider rejects (a result
        with no call before it). The tail starts at the call instead.
        """
        while 0 < boundary < len(messages) and self._is_tool_result_message(messages[boundary]):
            boundary -= 1
        return boundary

    def _tail_tokens(self, messages: list, boundary: int) -> int:
        """Tokens held by the protected tail beginning at *boundary*."""
        return sum(self._message_tokens(m) for m in messages[boundary:])

    def _protected_boundary(
        self, messages: list, tail_budget: int | None = None
    ) -> int:
        """Index of the first message of the protected tail: the last
        ``protect_recent_turns`` user-role messages and everything after the
        earliest of them never compact (microcompaction's fresh-window
        convention).

        *tail_budget* is the room the tail has to fit in. Without one the
        tail is protected unconditionally — and an unconditional tail is how
        a compaction pass becomes a no-op that still costs a turn. Once the
        protected region ALONE exceeds the threshold, no span chosen in front
        of it can bring the total under, so the pass re-runs on every
        subsequent turn: same prefix re-summarised, a summary of a summary,
        and a model reading an ever more lossy history while the logs report
        each pass as a success.

        Measured on a live session: 36-40 messages collapsed to one summary
        every turn, landing at 22.7k-23.5k against a 21,504 threshold —
        over by 1.2k-2k each time, for hours.

        So when a budget is given and the tail busts it, the protection
        yields one user turn at a time. The floor is the most recent user
        turn: the question being answered is never summarised away.
        """
        boundary = self._boundary_for(messages, self._protect_recent_turns)
        if tail_budget is None:
            return boundary
        if self._tail_tokens(messages, boundary) <= tail_budget:
            return boundary
        for keep in range(self._protect_recent_turns - 1, 0, -1):
            candidate = self._boundary_for(messages, keep)
            if candidate <= boundary:
                continue
            boundary = candidate
            if self._tail_tokens(messages, candidate) <= tail_budget:
                break
        return boundary

    def _plan_tail(self, messages: list, tail_budget: int) -> tuple[int, list, int]:
        """§6 (#773): ``(boundary, view, trimmed)`` for a tail that must fit *tail_budget*.

        In the spec's order: first trim the largest OLDER tool results inside
        the protected tail (the last complete tool round is never trimmed);
        only if that is not enough, yield user turns into the summarised span,
        down to the floor — the last complete tool round. ``view`` is a copy of
        ``messages`` with the trims applied; ``messages`` itself is untouched.
        """
        boundary = self._boundary_for(messages, self._protect_recent_turns)
        if self._tail_tokens(messages, boundary) <= tail_budget:
            return boundary, messages, 0
        view, trimmed = self._trim_tail_results(messages, boundary, tail_budget)
        if self._tail_tokens(view, boundary) <= tail_budget:
            return boundary, view, trimmed
        for keep in range(self._protect_recent_turns - 1, 0, -1):
            candidate = self._boundary_for(messages, keep)
            if candidate <= boundary:
                continue
            boundary = candidate
            if self._tail_tokens(view, candidate) <= tail_budget:
                break
        return boundary, view, trimmed

    @staticmethod
    def _is_tool_result_message(msg: Any) -> bool:
        content = getattr(msg, "content", None) or ()
        return bool(content) and all(getattr(b, "type", "") == "tool_result" for b in content)

    def _current_turn_start(self, messages: list) -> int | None:
        """Index of the last user message that is a real turn, not tool results."""
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if getattr(msg, "role", "") == "user" and not self._is_tool_result_message(msg):
                return i
        return None

    def _select_span_end(
        self, messages: list, tail_budget: int | None = None
    ) -> int:
        """End index (exclusive) of the compactable prefix span.

        The span is the contiguous prefix of span-transparent messages (see
        :data:`_SPAN_TRANSPARENT`) up to the protected boundary. Any other
        provenance (task_supervisor, cron, orchestrator, compactor, …) is a
        BARRIER: trust-tagged injections and managed-task context are never
        summarized away, and the single-layer guarantee holds because
        synthetic summaries can never enter a span.
        """
        boundary = self._protected_boundary(messages, tail_budget)
        end = 0
        for i in range(boundary):
            if getattr(messages[i], "provenance", "user") not in _SPAN_TRANSPARENT:
                break
            end = i + 1
        return end

    @staticmethod
    def _span_key(session_id: str, span: list) -> str:
        h = hashlib.sha256()
        h.update(session_id.encode("utf-8", "replace"))
        for msg in span:
            h.update(b"\x00")
            h.update(getattr(msg, "role", "").encode("utf-8", "replace"))
            h.update(b"\x01")
            h.update(getattr(msg, "provenance", "user").encode("utf-8", "replace"))
            h.update(b"\x02")
            try:
                h.update(msg.content_json.encode("utf-8", "replace"))
            except Exception:
                h.update((getattr(msg, "text", "") or "").encode("utf-8", "replace"))
        return h.hexdigest()

    # -- summarizer input --------------------------------------------------

    @staticmethod
    def _render_message(msg: "ConversationMessage") -> str:
        parts: list[str] = []
        for block in getattr(msg, "content", None) or ():
            btype = getattr(block, "type", "")
            if btype == "text":
                parts.append(block.text)
            elif btype == "tool_use":
                parts.append(f"[called {block.name}({block.input})]")
            elif btype == "tool_result":
                flag = " ERROR" if getattr(block, "is_error", False) else ""
                parts.append(f"[result{flag}: {block.content}]")
        text = " ".join(p for p in parts if p)
        return f"{getattr(msg, 'role', '?')}: {text}"

    def _summarizer_window(self) -> int:
        """The summarizer's OWN context window: the model this compactor calls."""
        return int(self.limit_for(self._model))

    def _final_budget(self, span_tokens: int) -> int:
        """Output budget for the summary that replaces *span_tokens* (#773).

        Grows with the span, from ``max_summary_tokens`` to a few times that:
        one 512-token summary standing in for hundreds of thousands of tokens
        loses almost everything. Never more than a third of the summarizer's
        window, so the request that writes it still has room for its input.
        """
        grown = max(self._max_summary_tokens, int(span_tokens) // _FINAL_SUMMARY_RATIO)
        capped = min(grown, self._max_summary_tokens * _FINAL_SUMMARY_MAX_MULTIPLE)
        return max(1, min(capped, self._summarizer_window() // 3))

    def _input_budget(self, template: str, out_tokens: int) -> int:
        """Input tokens one summarizer request may carry beside *template* and its output."""
        window = self._summarizer_window()
        overhead = estimate_tokens(template.format(conversation="", parts="")) + 16
        return max(64, int(window * (1 - _WINDOW_MARGIN)) - overhead - out_tokens)

    def _chunk_lines(self, lines: list[str], budget: int) -> tuple[list[list[str]], int]:
        """Pack *lines* into chunks of at most *budget* tokens, in order.

        A single line bigger than a whole chunk is cut to fit (counted and
        marked — the only content a pass may shorten); no line is ever dropped.
        """
        chunks: list[list[str]] = []
        current: list[str] = []
        used = 0
        truncated = 0
        for line in lines:
            tokens = estimate_tokens(line)
            if tokens > budget:
                keep = max(1, len(line) * budget // max(1, tokens) - 40)
                line = line[:keep] + " …[cut to fit the summarizer]"
                tokens = estimate_tokens(line)
                truncated += 1
            if current and used + tokens > budget:
                chunks.append(current)
                current, used = [], 0
            current.append(line)
            used += tokens
        if current:
            chunks.append(current)
        return chunks, truncated

    async def _call_summarizer(
        self, prompt: str, max_tokens: int, session_id: str, span_len: int, part: str,
    ) -> str | None:
        raw = await self._envelope.call(
            provider=self._provider,
            model=self._model,
            prompt=prompt,
            max_tokens=max_tokens,
            operation="summarize_span",
            context={"session_id": session_id, "span_messages": span_len, "part": part},
            # The row's own column (WP-X.21 T10), under the loop's rule:
            # the conversation, and none at all on an ephemeral turn. The
            # anchor and cache keep the full id either way.
            session_id=recorded_session_id(session_id),
        )
        if raw is None or not str(raw).strip():
            return None
        return str(raw).strip()

    async def _summarize_span(
        self, span: list, span_tokens: int, session_id: str,
    ) -> tuple[str | None, int, int]:
        """Chunk-then-combine (#773). Returns ``(summary, chunks, truncated)``.

        Every message reaches some summarizer request, each sized to the
        summarizer's own window. One chunk: one call writes the final summary.
        More: each chunk is summarized, then the part summaries are merged —
        in further rounds while they still do not fit one request.
        """
        final_budget = self._final_budget(span_tokens)
        lines = [self._render_message(m) for m in span]
        budget = self._input_budget(_SUMMARY_PROMPT, max(final_budget, self._max_summary_tokens))
        chunks, truncated = self._chunk_lines(lines, budget)
        if len(chunks) == 1:
            summary = await self._call_summarizer(
                _SUMMARY_PROMPT.format(conversation="\n".join(chunks[0])),
                final_budget, session_id, len(span), "1/1")
            return summary, 1, truncated
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            part = await self._call_summarizer(
                _SUMMARY_PROMPT.format(conversation="\n".join(chunk)),
                self._max_summary_tokens, session_id, len(span), f"{i}/{len(chunks)}")
            if part is None:
                return None, len(chunks), truncated
            parts.append(part)
        combine_budget = self._input_budget(_COMBINE_PROMPT, final_budget)
        level = 0
        while True:
            level += 1
            labelled = [f"Part {i}: {p}" for i, p in enumerate(parts, 1)]
            groups, _ = self._chunk_lines(labelled, combine_budget)
            if len(groups) == 1:
                summary = await self._call_summarizer(
                    _COMBINE_PROMPT.format(parts="\n".join(groups[0])),
                    final_budget, session_id, len(span), f"combine {level}")
                return summary, len(chunks), truncated
            merged: list[str] = []
            for j, group in enumerate(groups, 1):
                piece = await self._call_summarizer(
                    _COMBINE_PROMPT.format(parts="\n".join(group)),
                    self._max_summary_tokens, session_id, len(span),
                    f"combine {level}.{j}")
                if piece is None:
                    return None, len(chunks), truncated
                merged.append(piece)
            if len(merged) >= len(parts):
                # A combine round that did not shrink the list would loop forever.
                return None, len(chunks), truncated
            parts = merged

    # -- §6: trimming inside the protected tail ---------------------------

    def _largest_recent_result_tokens(self, messages: list) -> int:
        """Largest tool result in the last :data:`RECENT_TOOL_ROUNDS` tool rounds."""
        largest = 0
        rounds = 0
        for msg in reversed(messages):
            if not self._is_tool_result_message(msg):
                continue
            rounds += 1
            for block in msg.content:
                largest = max(largest, estimate_tokens(str(getattr(block, "content", ""))))
            if rounds >= RECENT_TOOL_ROUNDS:
                break
        return largest

    def _trim_tail_results(
        self, out: list, start: int, budget: int,
    ) -> tuple[list, int]:
        """Trim the largest older tool results in ``out[start:]`` until it fits *budget* tokens.

        The last complete tool round keeps its results — it is what the model is
        acting on now. Trimmed results keep their ``tool_use_id`` (the provider
        pairs them) and say how much was removed. Render view only: messages
        are copied, never edited.
        """
        last_round = None
        for i in range(len(out) - 1, start - 1, -1):
            if self._is_tool_result_message(out[i]):
                last_round = i
                break
        candidates: list[tuple[int, int, int]] = []  # (tokens, msg index, block index)
        for i in range(start, len(out)):
            if i == last_round or not self._is_tool_result_message(out[i]):
                continue
            for j, block in enumerate(out[i].content):
                tokens = estimate_tokens(str(getattr(block, "content", "")))
                if tokens > 64:
                    candidates.append((tokens, i, j))
        candidates.sort(reverse=True)
        trimmed = 0
        out = list(out)
        total = self._tail_tokens(out, start)
        for tokens, i, j in candidates:
            if total <= budget:
                break
            msg = out[i]
            blocks = list(msg.content)
            marker = TRIMMED_RESULT_MARKER.format(n=tokens)
            blocks[j] = blocks[j].model_copy(update={"content": marker})
            out[i] = msg.model_copy(update={"content": blocks})
            total -= tokens - estimate_tokens(marker)
            trimmed += 1
        return out, trimmed

    def _floor_key(self, messages: list) -> str:
        """Identity of the protected floor (the current user turn), for §6's stuck mark."""
        start = self._current_turn_start(messages)
        if start is None:
            return ""
        return self._span_key("floor", [messages[start]]) + f":{start}"

    # -- telemetry ---------------------------------------------------------

    async def _record_event(self, signal_type: str, payload: dict) -> None:
        """One durable write path (Sprint-A convention): the SignalBus
        persists to signal_events before broadcasting; without a bus, write
        the telemetry row directly. Never both."""
        try:
            if self._signal_bus is not None:
                from prometheus.sentinel.signals import ActivitySignal

                await self._signal_bus.emit(ActivitySignal(
                    kind=signal_type,
                    payload=payload,
                    source="context_compactor",
                ))
                return
            if self._telemetry is not None:
                self._telemetry.record_signal_event(
                    signal_type=signal_type,
                    payload=payload,
                    source_subsystem="context_compactor",
                )
        except Exception:
            log.exception("ContextCompactor: telemetry event write failed")

    async def _fail_loud(self, reason: str, session_id: str, span_len: int) -> None:
        log.error(
            "ContextCompactor FAILED (%s) for session %s — falling back to "
            "the pre-existing behavior: sending the UNCOMPACTED prompt "
            "(provider-side overflow possible)", reason, session_id,
        )
        if self._telemetry is not None:
            try:
                self._telemetry.record_silent_failure(
                    subsystem="context_compactor",
                    operation="summarize_span",
                    exc=RuntimeError(reason),
                    context={"session_id": session_id, "span_messages": span_len},
                )
            except Exception:
                log.warning("ContextCompactor: silent_failure write failed",
                            exc_info=True)
        await self._record_event("context_compaction_failed", {
            "session_id": session_id,
            "span_messages": span_len,
            "reason": reason,
        })

    # -- the compaction ----------------------------------------------------

    async def apply(
        self,
        messages: list,
        *,
        session_id: str = "",
        system_prompt: str = "",
        tools_chars: int = 0,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> list:
        """Return the render view: ``messages`` itself when no compaction is
        needed, or a NEW list with the oldest span replaced by one synthetic
        summary message (and, if that is not enough, older tool results in the
        current turn trimmed). NEVER mutates ``messages`` or any message in it.

        *model* is the model actually serving THIS turn, which is not
        necessarily the one this compactor was built with — a per-session
        override (/claude, /qwen, …) routes elsewhere while the compactor
        instance is shared process-wide. Passing it lets the budget follow
        the session instead of being frozen at daemon start.

        Fires when the request is over the model's threshold, OR when the NEXT
        round would not fit: current size + this request's *max_tokens* + the
        largest recent tool result over the window (#773). Aims at
        :data:`TARGET_FRACTION` of the line, not just under it.
        """
        tokens_before = self.estimate_total(system_prompt, messages, tools_chars)
        threshold, threshold_source = self.threshold_for(model)
        limit = self.limit_for(model)
        out_room = int(max_tokens) if max_tokens else self._reserve_tokens
        largest = self._largest_recent_result_tokens(messages)
        projected_fits = tokens_before + out_room + largest <= limit
        if tokens_before <= threshold and projected_fits:
            return messages
        trigger = "threshold" if tokens_before > threshold else "projected"
        # The line this pass must get under, and the lower point it aims at.
        ceiling = max(1, min(threshold, limit - out_room - largest))
        target = int(ceiling * TARGET_FRACTION)

        # Room the protected tail is allowed to occupy: the target less the
        # fixed costs that are there whatever we do (system prompt, tool
        # schemas). Deliberately NOT less the summary's worst case: that made
        # the tail yield while convergence was still reachable.
        fixed_tokens = (
            estimate_tokens(system_prompt or "") + max(0, int(tools_chars)) // 4
        )
        tail_budget = max(0, target - fixed_tokens)

        # §6: a session marked stuck on this same floor pays for no new summary.
        floor_key = self._floor_key(messages)
        frozen = bool(floor_key) and self._stuck.get(session_id) == floor_key

        boundary, view, trimmed = self._plan_tail(messages, tail_budget)
        max_end = 0
        for i in range(boundary):
            if getattr(messages[i], "provenance", "user") not in _SPAN_TRANSPARENT:
                break
            max_end = i + 1

        # Anchored span: reuse the previously summarized span while the
        # compacted estimate stays under the line (zero model calls); extend
        # only when it re-crosses.
        span_end = max_end
        anchored = self._session_spans.get(session_id)
        if anchored is not None and MIN_SPAN_MESSAGES <= anchored <= max_end:
            anchored_key = self._span_key(session_id, messages[:anchored])
            cached = self._cache.get(anchored_key)
            if cached is not None:
                anchored_tokens = (
                    tokens_before
                    - sum(self._message_tokens(m) for m in messages[:anchored])
                    + estimate_tokens(cached)
                )
                if anchored_tokens <= ceiling or frozen:
                    span_end = anchored

        span: list = []
        summary: str | None = None
        cache_hit = False
        chunks = 0
        truncated = 0
        duration_ms = 0.0
        if span_end >= MIN_SPAN_MESSAGES:
            span = messages[:span_end]
            span_tokens = sum(self._message_tokens(m) for m in span)
            key = self._span_key(session_id, span)
            summary = self._cache.get(key)
            cache_hit = summary is not None
            if not cache_hit and not frozen:
                started = time.time()
                summary, chunks, truncated = await self._summarize_span(
                    span, span_tokens, session_id)
                duration_ms = (time.time() - started) * 1000.0
                # Sanity gate: empty, or not actually smaller than what it
                # replaces → loud failure + explicit fallback.
                if summary is None:
                    await self._fail_loud(
                        "summarizer returned empty/failed", session_id, len(span))
                    return messages
                if estimate_tokens(summary) >= span_tokens:
                    await self._fail_loud(
                        f"summary ({estimate_tokens(summary)} tok) not smaller "
                        f"than the span it replaces ({span_tokens} tok)",
                        session_id, len(span),
                    )
                    return messages
                self._cache[key] = summary
                while len(self._cache) > _CACHE_MAX_ENTRIES:
                    self._cache.popitem(last=False)
                self._session_spans[session_id] = span_end
            if summary is None:
                span = []

        out: list
        if span and summary is not None:
            # Build the substitution — a single clearly-marked synthetic message,
            # tagged compactor-generated (closed Provenance enum), trusted (the
            # agent's own machinery; untrusted turns can never be IN a span).
            from prometheus.engine.messages import ConversationMessage, TextBlock

            text = f"{SUMMARY_MARKER_PREFIX} 1–{len(span)}] {summary}"
            # The question the current turn is answering is never summarised
            # away (§6): when the span covers it, it rides along verbatim.
            question_at = self._current_turn_start(messages)
            if question_at is not None and question_at < len(span):
                text += (
                    "\n\n[The user's current request, verbatim:]\n"
                    + (messages[question_at].text or "")
                )
            synthetic = ConversationMessage(
                role="user",
                content=[TextBlock(text=text)],
                provenance="compactor",
                is_trusted=True,
            )
            out = [synthetic] + list(view[len(span):])
        else:
            out = list(view)

        tokens_after = self.estimate_total(system_prompt, out, tools_chars)

        converged = tokens_after <= ceiling
        if not span and not trimmed:
            # Over the line with no remedy at all. WARNING, not DEBUG.
            log.warning(
                "ContextCompactor: over the line (%d tokens, ceiling %d, trigger %s) "
                "for %s but nothing is compactable.",
                tokens_before, ceiling, trigger, session_id,
            )
            await self._record_event("context_compaction_noop", {
                "session_id": session_id,
                "tokens_before": tokens_before,
                "threshold": threshold,
                "span_messages": max_end,
                "over_by": tokens_before - ceiling,
                "trigger": trigger,
            })
            await self._mark_stuck(session_id, floor_key, frozen, tokens_before, ceiling)
            return messages

        log.info(
            "ContextCompactor: %d msgs → 1 summary, %d tool result(s) trimmed (%s); "
            "tokens %d → %d (ceiling %d, target %d, trigger %s)%s",
            len(span), trimmed, session_id, tokens_before, tokens_after, ceiling,
            target, trigger, " [cache]" if cache_hit else "",
        )
        if converged:
            self._stuck.pop(session_id, None)
        else:
            # A pass that lands over the line must not read like one that lands
            # under it. Said every time it happens; the stuck mark (once) is
            # what stops it costing a new summary every round.
            log.warning(
                "ContextCompactor: pass did NOT converge for %s — %d tokens "
                "after compaction still exceeds the %d threshold by %d.",
                session_id, tokens_after, ceiling, tokens_after - ceiling,
            )
            await self._mark_stuck(session_id, floor_key, frozen, tokens_after, ceiling)
        await self._record_event("context_compaction", {
            "session_id": session_id,
            "span_messages": len(span),
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "ceiling": ceiling,
            "target": target,
            "trigger": trigger,
            "converged": converged,
            "over_by": max(0, tokens_after - ceiling),
            "duration_ms": round(duration_ms, 1),
            "cache_hit": cache_hit,
            "chunks": chunks,
            # #773: always 0 now — the span is chunked, never cut.
            "summarizer_input_omitted": 0,
            "summarizer_input_truncated": truncated,
            "tool_results_trimmed": trimmed,
        })
        return out

    async def _mark_stuck(
        self, session_id: str, floor_key: str, frozen: bool, tokens: int, ceiling: int,
    ) -> None:
        """§6: say ONCE that this session cannot get under the line, and stop paying for it."""
        if frozen or not floor_key:
            return
        self._stuck[session_id] = floor_key
        log.warning(
            "ContextCompactor: session %s cannot get under the line (%d tokens, ceiling %d) "
            "even after summarising everything before the current turn and trimming its "
            "older tool results. No new summaries until the turn changes; the provider "
            "may refuse the request for size.", session_id, tokens, ceiling,
        )
        await self._record_event("context_compaction_stuck", {
            "session_id": session_id,
            "tokens": tokens,
            "ceiling": ceiling,
            "over_by": max(0, tokens - ceiling),
        })
