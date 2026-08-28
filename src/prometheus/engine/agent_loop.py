# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/engine/query.py
# License: MIT
# Modified: decoupled from Anthropic API — replaced SupportsStreamingMessages + openharness.api.client
#           with abstract ModelProvider from prometheus.providers.base;
#           renamed all imports (openharness → prometheus);
#           removed auto-compact (Sprint 4 concern — openharness.services.compact not yet ported);
#           wrapped run_query() async generator into AgentLoop class with run() sync entry point;
#           ToolRegistry / PermissionChecker are optional (stubs used when not provided)

"""Core tool-aware agent loop."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from prometheus.adapter import markup_guard
from prometheus.engine.messages import (
    ConversationMessage,
    ToolResultBlock,
    render_messages_for_model,
)
from prometheus.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    StreamEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from prometheus.config.ephemeral import is_session_ephemeral
from prometheus.config.shipped_defaults import SHIPPED_MAX_TOOL_ITERATIONS
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.context.system_prompt import rewrite_model_identity
from prometheus.engine.fallback import stream_round_with_fallback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Sprint 10 / Phase 2: Model Router + Divergence Detector
    # Lazy-imported at runtime to avoid circular import
    # (coordinator.__init__ → subagent → engine.agent_loop)
    from prometheus.router import ModelRouter, RouteDecision
    from prometheus.coordinator.divergence import DivergenceDetector, CheckpointStore

log = logging.getLogger(__name__)

# SPRINT-4 audit HIGH-RISK #13 — one-shot WARN guard for legacy
# permission_checker.evaluate signature (without `origin` kwarg). Flipped
# True the first time the TypeError fallback fires per process so the
# deprecation surfaces without spamming logs every tool call.
_LEGACY_PERMISSION_CHECKER_WARNED: bool = False

PermissionPrompt = Callable[[str, str], Awaitable[bool]]
AskUserPrompt = Callable[[str], Awaitable[str]]


@dataclass
class RunResult:
    """The outcome of a completed agent run."""

    text: str
    messages: list[ConversationMessage]
    usage: UsageSnapshot = field(default_factory=UsageSnapshot)
    turns: int = 0


class _IterationReason:
    """Constants for why the agent loop continued on each iteration."""
    TOOL_SUCCESS = "tool_success"
    TOOL_ERROR_RETRY = "tool_error_retry"
    GRAMMAR_REPAIR = "grammar_repair"
    CIRCUIT_BREAKER_TRIP = "circuit_breaker_trip"
    MAX_ITERATIONS_HIT = "max_iterations_hit"
    MODEL_FALLBACK = "model_fallback"
    # Provider dropped every tool call as malformed-empty (no name); the
    # loop fed structured guidance back and is retrying, breaker-bounded.
    MALFORMED_DROPPED = "malformed_dropped"
    EMPTY_RESPONSE = "empty_response"
    # Final-text hygiene stripped a tool-call envelope to nothing because the
    # extractor could not parse what the stripper could delete. Distinct from
    # EMPTY_RESPONSE on purpose: the model produced output, we removed it.
    # Telling these apart in telemetry is the whole point — 13 of these were
    # filed as empty_response and read as "the rig is broken".
    STRIPPED_TO_EMPTY = "stripped_to_empty"
    # LONGHAUL-1: the turn kept re-issuing the SAME call and kept getting the
    # SAME thing back. Distinct from MAX_ITERATIONS_HIT: that one fires on
    # round count alone and cannot tell a long productive run from a stuck
    # one. This fires on lack of PROGRESS, regardless of how many rounds in.
    UNPRODUCTIVE_REPEAT = "unproductive_repeat"


_FAILURE_CATEGORIES = (
    "empty_output",
    "raw_text",
    "special_char_escape",
    "malformed_json",
    "wrong_schema",
    "other",
)


def _categorize_failure(raw: str) -> str:
    """Categorize a failed tool-call attempt from its raw error/output text.

    Circuit Breaker Self-Diagnosis sprint. Pure logic, no LLM call.
    Exposed at module scope for direct unit testing.

    Categories:
      - empty_output: output is empty or whitespace
      - raw_text: no JSON-like delimiters at all (plain prose)
      - special_char_escape: JSON-like, but contains unescaped special chars
                            (%, backticks, literal newlines) that break parsing
      - malformed_json: JSON-like brackets but doesn't parse
      - wrong_schema: parses as JSON but isn't a tool-call shape
      - other: catch-all

    TODO(phase-4-followup): ``malformed_json`` conflates two distinct root
    causes that look identical from this function's perspective:
      (a) the MODEL emitted syntactically broken JSON
      (b) something UPSTREAM (e.g. a provider's streaming tool_use parser)
          dropped the model's real input and left a bare ``{}`` to be
          categorized — the model's output was fine, our code ate it.
    Case (b) is what the Haiku ``/claude`` bug looked like: the label sent
    users debugging Haiku when the bug was in AnthropicProvider's SSE
    finalize step. Worth a separate sprint to plumb a richer signal up from
    the provider layer so this function can return "parser_dropped_input"
    vs "model_bad_json" instead of the ambiguous shared label.
    """
    import json as _json
    import re as _re

    if raw is None:
        return "empty_output"
    stripped = raw.strip()
    if not stripped:
        return "empty_output"

    has_brackets = "{" in stripped or "[" in stripped
    if not has_brackets:
        return "raw_text"

    # Special-char detection — unescaped % or ` inside what looks like JSON args
    # These routinely break llama.cpp JSON emission (e.g. `ps -o %cpu,%mem`).
    # We check the original raw (not stripped) to catch cases like '"command":
    # "ps %cpu"'.
    if _re.search(r'(?<!\\)[%`]', raw):
        # If there's a brackets + unescaped % or `, that's the signature
        return "special_char_escape"

    # Try to parse. If it parses, decide between wrong_schema and other.
    try:
        parsed = _json.loads(stripped)
    except _json.JSONDecodeError:
        return "malformed_json"

    # Parsed — check if it looks like a tool call
    if isinstance(parsed, dict):
        keys = set(parsed.keys())
        # Rough tool-call shape markers
        if not (keys & {"name", "tool", "tool_name", "tool_use", "function"}):
            return "wrong_schema"
        return "wrong_schema"
    return "other"


@dataclass
class _CircuitBreaker:
    """Detect repeated tool-call failures and break the loop.

    Two thresholds:
      - max_identical: consecutive IDENTICAL errors (same tool+error) → stop
      - max_any: consecutive errors of ANY kind in a single turn → hard stop

    Circuit Breaker Self-Diagnosis sprint: the breaker now retains the last
    few raw error payloads and can run ONE diagnose-and-recover attempt via
    ``diagnose_and_recover()`` before the loop gives up.
    """

    max_identical: int = 3
    max_any: int = 5
    _last_error_key: str = ""
    _identical_count: int = 0
    _any_error_count: int = 0
    # Circuit Breaker Self-Diagnosis sprint state:
    _recent_errors: list[str] = field(default_factory=list)   # last 3 raw messages
    recovery_attempted: bool = False                           # one-shot guard
    last_failure_category: str = ""                            # from diagnose()

    def record_error(self, tool_name: str, error_msg: str) -> str | None:
        """Record an error. Returns a trip reason string, or None if OK."""
        error_key = f"{tool_name}:{error_msg[:120]}"
        self._any_error_count += 1

        # Retain last 3 raw messages for diagnose_and_recover()
        self._recent_errors.append(error_msg)
        if len(self._recent_errors) > self.max_identical:
            self._recent_errors = self._recent_errors[-self.max_identical:]

        if error_key == self._last_error_key:
            self._identical_count += 1
        else:
            self._last_error_key = error_key
            self._identical_count = 1

        if self._identical_count >= self.max_identical:
            return (
                f"{self._identical_count} consecutive identical errors "
                f"({self._last_error_key[:200]})"
            )
        if self._any_error_count >= self.max_any:
            return (
                f"{self._any_error_count} consecutive errors of mixed types "
                f"(last: {self._last_error_key[:200]})"
            )
        return None

    def record_success(self) -> None:
        """Reset all counters on successful tool execution."""
        self._last_error_key = ""
        self._identical_count = 0
        self._any_error_count = 0
        self._recent_errors = []
        # Note: recovery_attempted intentionally NOT reset — a successful tool
        # call after recovery should not re-arm the breaker for another
        # recovery in the same run. That's the "only ONE recovery attempt" rule.

    @property
    def is_formatting_error(self) -> bool:
        """True if the last trip was likely a tool-call formatting issue."""
        key = self._last_error_key.lower()
        return any(s in key for s in ("empty tool name", "unknown tool: ''", "malformed"))

    def diagnose_and_recover(
        self,
        *,
        context: "LoopContext",
        tool_name: str,
        intended_action: str = "",
    ) -> "_RecoveryResult":
        """Diagnose the circuit-breaker trip and attempt ONE recovery.

        Steps (per Circuit Breaker Self-Diagnosis sprint):
          1. Classify each retained raw error into a failure_category
          2. Log a diagnostic row to telemetry SQLite
          3. Attempt recovery (tier bump) if possible and not already tried
          4. Return a structured result with a user-facing message

        Never raises — any failure inside this method logs and returns a
        "recovery not possible" result so the caller can fall through to the
        normal circuit-breaker-trip message.
        """
        try:
            return _do_diagnose_and_recover(
                breaker=self,
                context=context,
                tool_name=tool_name,
                intended_action=intended_action,
            )
        except Exception as exc:
            log.warning("diagnose_and_recover crashed, suppressing: %s", exc, exc_info=True)
            # SPRINT-4 audit HIGH-RISK #10 fix: write a silent_failures row
            # alongside the WARN so /health can detect when the breaker's
            # own diagnostic path is failing. Pre-fix, the WARN-only path
            # meant a chronically-broken diagnose_and_recover would never
            # show up in telemetry. Best-effort — never raises.
            tel = getattr(context, "telemetry", None)
            if tel is not None and hasattr(tel, "record_silent_failure"):
                try:
                    tel.record_silent_failure(
                        subsystem="circuit_breaker",
                        operation="diagnose_and_recover",
                        exc=exc,
                        context={
                            "tool_name": tool_name,
                            "intended_action": intended_action,
                            "identical_count": self._identical_count,
                            "last_error_key": self._last_error_key[:200],
                        },
                    )
                except Exception:
                    log.debug(
                        "circuit_breaker: silent_failure write failed",
                        exc_info=True,
                    )
            return _RecoveryResult(
                recovered=False,
                recovery_method="error",
                failure_category="other",
                diagnostic_message=(
                    f"⚠️ Tool call failed {self._identical_count} times. "
                    f"Diagnosis unavailable (internal error)."
                ),
            )


@dataclass
class _RecoveryResult:
    """Structured return value from _CircuitBreaker.diagnose_and_recover().

    Used by run_loop to decide whether to continue (recovered=True) or give
    up while reporting the diagnostic message to the user.
    """
    recovered: bool
    recovery_method: str                    # "tier_bump", "none", "error", etc.
    failure_category: str
    diagnostic_message: str
    config_drift: bool = False
    new_tier: str | None = None             # set when tier_bump succeeded


_TIER_BUMP_LADDER: dict[str, str] = {
    "off": "light",
    "light": "full",
    # "full" has no bump target — full tier + still failing = give up
}


def _provider_name_for_telemetry(provider_instance: object) -> str:
    """Best-effort provider-name string for golden-trace flagging.

    Golden Trace Capture sprint: we need to know whether the current
    provider is cloud (anthropic/openai/gemini/xai/groq) vs local
    (llama_cpp/ollama) so ``ToolCallTelemetry.record()`` can compute
    ``is_golden``. Provider instances don't uniformly expose a
    ``provider_name`` attribute, so this falls back to mapping the
    concrete class name.

    Returns the provider name string (or "" if unknown — in which case
    golden flagging auto-fails, which is the safe default).
    """
    if provider_instance is None:
        return ""
    explicit = getattr(provider_instance, "provider_name", None)
    if explicit:
        return str(explicit)
    class_name = type(provider_instance).__name__.lower()
    if "anthropic" in class_name:
        return "anthropic"
    if "llamacpp" in class_name:
        return "llama_cpp"
    if "ollama" in class_name:
        return "ollama"
    if "openaicompat" in class_name:
        # OpenAICompatProvider is shared by openai / gemini / xai / groq.
        # Without access to the original config dict, we cannot distinguish
        # them — return "openai" as the most common case. Callers who need
        # precision should set provider_name on the instance explicitly.
        return "openai"
    if "stub" in class_name:
        return "stub"
    return ""


def _detect_config_drift(active_model: str) -> bool:
    """Compare the live model id against what's on disk in prometheus.yaml.

    Returns True iff the on-disk config specifies a model different from the
    one currently running. Silent False when no config file found or the
    file can't be parsed — we never block or mutate on this signal.

    Note: we intentionally do NOT auto-fix the config (denied_paths protects
    prometheus.yaml). The result just feeds the user-facing diagnostic.
    """
    import yaml
    candidates = [
        Path("config") / "prometheus.yaml",
        Path.home() / ".prometheus" / "prometheus.yaml",
    ]
    for candidate in candidates:
        # SPRINT-4 audit HIGH-RISK #9 fix: narrow the catch to genuine I/O +
        # YAML-parse errors and log them. Pre-fix, ``except Exception:
        # continue`` silently swallowed any error (including programming
        # bugs in the parser walker) — config drift would go undetected
        # without a warning. Same fix shape as PR #3 / 035f1fb.
        try:
            if not candidate.is_file():
                continue
            raw = candidate.read_text(encoding="utf-8")
            cfg = yaml.safe_load(raw) or {}
            expected = (cfg.get("model") or {}).get("model", "")
            if expected and expected != active_model:
                return True
            return False
        except (OSError, yaml.YAMLError) as exc:
            log.warning(
                "_detect_config_drift: could not read/parse %s (%s: %s); "
                "skipping this candidate",
                candidate, type(exc).__name__, exc, exc_info=True,
            )
            continue
    return False


def _format_diagnostic_message(
    *,
    trip_count: int,
    model_id: str,
    adapter_tier: str,
    failure_category: str,
    config_drift: bool,
    recovery_status: str,
    intended_action: str = "",
    golden_reference: str | None = None,
) -> str:
    """Build the user-facing circuit-breaker diagnostic message.

    Replaces the pre-sprint cryptic 'Circuit breaker tripped: ...' with a
    structured report the user can act on.

    Golden Trace Capture sprint: when ``golden_reference`` is provided, a
    "Reference" line is appended showing a cloud-teacher's successful call
    shape. This is purely additive — the diagnostic works unchanged when
    no golden trace exists.
    """
    lines = [
        f"⚠️ Tool call failed {trip_count} times. Diagnosis:",
        f" - Model: {model_id or 'unknown'}",
        f" - Adapter tier: {adapter_tier or 'unknown'}",
        f" - Failure type: {failure_category}",
        f" - Config drift: {'yes' if config_drift else 'no'}",
        f" - Recovery: {recovery_status}",
    ]
    if intended_action and recovery_status.startswith(("not possible", "attempted, failed")):
        lines.append(f" - Intended action: {intended_action[:300]}")
    if golden_reference:
        # Truncate for safety — the full reference is on the SQLite row.
        lines.append(f" - Reference (cloud teacher's call shape): {golden_reference[:400]}")
    return "\n".join(lines)


def _do_diagnose_and_recover(
    *,
    breaker: "_CircuitBreaker",
    context: "LoopContext",
    tool_name: str,
    intended_action: str,
) -> _RecoveryResult:
    """Actual diagnose + recover implementation (pulled out for try/except wrap)."""
    # ── Step 1: Diagnose ──────────────────────────────────────────
    raw_samples = list(breaker._recent_errors)
    category = (
        _categorize_failure(raw_samples[-1]) if raw_samples else "empty_output"
    )
    breaker.last_failure_category = category

    active_tier = getattr(context.adapter, "tier", "unknown") if context.adapter else "none"
    active_model_id = context.model or ""
    config_drift = _detect_config_drift(active_model_id)

    # ── Step 3: Recover (decide method, then act) ─────────────────
    recovery_method = "none"
    recovered = False
    new_tier: str | None = None

    if breaker.recovery_attempted:
        # Already tried once this run — give up cleanly per the one-shot rule.
        recovery_method = "already_attempted"
    elif category == "special_char_escape":
        # v1: we diagnose it but don't auto-rewrite the arguments. The user's
        # diagnostic message surfaces the category so they know what broke.
        recovery_method = "diagnostic_only:special_char_escape"
    elif active_tier in _TIER_BUMP_LADDER and context.adapter is not None:
        # Bump the adapter tier one rung up (off → light, light → full).
        next_tier = _TIER_BUMP_LADDER[active_tier]
        try:
            context.adapter.tier = next_tier
            new_tier = next_tier
            recovery_method = f"tier_bump:{active_tier}->{next_tier}"
            recovered = True
            breaker.record_success()   # clear counters so the loop can continue
        except Exception as exc:
            log.warning("Tier bump failed: %s", exc, exc_info=True)
            recovery_method = "tier_bump_failed"
    else:
        # Tier is already full (or unknown) — no recovery available.
        recovery_method = "no_recovery_available"

    # Mark as attempted regardless of outcome so we don't loop the diagnosis.
    breaker.recovery_attempted = True

    # ── Golden Trace Capture sprint: fetch golden reference for this tool ─
    # If cloud teacher models have successfully called this tool with zero
    # adapter retries, surface the most recent "parsed_tool_call" JSON in
    # the user-facing diagnostic and persist it on the SQLite row.
    golden_reference: str | None = None
    if context.telemetry is not None and hasattr(context.telemetry, "get_golden_traces"):
        try:
            golden = context.telemetry.get_golden_traces(tool_name=tool_name, limit=3)
            if golden:
                best = golden[0]
                golden_reference = best.get("parsed_tool_call") or None
        except Exception:
            log.debug("get_golden_traces failed in diagnose", exc_info=True)

    # ── Step 2: Log (write SQLite row AFTER we know recovery outcome) ─
    if context.telemetry is not None and hasattr(context.telemetry, "record_diagnosis"):
        try:
            context.telemetry.record_diagnosis(
                model_id=active_model_id,
                adapter_tier=str(active_tier),
                tool_name=tool_name,
                failure_category=category,
                config_drift=config_drift,
                raw_sample=(raw_samples[-1] if raw_samples else None),
                recovered=recovered,
                recovery_method=recovery_method,
                golden_reference=golden_reference,
            )
        except Exception:
            log.warning("Telemetry record_diagnosis failed", exc_info=True)

    # ── Step 4: Build user-facing message ─────────────────────────
    if recovered:
        status = f"attempted ({recovery_method}), will retry once"
    elif recovery_method == "already_attempted":
        status = "not possible (recovery already tried once this run)"
    elif recovery_method == "diagnostic_only:special_char_escape":
        status = (
            "not possible automatically — the model's arguments contained "
            "unescaped special characters (%, backticks). Simplify the "
            "command by hand and retry."
        )
    elif recovery_method == "no_recovery_available":
        status = (
            f"not possible at tier '{active_tier}' — already at the strictest "
            f"adapter configuration."
        )
    elif recovery_method == "tier_bump_failed":
        status = "attempted, failed (could not bump adapter tier)"
    else:
        status = f"not possible ({recovery_method})"

    message = _format_diagnostic_message(
        trip_count=breaker._identical_count or len(raw_samples),
        model_id=active_model_id,
        adapter_tier=str(active_tier),
        failure_category=category,
        config_drift=config_drift,
        recovery_status=status,
        intended_action=intended_action,
        golden_reference=golden_reference,
    )

    return _RecoveryResult(
        recovered=recovered,
        recovery_method=recovery_method,
        failure_category=category,
        diagnostic_message=message,
        config_drift=config_drift,
        new_tier=new_tier,
    )


@dataclass
class LoopContext:
    """Context shared across a loop run."""

    provider: ModelProvider
    model: str
    system_prompt: str
    max_tokens: int
    tool_registry: object | None = None       # ToolRegistry — wired in Sprint 2
    permission_checker: object | None = None  # PermissionChecker — wired in Sprint 4
    hook_executor: object | None = None       # HookExecutor — wired in Sprint 2
    adapter: object | None = None             # ModelAdapter — wired in Sprint 3
    telemetry: object | None = None           # ToolCallTelemetry — wired in Sprint 3
    cwd: Path = field(default_factory=Path.cwd)
    max_turns: int = 200
    max_tool_iterations: int = SHIPPED_MAX_TOOL_ITERATIONS
    # SPRINT-coding-mode v2 (scope item 1): per-RUN thinking override,
    # forwarded on every model call this context makes. None = provider
    # default (the global suppress_thinking config, normally True for
    # gemma); False = this run's calls think, WITHOUT flipping the global
    # default. The F1 envelope records the effective flag per call
    # (subsystem_runs.thinking), which is how tests assert it.
    suppress_thinking: bool | None = None
    # M5: hard ceiling on a single tool's execution. A hung tool (browser,
    # LSP, MCP, TTS — anything without its own deadline) otherwise freezes the
    # turn AND the session. Generous default so legitimate slow tools survive;
    # a tool can raise its own bar via the ``execution_timeout_seconds`` class
    # attribute on BaseTool.
    tool_timeout_seconds: float = 300.0
    # Cloud providers (adapter.tier == "off") typically chew through
    # iterations faster — Claude plans multi-step sequences that Gemma
    # would never attempt. When set, this limit applies whenever the
    # active adapter is at TIER_OFF; otherwise we fall back to the
    # local limit above.
    max_tool_iterations_cloud: int | None = None
    permission_prompt: PermissionPrompt | None = None
    ask_user_prompt: AskUserPrompt | None = None
    tool_metadata: dict[str, object] | None = None
    # Sprint 10: Model Router + Divergence Detector
    model_router: object | None = None
    divergence_detector: object | None = None
    # Sprint 20: LSP post-result hooks (modify tool result after execution)
    post_result_hooks: list[object] | None = None
    # Tool Calling Middle Layer sprint
    tool_loader: object | None = None     # DynamicToolLoader for deferred loading
    compactor: object | None = None       # ContextCompactor (compaction.enabled)
    # SPRINT-provider-fallback: where a turn goes when the provider fails TERMINALLY (auth /
    # billing — never a 429, which is already retried a layer down). None disables it, which is
    # also the Phase 4 opt-out coding mode relies on to keep its pinned-provider guarantee.
    fallback: object | None = None        # FallbackTarget
    # H1: per-result cap (tokens) applied to EACH tool result before injection,
    # via ToolResultTruncator's per-tool strategies. 0 disables (back-compat for
    # tests/benchmarks). Runs before the cross-result turn budget below, and
    # unlike that budget it also caps error results.
    tool_result_max: int = 0
    tool_results_turn_budget: int = 8000  # max tokens across ALL results per turn
    microcompact_after_turns: int = 3     # compact tool results older than N turns
    # Rewriting history invalidates the provider's cached prompt prefix. That
    # trade is worth it on a small local window and a clear loss on a cloud
    # one (see _microcompact_old_results), so cloud tiers skip it by default.
    # Set True to force the old behavior everywhere.
    microcompact_on_cloud: bool = False
    microcompact_keep_chars: int = 200    # chars to keep per compacted result
    microcompact_keep_chars_no_lcm: int = 500  # chars if LCM hasn't ingested
    lcm_engine: object | None = None      # LCMEngine for microcompaction checks
    # Agent profile (selector survey → wired): a zero-arg callable returning
    # the ACTIVE AgentProfile (or None = unfiltered). A RESOLVER rather than a
    # profile so /profile and Beacon switches reach the NEXT run through this
    # long-lived context — per-session values must be per-call parameters.
    profile_resolver: object | None = None
    # Phase 3.5: session_id used by the router's per-session override lookup.
    # Telegram: str(chat_id). Slack: str(channel_id). CLI: "cli". Web: "web".
    # Reserved: None and "system" never match any override (eval/benchmark/
    # cron/SENTINEL-adjacent paths use these so user commands never leak in).
    session_id: str | None = None
    # SPRINT-2 WS1: live ChatSession handle. When wired, run_loop drains the
    # ``queued_steers`` list before each model call and appends them to the
    # system prompt as a "[STEER FROM USER, mid-turn]: ..." addendum. None
    # when run from contexts that don't have a persistent session
    # (benchmarks, evals, cron). See engine/session.py module docstring.
    session_state: object | None = None
    # SPRINT-2 WS2: file-mutation verifier. When wired, the loop calls
    # ``pre_tool_use`` / ``post_tool_use`` around every tool dispatch and
    # ``post_turn`` when a turn ends without further tool calls. A summary of
    # claimed-vs-actual filesystem changes is appended as an injected turn
    # (``provenance="file_mutation_verifier"``) so the agent sees it on its
    # next turn. See prometheus/hooks/file_mutation_verifier.py.
    #
    # UNIQUE AMONG THE FIELDS HERE: this one is mutable STATE, not config, and
    # the daemon shares ONE instance across every surface. It is safe on this
    # shared context only because ``run_loop`` scopes every call to a per-turn
    # key. Anything else stateful added here needs the same treatment.
    file_mutation_verifier: object | None = None
    # Repair-pair flywheel: failed-call stash awaiting a matching success
    # within this loop run (keyed by tool name; "_malformed" for provider-
    # dropped empty envelopes). Lazily initialized; see learning/pair_capture.
    pair_pending: dict | None = None
    # Gym dual-scoring seam (series-2). Optional, None in production — the
    # ONLY caller is the gym runner, which scores the model's RAW emission
    # separately from the call that ACTUALLY EXECUTED (post adapter
    # repair/unwrap). messages keep the raw ToolUseBlock; the repaired input
    # is local to _execute_tool_call, so this callback is the one place both
    # are correlated by tool_use_id. Signature:
    #   observer(tool_use_id: str, raw: {"name","input"}, executed: {"name","input"})
    # Fires once per call that reaches execution, after all input mutations.
    # Inert by default → no live model-facing behavior change (Hard Rule 5).
    tool_call_observer: object | None = None
    # PASSIVE RECALL (MEMORY-3 follow-up): optional MemoryRecall. When wired,
    # run_loop matches the latest user message against memory.db facts and
    # appends a "# Recalled memory" section to THIS run's request-only system
    # prompt (the steer/nudge channel) — never to messages/history, or the
    # extractor would re-ingest its own output next cycle. None (benchmarks,
    # evals, coding mode, gym) = no recall, byte-identical prompts.
    memory_recall: object | None = None
    # SUNRISE: PeriodicNudge — a self-reflection prompt every N completed
    # assistant rounds within a run. Lives HERE, not on AgentLoop, because
    # ``AgentLoop.run_async`` is not the only entry point: the web bridge
    # (web/ws_server.py:_run_agent) drives ``run_loop`` directly with a
    # pre-built context, so anything the AgentLoop wrapper does around the
    # loop is invisible to every web / Beacon / Bridge turn. See the module
    # docstring of tests/test_run_async_web_parity.py.
    nudge: object | None = None


def _effective_max_tool_iterations(context: LoopContext) -> int:
    """Resolve the iteration limit for the currently-active provider.

    Cloud providers (adapter.tier == "off") use
    ``max_tool_iterations_cloud`` when configured; everything else —
    tier=light, tier=full, or no adapter — uses the local
    ``max_tool_iterations``. If ``max_tool_iterations_cloud`` is None,
    both paths share the local limit (fully backward-compatible).

    The resolver runs at each guard check rather than at context
    construction because the active provider/adapter can swap mid-loop
    via model fallback (see ``_try_model_fallback``).
    """
    if context.max_tool_iterations_cloud is None:
        return context.max_tool_iterations
    adapter = context.adapter
    if adapter is not None and getattr(adapter, "tier", None) == "off":
        return context.max_tool_iterations_cloud
    return context.max_tool_iterations


async def run_loop(
    context: LoopContext,
    messages: list[ConversationMessage],
    *,
    mode: str = "agent",
    session_id: str | None = None,
    tool_choice: object | None = None,
) -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]:
    """Run the conversation loop until the model stops requesting tools.

    Yields (StreamEvent, UsageSnapshot | None) tuples. The loop exits when
    the assistant returns a response with no tool_uses, or after max_turns.

    Thin wrapper around :func:`_run_loop` whose only job is the
    file-mutation verifier's turn scope. ONE verifier instance is shared by
    every surface (telegram, CLI, cron, web/Beacon), so its state is keyed by
    a token minted here — one per invocation, because one ``run_loop`` call
    IS one turn. The ``finally`` is the point of the wrapper: ``_run_loop``
    has five ``return`` sites plus max_turns exhaustion plus cancellation
    (the Stop button closes the generator mid-iteration), and a turn that
    exits any of those ways without dropping its record would leak snapshots
    into the shared instance.

    It also resolves the turn's EPHEMERAL flag, for the same reason and with
    the same scope: one ``run_loop`` call is one turn on one session. This is
    deliberately NOT a ``LoopContext`` field. ``run_daemon`` builds the web
    ``LoopContext`` ONCE at startup and every Beacon/Bridge session shares that
    single instance — which is exactly why ``mode`` and ``session_id`` are
    per-call arguments here rather than context fields (see
    ``ws_server._run_agent``). A per-chat privacy flag parked on the shared
    context would leak across concurrent sessions: one ephemeral chat would
    silently suppress persistence for every other web session, or worse, an
    ordinary chat would inherit ``False`` and persist a turn the user had
    flagged. Resolving it HERE means both loop construction sites get it by
    construction, with nothing to remember to pass and nothing to cross-talk.
    """
    # A duck-typed verifier without ``new_turn_key`` predates turn scoping;
    # it keeps its single accumulator and we never pass it a key it can't
    # accept. Capability is resolved ONCE here so the call sites below don't
    # each have to guess (and can't fall back silently mid-turn).
    fmv = getattr(context, "file_mutation_verifier", None)
    turn_key: str | None = None
    if fmv is not None and hasattr(fmv, "new_turn_key"):
        turn_key = fmv.new_turn_key(session_id or context.session_id)
    # The EFFECTIVE session id, same idiom as the verifier key above and the
    # router's override lookup below: the per-call argument wins, because on
    # the web path ``context.session_id`` belongs to the shared context and is
    # not this turn's session.
    ephemeral = is_session_ephemeral(session_id or context.session_id)
    # The same effective id, NAMED — because two telemetry writers used to read
    # ``context.session_id`` raw and recorded the literal "web" for every web
    # turn, which is the routing namespace daemon.py pins on the shared web
    # context, not this turn's conversation. tool_calls.session_id exists to
    # join back to lcm_messages; keyed on "web" that join matches nothing.
    #
    # DESCRIPTIVE READERS ONLY. Do NOT feed this to origin_from_session_id:
    # "web" is a load-bearing member of _USER_SESSION_LITERALS while "web:" is
    # NOT in _USER_SESSION_PREFIXES, so the real id classifies as SYSTEM where
    # the literal classifies as USER — and that classification decides whether
    # a human is treated as present to sanction the next tool call. Enforced by
    # tests/test_session_id_descriptive_only.py, not by this comment.
    effective_session_id = session_id or context.session_id
    # FL-4: the divergence detector's task scope, minted here for exactly the
    # reasons the verifier's turn_key is (one shared instance, one run_loop
    # call = one task) — and HERE rather than in ``AgentLoop.run_async``,
    # because run_async is only one of five callers. Wiring it there would
    # have left web/Beacon, the gym runner, the coding session and both CLI
    # paths untracked: CROSS-CUTTING §2, the two-loop defect, rebuilt.
    # ``start_task`` is what ``current_task_id`` was always missing — without
    # it every checkpoint and every evaluation returned at its first guard.
    div = getattr(context, "divergence_detector", None)
    div_task_id: str | None = None
    if div is not None and getattr(div, "enabled", False):
        try:
            div_task_id = div.new_task_id(session_id or context.session_id)
            div.start_task(div_task_id, _goal_message_from(messages))
        except Exception:
            # Fail-open: divergence is observational. It must never be able
            # to break a turn.
            log.debug("DivergenceDetector.start_task raised", exc_info=True)
            div_task_id = None
    try:
        async for item in _run_loop(
            context,
            messages,
            mode=mode,
            session_id=session_id,
            tool_choice=tool_choice,
            fmv_turn_key=turn_key,
            ephemeral=ephemeral,
            div_task_id=div_task_id,
            effective_session_id=effective_session_id,
        ):
            yield item
    finally:
        if turn_key is not None:
            try:
                fmv.discard_turn(turn_key=turn_key)
            except Exception:
                log.debug(
                    "FileMutationVerifier.discard_turn raised", exc_info=True,
                )
        if div_task_id is not None:
            # Same reason the verifier drains in a ``finally``: _run_loop has
            # five return sites plus max_turns exhaustion plus cancellation,
            # and a task that exits any of those without dropping its record
            # leaks state into the shared detector.
            try:
                div.end_task(div_task_id)
            except Exception:
                log.debug("DivergenceDetector.end_task raised", exc_info=True)


def _goal_message_from(messages: list[ConversationMessage]) -> str:
    """The task goal: the most recent user text in the turn's history.

    Same reverse-scan idiom ``AgentLoop.run_async`` uses to recover
    ``user_message`` from a supplied history. Empty string when there is no
    user turn (a cron/eval kickoff with a system prompt only), which
    ``GoalTracker.set_goal`` handles — it yields an empty objective set and
    an alignment score that never accuses.
    """
    for msg in reversed(messages):
        if getattr(msg, "role", None) == "user":
            return msg.text or ""
    return ""


async def _run_loop(
    context: LoopContext,
    messages: list[ConversationMessage],
    *,
    mode: str = "agent",
    session_id: str | None = None,
    tool_choice: object | None = None,
    fmv_turn_key: str | None = None,
    ephemeral: bool = False,
    div_task_id: str | None = None,
    effective_session_id: str | None = None,
) -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]:
    """The loop body. See :func:`run_loop` — call that, not this.

    ``ephemeral`` is resolved by :func:`run_loop` for the turn's session and
    threaded down to the tool-execution path, where it nulls the content
    columns on ``telemetry.tool_calls`` and skips repair-pair capture.

    ``div_task_id`` scopes every divergence call below to THIS task; None
    when the detector is absent or disabled.
    """
    # Scopes every verifier call below to THIS turn. Empty for a duck-typed
    # verifier that predates turn scoping (see run_loop).
    _fmv_kw: dict[str, str] = {} if fmv_turn_key is None else {"turn_key": fmv_turn_key}
    # Sprint B / Piece 2 + force-search: resolve the per-call tool directive. `tool_choice`
    # (auto|none|required|{"tool":X}) is the lever; `mode` is sugar (agent->auto, chat->none).
    # An explicit tool_choice wins, else `mode` resolves; unknown/None -> auto, so an
    # unrecognized value can never silently drop tools (the byte-identical default). "none"
    # empties the schema AND sets suppress_tools (provider drops the grammar); auto/required/
    # {tool} offer the full schema and let the provider's grammar SELECTION do the constraining.
    from prometheus.api.tool_choice import NONE as _TC_NONE, resolve_mode_to_tool_choice
    effective_tool_choice = tool_choice if tool_choice is not None else resolve_mode_to_tool_choice(mode)
    tools_enabled = effective_tool_choice != _TC_NONE
    # FIRST-ROUND FORCING (post-IGNITION follow-up 1): a forced directive
    # (required / {tool:X}) binds ONLY the first substantive model call of the
    # turn, then relaxes to auto — so a forced turn makes its forced call, sees
    # the result, and can conclude in prose instead of burning rounds to the
    # iteration cap (the live proof's every-round-forcing finding). "none" and
    # "auto" are turn-wide as before (byte-identical dormant path). An
    # empty-response retry round does NOT consume the force — the force is
    # spent by the first round that yields text or a tool call.
    _force_directive = (
        effective_tool_choice
        if effective_tool_choice == "required" or isinstance(effective_tool_choice, dict)
        else None
    )
    _force_spent = False

    # Tool advertisement (feat/deferred-tools-tier-aware). Resolved ONCE here,
    # before the round loop, and tool_schema is never reassigned below — the
    # advertised catalog is frozen for the run. Changing it mid-run is the
    # #120 prefix-mutation bug class (it invalidates the provider's cached
    # prompt prefix from the tools block onward, i.e. everything).
    #
    # This call site previously used active_schemas() with no arguments, which
    # could not see the adapter — so tier-aware ("auto") resolution was
    # structurally impossible and every registered tool shipped every round
    # (measured: 49 schemas, ~9.6k tokens, 60.7% of round 0 on the EMBERFALL
    # baseline, of which 3 tools were used).
    tool_schema: list[dict] = []
    deferred_tools_active = False
    deferred_source = "tools disabled"
    if tools_enabled:
        if context.tool_loader is not None and hasattr(context.tool_loader, "resolve_deferred"):
            deferred_tools_active, deferred_source = (
                context.tool_loader.resolve_deferred(context.adapter)
            )
            tool_schema = context.tool_loader.schemas_for_run(deferred_tools_active)
        elif context.tool_loader is not None and hasattr(context.tool_loader, "active_schemas"):
            # Duck-typed loaders (tests, plugins) without the tri-state API.
            deferred_source = "legacy loader (no tri-state support)"
            tool_schema = context.tool_loader.active_schemas()
        elif context.tool_registry is not None and hasattr(context.tool_registry, "to_api_schema"):
            deferred_source = "no tool loader (registry direct)"
            tool_schema = context.tool_registry.to_api_schema()

    # Agent profile filter (selector survey → wired). Applied HERE — after
    # source resolution so it treats all three sources identically, before
    # the freeze so the run stays prefix-stable, and before the telemetry row
    # so "advertised" states what the model actually saw. Resolved per run:
    # a /profile or Beacon switch affects the next run, not a restart.
    active_profile = None
    if tools_enabled and tool_schema and context.profile_resolver is not None:
        try:
            active_profile = context.profile_resolver()
        except Exception:
            log.warning("profile resolver failed — advertising unfiltered", exc_info=True)
        if active_profile is not None:
            from prometheus.config.profiles import filter_tools_by_profile

            filtered = filter_tools_by_profile(tool_schema, active_profile)
            if filtered:
                tool_schema = filtered
            else:
                # A profile that filters the catalog to NOTHING is a config
                # error wearing a quiet face — advertising zero tools is the
                # vault_search failure shape (the model concludes the
                # capability does not exist). Fail loud, run unfiltered.
                log.error(
                    "profile %r filtered all %d advertised tools — "
                    "advertising unfiltered instead; fix the profile's tool "
                    "names", active_profile.name, len(tool_schema),
                )
                active_profile = None

    # A/B measurability: one row per run stating what was advertised and why,
    # so deferred-vs-full comparisons come straight out of the DB instead of
    # being reconstructed from token counts.
    if context.telemetry is not None and tools_enabled:
        try:
            _registered_total = (
                len(context.tool_registry.list_tools())
                if context.tool_registry is not None
                and hasattr(context.tool_registry, "list_tools")
                else None
            )
            context.telemetry.record_run(
                subsystem="agent_loop",
                operation="tool_advertisement",
                outcome="success",
                session_id=getattr(context, "session_id", None),
                model=getattr(context, "model", None),
                summary={
                    "deferred_active": deferred_tools_active,
                    "source": deferred_source,
                    "advertised": len(tool_schema),
                    "registered_total": _registered_total,
                    "profile": getattr(active_profile, "name", None),
                },
            )
        except Exception:
            log.debug("tool_advertisement telemetry write failed", exc_info=True)

    # Sprint 10 / Phase 2: route the first user message through ModelRouter.
    # The canonical router returns a RouteDecision with pre-instantiated
    # provider + adapter. For the default/primary path the decision's provider
    # is the same instance already on the context (no-op swap). When a rule,
    # smart-routing, override, or escalation branch fires, the swap activates.
    # Phase 3.5: session_id threaded via context dict so the router's
    # per-session override lookup can fire (or, for session_id in (None,
    # "system"), always resolve to primary).
    if context.model_router is not None and messages:
        # M4: route on the MOST RECENT user message, not the first. In a long
        # session the first message is stale — the current turn's request is the
        # latest user turn, which is what smart-routing / task classification
        # should see. (Per-session overrides are keyed on session_id and are
        # unaffected by which message text we pass.)
        latest_user = next(
            (m.text for m in reversed(messages) if m.role == "user" and m.text),
            None,
        )
        if latest_user:
            try:
                # Per-session override is keyed on the ACTUAL turn's session_id,
                # threaded per-call (like `mode`): callers share one LoopContext whose
                # `.session_id` is NOT the live turn's session, so reading it here made
                # REST/WS turns look the override up under a stale id and silently run
                # the primary. Fall back to context.session_id for callers (CLI, coding,
                # gym) that don't pass one.
                # Bound once and reused for BOTH the routing call and the log line
                # below, so the audit record can never name a different session than
                # the one the override was actually looked up under.
                route_session = (
                    session_id if session_id is not None else context.session_id
                )
                decision = context.model_router.route(
                    latest_user,
                    context={"session_id": route_session},
                )
                reason_repr = (
                    decision.reason.value
                    if hasattr(decision.reason, "value")
                    else decision.reason
                )
                # INFO, not DEBUG: which model served a turn is the first question asked
                # when a user says "I switched models and it did not take", and at DEBUG
                # the answer does not exist. Checked 2026-08-25 against a daemon up four
                # days: three days of journal held ZERO of these lines, so a session's
                # routing had to be inferred sideways from a provider-build side effect
                # that only fires on the FIRST turn after a switch — the built provider
                # is cached on the override, so every later turn is silent. One line per
                # turn: this block runs once, before the tool-iteration loop.
                #
                # The user's message text is deliberately NOT in it. The old DEBUG line
                # carried a 60-char excerpt of the prompt; promoting that verbatim would
                # write conversation content into the daemon journal at the default
                # level. The routing decision is what needs auditing, not what was said.
                log.info(
                    "ModelRouter: session=%s → %s/%s (%s)",
                    route_session,
                    decision.provider_name,
                    decision.model_name,
                    reason_repr,
                )
                if decision.provider is not None:
                    context.provider = decision.provider
                if decision.adapter is not None:
                    context.adapter = decision.adapter
                if decision.model_name:
                    context.model = decision.model_name
                # Phase 4 fix: after the router swap, rewrite the identity
                # line in the system prompt ("- Model: <name> (provider: <p>)")
                # to match the *active* provider. Without this, a primary-
                # baked system prompt says "Model: gemma4-26b" and Claude/GPT
                # dutifully impersonate the primary when the user asks "what
                # model is this?". The line is emitted by
                # prometheus.context.system_prompt._format_environment_section;
                # we rewrite it in-place rather than rebuilding the whole
                # prompt to avoid pulling environment detection into the hot
                # path of every request.
                if decision.provider_name or decision.model_name:
                    # Shared with the fallback handler rather than copied: a second copy is how
                    # a swapped-in model ends up claiming to be the primary, which is the bug
                    # this rewrite exists to prevent.
                    #
                    # `reason_repr == "primary"` is passed as the ANSWER to "is the serving
                    # model the local backend", which is what the old `!= "primary"` gate
                    # actually meant. See rewrite_model_identity's docstring — the fallback is a
                    # caller where route-reason and local-backend stop agreeing.
                    context.system_prompt = rewrite_model_identity(
                        context.system_prompt,
                        model_name=decision.model_name or "unknown",
                        provider_name=decision.provider_name or "unknown",
                        serving_is_local_backend=(reason_repr == "primary"),
                    )
            except Exception:
                # Phase 4: elevated from DEBUG → WARNING. A silent DEBUG here
                # hid a real production bug (stale-system-prompt identity)
                # from the logs. Any exception in route() means the user's
                # override (or task rule, or escalation) was NOT applied and
                # we silently fell through to primary — that's not something
                # we should discover by reading source code.
                log.warning(
                    "ModelRouter: route() raised — falling back to primary. "
                    "session_id=%r, latest_user=%r",
                    context.session_id,
                    (latest_user or "")[:60],
                    exc_info=True,
                )

    # Sprint 3: format tools + system prompt for the target model
    active_system_prompt = context.system_prompt
    active_tools = tool_schema
    if context.adapter is not None and hasattr(context.adapter, "format_request"):
        active_system_prompt, active_tools = context.adapter.format_request(
            context.system_prompt, tool_schema
        )

    # PASSIVE RECALL (MEMORY-3 follow-up): surface stored facts relevant to
    # the latest user message as a request-only system-prompt section for
    # this run. Same never-persisted channel as steers and the empty-retry
    # nudge — recalled facts committed to durable history would be
    # re-extracted on the next memory cycle (a feedback loop). Fail-open:
    # a recall error must never block the turn.
    if context.memory_recall is not None:
        _latest_user_text = next(
            (m.text for m in reversed(messages) if m.role == "user" and m.text),
            None,
        )
        if _latest_user_text:
            try:
                _recall_block = context.memory_recall.recall(_latest_user_text)
            except Exception:
                _recall_block = ""
                log.warning(
                    "MemoryRecall raised — continuing without recall",
                    exc_info=True,
                )
            if _recall_block:
                active_system_prompt = (
                    f"{active_system_prompt}\n\n{_recall_block}"
                )

    circuit_breaker = _CircuitBreaker(max_identical=3, max_any=5)
    # Per-turn repeat guard. The circuit breaker above trips at max_identical
    # consecutive errors, but only AFTER each one runs to its full timeout — so
    # a doomed command (an unbounded ``find`` that always hits the 300s tool
    # timeout) burns minutes per strike and the turn looks frozen. This map
    # counts how many times each exact (name, input) signature has failed this
    # turn; once it reaches _REPEAT_FAIL_LIMIT the call is refused WITHOUT
    # executing, so further identical retries cost ~0s. A success clears it.
    failed_call_signatures: dict[str, int] = {}
    _REPEAT_FAIL_LIMIT = 2
    # LONGHAUL-1 progress-aware repeat detector. Complements the guard above
    # rather than replacing it: that one counts FAILURES and clears on success,
    # so a call that keeps SUCCEEDING with identical output slips past it and
    # burns the flat iteration cap instead. Per-turn, same as everything else
    # here -- see the reset-scope note on tool_iteration.
    repeat_detector = _ProgressRepeatDetector()
    tool_iteration = 0
    # Empty-response guard: at most one model-call retry per run_loop on a
    # genuinely empty assistant reply (no text, no tool calls) before surfacing.
    # The retry's nudge rides ONLY the per-call request (per_call_system_prompt,
    # the /steer channel) — never appended to messages, so it can't leak into
    # session history.
    empty_retried = False
    pending_empty_nudge = False

    # SUNRISE PeriodicNudge — completed assistant rounds THIS run, and the
    # nudge text (if any) owed to the next model call. Counted in the loop
    # rather than in AgentLoop.run_async's `async for` so the web bridge's
    # direct run_loop call gets it too.
    nudge_turns = 0
    pending_periodic_nudge: str | None = None

    # SPRINT-loop-envelope (F1): the loop's model calls run inside the shared
    # LLMCallEnvelope like every other _call_model path — silent-failure
    # capture + a per-round usage row (tokens, round, session, model,
    # effective thinking flag) in subsystem_runs. Observation only: the
    # request and every stream event pass through unchanged (lazy import
    # matches this file's convention for cross-package deps).
    from prometheus.learning.llm_envelope import LLMCallEnvelope
    loop_envelope = LLMCallEnvelope("agent_loop", telemetry=context.telemetry)

    for turn in range(context.max_turns):
        # MicroCompaction: compact old tool results (free, no LLM calls)
        if turn > 0 and context.microcompact_after_turns > 0:
            _microcompact_old_results(context, messages, turn)

        final_message: ConversationMessage | None = None
        usage = UsageSnapshot()
        dropped_malformed = 0
        # Blocks that final-text hygiene stripped to nothing, and a truncated
        # sample of what they held. A SEPARATE counter from dropped_malformed
        # deliberately: both mean "we deleted what the model said" and both
        # take the same breaker-bounded retry below, but they are different
        # failures and need different corrections. dropped_malformed means
        # "you sent a call with no name"; this means "your envelope did not
        # parse". Sharing one integer would relabel a parse disagreement as a
        # provider drop and hand the model the wrong instruction.
        stripped_to_empty = 0
        stripped_samples: list[str] = []
        served_model_this_turn: str | None = None
        # Per-TURN, like served_model above. A list rather than a plain name because the
        # on_degrade callback below is a closure, and rebinding through it would need nonlocal.
        degrade_notice_this_turn: list[str] = []

        # SPRINT-2 WS1: drain queued steers as a system-prompt addendum
        # for THIS model call only. Steers arrive from the gateway's
        # /steer command path and accumulate while the loop is mid-turn
        # awaiting a tool result. Combined here so the model sees them
        # on its very next iteration — no role-alternation violation,
        # no fresh user turn.
        per_call_system_prompt = active_system_prompt
        _drain_steers = getattr(context.session_state, "drain_steers", None)
        if callable(_drain_steers):
            try:
                steer_text = _drain_steers()
            except Exception:
                steer_text = None
                log.debug(
                    "run_loop: drain_steers raised, treating as no-op",
                    exc_info=True,
                )
            if steer_text:
                per_call_system_prompt = (
                    f"{active_system_prompt}\n\n"
                    f"[STEER FROM USER, mid-turn]: {steer_text}"
                )

        # Empty-response retry nudge — request-only, same per-call channel as
        # steers. Set when the previous iteration returned an empty reply; rides
        # THIS call's system prompt only, never messages/history.
        if pending_empty_nudge:
            per_call_system_prompt = (
                f"{per_call_system_prompt}\n\n"
                "[RETRY: your previous response was empty — no text and no tool "
                "call. Respond now with a message or a tool call.]"
            )
            pending_empty_nudge = False

        # SUNRISE PeriodicNudge — the same request-only channel as steers,
        # the empty-retry nudge and passive recall. It rides THIS call's
        # system prompt and is never appended to ``messages``; see
        # _maybe_periodic_nudge for why the old user-message channel was
        # wrong on all three of the wire, the history and the UI.
        if pending_periodic_nudge:
            per_call_system_prompt = (
                f"{per_call_system_prompt}\n\n{pending_periodic_nudge}"
            )
            pending_periodic_nudge = None

        # H2: tier-full models lack native tool calling — their tools live in
        # the (formatter-augmented) system prompt and the GBNF grammar set on
        # the provider constrains output to valid tool-call JSON (or prose).
        # Sending tools in the payload would make llama.cpp drop the grammar
        # (see LlamaCppProvider._build_request_payload), so we withhold them at
        # tier full only. tier light (native + --jinja) and tier off (cloud)
        # are unchanged — they send tools and use no grammar.
        _payload_tools = active_tools
        if (
            context.adapter is not None
            and getattr(context.adapter, "tier", None) == "full"
        ):
            _payload_tools = []

        # FIRST-ROUND FORCING: this round's directive — the force until it is
        # spent, then auto (never touches "none"/"auto" turns).
        round_tool_choice = effective_tool_choice
        if _force_directive is not None and _force_spent:
            round_tool_choice = "auto"

        # LOCAL {tool:X}/required ENFORCEMENT VIA GRAMMAR (follow-up 2): the live
        # llama-server build silently ignores OpenAI-shape function-forcing on the
        # native tools path (IGNITION finding: forced web_search -> 11x read_file).
        # For the FORCED round only, when the current provider can enforce the
        # directive deterministically via GBNF (LlamaCppProvider with a wired
        # grammar source), withhold native tools so the provider's grammar path
        # fires — the grammar CANNOT be ignored by the server. The relaxed rounds
        # return to the native tools path. Cloud providers (no grammar) keep the
        # native param, which they honor (live-proven on anthropic).
        if (
            round_tool_choice is not None
            and round_tool_choice != "auto"
            and _payload_tools
            and hasattr(context.provider, "can_force_via_grammar")
            and context.provider.can_force_via_grammar(round_tool_choice)
        ):
            _payload_tools = []

        # Visible-stream hygiene: local tiers emit tool calls as inline
        # <tool_call> markup, and the raw token stream goes to every gateway —
        # without this filter the grammar tags render verbatim in chat bubbles
        # (Beacon web/desktop, Telegram). Streaming-only: the dispatch path
        # parses the COMPLETE turn text (final_message / raw_model_output),
        # which stays unfiltered. Tier off (cloud) streams no markup → no
        # filter, so quoted tags in cloud prose are never eaten.
        _markup_filter = None
        if (
            context.adapter is not None
            and getattr(context.adapter, "tier", None) != "off"
        ):
            from prometheus.adapter.formatter import ToolCallMarkupFilter
            _markup_filter = ToolCallMarkupFilter()

        # SPRINT-CONTEXT-COMPACTOR: when wired (compaction.enabled), replace
        # the oldest span with its cached summary in the RENDER VIEW only —
        # ``messages``, the session, and lcm.db are never mutated. The
        # compactor fails loud internally (telemetry + signal event) and
        # falls back to the pre-existing behavior: send unmodified.
        render_source = messages
        if context.compactor is not None:
            try:
                render_source = await context.compactor.apply(
                    messages,
                    session_id=context.session_id or "",
                    system_prompt=per_call_system_prompt,
                    tools_chars=len(str(_payload_tools)) if _payload_tools else 0,
                    # The model serving THIS turn — a per-session override
                    # (/claude, /qwen, …) routes elsewhere while the compactor
                    # instance is shared process-wide. Without this the budget
                    # stays frozen at the LOCAL model's window, so a cloud
                    # session with a far larger context was compacted as if it
                    # were the local GGUF.
                    model=context.model,
                )
            except Exception:
                log.exception(
                    "ContextCompactor.apply raised — sending uncompacted")
                render_source = messages

        # One round, degrading to context.fallback on a terminal provider failure. The envelope
        # still observes and re-raises; this is the recovery its docstring says the loop owns.
        def _build_request(_model: str) -> ApiMessageRequest:
            # Rebuilt per attempt: the model name is part of the payload, so reusing the failed
            # request would ask the local backend to serve the cloud model's name.
            return ApiMessageRequest(
                model=_model,
                # Context-assembly: fence any untrusted injected turns (task
                # output, watched-file contents, cron data) with the derived
                # banner before the provider serializes them. The session/LCM
                # copies stay clean — this projection is per-call only.
                messages=render_messages_for_model(render_source),
                system_prompt=per_call_system_prompt,
                max_tokens=context.max_tokens,
                tools=_payload_tools,
                suppress_thinking=context.suppress_thinking,
                suppress_tools=not tools_enabled,
                tool_choice=round_tool_choice,
            )

        def _window_for(_model: str) -> tuple[int, bool]:
            if context.compactor is None:
                return 0, False
            # Measured only for a local backend — llama.cpp publishes n_ctx at /props. Cloud
            # APIs do not publish context length at all, so that side is a configured floor and
            # the refusal message must say which it had.
            measured = bool(getattr(context.fallback, "is_local_backend", False))
            return context.compactor.limit_for(_model), measured

        def _estimate_tokens() -> int:
            if context.compactor is None:
                return 0
            return context.compactor.estimate_total(
                per_call_system_prompt,
                render_source,
                len(str(_payload_tools)) if _payload_tools else 0,
            )

        def _on_degrade(_decision) -> None:
            degrade_notice_this_turn.append(_decision.message)
            log.warning(
                "provider fallback: %s -> %s (%s) session=%r",
                context.model, _decision.model, _decision.message, context.session_id,
            )
            # The serving model is the fallback from here on, so the prompt's identity line has
            # to follow — shared with the router rather than copied, and keyed on whether the
            # SERVING model is the local backend (not on why it changed).
            context.system_prompt = rewrite_model_identity(
                context.system_prompt,
                model_name=_decision.model or "unknown",
                provider_name=_decision.provider_name or "unknown",
                serving_is_local_backend=bool(
                    getattr(context.fallback, "is_local_backend", False)
                ),
            )

        async for event in stream_round_with_fallback(
            envelope=loop_envelope,
            provider=context.provider,
            model=context.model,
            build_request=_build_request,
            target=context.fallback,
            enabled=context.fallback is not None,
            window_for=_window_for,
            estimate_tokens=_estimate_tokens,
            on_degrade=_on_degrade,
            operation="loop_round",
            round_index=turn,
            session_id=effective_session_id,
        ):
            if isinstance(event, ApiTextDeltaEvent):
                if _markup_filter is not None:
                    visible = _markup_filter.feed(event.text)
                    if visible:
                        yield AssistantTextDelta(text=visible), None
                else:
                    yield AssistantTextDelta(text=event.text), None
                continue

            if isinstance(event, ApiMessageCompleteEvent):
                final_message = event.message
                usage = event.usage
                # malformed_empty guard: count of tool-call entries the
                # provider dropped at parse time (empty function name).
                # getattr for providers predating the field.
                dropped_malformed = getattr(event, "dropped_malformed", 0)
                # Per-TURN, never stored on the shared LoopContext: concurrent
                # turns would cross-talk (the per-message `mode` precedent).
                # Threaded as a parameter exactly like raw_model_output.
                served_model_this_turn = getattr(event, "served_model", None)

        if _markup_filter is not None:
            # Release any withheld tag-lookalike text now that the stream ended.
            _tail = _markup_filter.flush()
            if _tail:
                yield AssistantTextDelta(text=_tail), None

        if final_message is None:
            raise RuntimeError("Model stream finished without a final message")

        # Golden Trace Capture sprint: capture the model's raw output BEFORE
        # the adapter's extract_tool_calls path rewrites final_message. This
        # string is what we'd want to train a local model to emit for the
        # current tool-calling task.
        raw_model_output_this_turn = final_message.text or ""

        # Sprint 3: try to extract tool calls from text when none came back structured
        if (
            not final_message.tool_uses
            and final_message.text
            and context.adapter is not None
        ):
            extracted = context.adapter.extract_tool_calls(
                final_message.text, context.tool_registry
            )
            if extracted:
                from prometheus.engine.messages import TextBlock
                # Keep residual prose around the markup (stream already showed it);
                # strip the tags so gateways that read result.text never see them.
                from prometheus.adapter.formatter import strip_tool_call_markup
                residual = strip_tool_call_markup(final_message.text).strip()
                content_blocks: list = list(extracted)
                if residual:
                    content_blocks.insert(0, TextBlock(text=residual))
                final_message = ConversationMessage(
                    role="assistant",
                    content=content_blocks,
                )

        # Final-text hygiene (local tiers only): gateways that deliver
        # result.text / AssistantTurnComplete.message.text (Telegram, Slack,
        # Discord, CLI) never saw the stream filter. Dual-emit paths
        # (structured tool_calls + leftover <tool_call> in content) and
        # extract-miss leftovers would otherwise leak raw grammar tags into
        # the chat bubble. raw_model_output_this_turn above stays unfiltered
        # for golden-trace capture. Tier off (cloud) leaves prose alone so
        # quoted tags in explanations are not eaten.
        if (
            _markup_filter is not None
            and final_message is not None
            and final_message.text
            and "<tool_call" in final_message.text
        ):
            from prometheus.adapter.formatter import strip_tool_call_markup
            from prometheus.engine.messages import TextBlock
            cleaned_blocks: list = []
            changed = False
            for block in final_message.content:
                if isinstance(block, TextBlock):
                    cleaned = strip_tool_call_markup(block.text)
                    if cleaned != block.text:
                        changed = True
                    if cleaned:
                        cleaned_blocks.append(TextBlock(text=cleaned))
                    else:
                        # ⚠ A BLOCK THAT STRIPS TO NOTHING IS A DROP, AND IT
                        # IS COUNTED. This line used to be the bare comment
                        # "empty after strip → drop the block" — no counter,
                        # no log, no telemetry — and it is where four turns
                        # of Will's went on 2026-08-17.
                        #
                        # Reaching here means the model DID emit a tool call
                        # and two components disagreed about it: the
                        # enforcer's extractor could not parse the envelope
                        # (or we would have replaced the message above), and
                        # the stripper could, so the stripper deleted it.
                        # The more permissive component wins by running
                        # second, and cleaning is destructive, so the
                        # evidence was gone before anything could count it.
                        # #77's family — there the grammar parser rejected
                        # what the admit-checker accepted.
                        stripped_to_empty += 1
                        if len(stripped_samples) < 3:
                            stripped_samples.append(block.text[:400])
                        log.warning(
                            "tool-call markup stripped to nothing — the "
                            "extractor could not parse an envelope the "
                            "stripper could delete. This is a PARSE "
                            "DISAGREEMENT, not an empty model response. "
                            "pre-strip text (truncated): %r",
                            block.text[:400],
                        )
                else:
                    cleaned_blocks.append(block)
            if changed:
                final_message = ConversationMessage(
                    role="assistant",
                    content=cleaned_blocks,
                )

        # ── Parse-disagreement guard ──
        # The model DID produce output; we deleted it. Handled BEFORE the
        # empty-response guard because it is not an empty response, and
        # reporting it as one is what sent Will "please rephrase and try
        # again" for four consecutive turns while the model was emitting tool
        # calls the whole time.
        #
        # RULING (asked for explicitly): this RETRIES with the text preserved
        # rather than surfacing a failure. The model produced output, so a
        # failure turn discards work and blames the operator for a
        # disagreement between two of our own components. It takes the same
        # breaker-bounded shape the dropped_malformed path already uses — the
        # retry carries structured feedback INCLUDING what the model actually
        # emitted, so it is an informed retry, not a blind one, and the
        # circuit breaker bounds it exactly as it bounds the malformed case.
        # A blind retry would risk the 66-minute-turn class; a fed-back retry
        # is the shape this loop already proved.
        if (
            stripped_to_empty
            and not (final_message.text or "").strip()
            and not final_message.tool_uses
        ):
            trip_reason = circuit_breaker.record_error(
                "_strip_disagreement",
                "tool-call envelope stripped to nothing (extractor missed)",
            )
            if trip_reason is None:
                _log_iteration(
                    context,
                    _IterationReason.STRIPPED_TO_EMPTY,
                    turn,
                    tool_iteration,
                    f"stripped={stripped_to_empty}",
                )
                messages.append(
                    ConversationMessage.from_injected(
                        _strip_disagreement_feedback(context, stripped_samples),
                        provenance="orchestrator",
                        is_trusted=True,
                    )
                )
                continue
            _log_iteration(
                context, _IterationReason.CIRCUIT_BREAKER_TRIP, turn, tool_iteration,
                f"strip_disagreement: {trip_reason}",
            )
            error_msg = _make_assistant_msg(
                "I emitted a tool call that could not be parsed, and the "
                "cleanup step removed it. This is a parsing fault on my side, "
                "not a problem with your message — no need to rephrase. The "
                "raw output is in the daemon log (search: PARSE DISAGREEMENT)."
            )
            messages.append(error_msg)
            yield AssistantTurnComplete(message=error_msg, usage=usage), usage
            return

        # ── Empty-response guard ──
        # A genuinely empty assistant turn — no text, no tool calls, and NOT the
        # malformed-tool-call case handled below (dropped_malformed). Committing
        # it poisons the history: llama.cpp then 400s "Assistant message must
        # contain either 'content' or 'tool_calls'" on every subsequent turn, and
        # the session is stuck until its in-memory state is cleared. Never let it
        # into the message list. Retry the model call ONCE with a nudge; if it's
        # still empty, surface a valid error turn instead. (A tool-only assistant
        # with empty text is valid — guarded by ``not final_message.tool_uses``.)
        if (
            not (final_message.text or "").strip()
            and not final_message.tool_uses
            and not dropped_malformed
        ):
            if not empty_retried:
                empty_retried = True
                pending_empty_nudge = True  # rides the retry REQUEST only (above)
                _log_iteration(
                    context, _IterationReason.EMPTY_RESPONSE, turn, tool_iteration
                )
                continue
            _log_iteration(
                context, _IterationReason.EMPTY_RESPONSE, turn, tool_iteration
            )
            error_msg = _make_assistant_msg(
                "The model returned an empty response twice — unable to complete "
                "this turn. Please rephrase and try again."
            )
            messages.append(error_msg)
            yield AssistantTurnComplete(message=error_msg, usage=usage), usage
            return

        # FIRST-ROUND FORCING: this round produced substance (text or a tool
        # call — the empty guard above already `continue`d otherwise), so the
        # force is spent; subsequent rounds relax to auto.
        if _force_directive is not None and not _force_spent:
            _force_spent = True
            # Fail-loud contract guard: a forced {tool:X} round that came back
            # with a call to a DIFFERENT tool means the provider path did not
            # enforce (e.g. a server that ignores function-forcing, with no
            # grammar available to override it). Never let a call to Y
            # masquerade as the forced X.
            if isinstance(round_tool_choice, dict) and final_message.tool_uses:
                _forced_name = round_tool_choice.get("tool")
                _wrong = sorted({t.name for t in final_message.tool_uses if t.name != _forced_name})
                if _wrong:
                    raise RuntimeError(
                        f"forced tool_choice {{'tool': {_forced_name!r}}} was not honored — "
                        f"the model called {_wrong} instead. The provider path could not "
                        "enforce the directive; refusing to proceed silently."
                    )

        # #65 — TOTAL no-empty-assistant-turn invariant at the commit point.
        # The empty-response guard above retries/surfaces a NON-malformed empty
        # turn, but excludes the malformed case (`and not dropped_malformed`),
        # so a malformed-empty reply — no text, no SURVIVING tool calls,
        # dropped_malformed>0 — fell through here and entered history as a
        # content-less assistant message the provider 400s on (verified live:
        # "Assistant message must contain either 'content' or 'tool_calls'").
        # Commit ONLY a non-empty turn; an empty one (malformed or not) never
        # reaches the wire. The malformed branch below still fires — it gates on
        # `not final_message.tool_uses`, committing its structured feedback and
        # driving recovery. A turn with surviving tool calls or residual prose
        # is NOT empty and commits normally, so the non-empty malformed case is
        # untouched.
        turn_is_empty = (
            not final_message.tool_uses
            and not (final_message.text or "").strip()
        )
        if not turn_is_empty:
            if degrade_notice_this_turn:
                # Local import: this function binds TextBlock locally in two other branches, so
                # a module-level name would be an unbound local here whenever those did not run.
                from prometheus.engine.messages import TextBlock

                # The notice reaches the live stream as a delta from the fallback wrapper; this
                # puts the SAME text into the message that becomes history, so a re-read shows
                # why the answer came from a different model. Without it the degrade is loud
                # once and silent forever after, which is the failure this sprint exists to
                # prevent.
                #
                # Injected HERE and nowhere earlier, deliberately. `raw_model_output_this_turn`
                # is captured above as "what we would want to train a local model to emit", and
                # `extract_tool_calls` parses the same text. Prepending before either would
                # teach a local model to emit our outage banner and hand the tool parser a
                # prefix the model never wrote.
                final_message = final_message.model_copy(
                    update={
                        "content": [
                            TextBlock(text=f"⚠ {degrade_notice_this_turn[0]}\n\n"),
                            *final_message.content,
                        ]
                    }
                )
            messages.append(final_message)
            yield AssistantTurnComplete(message=final_message, usage=usage), usage
            # SUNRISE PeriodicNudge: one completed assistant round. Ask the
            # nudge whether this round is a multiple of its interval and, if
            # so, owe the text to the NEXT model call (below). A run that
            # ends here never spends it — there is nothing left to reflect on.
            nudge_turns += 1
            pending_periodic_nudge = _maybe_periodic_nudge(context, nudge_turns)

        if not final_message.tool_uses:
            # malformed_empty guard: the provider dropped every tool call in
            # this turn as structurally empty and the model produced no prose
            # either. Silently ending the turn here is the old dead-turn
            # outcome (D1 collapse arcs) — instead, feed structured guidance
            # back and retry, bounded by the same circuit breaker that guards
            # tool-error loops.
            if dropped_malformed and not final_message.text.strip():
                trip_reason = circuit_breaker.record_error(
                    "_malformed", "empty tool-call envelope (dropped at provider)"
                )
                if trip_reason is None:
                    _log_iteration(
                        context,
                        _IterationReason.MALFORMED_DROPPED,
                        turn,
                        tool_iteration,
                        f"dropped={dropped_malformed}",
                    )
                    # Repair-pair flywheel: if the model recovers with a
                    # working call after this feedback, pair it against the
                    # empty envelope it just emitted.
                    _stash_pending_pair(
                        context,
                        "_malformed",
                        rejected_name="",
                        rejected_input={},
                        error="empty tool-call envelope (dropped at provider)",
                        source="malformed_recovery",
                    )
                    messages.append(
                        ConversationMessage.from_injected(
                            _malformed_retry_feedback(context, dropped_malformed),
                            provenance="orchestrator",
                            is_trusted=True,
                        )
                    )
                    continue
                _log_iteration(
                    context,
                    _IterationReason.CIRCUIT_BREAKER_TRIP,
                    turn,
                    tool_iteration,
                    trip_reason,
                )
                error_msg = _make_assistant_msg(
                    f"Circuit breaker tripped: {trip_reason}. "
                    f"The model cannot produce valid tool calls for this request."
                )
                messages.append(error_msg)
                yield AssistantTurnComplete(message=error_msg, usage=usage), usage
                return

            # SPRINT-2 WS2: turn ended without further tool calls — drain
            # THIS turn's mutations from the shared verifier and append the
            # summary as an injected turn so the model sees it on its next
            # turn. Same channel as PeriodicNudge. None when no mutations
            # were tracked.
            #
            # NOT from_user_text: that tagged the summary provenance="user",
            # is_trusted=True, i.e. indistinguishable from something the
            # human typed. It reached LCM that way, so the REST history
            # replayed it as role:"user" (a chat bubble nobody wrote) and the
            # MemoryExtractor — which mines user-provenance rows — banked
            # "[FILE MUTATION VERIFIER] ..." as user facts. is_trusted stays
            # True: this is machinery-authored, not third-party data, so it
            # keeps its current banner-free rendering to the model.
            fmv = getattr(context, "file_mutation_verifier", None)
            escapes: list[str] = []
            if fmv is not None:
                # THE TEETH (outcome layer). Read the paths that ACTUALLY
                # changed on disk BEFORE post_turn drains the record, and ask
                # the gate about each one. This is the only check in the
                # system that survives tool substitution: it watches the
                # filesystem, not the toolset, so `bash -c "echo x > path"`
                # is seen exactly as `write_file` would be.
                #
                # It is DETECTION, not containment. `_Snapshot` holds no
                # content, so nothing here can undo a write — which is why the
                # response is to end the turn and say so plainly, and why the
                # message must never read as though the write was stopped.
                try:
                    escapes = _boundary_escapes(context, fmv, _fmv_kw)
                except Exception:
                    log.debug("boundary escape check raised", exc_info=True)
                try:
                    summary = fmv.post_turn(**_fmv_kw)
                except Exception:
                    summary = None
                    log.debug(
                        "FileMutationVerifier.post_turn raised — "
                        "skipping summary append", exc_info=True,
                    )
                if summary:
                    messages.append(ConversationMessage.from_injected(
                        summary,
                        provenance="file_mutation_verifier",
                        is_trusted=True,
                    ))
            if escapes:
                log.error(
                    "BOUNDARY ESCAPE: %d file(s) changed outside the permitted "
                    "area this turn: %s", len(escapes), ", ".join(escapes),
                )
                error_msg = _make_assistant_msg(_boundary_escape_text(escapes))
                messages.append(error_msg)
                yield AssistantTurnComplete(message=error_msg, usage=usage), usage
                return
            return

        tool_calls = final_message.tool_uses
        tool_iteration += len(tool_calls)

        # --- Guard: max_tool_iterations ---
        effective_iter_limit = _effective_max_tool_iterations(context)
        if tool_iteration > effective_iter_limit:
            _log_iteration(context, _IterationReason.MAX_ITERATIONS_HIT, turn, tool_iteration)
            error_msg = _make_assistant_msg(
                f"Tool iteration limit reached ({tool_iteration}/{effective_iter_limit}). "
                f"Stopping to prevent runaway loops."
            )
            messages.append(error_msg)
            yield AssistantTurnComplete(message=error_msg, usage=usage), usage
            return

        # SPRINT-2 WS2: snapshot every path the upcoming tool batch may
        # touch BEFORE dispatch. Best-effort; verifier swallows its own
        # exceptions so loop progress is unaffected.
        fmv = getattr(context, "file_mutation_verifier", None)
        if fmv is not None:
            for _tc in tool_calls:
                try:
                    fmv.pre_tool_use(_tc.name, _tc.input or {}, _tc.id, **_fmv_kw)
                except Exception:
                    log.debug(
                        "FileMutationVerifier.pre_tool_use raised",
                        exc_info=True,
                    )

        # Per-turn repeat guard: refuse any tool call whose exact (name, input)
        # already failed _REPEAT_FAIL_LIMIT times this turn, WITHOUT executing it
        # again. The blocked result carries a directive telling the model to
        # change approach; the circuit breaker below still sees it as an error
        # and trips, but we no longer pay the tool timeout on every retry.
        _blocked: dict[int, ToolResultBlock] = {}
        _runnable: list = []
        for _i, _tc in enumerate(tool_calls):
            if failed_call_signatures.get(_tool_call_signature(_tc), 0) >= _REPEAT_FAIL_LIMIT:
                _n = failed_call_signatures[_tool_call_signature(_tc)]
                _blocked[_i] = ToolResultBlock(
                    tool_use_id=_tc.id,
                    content=(
                        f"BLOCKED: this exact {_tc.name} call already failed "
                        f"{_n} times this turn. Do NOT run it again — change your "
                        f"approach (different command, arguments, or tool) or tell "
                        f"the user you cannot complete this request."
                    ),
                    is_error=True,
                )
                log.warning(
                    "Repeat guard: blocked re-run of %s (failed %d× this turn)",
                    _tc.name, _n,
                )
            else:
                _runnable.append(_tc)

        _ran = (
            await _dispatch_tool_calls(
                context, _runnable,
                raw_model_output=raw_model_output_this_turn,
                served_model=served_model_this_turn,
                ephemeral=ephemeral,
                div_task_id=div_task_id,
                effective_session_id=effective_session_id,
            )
            if _runnable
            else []
        )

        # Reassemble in the original tool_calls order, splicing blocked results
        # back in where they belong.
        _ran_iter = iter(_ran)
        tool_results = [
            _blocked[_i] if _i in _blocked else next(_ran_iter)
            for _i in range(len(tool_calls))
        ]

        # Update the per-turn failure tally: a fresh failure increments its
        # signature; any success clears it (so a flaky-then-fixed call recovers).
        for _tc, _r in zip(tool_calls, tool_results):
            _sig = _tool_call_signature(_tc)
            if _r.is_error:
                failed_call_signatures[_sig] = failed_call_signatures.get(_sig, 0) + 1
            else:
                failed_call_signatures.pop(_sig, None)

        # LONGHAUL-1: feed the progress detector BEFORE _apply_cross_result_budget
        # crops anything -- truncation can make two different results identical
        # and manufacture a repeat that never happened. The verdict is held and
        # acted on further down, after the results are appended to history, so a
        # halt never leaves tool_use blocks without matching tool_result blocks.
        repeat_trip = None
        for _i, (_tc, _r) in enumerate(zip(tool_calls, tool_results)):
            repeat_trip = repeat_detector.record(
                _tc, _r.content, blocked=(_i in _blocked),
            )
            if repeat_trip is not None:
                break

        # SPRINT-2 WS2: post-snapshot + diff after every tool result.
        if fmv is not None:
            for _tc, _r in zip(tool_calls, tool_results):
                try:
                    fmv.post_tool_use(
                        _tc.name, _tc.input or {}, _tc.id,
                        output=_r.content, is_error=_r.is_error,
                        **_fmv_kw,
                    )
                except Exception:
                    log.debug(
                        "FileMutationVerifier.post_tool_use raised",
                        exc_info=True,
                    )

        # --- Circuit breaker ---
        all_errors = all(r.is_error for r in tool_results)
        if all_errors:
            # Build composite key from all tool results in this dispatch
            trip_reasons = []
            for tc, r in zip(tool_calls, tool_results):
                reason = circuit_breaker.record_error(tc.name, r.content)
                if reason:
                    trip_reasons.append(reason)

            if trip_reasons:
                trip_msg = trip_reasons[0]
                _log_iteration(context, _IterationReason.CIRCUIT_BREAKER_TRIP, turn, tool_iteration, trip_msg)

                # Try model fallback for formatting errors before giving up.
                # Phase 2: _try_model_fallback now returns a RouteDecision (or None)
                # whose .provider and .adapter are pre-instantiated.
                if circuit_breaker.is_formatting_error and context.model_router is not None:
                    fallback = _try_model_fallback(context)
                    if fallback is not None:
                        _log_iteration(
                            context,
                            _IterationReason.MODEL_FALLBACK,
                            turn,
                            tool_iteration,
                            f"{context.model} → {fallback.model_name}",
                        )
                        context.provider = fallback.provider
                        context.model = fallback.model_name
                        if fallback.adapter is not None:
                            context.adapter = fallback.adapter
                        circuit_breaker.record_success()
                        # Re-format for the new model's adapter if needed
                        if context.adapter is not None and hasattr(context.adapter, "format_request"):
                            active_system_prompt, active_tools = context.adapter.format_request(
                                context.system_prompt, tool_schema
                            )
                        # Feed error results back so the fallback model sees them
                        messages.append(ConversationMessage(role="user", content=tool_results))
                        continue

                # Circuit Breaker Self-Diagnosis sprint: before reporting
                # failure to the user, run ONE diagnose-and-recover pass.
                # If recovery succeeded (tier bump), continue the loop once
                # more. If not, report the structured diagnostic instead of
                # the old cryptic "Circuit breaker tripped" message.
                if not circuit_breaker.recovery_attempted:
                    # Pull the failing tool name + args for the diagnostic
                    first_failed_tc = tool_calls[0] if tool_calls else None
                    failing_name = first_failed_tc.name if first_failed_tc else "unknown"
                    import json as _json
                    try:
                        intended = _json.dumps(
                            first_failed_tc.input if first_failed_tc else {},
                            default=str,
                        )
                    except Exception:
                        intended = str(first_failed_tc.input if first_failed_tc else "")

                    recovery = circuit_breaker.diagnose_and_recover(
                        context=context,
                        tool_name=failing_name,
                        intended_action=intended,
                    )
                    if recovery.recovered:
                        _log_iteration(
                            context,
                            _IterationReason.CIRCUIT_BREAKER_TRIP,
                            turn,
                            tool_iteration,
                            f"recovered via {recovery.recovery_method}",
                        )
                        # Re-format for the new tier, feed error back, continue.
                        if context.adapter is not None and hasattr(context.adapter, "format_request"):
                            active_system_prompt, active_tools = context.adapter.format_request(
                                context.system_prompt, tool_schema
                            )
                        messages.append(ConversationMessage(role="user", content=tool_results))
                        continue

                    # Recovery not possible — emit the structured diagnostic.
                    error_msg = _make_assistant_msg(recovery.diagnostic_message)
                    messages.append(error_msg)
                    yield AssistantTurnComplete(message=error_msg, usage=usage), usage
                    return

                error_msg = _make_assistant_msg(
                    f"Circuit breaker tripped: {trip_msg}. "
                    f"The model cannot produce valid tool calls for this request."
                )
                messages.append(error_msg)
                yield AssistantTurnComplete(message=error_msg, usage=usage), usage
                return
            else:
                _log_iteration(context, _IterationReason.TOOL_ERROR_RETRY, turn, tool_iteration)
        else:
            circuit_breaker.record_success()
            _log_iteration(context, _IterationReason.TOOL_SUCCESS, turn, tool_iteration)

        # Cross-result token budget: proportional truncation across all results
        if context.tool_results_turn_budget > 0:
            tool_results = _apply_cross_result_budget(context, tool_calls, tool_results)

        for tc, result in zip(tool_calls, tool_results):
            yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input, tool_use_id=tc.id), None
            yield ToolExecutionCompleted(
                tool_name=tc.name,
                output=result.content,
                is_error=result.is_error,
                tool_use_id=tc.id,
            ), None

        messages.append(ConversationMessage(role="user", content=tool_results))

        # LONGHAUL-1: unproductive repetition halts the turn LOUDLY. Deliberately
        # after the append above so the model's tool_use blocks keep their
        # matching tool_result blocks, and after the circuit breaker.
        #
        # ``not all_errors`` is the division of labour, and it is load-bearing:
        # a round where EVERYTHING errored belongs to the breaker, which can
        # still run its one-shot diagnose-and-recover and SALVAGE the turn.
        # Halting ahead of it traded a recoverable turn for a dead one --
        # tests/test_wiring.py::test_trip_handler_calls_diagnose_and_recover
        # caught exactly that. A doomed call is still bounded there (the breaker
        # trips at max_any consecutive errors); this detector exists for the
        # calls that come back CLEAN and still get nowhere, plus mixed rounds
        # the breaker never inspects.
        if repeat_trip is not None and not all_errors:
            _log_iteration(
                context,
                _IterationReason.UNPRODUCTIVE_REPEAT,
                turn,
                tool_iteration,
                f"{repeat_trip.tool_name} x{repeat_trip.count}",
            )
            log.warning(
                "Unproductive repeat: %s called %d\u00d7 with identical arguments, "
                "%s each time \u2014 halting turn.",
                repeat_trip.tool_name,
                repeat_trip.count,
                "empty result" if repeat_trip.empty else "identical result",
            )
            _tel = getattr(context, "telemetry", None)
            if _tel is not None and hasattr(_tel, "record_run"):
                try:
                    _tel.record_run(
                        subsystem="repeat_detector",
                        operation="trip",
                        outcome="failed",
                        summary={
                            "tool_name": repeat_trip.tool_name,
                            "repeat_count": repeat_trip.count,
                            "empty_result": repeat_trip.empty,
                            "tool_iteration": tool_iteration,
                            "turn": turn,
                        },
                        session_id=effective_session_id or context.session_id,
                        model=context.model,
                    )
                except Exception:
                    # Telemetry must never be why a halt fails to happen.
                    log.debug("repeat_detector: record_run failed", exc_info=True)
            error_msg = _make_assistant_msg(_repeat_trip_text(repeat_trip))
            messages.append(error_msg)
            yield AssistantTurnComplete(message=error_msg, usage=usage), usage
            return

        # Sprint 10 / FL-4: checkpoint + divergence evaluation after tool
        # dispatch. Observational only — the score is REPORTED, never acted
        # on. See coordinator/divergence.py's module docstring for why the
        # rollback half was retired rather than finished.
        if context.divergence_detector is not None and div_task_id is not None:
            dd = context.divergence_detector
            try:
                msg_dicts = [
                    {"role": m.role, "content": m.text or ""}
                    for m in messages
                    if hasattr(m, "role")
                ]
                dd.maybe_checkpoint(msg_dicts, task_id=div_task_id)

                # Evaluate divergence (only after 3+ steps to gather signal)
                if dd.steps(div_task_id) > 3:
                    tool_result_dicts = [
                        {"result": tr.content, "success": not tr.is_error}
                        for tr in tool_results
                    ]
                    div_result = dd.evaluate(
                        msg_dicts, tool_result_dicts, task_id=div_task_id,
                    )
                    if div_result.diverged:
                        # The signal's only consumer. WARNING because the
                        # point of the feature is that someone reading the
                        # journal can see the agent going in circles.
                        log.warning(
                            "Divergence: task=%s %s", div_task_id, div_result.reason,
                        )
            except Exception:
                # Fail-open, same posture as start_task/end_task: an
                # observational subsystem must never break a turn.
                log.debug("Divergence evaluation raised", exc_info=True)

    raise RuntimeError(f"Exceeded maximum turn limit ({context.max_turns})")


# ---------------------------------------------------------------------------
# Helpers for run_loop
# ---------------------------------------------------------------------------

def _boundary_escapes(context, fmv, fmv_kw: dict) -> list[str]:
    """Paths that CHANGED this turn and the gate would not have permitted.

    Asks the SecurityGate, which is the single holder of the policy — the FMV
    stays a reporter and learns nothing about denied_paths or workspace roots.

    Only paths the gate would have DENIED or made the user APPROVE count. A
    path inside the permitted area is ordinary work.
    """
    gate = getattr(context, "permission_checker", None)
    if gate is None or not hasattr(fmv, "landed_paths"):
        return []
    escaped: list[str] = []
    for path in fmv.landed_paths(**fmv_kw):
        try:
            decision = gate.evaluate("write_file", file_path=path, origin="user")
        except Exception:
            # UNCLASSIFIABLE, and the direction is a deliberate choice rather
            # than whatever the exception happens to produce (CROSS-CUTTING
            # §8). A path the gate cannot rule on is NOT treated as an escape
            # — ending a turn on a classification error is an over-refusal,
            # and this layer only ever detects. But it must not be silent
            # either: an unseen path is a hole in the layer's coverage, so it
            # is logged at WARNING with the path named. Silent `continue` here
            # was fail-open detection with no trace.
            log.warning(
                "boundary check could not classify %s — this turn's coverage "
                "has a gap and no escape can be ruled out for that path",
                path, exc_info=True,
            )
            continue
        if not decision.allowed:
            escaped.append(path)
    return escaped


def _boundary_escape_text(escapes: list[str]) -> str:
    """The turn's final answer when a write escaped the boundary.

    WORDING IS THE FEATURE. This layer cannot undo anything — the verifier
    captures size/mtime/mode and no content — so the message must state that
    the change LANDED and is not recoverable. A detection layer that reads
    like a prevention layer ("blocked", "prevented", "refused") is the same
    overclaim that made the workspace boundary a liability until PR #177
    relabelled it, one layer further in.
    """
    listed = "\n".join(f"  - {p}" for p in escapes)
    n = len(escapes)
    noun = "file" if n == 1 else "files"
    return (
        f"TURN ENDED — {n} {noun} outside the permitted area {'was' if n == 1 else 'were'} "
        f"CHANGED ON DISK:\n{listed}\n\n"
        f"These writes ALREADY HAPPENED and CANNOT BE UNDONE — this check runs "
        f"after the fact and does not hold the previous contents. Nothing was "
        f"blocked or prevented.\n\n"
        f"The turn is stopped here so the change is not built on. If it was "
        f"intended, say so and it can be redone through a permitted path; if "
        f"it was not, the {noun} above need restoring from version control or "
        f"a backup."
    )


def _make_assistant_msg(text: str) -> ConversationMessage:
    """Build a synthetic assistant message."""
    from prometheus.engine.messages import TextBlock
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


def _maybe_periodic_nudge(context: LoopContext, turn_count: int) -> str | None:
    """Return the PeriodicNudge text owed after ``turn_count`` rounds, or None.

    Request-only by design. Until 2026-08-01 the nudge was injected by
    ``AgentLoop.run_async`` as ``ConversationMessage.from_user_text(...)``
    appended to ``messages`` mid-iteration, which was wrong three ways:

    * **On the wire.** The append landed while ``run_loop`` was suspended at
      the ``AssistantTurnComplete`` yield — i.e. AFTER the assistant's
      ``tool_use`` turn but BEFORE the loop appends the matching
      ``tool_result``. Anthropic requires the tool_result to be the next
      message, so a 15th-round nudge split the pair and 400'd the turn
      (``tool_use ids were found without tool_result blocks immediately
      after``). It also produced two consecutive user turns.
    * **In history.** ``provenance="user", is_trusted=True`` made it
      indistinguishable from something the human typed, and the trailing
      nudge on the last round was persisted to LCM by every gateway.
    * **In the UI.** On the web path that persisted turn comes back from
      ``GET /api/sessions/{id}/messages`` as ``role: "user"`` and renders in
      Beacon as a message Will never wrote.

    The steer channel (``per_call_system_prompt``) has none of those
    problems and is already the established home for exactly this kind of
    machinery-authored, model-facing, never-persisted text — the
    empty-response retry nudge and passive recall both ride it.

    Fail-open: a nudge that raises must never break the turn.
    """
    nudge = getattr(context, "nudge", None)
    if nudge is None:
        return None
    try:
        payload = nudge.maybe_inject(turn_count)
    except Exception:
        log.debug("PeriodicNudge: maybe_inject raised", exc_info=True)
        return None
    if not payload:
        return None
    text = payload.get("content") if isinstance(payload, dict) else None
    if not text:
        return None
    log.debug("PeriodicNudge: armed for the next call at round %d", turn_count)
    return str(text)


def _strip_disagreement_feedback(
    context: LoopContext, samples: list[str]
) -> str:
    """Guidance after a tool-call envelope was stripped to nothing.

    ⚠ THE MODEL'S OWN TEXT GOES BACK TO IT. The failure is that we could not
    parse what it emitted, so the one thing it needs is to see what it
    emitted — quoting it back is the difference between "try again" and "this
    exact string did not parse". Truncated, because a runaway envelope must
    not be echoed whole into the next prompt.

    Distinct from ``_malformed_retry_feedback``: that one means "you sent a
    call with no name", this one means "your envelope did not parse". Same
    accounting, different correction.
    """
    names = ""
    registry = context.tool_registry
    if registry is not None and hasattr(registry, "list_tools"):
        try:
            names = ", ".join(t.name for t in registry.list_tools())
        except Exception:
            names = ""
    parts = [
        "Your previous response contained tool-call markup that could not be "
        "parsed, so it was discarded and nothing ran.",
    ]
    if samples:
        quoted = " | ".join(s.replace("\n", " ")[:200] for s in samples[:2])
        parts.append(f"What you emitted (truncated): {quoted}")
    parts.append(
        "Emit exactly one call as "
        '<tool_call>{"name": "<tool_name>", "arguments": {...}}</tool_call> '
        "with valid JSON, or answer in plain text with no <tool_call> markup."
    )
    if names:
        parts.append(f"Available tools: {names}.")
    return " ".join(parts)


def _malformed_retry_feedback(context: LoopContext, dropped: int) -> str:
    """Structured guidance after the provider dropped malformed-empty calls.

    Mirrors the validator's structured-error shape (what went wrong, the
    expected format, the available names) so the model has something to
    self-correct against — the old path fed it the bare "Unknown tool: ".
    """
    names = ""
    registry = context.tool_registry
    if registry is not None and hasattr(registry, "list_tools"):
        try:
            names = ", ".join(t.name for t in registry.list_tools())
        except Exception:
            names = ""
    parts = [
        f"Your previous response contained {dropped} malformed tool call(s) "
        "with an empty name — they could not be executed.",
        "Either answer in plain text, or emit a valid tool call: "
        '{"name": "<tool_name>", "arguments": {...}}.',
    ]
    if names:
        parts.append(f"Available tools: {names}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Repair-pair flywheel capture helpers (learning/pair_capture)
# ---------------------------------------------------------------------------

_PAIR_PENDING_TTL_S = 600.0


def _pending_pairs(context: LoopContext) -> dict:
    if context.pair_pending is None:
        context.pair_pending = {}
    return context.pair_pending


def _tool_schema_json(context: LoopContext, tool_name: str) -> str | None:
    """The tool's JSON schema as the model saw it, for golden-trace export.

    Full ``parameters`` here, not the property-name digest ``_pair_context``
    keeps: a fine-tuning example needs the schema the model was actually
    conditioned on, and types/enums/descriptions are most of that signal.
    Best-effort — a missing schema costs one export row, never a turn.
    """
    import json as _json

    try:
        registry = context.tool_registry
        tool = registry.get(tool_name) if registry is not None else None
        if tool is None:
            return None
        return _json.dumps(
            {
                "name": tool_name,
                "description": (getattr(tool, "description", "") or "")[:1000],
                "parameters": tool.input_model.model_json_schema(),
            },
            default=str,
        )
    except Exception:
        return None


def _pair_context(context: LoopContext, tool_name: str) -> dict:
    """Compact reproducible context: LCM reference + the tool schema the
    model saw. Sessions without LCM persistence still get the schema."""
    ctx: dict = {
        "kind": "lcm_ref",
        "session_id": context.session_id,
        "ts": time.time(),
        "model": context.model,
    }
    try:
        tool = context.tool_registry.get(tool_name) if context.tool_registry else None
        if tool is not None:
            schema = tool.input_model.model_json_schema()
            ctx["tool_schema"] = {
                "name": tool_name,
                "properties": list(schema.get("properties", {})),
                "required": schema.get("required", []),
            }
    except Exception:
        pass
    return ctx


def _stash_pending_pair(
    context: LoopContext,
    key: str,
    *,
    rejected_name: str,
    rejected_input: object,
    error: str,
    source: str,
) -> None:
    """Remember a failed call so a near-future success can complete the pair."""
    try:
        _pending_pairs(context)[key] = {
            "rejected": {"name": rejected_name, "input": rejected_input},
            "error": error[:500],
            "ts": time.time(),
            "source": source,
        }
    except Exception:
        log.debug("pair stash failed", exc_info=True)


def _capture_success_pairs(
    context: LoopContext,
    tool_name: str,
    original_tool_name: str,
    tool_input: dict,
    provider_name: str,
) -> None:
    """On a successful execution: complete any pending pairs + cloud goldens.

    Fail-loud-but-non-blocking (pair_capture handles the loudness)."""
    try:
        from prometheus.learning.pair_capture import (
            capture_pair,
            cloud_golden_enabled,
            get_store,
        )
        if get_store() is None:
            return
        chosen = {"name": tool_name, "input": tool_input}
        pending = _pending_pairs(context)
        now = time.time()
        matched: list[dict] = []
        for key in {tool_name, original_tool_name, "_malformed"}:
            entry = pending.pop(key, None)
            if entry and now - entry["ts"] <= _PAIR_PENDING_TTL_S:
                matched.append(entry)
        # An unknown-NAME failure (model invented a tool, repair refused,
        # retry prompt listed the real names) recovers under a DIFFERENT,
        # correct name — exact-key matching can't see that. Pair it only in
        # the unambiguous case: exactly one pending left and its key is not
        # a registered tool.
        if not matched and len(pending) == 1:
            key = next(iter(pending))
            registry = context.tool_registry
            if registry is not None and registry.get(key) is None:
                entry = pending.pop(key)
                if now - entry["ts"] <= _PAIR_PENDING_TTL_S:
                    matched.append(entry)
        for entry in matched:
            capture_pair(
                pair_source=entry["source"],
                model_id=context.model,
                tool_name=tool_name,
                context=_pair_context(context, tool_name),
                rejected=entry["rejected"],
                chosen=chosen,
                meta={"error_feedback": entry["error"]},
                telemetry=context.telemetry,
            )
        if cloud_golden_enabled():
            from prometheus.telemetry.tracker import _CLOUD_PROVIDERS

            if provider_name in _CLOUD_PROVIDERS:
                capture_pair(
                    pair_source="cloud_golden",
                    model_id=context.model,
                    tool_name=tool_name,
                    context=_pair_context(context, tool_name),
                    rejected=None,
                    chosen=chosen,
                    telemetry=context.telemetry,
                )
    except Exception:
        log.error("pair capture (success path) failed", exc_info=True)


def _log_iteration(
    context: LoopContext,
    reason: str,
    turn: int,
    tool_iteration: int,
    detail: str = "",
) -> None:
    """Log why the agent loop continued (or stopped) on this iteration."""
    log.debug("loop turn=%d iter=%d reason=%s %s", turn, tool_iteration, reason, detail)
    if context.telemetry is not None:
        context.telemetry.record(
            model=context.model,
            tool_name="_loop_transition",
            success=(reason == _IterationReason.TOOL_SUCCESS),
            error_type=reason if reason != _IterationReason.TOOL_SUCCESS else None,
            error_detail=detail or None,
        )


async def _try_escalate_tool_call(
    context: LoopContext,
    tool_name: str,
    tool_input: dict,
    tool_use_id: str,
    last_error: str,
) -> ToolResultBlock | None:
    """Phase 3: escalate a repeatedly-failing tool call to a stronger provider.

    Spawns a SubagentSpawner with the router's configured escalation provider
    and asks it to execute the failing tool. The subagent runs in isolation
    (fresh context, curated tool subset = just the failing tool). The main
    agent loop keeps running on the primary provider; only the single tool
    call is delegated.

    Returns:
        A ToolResultBlock with the subagent's result (success or error), or
        None if the router has no escalation configured OR if spawning the
        subagent raises. In the None case, the caller falls through to the
        normal ABORT error path — escalation is best-effort, never fatal.
    """
    import json

    if context.model_router is None or not hasattr(context.model_router, "get_escalation_decision"):
        return None

    try:
        decision = context.model_router.get_escalation_decision()
    except Exception:
        log.warning("Escalation lookup failed", exc_info=True)
        return None

    if decision is None:
        return None

    try:
        # Lazy import to avoid circular dependency
        # (coordinator.subagent → engine.agent_loop)
        from prometheus.coordinator.subagent import SubagentSpawner

        spawner = SubagentSpawner(
            provider=decision.provider,
            parent_tool_registry=context.tool_registry,
            model=decision.model_name or "unknown",
            max_tokens=context.max_tokens,
            cwd=context.cwd,
            adapter=decision.adapter,
            telemetry=context.telemetry,
        )

        try:
            tool_input_json = json.dumps(tool_input, default=str)
        except Exception:
            tool_input_json = str(tool_input)

        task_prompt = (
            f"The primary model failed validation for the tool `{tool_name}` "
            f"after repeated attempts. Original arguments: {tool_input_json}\n\n"
            f"Last error: {last_error}\n\n"
            f"Please invoke `{tool_name}` with corrected arguments and return "
            f"its output."
        )

        log.warning(
            "Escalating tool call %s to %s/%s (retries exhausted)",
            tool_name,
            decision.provider_name,
            decision.model_name,
        )

        result = await spawner.spawn(
            task=task_prompt,
            agent_type="general-purpose",
            tools_subset=[tool_name],
        )

        if result.success and result.text:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=result.text,
                is_error=False,
            )

        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=(
                f"Escalation attempt failed ({decision.provider_name}/"
                f"{decision.model_name}): {result.error or 'no result text'}"
            ),
            is_error=True,
        )
    except Exception as exc:
        log.warning("Escalation raised, falling through: %s", exc, exc_info=True)
        return None


def _try_model_fallback(context: LoopContext):
    """Attempt to switch to a fallback provider for tool-call formatting errors.

    Phase 2: the canonical ModelRouter.get_fallback() returns a RouteDecision
    with pre-instantiated provider + adapter (no ProviderRegistry.create call
    needed here). Returns the RouteDecision or None if no fallback is available.
    """
    if context.model_router is None or not hasattr(context.model_router, "get_fallback"):
        return None

    # Determine current provider name from config or model_router defaults
    current_provider = getattr(context.provider, "provider_name", None) or "llama_cpp"
    try:
        decision = context.model_router.get_fallback(current_provider)
    except Exception:
        log.warning("Failed to get fallback from router", exc_info=True)
        return None

    if decision is None:
        return None

    log.warning(
        "Model fallback: %s → %s/%s (tool formatting errors)",
        current_provider,
        decision.provider_name,
        decision.model_name,
    )
    return decision


def _apply_cross_result_budget(
    context: LoopContext,
    tool_calls: list,
    tool_results: list[ToolResultBlock],
) -> list[ToolResultBlock]:
    """Enforce a total token budget across all tool results in a single turn.

    Runs AFTER individual per-result truncation but BEFORE injection into
    conversation history. Prioritizes mutating tool results over read-only.
    """
    from prometheus.context.token_estimation import estimate_tokens

    budget = context.tool_results_turn_budget
    if budget <= 0:
        return tool_results

    # Calculate total tokens
    result_tokens = [(r, estimate_tokens(r.content)) for r in tool_results]
    total = sum(t for _, t in result_tokens)
    if total <= budget:
        return tool_results

    # Classify read-only vs mutating for priority
    ro_indices: list[int] = []
    mut_indices: list[int] = []
    for i, tc in enumerate(tool_calls):
        tool = context.tool_registry.get(tc.name) if context.tool_registry else None
        if tool is not None and _is_tool_read_only(tool, tc.input):
            ro_indices.append(i)
        else:
            mut_indices.append(i)

    # Truncate read-only results first, then mutating if still over budget
    new_results = list(tool_results)
    remaining = total

    for idx_group in (ro_indices, mut_indices):
        if remaining <= budget:
            break
        for i in idx_group:
            if remaining <= budget:
                break
            r, tokens = result_tokens[i]
            if r.is_error or tokens == 0:
                continue
            # Proportionally reduce this result
            share = max(100, int(budget * tokens / total))
            char_limit = share * 4  # estimate_tokens uses chars/4
            if len(r.content) > char_limit:
                # NOTICE CONTRACT (selector survey 2026-08-11): the old text
                # prescribed "lcm_expand or re-read". lcm_expand expands LCM
                # summary nodes and cannot recover a tool result truncated
                # before injection (it was never stored), and re-reading
                # returns the same head — advice that cannot be followed is
                # worse than none. Say what happened and what actually works.
                trimmed = r.content[:char_limit] + \
                    "\n[truncated to fit the per-turn tool-result budget — " \
                    "the rest was not retained; re-run the tool with " \
                    "narrower arguments if needed]"
                new_results[i] = ToolResultBlock(
                    tool_use_id=r.tool_use_id,
                    content=trimmed,
                    is_error=r.is_error,
                )
                remaining -= (tokens - estimate_tokens(trimmed))

    log.debug("Cross-result budget: %d → %d tokens (budget %d)", total, remaining, budget)
    return new_results


def _classify_tool_error(
    *, is_error: bool, metadata: dict | None
) -> str | None:
    """Separate "the tool ran and reported failure" from "the call failed".

    ``pytest`` exiting 1 on failing tests is the tool working correctly — the
    command executed, produced output, and reported a real result the model can
    act on. That was previously indistinguishable from a call that never ran
    (bad arguments, missing tool, an exception), and both landed as
    ``tool_error``. On the EMBERFALL baseline this dragged bash to 82% "success"
    when zero of the failures were model- or call-level faults, which both
    understates reliability and hides genuine breakage in the same bucket.

    A tool that ran to completion exposes its exit status in
    ``ToolResult.metadata['returncode']``; a call that never got there has no
    returncode at all. That presence check is the discriminator.

    Returns None on success, ``"nonzero_exit"`` when the tool ran and exited
    non-zero, else ``"tool_error"``.
    """
    if not is_error:
        return None
    if isinstance(metadata, dict):
        rc = metadata.get("returncode")
        # rc == 0 with is_error set means the tool asserted failure on its own
        # terms rather than via exit status — that is a real tool error.
        if isinstance(rc, int) and not isinstance(rc, bool) and rc != 0:
            return "nonzero_exit"
    return "tool_error"


def _microcompact_old_results(
    context: LoopContext,
    messages: list[ConversationMessage],
    current_turn: int,
) -> None:
    """Compact old tool result messages in-place to save context tokens.

    Runs BEFORE LCM compaction and compression — it's free (no LLM calls).
    Only touches ToolResultBlock content in messages older than N turns.
    """
    if current_turn < context.microcompact_after_turns:
        return

    # CACHE-AWARE GATE (fix/history-append-only). Rewriting history mid-run
    # invalidates the provider's cached prompt prefix from that point on, and
    # every cached token then gets re-billed at full rate. Measured on the
    # EMBERFALL baseline: this fired at turn 4, saved ~3.6k tokens, and put
    # ~92%-cacheable context (492k of 535k input tokens) back at full price —
    # a large net loss under cached-input pricing.
    #
    # Microcompaction exists to protect SMALL LOCAL CONTEXT WINDOWS, where a
    # few thousand tokens genuinely decide whether a run survives. Cloud tiers
    # have windows orders of magnitude larger, so the saving is noise and the
    # cache cost is not. Default: skip on cloud, keep for local.
    #
    # Unknown/missing adapter => compact (never silently disable a context
    # safeguard just because provenance is unclear).
    adapter = getattr(context, "adapter", None)
    is_cloud = adapter is not None and getattr(adapter, "tier", None) == "off"
    if is_cloud and not getattr(context, "microcompact_on_cloud", False):
        return

    from prometheus.engine.messages import ToolResultBlock as TRB

    # Count user messages from the end to identify the "fresh" window
    user_msg_count = 0
    fresh_boundary = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if hasattr(msg, "role") and msg.role == "user":
            user_msg_count += 1
            if user_msg_count >= context.microcompact_after_turns:
                fresh_boundary = i
                break

    compacted = 0
    chars_before = 0
    chars_after = 0
    for i in range(fresh_boundary):
        msg = messages[i]
        if not hasattr(msg, "content") or not isinstance(msg.content, list):
            continue
        for j, block in enumerate(msg.content):
            if not isinstance(block, TRB):
                continue
            if block.is_error:
                continue
            content = block.content
            if "[content pruned" in content or "[microcompacted]" in content:
                continue  # Already compacted by compression.py or us
            if len(content) <= context.microcompact_keep_chars:
                continue

            # Check LCM ingestion for keep_chars decision. PR
            # fix/memory-lcm-full-rewire removed the hasattr guard now
            # that LCMEngine.is_ingested exists. Note: until the
            # follow-up that maps tool_use_id ↔ message_id, is_ingested
            # returns False for every tool_use_id, so this branch
            # effectively always takes the no-LCM-coverage path for
            # tool results — same behaviour as before the guard removal,
            # just no longer hidden behind defensive ``hasattr``.
            keep_chars = context.microcompact_keep_chars
            if context.lcm_engine is None or not context.lcm_engine.is_ingested(
                getattr(block, "tool_use_id", "")
            ):
                keep_chars = context.microcompact_keep_chars_no_lcm

            # Extract tool name from the block or content
            first_line = content.split("\n", 1)[0][:80]
            summary = content[:keep_chars]
            replacement = f"[microcompacted] {first_line}...\n{summary}"
            msg.content[j] = TRB(
                tool_use_id=block.tool_use_id,
                content=replacement,
                is_error=False,
            )
            compacted += 1
            chars_before += len(content)
            chars_after += len(replacement)

    if compacted:
        dropped = chars_before - chars_after
        log.info(
            "Microcompacted %d old tool results at turn %d (-%d chars) — "
            "the prompt prefix changed, so any provider cache is invalidated "
            "from this round on",
            compacted, current_turn, dropped,
        )
        # Record it. This used to be log.debug ONLY, which is why a 3,595-token
        # mid-run history shrink was invisible until someone diffed per-round
        # token counts by hand. A rewrite of history must always leave a trace.
        if context.telemetry is not None:
            try:
                context.telemetry.record_run(
                    subsystem="agent_loop",
                    operation="microcompact",
                    outcome="success",
                    session_id=getattr(context, "session_id", None),
                    model=getattr(context, "model", None),
                    round_index=current_turn,
                    summary={
                        "results_compacted": compacted,
                        "chars_dropped": dropped,
                        "chars_before": chars_before,
                        "chars_after": chars_after,
                        "cache_prefix_invalidated": True,
                    },
                )
            except Exception:
                log.warning("microcompact telemetry write failed", exc_info=True)


async def _safe_execute(
    context: LoopContext,
    tc: object,
    raw_model_output: str | None,
    served_model: str | None = None,
    *,
    ephemeral: bool = False,
    div_task_id: str | None = None,
    effective_session_id: str | None = None,
) -> ToolResultBlock:
    """Run one tool call, always returning a correctly-correlated
    ``ToolResultBlock`` and never raising.

    Failure isolation (audit H4). Without this, an exception escaping
    ``_execute_tool_call`` (a bug in tool code, a hook, or the permission gate)
    either killed the whole turn on the sequential path, or — inside the
    parallel ``gather`` — lost its index and id (the old ``-1`` /
    ``tool_use_id="error"`` block sorted to the front), so every downstream
    ``zip(tool_calls, tool_results)`` re-paired results with the WRONG calls and
    the model reasoned over misattributed tool output.
    """
    try:
        return await _execute_tool_call(
            context, tc.name, tc.id, tc.input,
            raw_model_output=raw_model_output,
            served_model=served_model,
            ephemeral=ephemeral,
            div_task_id=div_task_id,
            effective_session_id=effective_session_id,
        )
    except Exception as exc:  # noqa: BLE001 — isolating tool failure is the point
        log.error(
            "Tool %s raised during execution: %s", tc.name, exc, exc_info=True,
        )
        if context.telemetry is not None:
            try:
                context.telemetry.record(
                    model=context.model,
                    tool_name=tc.name,
                    success=False,
                    error_type="tool_exception",
                    # The row stays (it is a denominator); the exception text
                    # goes, because a tool exception routinely quotes the input
                    # that produced it.
                    error_detail=None if ephemeral else str(exc)[:2000],
                    served_model=served_model,
                )
            except Exception:  # pragma: no cover - telemetry must not mask result
                log.debug("telemetry.record failed in _safe_execute", exc_info=True)
        return ToolResultBlock(
            tool_use_id=tc.id,
            content=f"Tool {tc.name} raised an exception: {exc}",
            is_error=True,
        )


def _tool_call_signature(tool_call: object) -> str:
    """Stable identity for a tool call's (name, input).

    Used by the per-turn repeat guard so an EXACT re-issue of a call that
    already failed this turn (e.g. the same unbounded ``find`` that keeps
    hitting the tool timeout) can be short-circuited instead of paying the full
    timeout again. Canonical JSON so key order never matters; falls back to
    ``repr()`` for anything not JSON-serialisable.
    """
    import json as _json
    name = getattr(tool_call, "name", "?")
    raw = getattr(tool_call, "input", None)
    try:
        body = _json.dumps(raw, sort_keys=True, default=str)
    except Exception:
        body = repr(raw)
    return f"{name}:{body}"


# LONGHAUL-1 tuning. The window matches coordinator/divergence.py's
# SCORING_WINDOW (10) on purpose: two detectors disagreeing about how far back
# "recent" reaches would be needless surface area. _REPEAT_TRIP is 3 because
# that is where the same signature stops looking like a retry and starts
# looking like a loop -- calibrated against 66 real tasks in the journal, where
# divergence's equivalent predicate fired on 3 of them (4.5%).
_REPEAT_WINDOW = 10
_REPEAT_TRIP = 3


def _result_fingerprint(content: object) -> str:
    """Identity of a tool RESULT for progress purposes.

    Returns ``""`` for an empty/whitespace-only result -- that is the "returned
    nothing" half of unproductive, and it is deliberately its own value rather
    than a hash so an empty result is unproductive from the FIRST occurrence
    (there is no prior to compare against).

    Everything else hashes the FULL stripped payload. Not a bounded prefix:
    two long results that differ only in their tail are PROGRESS, and a prefix
    hash would call them identical and halt a working run.
    """
    text = ("" if content is None else str(content)).strip()
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


@dataclass
class _RepeatTrip:
    """One unproductive-repetition verdict, carried to the halt site."""
    tool_name: str
    tool_input: object
    count: int
    empty: bool


@dataclass
class _ProgressRepeatDetector:
    """Halt a turn that re-issues the SAME call and gets the SAME thing back.

    The gap this closes: ``failed_call_signatures`` counts only FAILURES and
    clears on success, so a call that succeeds three times returning identical
    bytes is invisible to it. ``max_tool_iterations`` sees that call, but only
    as round count -- it cannot tell it from a long productive run, so both
    halt at the same number and long legitimate work pays for the loop.

    PROGRESS IS THE RESET. A repeat that returns new data is not a loop; it
    clears the signature's history outright rather than merely failing to
    increment, so ``read(A), read(A), read(B)`` leaves no loaded gun behind.
    That is not a refinement -- it is what keeps the smoke test alive, whose
    reads bracket an ``edit_file`` and legitimately return different bytes.

    KNOWN BLIND SPOT, recorded rather than fixed: keyed on the exact
    (name, input) signature, so a flail with DIFFERENT arguments every round is
    invisible here. coordinator/divergence.py:704 makes the same observation
    from the other side. ``max_tool_iterations`` remains the backstop for that
    shape and must NOT be read as redundant.
    """

    window: int = _REPEAT_WINDOW
    trip_at: int = _REPEAT_TRIP
    _recent: list[str] = field(default_factory=list)
    _last_fp: dict[str, str] = field(default_factory=dict)

    def record(
        self,
        tool_call: object,
        content: object,
        *,
        blocked: bool = False,
    ) -> "_RepeatTrip | None":
        """Feed one executed call+result. Returns a trip verdict or None.

        Must be fed the UNTRUNCATED result: ``_apply_cross_result_budget``
        can crop two different results down to the same bytes, and a
        fingerprint taken after that would manufacture a false repeat.

        ``blocked`` marks a result the per-turn repeat guard synthesised
        WITHOUT executing the tool. Such a result is unproductive by
        construction -- no tool ran, so no new information can exist -- and
        must be judged on that fact rather than on its text. Its text carries
        the running failure count, which CHANGES every round; fingerprinting it
        would read as fresh data and reset the very counter that should be
        climbing. CI caught this: a call failing every round rode the flat cap
        to exhaustion instead of tripping at 3.
        """
        signature = _tool_call_signature(tool_call)
        fp = _result_fingerprint(content)
        prior = self._last_fp.get(signature)
        self._last_fp[signature] = fp

        # Unproductive = returned nothing, or returned exactly what it
        # returned last time. Same predicate as divergence.py:718 --
        # duplicated deliberately, see the PR body.
        unproductive = blocked or (fp == "") or (prior is not None and fp == prior)

        if not unproductive:
            # Progress. Forget this signature's history entirely.
            self._recent = [s for s in self._recent if s != signature]
            self._recent.append(signature)
            self._trim()
            return None

        self._recent.append(signature)
        self._trim()
        count = sum(1 for s in self._recent if s == signature)
        if count >= self.trip_at:
            return _RepeatTrip(
                tool_name=getattr(tool_call, "name", "?"),
                tool_input=getattr(tool_call, "input", None),
                count=count,
                empty=(fp == ""),
            )
        return None

    def _trim(self) -> None:
        if len(self._recent) > self.window:
            self._recent = self._recent[-self.window:]


def _repeat_trip_text(trip: "_RepeatTrip") -> str:
    """Operator-facing halt message. Names the tool, the args and the count."""
    import json as _json
    try:
        args = _json.dumps(trip.tool_input, sort_keys=True, default=str)
    except Exception:
        args = repr(trip.tool_input)
    if len(args) > 500:
        args = args[:500] + "... (truncated)"
    got = "returned nothing" if trip.empty else "returned an identical result"
    return (
        f"Halted: no progress. `{trip.tool_name}` was called {trip.count} times "
        f"with identical arguments and {got} every time.\n\n"
        f"Arguments: {args}\n\n"
        f"Repeating this call cannot produce new information. Change approach "
        f"\u2014 different arguments, a different tool, or tell the user what is "
        f"blocking you."
    )


async def _dispatch_tool_calls(
    context: LoopContext,
    tool_calls: list,
    raw_model_output: str | None = None,
    served_model: str | None = None,
    *,
    ephemeral: bool = False,
    div_task_id: str | None = None,
    effective_session_id: str | None = None,
) -> list[ToolResultBlock]:
    """Dispatch tool calls with parallel execution for read-only tools.

    Read-only tools are executed simultaneously via ``asyncio.gather``.
    Mutating tools are executed sequentially afterwards to preserve order.
    Single tool calls skip partitioning entirely.

    Every call goes through ``_safe_execute``, which guarantees a
    correctly-correlated ToolResultBlock and never raises — so a tool blowing up
    can never scramble result↔call correlation or kill the turn (audit H4).

    Golden Trace Capture sprint: ``raw_model_output`` is the text the
    model produced for this turn (before adapter parsing). Forwarded to
    each ``_execute_tool_call`` so successful cloud-provider calls get
    captured as golden traces in telemetry.

    ``ephemeral`` rides every ``_safe_execute`` for the same reason
    ``raw_model_output`` does: the decision belongs to the turn, and the only
    place that can act on it is the per-call telemetry write at the bottom of
    ``_execute_tool_call``.

    ``div_task_id`` rides along for the same reason again — the divergence
    detector is process-wide, so its step counter has to be told which task
    a call belongs to. Note the read-only branch below runs concurrently
    under ``gather``: that is precisely why the counter cannot be an
    attribute on the shared detector.
    """
    if len(tool_calls) == 1:
        return [
            await _safe_execute(
                context, tool_calls[0], raw_model_output, served_model,
                ephemeral=ephemeral, div_task_id=div_task_id,
                effective_session_id=effective_session_id,
            )
        ]

    # Partition into read-only and mutating based on tool.is_read_only()
    read_only: list[tuple[int, object]] = []   # (original_index, tool_call)
    mutating: list[tuple[int, object]] = []

    for i, tc in enumerate(tool_calls):
        tool = context.tool_registry.get(tc.name) if context.tool_registry else None
        if tool is not None and _is_tool_read_only(tool, tc.input):
            read_only.append((i, tc))
        else:
            mutating.append((i, tc))

    results: list[tuple[int, ToolResultBlock]] = []

    # Run all read-only tools in parallel. ``_safe_execute`` never raises, so
    # every gathered item is an (index, ToolResultBlock) tuple — no
    # return_exceptions backstop is needed and the index is never lost. A
    # CancelledError during loop teardown still propagates, which is correct.
    if read_only:
        async def _run_ro(idx, tc):
            return idx, await _safe_execute(
                context, tc, raw_model_output, served_model,
                ephemeral=ephemeral, div_task_id=div_task_id,
                effective_session_id=effective_session_id,
            )

        results.extend(
            await asyncio.gather(*[_run_ro(idx, tc) for idx, tc in read_only])
        )

    # Run mutating tools sequentially (order matters)
    for idx, tc in mutating:
        results.append((
            idx,
            await _safe_execute(
                context, tc, raw_model_output, served_model,
                ephemeral=ephemeral, div_task_id=div_task_id,
                effective_session_id=effective_session_id,
            ),
        ))

    # Restore original order
    results.sort(key=lambda x: x[0])
    return [r for _, r in results]


def _is_tool_read_only(tool: object, tool_input: dict) -> bool:
    """Check if a tool call is read-only, handling both method and attribute patterns."""
    if callable(getattr(tool, "is_read_only", None)):
        try:
            parsed = tool.input_model.model_validate(tool_input)
            return tool.is_read_only(parsed)
        except Exception as exc:
            # SPRINT-4 audit HIGH-RISK #11 fix: surface validator / handler
            # failures at debug level. ``return False`` keeps the fail-safe
            # direction (unknown → treat as write, requires permission), so
            # the silent swallow doesn't change the security stance — but
            # without the log line we'd never know a tool's input_model is
            # broken until the user files a "why does this tool keep
            # prompting" bug.
            log.debug(
                "_is_tool_read_only: %s.is_read_only check failed (%s: %s); "
                "defaulting to False (treat as write)",
                type(tool).__name__, type(exc).__name__, exc,
                exc_info=True,
            )
            return False
    return getattr(tool, "is_read_only", False)


async def _maybe_suggest_printing_press(
    press: object, bash_output: str
) -> str | None:
    """If a bash failure looks like ``command not found: <cli>``, ask
    the Printing Press registry whether it has a matching CLI and
    return a one-line suggestion the model can relay to the user.

    Returns ``None`` for any of:
      • registry is unavailable (no library clone)
      • no ``command not found`` pattern in the output
      • no matching CLI found
      • the matched CLI is already installed (so the failure is something else)

    The hook never installs anything — it just surfaces the option.
    Installation requires explicit user action (``/press install <name>``)
    which routes through ApprovalQueue.
    """
    try:
        from prometheus.tools.printing_press import detect_command_not_found
    except Exception:
        return None
    if not hasattr(press, "is_available") or not press.is_available():
        return None
    missing = detect_command_not_found(bash_output)
    if not missing:
        return None
    # Strip common suffixes models tend to type (-pp-cli, -cli) for the search
    candidates = [missing]
    for suffix in ("-pp-cli", "-cli", "-pp"):
        if missing.endswith(suffix):
            candidates.append(missing[: -len(suffix)])
    seen: set[str] = set()
    matches = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            matches = press.search(cand, limit=3)
        except Exception:
            continue
        if matches:
            break
    if not matches:
        return None
    best = matches[0]
    if getattr(best, "installed", False):
        return None
    return (
        f"💡 Printing Press has a CLI for this: **{best.name}** "
        f"({best.category or 'cli'}). To install, the user can run "
        f"`/press install {best.name}` — installation requires their "
        f"explicit approval and will not happen automatically."
    )


async def _execute_tool_call(
    context: LoopContext,
    tool_name: str,
    tool_use_id: str,
    tool_input: dict[str, object],
    *,
    raw_model_output: str | None = None,
    served_model: str | None = None,
    ephemeral: bool = False,
    div_task_id: str | None = None,
    effective_session_id: str | None = None,
) -> ToolResultBlock:
    """Execute a single tool call, running hooks if configured.

    Golden Trace Capture sprint: ``raw_model_output`` is the text the
    model produced BEFORE adapter parsing (enforcer/formatter) for this
    turn. Passed through to ``telemetry.record()`` on the success path
    so cloud-provider wins with zero adapter retries get flagged as
    ``is_golden=1`` for later fine-tuning use.

    ``ephemeral`` (per-turn, resolved in :func:`run_loop`) is the "Prometheus
    won't remember this" flag. It does NOT suppress the telemetry row — the
    row is the denominator of every tool success rate, and dropping it would
    silently bias those rates with no marker to correct for. It nulls the
    three columns that carry content instead:

    * ``raw_model_output`` — the model's complete turn text
    * ``parsed_tool_call`` — the entire tool input, verbatim
    * ``error_detail`` — up to 2 000 chars of the tool's own output

    Nulling ``raw_model_output`` also makes ``is_golden`` False by
    construction (``ToolCallTelemetry.record`` requires it to be non-None), so
    an ephemeral call cannot be picked up by the nightly golden-trace export
    into ``~/.prometheus/trajectories/`` either. That is a consequence worth
    stating out loud rather than a coincidence to rely on silently.

    It also skips repair-pair capture (``training.db``), whose ``chosen`` /
    ``rejected`` payloads are full tool-call JSON.
    """
    # Pre-tool hook (Sprint 2)
    if context.hook_executor is not None:
        from prometheus.hooks import HookEvent
        pre = await context.hook_executor.execute(
            HookEvent.PRE_TOOL_USE,
            {"tool_name": tool_name, "tool_input": tool_input, "event": HookEvent.PRE_TOOL_USE.value},
        )
        if pre.blocked:
            if context.telemetry is not None:
                context.telemetry.record(
                    model=context.model,
                    tool_name=tool_name,
                    success=False,
                    error_type="hook_blocked",
                    error_detail=pre.reason or f"pre_tool_use hook blocked {tool_name}",
                    served_model=served_model,
                )
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=pre.reason or f"pre_tool_use hook blocked {tool_name}",
                is_error=True,
            )

    if context.tool_registry is None:
        if context.telemetry is not None:
            context.telemetry.record(
                model=context.model,
                tool_name=tool_name,
                success=False,
                error_type="no_registry",
                error_detail="No tool registry configured",
                served_model=served_model,
            )
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"No tool registry configured — cannot execute {tool_name}",
            is_error=True,
        )

    # Sprint 3: validate + auto-repair the tool call before execution
    retries_used = 0
    repair_log: list[str] = []
    _original_tool_name = tool_name
    # Repair-pair flywheel: the repair return overwrites tool_input, so the
    # as-emitted call must be copied BEFORE validate_and_repair (D4 finding).
    _original_tool_input = dict(tool_input) if isinstance(tool_input, dict) else tool_input
    _adapter_tier = getattr(context.adapter, "tier", None) if context.adapter else None
    if context.adapter is not None and _adapter_tier != "off":
        try:
            tool_name, tool_input, repair_log = context.adapter.validate_and_repair(
                tool_name, tool_input, context.tool_registry
            )
            # M2: a name-changing repair (fuzzy match) silently executes a
            # DIFFERENT tool than the model named — for a mutating tool that's a
            # real surprise. Surface it; the repair count reaches telemetry on
            # the success record() below.
            if repair_log and tool_name != _original_tool_name:
                log.warning(
                    "Adapter repaired tool name %r → %r before execution: %s",
                    _original_tool_name, tool_name, "; ".join(repair_log),
                )
            if repair_log and not ephemeral:
                # An adapter repair IS a labeled pair: as-emitted vs repaired.
                # Both halves carry the verbatim tool input, so an ephemeral
                # turn contributes no training pair at all.
                try:
                    from prometheus.learning.pair_capture import capture_pair, get_store
                    if get_store() is not None:
                        capture_pair(
                            pair_source=(
                                "levenshtein_repair"
                                if tool_name != _original_tool_name
                                else "schema_repair"
                            ),
                            model_id=context.model,
                            tool_name=tool_name,
                            context=_pair_context(context, tool_name),
                            rejected={
                                "name": _original_tool_name,
                                "input": _original_tool_input,
                            },
                            chosen={"name": tool_name, "input": tool_input},
                            meta={"repair_log": repair_log},
                            telemetry=context.telemetry,
                        )
                except Exception:
                    log.error("pair capture (repair path) failed", exc_info=True)
        except ValueError as exc:
            # Validation failed and repair failed — ask retry engine
            action, retry_prompt = context.adapter.handle_retry(
                tool_name, str(exc), context.tool_registry
            )
            retries_used = 1
            # Repair-pair flywheel: remember the failed call; if the model's
            # retry of this tool succeeds within the TTL, that's a pair.
            _stash_pending_pair(
                context,
                tool_name,
                rejected_name=_original_tool_name,
                rejected_input=_original_tool_input,
                error=str(exc),
                source="retry_success",
            )
            if context.telemetry is not None:
                import json as _json
                try:
                    _failed_call = _json.dumps(
                        {"name": _original_tool_name, "input": _original_tool_input},
                        default=str,
                    )
                except Exception:
                    _failed_call = None
                context.telemetry.record(
                    model=context.model,
                    tool_name=tool_name,
                    success=False,
                    retries=retries_used,
                    latency_ms=0.0,
                    error_type="validation_failed",
                    error_detail=str(exc),
                    # forensics + future mining: the as-emitted call (D1 had
                    # to dig the LCM because failure rows lacked this)
                    parsed_tool_call=_failed_call,
                    served_model=served_model,
                )

            # Phase 3: ESCALATE — retries exhausted + router has escalation
            # configured. Spawn a subagent with the escalation provider to
            # attempt the failing tool call. Main agent keeps running on
            # the primary provider.
            from prometheus.adapter.retry import RetryAction
            if action == RetryAction.ESCALATE:
                escalated = await _try_escalate_tool_call(
                    context, tool_name, tool_input, tool_use_id, str(exc)
                )
                if escalated is not None:
                    return escalated

            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=retry_prompt,
                is_error=True,
            )

    tool = context.tool_registry.get(tool_name)
    if tool is None:
        if context.telemetry is not None:
            context.telemetry.record(
                model=context.model,
                tool_name=tool_name,
                success=False,
                error_type="unknown_tool",
                error_detail=f"Unknown tool: {tool_name}",
                served_model=served_model,
            )
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Unknown tool: {tool_name}",
            is_error=True,
        )

    # Lucky guess: tool is registered but wasn't in the active prompt schema.
    # Recomputes the run-start resolution rather than reading stored state:
    # resolve_deferred(config, adapter) is deterministic within a run, and the
    # shared LoopContext must not carry per-run mutable state (concurrent
    # turns cross-talk — see the per-message `mode` precedent). A config flip
    # mid-run can at worst mislabel THIS telemetry row; it can never change
    # the advertised catalog, which was frozen at run start.
    if context.tool_loader is not None and hasattr(context.tool_loader, "resolve_deferred"):
        _lg_deferred, _ = context.tool_loader.resolve_deferred(context.adapter)
        if _lg_deferred:
            loaded_names = {
                s["name"] for s in context.tool_loader.schemas_for_run(True)
            }
            if tool_name not in loaded_names:
                log.info("Lucky guess: model called deferred tool %s", tool_name)
                if context.telemetry is not None:
                    context.telemetry.record(
                        model=context.model,
                        tool_name=tool_name,
                        success=True,
                        error_type="lucky_guess",
                        error_detail=f"Tool {tool_name} called without being in prompt schema",
                        served_model=served_model,
                    )

    # Content gate. GBNF validated structure, json.loads validated syntax and
    # pydantic is about to validate type — none of the three asks what a string
    # SAYS, so leaked chat-template markup satisfies all of them and executes.
    # It already has: 13 recorded calls carried it and every one succeeded,
    # including a live wiki_compile whose entity_name was raw decoder artifacts.
    # Runs BEFORE model_validate because the corrupt value is a perfectly valid
    # `str` — the type check cannot be the thing that catches this.
    _markup = markup_guard.scan_arguments(tool_input)
    if _markup:
        log.warning(
            "Rejected %s: chat-template markup in arguments — %s",
            tool_name, markup_guard.describe(_markup),
        )
        # Rejected, never stripped: a repaired-in-place value is a value nobody
        # saw. Feed the model specifics and let the retry loop work — which
        # also files the corrupt attempt as the REJECTED side of a pair, the
        # inverse of how these values used to be banked as `chosen`.
        _stash_pending_pair(
            context,
            tool_name,
            rejected_name=tool_name,
            rejected_input=tool_input,
            error=markup_guard.describe(_markup),
            source="self_correction",
        )
        if context.telemetry is not None:
            import json as _json
            try:
                _markup_call = _json.dumps(
                    {"name": tool_name, "input": tool_input}, default=str
                )
            except Exception:
                _markup_call = None
            context.telemetry.record(
                model=context.model,
                tool_name=tool_name,
                success=False,
                error_type="template_markup",
                error_detail=markup_guard.describe(_markup),
                parsed_tool_call=_markup_call,
                served_model=served_model,
            )
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=markup_guard.rejection_message(tool_name, _markup),
            is_error=True,
        )

    try:
        parsed_input = tool.input_model.model_validate(tool_input)
    except Exception as exc:
        # Phase 4 (config-gated, default off): conservative dict-wrap
        # unwrapping. Only runs because the original input FAILED validation;
        # only accepted if the unwrapped form PASSES. The observed live
        # pathology is the model inventing nesting against flat schemas
        # ({"status": {"status": null}}, {"prompt": {...actual args...}}).
        _unwrapped = None
        if (
            context.adapter is not None
            and tool_name in getattr(context.adapter, "unwrap_tools", ())
        ):
            from prometheus.adapter.unwrap import try_unwrap_arguments
            _unwrapped = try_unwrap_arguments(tool, tool_input)
        if _unwrapped is not None:
            _rejected_input = tool_input
            tool_input, _unwrap_log = _unwrapped
            parsed_input = tool.input_model.model_validate(tool_input)
            repair_log = list(repair_log) + _unwrap_log  # counts as repairs
            try:
                from prometheus.learning.pair_capture import capture_pair, get_store
                if get_store() is not None and not ephemeral:
                    capture_pair(
                        pair_source="schema_repair",
                        model_id=context.model,
                        tool_name=tool_name,
                        context=_pair_context(context, tool_name),
                        rejected={"name": tool_name, "input": _rejected_input},
                        chosen={"name": tool_name, "input": tool_input},
                        meta={"unwrap_log": _unwrap_log},
                        telemetry=context.telemetry,
                    )
            except Exception:
                log.error("pair capture (unwrap path) failed", exc_info=True)
        else:
            # Repair-pair flywheel: pydantic rejection + a later success on
            # the same tool = a self-correction pair (the dominant live
            # recovery shape — see D4: the adapter machinery was dormant; the
            # model corrects conversationally after structured feedback).
            _stash_pending_pair(
                context,
                tool_name,
                rejected_name=tool_name,
                rejected_input=tool_input,
                error=str(exc),
                source="self_correction",
            )
            if context.telemetry is not None:
                import json as _json
                try:
                    _failed_call = _json.dumps(
                        {"name": tool_name, "input": tool_input}, default=str
                    )
                except Exception:
                    _failed_call = None
                context.telemetry.record(
                    model=context.model,
                    tool_name=tool_name,
                    success=False,
                    error_type="input_validation",
                    error_detail=str(exc),
                    # forensics + future mining (input_validation rows had
                    # 0/21 parsed_tool_call coverage in all history)
                    parsed_tool_call=_failed_call,
                    served_model=served_model,
                )
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=f"Invalid input for {tool_name}: {exc}",
                is_error=True,
            )

    # Gym dual-scoring seam: hand off raw-emitted vs about-to-execute call,
    # correlated by tool_use_id. tool_name/tool_input are now final (post
    # repair + unwrap). None in production; exception-isolated so a buggy
    # observer can never affect a real turn.
    if context.tool_call_observer is not None:
        try:
            context.tool_call_observer(
                tool_use_id,
                {"name": _original_tool_name, "input": _original_tool_input},
                {"name": tool_name, "input": tool_input},
            )
        except Exception:
            log.debug("tool_call_observer raised (ignored)", exc_info=True)

    # Permission check (Sprint 4 + TRUST-CONTEXT)
    if context.permission_checker is not None:
        # THE DEFECT THIS REPLACES: `tool_input.get("file_path")` — a key NO
        # registered tool declares. The gate got None on every call, so
        # denied_paths and the workspace boundary were both skipped, from the
        # initial commit until 2026-08-13. See permissions/tool_paths.py.
        from prometheus.permissions.tool_paths import gate_path_for
        # The tool's own schema says which params are paths (never guessed
        # from the name — that mistake has now been made three times), and
        # `base` is what a relative DIRECTORY root resolves against: the same
        # cwd the tool itself resolves against, so the gate rules on the path
        # the tool will actually read.
        _gate_tool = (
            context.tool_registry.get(tool_name)
            if context.tool_registry is not None else None
        )
        _gate_schema = None
        if _gate_tool is not None:
            try:
                _gate_schema = _gate_tool.input_model.model_json_schema()
            except Exception:
                # The schema IS the fallback's evidence — losing it silently
                # would reinstate the four-month failure in a new place, so
                # say so rather than degrade quietly.
                log.warning(
                    "SecurityGate: could not read %r's schema — the "
                    "unmapped-path fallback cannot run for this call",
                    tool_name, exc_info=True,
                )
        _file_path, _path_unknown = gate_path_for(
            tool_name, tool_input, schema=_gate_schema, base=context.cwd,
        )
        _command = str(tool_input.get("command", "")) or None
        # TRUST-CONTEXT: derive origin from the session_id already
        # threaded through LoopContext (agent_loop.py:538-542 convention).
        # User-initiated calls (telegram:/cli/web) skip the
        # ExfiltrationDetector and the network/install approve-patterns;
        # background/automated calls (system, None, SYMBIOTE/GEPA/SENTINEL
        # uuids) get the full restriction set.
        try:
            from prometheus.permissions.checker import origin_from_session_id
            _origin = origin_from_session_id(context.session_id)
        except Exception as exc:
            # SPRINT-4 audit HIGH-RISK #12 fix: log a WARN with a session_id
            # snippet so the next deploy that regresses origin resolution
            # surfaces. Falling back to "system" is the correct SAFE choice
            # (system origin gets the FULL restriction set, not the relaxed
            # user-initiated one), but a chronic regression would silently
            # restrict everything without anyone noticing. Truncate the
            # session_id so we don't leak full chat ids into logs.
            _session_snippet = (str(context.session_id) or "")[:16]
            log.warning(
                "origin_from_session_id failed for session=%s... (%s: %s); "
                "defaulting to 'system' origin (full restrictions apply)",
                _session_snippet, type(exc).__name__, exc, exc_info=True,
            )
            _origin = "system"
        try:
            decision = context.permission_checker.evaluate(
                tool_name,
                is_read_only=tool.is_read_only(parsed_input),
                file_path=_file_path,
                command=_command,
                origin=_origin,
            )
        except TypeError:
            # Older permission_checker implementations don't accept origin.
            # Fall back to the legacy call shape so third-party gates keep working.
            # SPRINT-4 audit HIGH-RISK #13 fix: one-shot WARN so the
            # deprecation is observable. Module-level guard avoids spamming
            # logs every tool call when a legacy gate is in place.
            global _LEGACY_PERMISSION_CHECKER_WARNED
            if not _LEGACY_PERMISSION_CHECKER_WARNED:
                log.warning(
                    "permission_checker.evaluate accepted as legacy signature "
                    "(no `origin` kwarg). %s should be updated to accept "
                    "`origin: str` per the TRUST-CONTEXT change; logging "
                    "this once per process.",
                    type(context.permission_checker).__name__,
                )
                _LEGACY_PERMISSION_CHECKER_WARNED = True
            decision = context.permission_checker.evaluate(
                tool_name,
                is_read_only=tool.is_read_only(parsed_input),
                file_path=_file_path,
                command=_command,
            )
        if _path_unknown and decision.allowed:
            # A path exists and could not be resolved to an absolute one
            # (relative target, or a tool nobody mapped). The gate could not
            # rule on it, so it must not be treated as cleared: UNKNOWN
            # prompts. Never silently allowed — that is the shape of the
            # defect this whole change closes.
            from prometheus.permissions.checker import PermissionDecision
            decision = PermissionDecision.approve(_path_unknown)
        if not decision.allowed:
            # The gate's own queue is the fallback prompt. Resolved HERE, in
            # the one function every surface goes through, rather than wired
            # at each construction site — that is CROSS-CUTTING §2's lesson,
            # and `permission_prompt` being populated by NO site on ANY
            # surface is what it looks like when you rely on wiring instead.
            _prompt = context.permission_prompt
            if _prompt is None and context.permission_checker is not None:
                _prompt = getattr(context.permission_checker, "request_approval", None)
            if decision.requires_confirmation and _prompt is not None:
                confirmed = await _prompt(tool_name, decision.reason)
                if not confirmed:
                    if context.telemetry is not None:
                        context.telemetry.record(
                            model=context.model,
                            tool_name=tool_name,
                            success=False,
                            error_type="permission_denied",
                            error_detail=f"User denied permission for {tool_name}",
                            served_model=served_model,
                        )
                    return ToolResultBlock(
                        tool_use_id=tool_use_id,
                        content=f"Permission denied for {tool_name}",
                        is_error=True,
                    )
            else:
                if context.telemetry is not None:
                    context.telemetry.record(
                        model=context.model,
                        tool_name=tool_name,
                        success=False,
                        error_type="permission_denied",
                        error_detail=decision.reason or f"Permission denied for {tool_name}",
                        served_model=served_model,
                    )
                return ToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=decision.reason or f"Permission denied for {tool_name}",
                    is_error=True,
                )

    from prometheus.tools.base import ToolExecutionContext
    # M5: bound tool execution so a hung tool can't freeze the turn/session.
    # The timeout wraps ONLY tool.execute() — not the permission prompt above,
    # which may legitimately wait on a human approval. A per-tool override
    # (execution_timeout_seconds) wins over the LoopContext default.
    _tool_override = getattr(tool, "execution_timeout_seconds", None)
    _timeout = _tool_override if _tool_override is not None else context.tool_timeout_seconds
    _t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(
            tool.execute(
                parsed_input,
                ToolExecutionContext(
                    cwd=context.cwd,
                    metadata={
                        "tool_registry": context.tool_registry,
                        "ask_user_prompt": context.ask_user_prompt,
                        # Managed tasks: the creating session id, so task_create
                        # can resolve session_id + notify_target from trusted
                        # context rather than from (injected) tool arguments.
                        "session_id": context.session_id,
                        **(context.tool_metadata or {}),
                    },
                ),
            ),
            timeout=_timeout,
        )
    except asyncio.TimeoutError:
        _latency_ms = (time.monotonic() - _t0) * 1000.0
        log.error(
            "Tool %s exceeded %.0fs timeout and was cancelled", tool_name, _timeout,
        )
        if context.telemetry is not None:
            context.telemetry.record(
                model=context.model,
                tool_name=tool_name,
                success=False,
                retries=retries_used,
                latency_ms=_latency_ms,
                error_type="tool_timeout",
                error_detail=f"Tool execution exceeded {_timeout:.0f}s timeout",
                repairs=len(repair_log),
                served_model=served_model,
            )
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=(
                f"Tool {tool_name} timed out after {_timeout:.0f}s and was "
                f"cancelled. If this tool legitimately needs longer, it can set "
                f"a higher execution_timeout_seconds."
            ),
            is_error=True,
        )
    _latency_ms = (time.monotonic() - _t0) * 1000.0

    # WEAVE-PRESS: when a user-initiated bash command fails with
    # "command not found", check the Printing Press library for a
    # matching CLI and append a suggestion to the tool result. The
    # model relays this to the user, who decides whether to type
    # ``/press install <name>``. Never auto-installs; never fires for
    # background/automated sessions.
    augmented_output = result.output
    if (
        tool_name == "bash"
        and result.is_error
        and (context.tool_metadata or {}).get("printing_press") is not None
    ):
        try:
            from prometheus.permissions.checker import (
                ORIGIN_USER, origin_from_session_id,
            )
            if origin_from_session_id(context.session_id) == ORIGIN_USER:
                press = context.tool_metadata["printing_press"]
                suggestion = await _maybe_suggest_printing_press(
                    press, result.output
                )
                if suggestion:
                    augmented_output = result.output + "\n\n" + suggestion
        except Exception:
            log.debug(
                "Printing Press suggestion hook failed", exc_info=True
            )

    # H1: per-result truncation (tool_result_max) BEFORE injection. Uses the
    # tool-aware strategies (bash tail, file head+tail, grep top-N, else hard
    # cap). Caps every result including errors — the cross-result budget skips
    # errors, so without this a giant error payload was unbounded. Telemetry
    # above already captured the untruncated error_detail for diagnostics.
    final_output = augmented_output
    if context.tool_result_max and context.tool_result_max > 0:
        from prometheus.context.truncation import ToolResultTruncator
        final_output = ToolResultTruncator(context.tool_result_max).truncate(
            tool_name, augmented_output,
        )

    tool_result = ToolResultBlock(
        tool_use_id=tool_use_id,
        content=final_output,
        is_error=result.is_error,
    )

    # Sprint 3 / Golden Trace Capture: record telemetry with raw + parsed output.
    provider_name = _provider_name_for_telemetry(context.provider)
    if context.telemetry is not None:
        import json as _json
        try:
            parsed_tool_json = _json.dumps({"name": tool_name, "input": tool_input}, default=str)
        except Exception:
            parsed_tool_json = None
        context.telemetry.record(
            model=context.model,
            tool_name=tool_name,
            success=not result.is_error,
            retries=retries_used,
            latency_ms=_latency_ms,
            # "ran and exited non-zero" is not "the call failed" — see
            # _classify_tool_error. Keeping both under tool_error made a
            # pytest run with failing tests look like a broken tool.
            error_type=_classify_tool_error(
                is_error=result.is_error,
                metadata=getattr(result, "metadata", None),
            ),
            # Capture the tool's own error message so `tool_error` rows are
            # diagnosable instead of blank (audit fix #4).
            #
            # EPHEMERAL: the row itself always lands — model, tool, success,
            # retries, latency, error_type, repairs are the denominators of
            # every success rate and cost report, and a missing row biases
            # them invisibly. Only the three content-bearing columns go null.
            # Note that `raw_model_output=None` also forces is_golden=0, which
            # keeps the row out of the nightly trajectories/ export.
            error_detail=(
                None if ephemeral
                else ((result.output or "")[:2000] if result.is_error else None)
            ),
            raw_model_output=None if ephemeral else raw_model_output,
            parsed_tool_call=None if ephemeral else parsed_tool_json,
            provider=provider_name,
            repairs=len(repair_log),
            served_model=served_model,
            # Fine-tuning capture: WHAT was called is not a trainable example
            # on its own — the situation that prompted it is the input half.
            # session_id joins back to lcm_messages for that context, and the
            # schema is stored as the model actually saw it rather than
            # re-derived from a registry that may have changed by export time.
            # Ephemeral turns null both, consistent with the content columns.
            session_id=None if ephemeral else (
                effective_session_id if effective_session_id is not None
                else context.session_id
            ),
            tool_schema=None if ephemeral else _tool_schema_json(context, tool_name),
        )

    # Repair-pair flywheel: a successful execution completes any pending
    # pairs for this tool (retry_success / self_correction / malformed
    # recovery) and, when enabled, harvests cloud goldens. An execution
    # FAILURE with mode-misuse shape stashes a pending self-correction —
    # task_create's "'command' is required for local_bash tasks" is the
    # canonical live example (D2).
    if not result.is_error:
        if not ephemeral:
            _capture_success_pairs(
                context, tool_name, _original_tool_name, tool_input, provider_name
            )
    elif "is required" in (result.output or "")[:200]:
        _stash_pending_pair(
            context,
            tool_name,
            rejected_name=tool_name,
            rejected_input=tool_input,
            error=(result.output or "")[:500],
            source="self_correction",
        )

    # Sprint 10 / FL-4: record tool call for divergence detection. Scoped to
    # the task minted in ``run_loop``; without a task id there is nothing to
    # record against, and recording anyway is what grew an unbounded buffer
    # on every daemon for four months (see divergence.record_tool_call).
    if context.divergence_detector is not None and div_task_id is not None:
        try:
            context.divergence_detector.record_tool_call(
                tool_name=tool_name,
                args=tool_input,
                result=tool_result.content,
                success=not tool_result.is_error,
                task_id=div_task_id,
            )
        except Exception:
            log.debug("DivergenceDetector.record_tool_call raised", exc_info=True)

    # Post-tool hook (Sprint 2)
    if context.hook_executor is not None:
        from prometheus.hooks import HookEvent
        await context.hook_executor.execute(
            HookEvent.POST_TOOL_USE,
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_result.content,
                "tool_is_error": tool_result.is_error,
                "event": HookEvent.POST_TOOL_USE.value,
            },
        )

    # Sprint 20: Post-result hooks (e.g., LSP diagnostics — can modify result)
    if context.post_result_hooks:
        for hook in context.post_result_hooks:
            try:
                tool_result = await hook(tool_name, tool_input, tool_result)
            except Exception:
                log.debug("Post-result hook failed", exc_info=True)

    return tool_result


class AgentLoop:
    """High-level agent loop that wraps run_loop().

    Usage:
        provider = StubProvider(base_url="http://localhost:8080")
        loop = AgentLoop(provider=provider)
        result = loop.run(
            system_prompt="You are a helpful assistant.",
            user_message="What is 2+2?",
        )
        print(result.text)
    """

    def __init__(
        self,
        provider: ModelProvider,
        model: str = "qwen3.5-32b",
        max_tokens: int = 4096,
        max_turns: int = 200,
        max_tool_iterations: int = SHIPPED_MAX_TOOL_ITERATIONS,
        max_tool_iterations_cloud: int | None = None,
        tool_registry=None,
        hook_executor=None,
        permission_checker=None,
        adapter=None,
        telemetry=None,
        cwd: Path | None = None,
        model_router: object | None = None,
        divergence_detector: object | None = None,
        post_result_hooks: list[object] | None = None,
        tool_loader: object | None = None,
        nudge: object | None = None,
        tool_metadata: dict[str, object] | None = None,
        file_mutation_verifier: object | None = None,
        tool_result_max: int = 0,
        compactor: object | None = None,
        memory_recall: object | None = None,
        tool_results_turn_budget: int = 8000,
        microcompact_after_turns: int = 3,
        microcompact_on_cloud: bool = False,
        microcompact_keep_chars: int = 200,
        microcompact_keep_chars_no_lcm: int = 500,
        lcm_engine: object | None = None,
        profile_resolver: object | None = None,
        fallback: object | None = None,
    ) -> None:
        self._provider = provider
        # SPRINT-provider-fallback: where a terminal provider failure degrades to.
        self._fallback = fallback
        self._model = model
        self._tool_result_max = tool_result_max
        self._compactor = compactor
        # Selector-survey 2026-08-11: these five were LoopContext fields with
        # config keys that only __main__.py threaded, so every daemon surface
        # ran on the dataclass defaults — invisible because the defaults
        # happened to EQUAL the live config values. Defaults here mirror the
        # LoopContext defaults so omitting them stays behavior-identical.
        self._tool_results_turn_budget = tool_results_turn_budget
        self._microcompact_after_turns = microcompact_after_turns
        self._microcompact_on_cloud = microcompact_on_cloud
        self._microcompact_keep_chars = microcompact_keep_chars
        self._microcompact_keep_chars_no_lcm = microcompact_keep_chars_no_lcm
        self._profile_resolver = profile_resolver
        self._max_tokens = max_tokens
        self._max_turns = max_turns
        self._max_tool_iterations = max_tool_iterations
        self._max_tool_iterations_cloud = max_tool_iterations_cloud
        self._tool_registry = tool_registry
        self._hook_executor = hook_executor
        self._permission_checker = permission_checker
        self._adapter = adapter
        self._telemetry = telemetry
        self._cwd = cwd or Path.cwd()
        self._post_task_hooks: list[Callable] = []
        self._tool_trace: list[dict] = []
        # Sprint 10
        self._model_router = model_router
        self._divergence_detector = divergence_detector
        # Sprint 20: LSP post-result hooks
        self._post_result_hooks = post_result_hooks
        # Tool Calling Middle Layer
        self._tool_loader = tool_loader
        # SUNRISE: PeriodicNudge for self-reflection every N turns. Forwarded
        # to LoopContext.nudge in run_async — run_loop owns the injection so
        # the web bridge's direct run_loop call is not left out.
        self._nudge = nudge
        # WEAVE-PRESS: opaque dict forwarded to ToolExecutionContext.metadata
        # so subsystems like the Printing Press hook can reach into the
        # registered registry without changing the LoopContext shape.
        self._tool_metadata = dict(tool_metadata) if tool_metadata else None
        # SPRINT-2 WS2: file-mutation verifier — see
        # prometheus/hooks/file_mutation_verifier.py. Optional; when None
        # the loop runs without verification (back-compat for benchmarks
        # and unit tests that build AgentLoop without supplying a config).
        self._file_mutation_verifier = file_mutation_verifier
        # PASSIVE RECALL (MEMORY-3 follow-up): PUBLIC — the daemon assigns
        # this after construction (same late-wiring pattern as
        # session_manager.lcm_engine) because the MemoryStore is built in
        # the extractor block, long after the AgentLoop. run_async reads it
        # at call time, so late assignment reaches every subsequent turn.
        self.memory_recall = memory_recall
        # LCM engine for the microcompactor's is_ingested check — PUBLIC for
        # the same reason as memory_recall above: the daemon builds the engine
        # ~370 lines after the AgentLoop and late-assigns it; run_async reads
        # the attribute per call.
        self.lcm_engine = lcm_engine

    def add_post_task_hook(self, hook: Callable) -> None:
        """Append a callback invoked after each completed task.

        Hooks fire sequentially in registration order. Each hook receives
        ``(task_description, tool_trace, final_text)`` — ``final_text`` is
        the assistant's final reply for the turn, the only place the
        semantic outcome lives when every tool call succeeded mechanically
        (SkillCreator's Stage 1 gate reads it) — and should return a
        coroutine. One hook's failure does not block subsequent hooks.
        """
        self._post_task_hooks.append(hook)

    def set_post_task_hook(self, hook: Callable) -> None:
        """Back-compat: replace the hook list with a single hook."""
        self._post_task_hooks = [hook]

    @property
    def post_task_hooks(self) -> list[Callable]:
        """Read-only view of registered post-task hooks (for tests)."""
        return list(self._post_task_hooks)

    async def run_async(
        self,
        system_prompt: str,
        user_message: str = "",
        *,
        messages: list[ConversationMessage] | None = None,
        tools: list | None = None,
        session_id: str | None = None,
        session_state: object | None = None,
    ) -> RunResult:
        """Run the agent loop asynchronously, return a RunResult.

        Phase 3.5: ``session_id`` is forwarded into LoopContext so the
        ModelRouter's per-session override lookup can fire (or bypass,
        for reserved IDs None/"system").

        SPRINT-2 WS1: ``session_state`` is the live ChatSession (or any
        object exposing ``drain_steers() -> str | None``). When supplied,
        the loop drains queued steers before each model call. None for
        contexts without a persistent session (benchmarks, evals, cron).
        """
        if messages is not None:
            messages = list(messages)  # shallow copy — run_loop mutates in place
            if not user_message:
                for msg in reversed(messages):
                    if msg.role == "user":
                        user_message = msg.text
                        break
        else:
            messages = [ConversationMessage.from_user_text(user_message)]

        context = LoopContext(
            provider=self._provider,
            model=self._model,
            fallback=self._fallback,
            system_prompt=system_prompt,
            max_tokens=self._max_tokens,
            max_turns=self._max_turns,
            max_tool_iterations=self._max_tool_iterations,
            max_tool_iterations_cloud=self._max_tool_iterations_cloud,
            tool_registry=self._tool_registry,
            hook_executor=self._hook_executor,
            permission_checker=self._permission_checker,
            adapter=self._adapter,
            telemetry=self._telemetry,
            cwd=self._cwd,
            model_router=self._model_router,
            divergence_detector=self._divergence_detector,
            post_result_hooks=self._post_result_hooks,
            tool_loader=self._tool_loader,
            tool_metadata=self._tool_metadata,
            session_id=session_id,
            session_state=session_state,
            file_mutation_verifier=self._file_mutation_verifier,
            tool_result_max=self._tool_result_max,
            tool_results_turn_budget=self._tool_results_turn_budget,
            microcompact_after_turns=self._microcompact_after_turns,
            microcompact_on_cloud=self._microcompact_on_cloud,
            microcompact_keep_chars=self._microcompact_keep_chars,
            microcompact_keep_chars_no_lcm=self._microcompact_keep_chars_no_lcm,
            compactor=self._compactor,
            memory_recall=self.memory_recall,
            lcm_engine=self.lcm_engine,
            profile_resolver=self._profile_resolver,
            # The nudge USED to be injected below, in the `async for` body.
            # That made it AgentLoop-only, so no web / Beacon / Bridge turn
            # ever saw it — the parity guard could not catch it either,
            # because it compares LoopContext FIELDS and this was not one.
            # It is now a field, so both loops get it and both guards apply.
            nudge=self._nudge,
        )

        last_text = ""
        last_usage = UsageSnapshot()
        turns = 0
        self._tool_trace = []

        async for event, usage in run_loop(context, messages):
            if isinstance(event, AssistantTurnComplete):
                last_text = event.message.text
                last_usage = event.usage
                turns += 1
            elif isinstance(event, ToolExecutionCompleted):
                self._tool_trace.append({
                    "tool_name": event.tool_name,
                    "result": (event.output or "")[:200],
                    "is_error": event.is_error,
                })
            elif isinstance(event, AssistantTextDelta):
                pass  # streaming deltas — consumed silently here

        result = RunResult(
            text=last_text,
            messages=messages,
            usage=last_usage,
            turns=turns,
        )

        # Post-task learning hooks — fire each in registration order;
        # a failing hook does not block subsequent hooks.
        #
        # EPHEMERAL: skipped entirely. The hooks are handed
        # ``hook(user_message, tool_trace, last_text)`` — ``user_message`` is
        # the raw text the user typed, ``last_text`` the assistant's final
        # reply (the turn's semantic outcome, which SkillCreator's Stage 1
        # gate reads). SkillCreator sends the message to the model as the
        # ``task_description`` of a skill-generation prompt and writes the
        # result to ``~/.prometheus/skills/auto/<name>.md``, then emits a
        # ``skill_created`` signal whose payload carries ``trigger_task`` (the
        # message's first 200 chars) into ``telemetry.signal_events``. The
        # trace is still drained below so it cannot leak into the next turn.
        if self._post_task_hooks and self._tool_trace:
            if is_session_ephemeral(session_id):
                log.debug(
                    "Ephemeral session %s — skipping %d post-task hook(s)",
                    session_id, len(self._post_task_hooks),
                )
            else:
                for hook in self._post_task_hooks:
                    try:
                        await hook(user_message, self._tool_trace, last_text)
                    except Exception:
                        log.debug(
                            "Post-task hook %s failed",
                            getattr(hook, "__qualname__", repr(hook)),
                            exc_info=True,
                        )
            self._tool_trace = []

        return result

    def run(
        self,
        system_prompt: str,
        user_message: str = "",
        *,
        messages: list[ConversationMessage] | None = None,
        tools: list | None = None,
        session_id: str | None = None,
    ) -> RunResult:
        """Synchronous entry point — wraps run_async() via asyncio.run()."""
        return asyncio.run(
            self.run_async(
                system_prompt,
                user_message,
                messages=messages,
                tools=tools,
                session_id=session_id,
            )
        )
