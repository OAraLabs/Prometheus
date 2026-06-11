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
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

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
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)

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
    max_tool_iterations: int = 25
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
    # H1: per-result cap (tokens) applied to EACH tool result before injection,
    # via ToolResultTruncator's per-tool strategies. 0 disables (back-compat for
    # tests/benchmarks). Runs before the cross-result turn budget below, and
    # unlike that budget it also caps error results.
    tool_result_max: int = 0
    tool_results_turn_budget: int = 8000  # max tokens across ALL results per turn
    microcompact_after_turns: int = 3     # compact tool results older than N turns
    microcompact_keep_chars: int = 200    # chars to keep per compacted result
    microcompact_keep_chars_no_lcm: int = 500  # chars if LCM hasn't ingested
    lcm_engine: object | None = None      # LCMEngine for microcompaction checks
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
    # ``post_turn`` when a turn ends without further tool calls. A summary
    # of claimed-vs-actual filesystem changes is appended as a synthetic
    # user-role message so the agent sees it on its next turn. See
    # prometheus/hooks/file_mutation_verifier.py.
    file_mutation_verifier: object | None = None
    # Repair-pair flywheel: failed-call stash awaiting a matching success
    # within this loop run (keyed by tool name; "_malformed" for provider-
    # dropped empty envelopes). Lazily initialized; see learning/pair_capture.
    pair_pending: dict | None = None


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
) -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]:
    """Run the conversation loop until the model stops requesting tools.

    Yields (StreamEvent, UsageSnapshot | None) tuples. The loop exits when
    the assistant returns a response with no tool_uses, or after max_turns.
    """
    tool_schema: list[dict] = []
    if context.tool_loader is not None and hasattr(context.tool_loader, "active_schemas"):
        tool_schema = context.tool_loader.active_schemas()
    elif context.tool_registry is not None and hasattr(context.tool_registry, "to_api_schema"):
        tool_schema = context.tool_registry.to_api_schema()

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
                decision = context.model_router.route(
                    latest_user,
                    context={"session_id": context.session_id},
                )
                reason_repr = (
                    decision.reason.value
                    if hasattr(decision.reason, "value")
                    else decision.reason
                )
                log.debug(
                    "ModelRouter: %s → %s/%s (%s)",
                    latest_user[:60],
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
                    import re as _re
                    provider_name = decision.provider_name or "unknown"
                    model_name = decision.model_name or "unknown"
                    new_line = f"- Model: {model_name} (provider: {provider_name})"
                    context.system_prompt = _re.sub(
                        r"^- Model: .*$",
                        new_line,
                        context.system_prompt,
                        count=1,
                        flags=_re.MULTILINE,
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

    circuit_breaker = _CircuitBreaker(max_identical=3, max_any=5)
    tool_iteration = 0

    for turn in range(context.max_turns):
        # MicroCompaction: compact old tool results (free, no LLM calls)
        if turn > 0 and context.microcompact_after_turns > 0:
            _microcompact_old_results(context, messages, turn)

        final_message: ConversationMessage | None = None
        usage = UsageSnapshot()
        dropped_malformed = 0

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
                )
            except Exception:
                log.exception(
                    "ContextCompactor.apply raised — sending uncompacted")
                render_source = messages

        async for event in context.provider.stream_message(
            ApiMessageRequest(
                model=context.model,
                # Context-assembly: fence any untrusted injected turns (task
                # output, watched-file contents, cron data) with the derived
                # banner before the provider serializes them. The session/LCM
                # copies stay clean — this projection is per-call only.
                messages=render_messages_for_model(render_source),
                system_prompt=per_call_system_prompt,
                max_tokens=context.max_tokens,
                tools=_payload_tools,
            )
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
                final_message = ConversationMessage(
                    role="assistant",
                    content=extracted,
                )

        messages.append(final_message)
        yield AssistantTurnComplete(message=final_message, usage=usage), usage

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
            # the file-mutation verifier and append its summary as a
            # synthetic user-role message so the model sees it on its
            # next turn. Same channel as PeriodicNudge. None when no
            # mutations were tracked.
            fmv = getattr(context, "file_mutation_verifier", None)
            if fmv is not None:
                try:
                    summary = fmv.post_turn()
                except Exception:
                    summary = None
                    log.debug(
                        "FileMutationVerifier.post_turn raised — "
                        "skipping summary append", exc_info=True,
                    )
                if summary:
                    messages.append(ConversationMessage.from_user_text(summary))
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
                    fmv.pre_tool_use(_tc.name, _tc.input or {}, _tc.id)
                except Exception:
                    log.debug(
                        "FileMutationVerifier.pre_tool_use raised",
                        exc_info=True,
                    )

        tool_results = await _dispatch_tool_calls(
            context, tool_calls, raw_model_output=raw_model_output_this_turn
        )

        # SPRINT-2 WS2: post-snapshot + diff after every tool result.
        if fmv is not None:
            for _tc, _r in zip(tool_calls, tool_results):
                try:
                    fmv.post_tool_use(
                        _tc.name, _tc.input or {}, _tc.id,
                        output=_r.content, is_error=_r.is_error,
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

        # Sprint 10: checkpoint + divergence evaluation after tool dispatch
        if context.divergence_detector is not None:
            dd = context.divergence_detector
            # Maybe create a checkpoint
            msg_dicts = [
                {"role": m.role, "content": m.text or ""}
                for m in messages
                if hasattr(m, "role")
            ]
            dd.maybe_checkpoint(msg_dicts)

            # Evaluate divergence (only after 3+ steps to gather signal)
            if dd.step_count > 3:
                tool_result_dicts = [
                    {"result": tr.content, "success": not tr.is_error}
                    for tr in tool_results
                ]
                div_result = dd.evaluate(msg_dicts, tool_result_dicts)
                if div_result.should_rollback and div_result.checkpoint:
                    trust = 1  # default to non-autonomous
                    rolled_back, restored = dd.rollback(div_result.checkpoint, trust)
                    if rolled_back:
                        log.warning(
                            "Divergence rollback: restoring %d messages",
                            len(restored),
                        )

    raise RuntimeError(f"Exceeded maximum turn limit ({context.max_turns})")


# ---------------------------------------------------------------------------
# Helpers for run_loop
# ---------------------------------------------------------------------------

def _make_assistant_msg(text: str) -> ConversationMessage:
    """Build a synthetic assistant message."""
    from prometheus.engine.messages import TextBlock
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


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
                trimmed = r.content[:char_limit] + \
                    "\n[truncated — use lcm_expand or re-read for full content]"
                new_results[i] = ToolResultBlock(
                    tool_use_id=r.tool_use_id,
                    content=trimmed,
                    is_error=r.is_error,
                )
                remaining -= (tokens - estimate_tokens(trimmed))

    log.debug("Cross-result budget: %d → %d tokens (budget %d)", total, remaining, budget)
    return new_results


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
            msg.content[j] = TRB(
                tool_use_id=block.tool_use_id,
                content=f"[microcompacted] {first_line}...\n{summary}",
                is_error=False,
            )
            compacted += 1

    if compacted:
        log.debug("Microcompacted %d old tool results (turn %d)", compacted, current_turn)


async def _safe_execute(
    context: LoopContext,
    tc: object,
    raw_model_output: str | None,
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
                    error_detail=str(exc)[:2000],
                )
            except Exception:  # pragma: no cover - telemetry must not mask result
                log.debug("telemetry.record failed in _safe_execute", exc_info=True)
        return ToolResultBlock(
            tool_use_id=tc.id,
            content=f"Tool {tc.name} raised an exception: {exc}",
            is_error=True,
        )


async def _dispatch_tool_calls(
    context: LoopContext,
    tool_calls: list,
    raw_model_output: str | None = None,
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
    """
    if len(tool_calls) == 1:
        return [await _safe_execute(context, tool_calls[0], raw_model_output)]

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
            return idx, await _safe_execute(context, tc, raw_model_output)

        results.extend(
            await asyncio.gather(*[_run_ro(idx, tc) for idx, tc in read_only])
        )

    # Run mutating tools sequentially (order matters)
    for idx, tc in mutating:
        results.append((idx, await _safe_execute(context, tc, raw_model_output)))

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
) -> ToolResultBlock:
    """Execute a single tool call, running hooks if configured.

    Golden Trace Capture sprint: ``raw_model_output`` is the text the
    model produced BEFORE adapter parsing (enforcer/formatter) for this
    turn. Passed through to ``telemetry.record()`` on the success path
    so cloud-provider wins with zero adapter retries get flagged as
    ``is_golden=1`` for later fine-tuning use.
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
            if repair_log:
                # An adapter repair IS a labeled pair: as-emitted vs repaired.
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
            )
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Unknown tool: {tool_name}",
            is_error=True,
        )

    # Lucky guess: tool is registered but wasn't in the active prompt schema
    if context.tool_loader is not None and hasattr(context.tool_loader, "_deferred_enabled"):
        if context.tool_loader._deferred_enabled:
            loaded_names = {s["name"] for s in context.tool_loader.active_schemas()}
            if tool_name not in loaded_names:
                log.info("Lucky guess: model called deferred tool %s", tool_name)
                if context.telemetry is not None:
                    context.telemetry.record(
                        model=context.model,
                        tool_name=tool_name,
                        success=True,
                        error_type="lucky_guess",
                        error_detail=f"Tool {tool_name} called without being in prompt schema",
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
                if get_store() is not None:
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
                )
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=f"Invalid input for {tool_name}: {exc}",
                is_error=True,
            )

    # Permission check (Sprint 4 + TRUST-CONTEXT)
    if context.permission_checker is not None:
        _file_path = str(tool_input.get("file_path", "")) or None
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
        if not decision.allowed:
            if decision.requires_confirmation and context.permission_prompt is not None:
                confirmed = await context.permission_prompt(tool_name, decision.reason)
                if not confirmed:
                    if context.telemetry is not None:
                        context.telemetry.record(
                            model=context.model,
                            tool_name=tool_name,
                            success=False,
                            error_type="permission_denied",
                            error_detail=f"User denied permission for {tool_name}",
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
            error_type="tool_error" if result.is_error else None,
            # Capture the tool's own error message so `tool_error` rows are
            # diagnosable instead of blank (audit fix #4).
            error_detail=(result.output or "")[:2000] if result.is_error else None,
            raw_model_output=raw_model_output,
            parsed_tool_call=parsed_tool_json,
            provider=provider_name,
            repairs=len(repair_log),
        )

    # Repair-pair flywheel: a successful execution completes any pending
    # pairs for this tool (retry_success / self_correction / malformed
    # recovery) and, when enabled, harvests cloud goldens. An execution
    # FAILURE with mode-misuse shape stashes a pending self-correction —
    # task_create's "'command' is required for local_bash tasks" is the
    # canonical live example (D2).
    if not result.is_error:
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

    # Sprint 10: record tool call for divergence detection
    if context.divergence_detector is not None:
        context.divergence_detector.record_tool_call(
            tool_name=tool_name,
            args=tool_input,
            result=tool_result.content,
            success=not tool_result.is_error,
        )

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
        max_tool_iterations: int = 25,
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
    ) -> None:
        self._provider = provider
        self._model = model
        self._tool_result_max = tool_result_max
        self._compactor = compactor
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
        # SUNRISE: PeriodicNudge for self-reflection every N turns
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

    def add_post_task_hook(self, hook: Callable) -> None:
        """Append a callback invoked after each completed task.

        Hooks fire sequentially in registration order. Each hook receives
        ``(task_description, tool_trace)`` and should return a coroutine.
        One hook's failure does not block subsequent hooks.
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
            compactor=self._compactor,
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
                # PeriodicNudge — inject self-reflection prompt every N turns.
                # The nudge appears as a [system-internal]-prefixed user message
                # so the model sees it on the next iteration.
                if self._nudge is not None:
                    try:
                        nudge_msg = self._nudge.maybe_inject(turns)
                    except Exception:
                        nudge_msg = None
                        log.debug("PeriodicNudge: maybe_inject raised", exc_info=True)
                    if nudge_msg:
                        messages.append(
                            ConversationMessage.from_user_text(nudge_msg["content"])
                        )
                        log.debug("PeriodicNudge: injected at turn %d", turns)
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
        if self._post_task_hooks and self._tool_trace:
            for hook in self._post_task_hooks:
                try:
                    await hook(user_message, self._tool_trace)
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
