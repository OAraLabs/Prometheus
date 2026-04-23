"""Model Router — selects provider + adapter for each request.

Combines:
- Hermes smart_model_routing (classify simple vs complex)
- OpenClaw fallback chain (graceful degradation)
- Claude Code subagent isolation (escalation to cloud)
- Prometheus adapter auto-adjustment (formatter/strictness per provider)

The router SELECTS. It does not EXECUTE. The agent loop still runs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

log = logging.getLogger(__name__)


# ── Task classification (relocated from adapter/router.py in Phase 1) ─────────
#
# Phase 1 of GRAFT-ROUTER-WIRE: TaskType, TaskClassification, and TaskClassifier
# live here now in addition to the adapter/router.py copy. This duplication is
# intentional for the zero-behavior-change guarantee — adapter/router.py is
# still the live path until Phase 2 flips the switch. Phase 1.5 will integrate
# TaskClassifier into ModelRouter.route() via a new _route_by_task_type branch.


class TaskType(Enum):
    """Task classification types for routing decisions."""
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    QUICK_ANSWER = "quick_answer"
    CREATIVE = "creative"
    TOOL_HEAVY = "tool_heavy"


@dataclass
class TaskClassification:
    """Result of classifying a user message."""
    task_type: TaskType
    confidence: float  # 0.0 - 1.0
    matched_tokens: list[str]
    reason: str


class TaskClassifier:
    """
    Classify incoming tasks for routing decisions.

    Uses token-based scoring similar to leaky's PortRuntime.route_prompt().
    No LLM calls — pure heuristics for speed.
    """

    # Token sets for each task type (like leaky's module name matching)
    TASK_TOKENS: dict[TaskType, set[str]] = {
        TaskType.CODE_GENERATION: {
            "write", "create", "implement", "code", "function", "class",
            "script", "program", "fix", "debug", "refactor", "optimize",
            "python", "javascript", "typescript", "rust", "go", "java",
            "api", "endpoint", "database", "query", "sql", "schema",
            "test", "unittest", "pytest", "module", "package", "library",
        },
        TaskType.REASONING: {
            "explain", "why", "how", "analyze", "compare", "evaluate",
            "assess", "think", "reason", "consider", "implications",
            "consequences", "pros", "cons", "tradeoffs", "advantages",
            "disadvantages", "strategy", "approach", "plan", "design",
            "architect", "review", "critique", "weigh",
        },
        TaskType.QUICK_ANSWER: {
            "what", "who", "when", "where", "which", "define",
            "definition", "meaning", "list", "name", "give",
            "is", "are", "does", "can", "will", "would",
        },
        TaskType.CREATIVE: {
            "story", "poem", "song", "essay", "article", "creative",
            "imaginative", "fictional", "narrative", "roleplay",
            "pretend", "imagine", "scenario", "character", "dialogue",
            "compose", "draft", "author",
        },
        TaskType.TOOL_HEAVY: {
            "search", "find", "look", "fetch", "download", "browse",
            "file", "directory", "folder", "read", "edit",
            "delete", "run", "execute", "bash", "shell", "command",
            "terminal", "dashboard", "serve", "http", "server",
            "git", "commit", "push", "pull", "deploy",
        },
    }

    # Message length thresholds
    SHORT_MSG_CHARS = 50
    LONG_MSG_CHARS = 500

    def classify(
        self,
        message: str,
        tool_mentions: Optional[list[str]] = None,
    ) -> TaskClassification:
        """
        Classify a message using token-based scoring.

        Algorithm (adapted from leaky runtime.py):
        1. Tokenize message (split on whitespace and punctuation)
        2. Score each task type by token overlap
        3. Apply length-based adjustments
        4. Apply tool-mention boosts
        5. Return highest-scoring type with confidence
        """
        # Tokenize (leaky pattern: split on / - and whitespace)
        tokens = set(
            token.lower()
            for token in re.split(r'[\s/\-_.,;:!?\'"()\[\]{}]+', message)
            if token and len(token) > 1
        )

        # Score each task type
        scores: dict[TaskType, float] = {}
        matched: dict[TaskType, list[str]] = {}

        for task_type, task_tokens in self.TASK_TOKENS.items():
            overlap = tokens & task_tokens
            scores[task_type] = len(overlap)
            matched[task_type] = list(overlap)

        # Apply adjustments
        msg_len = len(message)

        # Short messages favor quick answers
        if msg_len < self.SHORT_MSG_CHARS:
            scores[TaskType.QUICK_ANSWER] += 1.0

        # Long messages favor reasoning/code
        if msg_len > self.LONG_MSG_CHARS:
            scores[TaskType.REASONING] += 0.5
            scores[TaskType.CODE_GENERATION] += 0.5

        # Code blocks strongly indicate code generation
        if "```" in message or "`" in message:
            scores[TaskType.CODE_GENERATION] += 2.0

        # Tool mentions boost TOOL_HEAVY
        if tool_mentions:
            scores[TaskType.TOOL_HEAVY] += len(tool_mentions) * 0.5

        # Find best match
        best_type = max(scores, key=lambda t: scores[t])
        best_score = scores[best_type]
        total_score = sum(scores.values()) or 1.0

        # Confidence is relative score
        confidence = min(best_score / total_score, 1.0) if best_score > 0 else 0.3

        # Default to REASONING if no clear signal
        if best_score == 0:
            best_type = TaskType.REASONING
            confidence = 0.3
            matched[best_type] = []

        return TaskClassification(
            task_type=best_type,
            confidence=confidence,
            matched_tokens=matched[best_type],
            reason=f"tokens={best_score:.1f}, len={msg_len}, conf={confidence:.2f}",
        )


class RouteReason(str, Enum):
    """Why a particular provider was chosen."""

    PRIMARY = "primary"
    USER_OVERRIDE = "user_override"
    SMART_SIMPLE = "smart_simple"
    SMART_COMPLEX = "smart_complex"
    TASK_RULE = "task_rule"        # Phase 1.5: TaskClassifier matched a config rule
    ESCALATION = "escalation"
    FALLBACK = "fallback"
    QUEUE = "queue"
    AUXILIARY = "auxiliary"


@dataclass
class RouteDecision:
    """Everything the agent loop needs to handle this turn."""

    provider: Any              # ModelProvider instance
    adapter: Any               # ModelAdapter instance
    reason: RouteReason
    use_subagent: bool = False
    model_name: str = ""
    provider_name: str = ""
    cost_warning: str | None = None


@dataclass
class RoutingRule:
    """A rule mapping a classified task type to a provider+model.

    Phase 1.5: consumed by ModelRouter._route_by_task_type(). Configured under
    `router.rules` in prometheus.yaml.
    """
    task_type: TaskType
    provider: str
    model: str
    base_url: Optional[str] = None
    min_confidence: float = 0.0


@dataclass
class RouterConfig:
    """Loaded from prometheus.yaml router: section."""

    fallback_chain: list[dict] = field(default_factory=list)

    # Task-type rules (Phase 1.5)
    task_rules: list[RoutingRule] = field(default_factory=list)

    # Smart routing
    smart_routing_enabled: bool = False
    max_simple_chars: int = 160
    max_simple_words: int = 28
    simple_provider: dict | None = None

    # Escalation
    escalation_enabled: bool = False
    escalation_provider: dict | None = None
    escalation_as_subagent: bool = True
    escalation_budget_usd: float = 1.00

    # Auxiliary
    auxiliary_vision: dict | None = None
    auxiliary_compression: dict | None = None
    auxiliary_summarization: dict | None = None


# -- Provider override presets for /claude, /gpt, etc. --

OVERRIDE_PRESETS: dict[str, dict[str, str]] = {
    "claude": {
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
    },
    "gpt": {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o",
    },
    "gemini": {
        "provider": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
    },
    "xai": {
        "provider": "xai",
        "api_key_env": "XAI_API_KEY",
        "model": "grok-3",
    },
}


class ModelRouter:
    """Select provider + adapter for each request.

    Usage::

        router = ModelRouter(config, primary_provider, primary_adapter)
        decision = router.route("refactor auth.py")
        # decision.provider, decision.adapter ready for AgentLoop
    """

    def __init__(
        self,
        config: RouterConfig,
        primary_provider: Any,
        primary_adapter: Any,
        primary_model: str = "local",
    ) -> None:
        self.config = config
        self.primary_provider = primary_provider
        self.primary_adapter = primary_adapter
        self.primary_model = primary_model

        # Lazy-built providers
        self._fallback_cache: list[tuple[dict, Any | None]] = [
            (cfg, None) for cfg in config.fallback_chain
        ]
        self._escalation_provider: Any | None = None
        self._simple_provider: Any | None = None
        self._auxiliary_cache: dict[str, Any] = {}

        # Task-type rule provider cache (Phase 1.5); keyed by "provider:model"
        self._task_rule_providers: dict[str, Any] = {}

        # Task classifier (Phase 1.5); used by _route_by_task_type
        self.classifier = TaskClassifier()

        # User override (set by /claude, /gpt; cleared by /local)
        self._override_config: dict | None = None
        self._override_provider: Any | None = None
        self._override_adapter: Any | None = None

    # ── Main entry point ──────────────────────────────────────────

    def route(self, message: str, context: dict | None = None) -> RouteDecision:
        """Select provider + adapter for this message.

        Args:
            message: User message text.
            context: Optional metadata (retry_count, is_subagent, etc.).
        """
        context = context or {}

        # 1. User override (/claude, /gpt, /local)
        if self._override_config:
            return self._route_override()

        # 2. Retry escalation
        retry_count = context.get("retry_count", 0)
        if retry_count >= 3 and self.config.escalation_enabled:
            return self._route_escalation()

        # 3. Smart routing (simple → cheap model)
        if self.config.smart_routing_enabled and self.config.simple_provider:
            if self._classify_complexity(message) == "simple":
                return self._route_simple()

        # 4. Task-type rules (Phase 1.5)
        # Cost-first ordering intentional: smart routing at #3 catches short/simple
        # messages before capability matching runs. See GRAFT-ROUTER-WIRE v3 Phase 1.5
        # design note for rationale.
        task_rule_decision = self._route_by_task_type(message)
        if task_rule_decision is not None:
            return task_rule_decision

        # 5. Primary (with fallback if needed)
        return self._route_primary()

    def route_auxiliary(self, task: str) -> RouteDecision:
        """Route an auxiliary task (vision, compression, summarization)."""
        aux_map = {
            "vision": self.config.auxiliary_vision,
            "compression": self.config.auxiliary_compression,
            "summarization": self.config.auxiliary_summarization,
        }
        aux_cfg = aux_map.get(task)
        if not aux_cfg:
            return RouteDecision(
                provider=self.primary_provider,
                adapter=self.primary_adapter,
                reason=RouteReason.AUXILIARY,
                model_name=self.primary_model,
            )

        provider = self._get_or_create_auxiliary(task, aux_cfg)
        adapter = _build_adapter_for(aux_cfg.get("provider", ""))
        return RouteDecision(
            provider=provider,
            adapter=adapter,
            reason=RouteReason.AUXILIARY,
            model_name=aux_cfg.get("model", "unknown"),
            provider_name=aux_cfg.get("provider", "unknown"),
        )

    # ── User override ─────────────────────────────────────────────

    def set_override(self, provider_config: dict) -> None:
        """Set user override (from /claude, /gpt commands)."""
        self._override_config = provider_config
        self._override_provider = None
        self._override_adapter = None

    def clear_override(self) -> None:
        """Clear user override (from /local command)."""
        self._override_config = None
        self._override_provider = None
        self._override_adapter = None

    @property
    def has_override(self) -> bool:
        return self._override_config is not None

    def _route_override(self) -> RouteDecision:
        assert self._override_config is not None
        if self._override_provider is None:
            from prometheus.providers.registry import ProviderRegistry
            self._override_provider = ProviderRegistry.create(self._override_config)
            pname = self._override_config.get("provider", "")
            self._override_adapter = _build_adapter_for(pname)
        return RouteDecision(
            provider=self._override_provider,
            adapter=self._override_adapter,
            reason=RouteReason.USER_OVERRIDE,
            model_name=self._override_config.get("model", "unknown"),
            provider_name=self._override_config.get("provider", "unknown"),
        )

    # ── Smart routing (Hermes pattern) ────────────────────────────

    def _classify_complexity(self, message: str) -> str:
        """Classify as 'simple' or 'complex'. Conservative — defaults to complex."""
        if len(message) > self.config.max_simple_chars:
            return "complex"
        if len(message.split()) > self.config.max_simple_words:
            return "complex"
        if "\n" in message.strip():
            return "complex"

        lowered = message.lower()
        complex_indicators = (
            "```", "def ", "class ", "import ", "function ",
            "refactor", "debug", "implement", "build", "create",
            "fix the", "edit the", "write a", "modify",
            "analyze", "explain in detail", "compare", "research",
            "plan", "architect", "design", "review",
        )
        if any(ind in lowered for ind in complex_indicators):
            return "complex"
        return "simple"

    def _route_simple(self) -> RouteDecision:
        assert self.config.simple_provider is not None
        if self._simple_provider is None:
            from prometheus.providers.registry import ProviderRegistry
            self._simple_provider = ProviderRegistry.create(self.config.simple_provider)
        pname = self.config.simple_provider.get("provider", "")
        return RouteDecision(
            provider=self._simple_provider,
            adapter=_build_adapter_for(pname),
            reason=RouteReason.SMART_SIMPLE,
            model_name=self.config.simple_provider.get("model", "unknown"),
            provider_name=pname,
        )

    # ── Escalation (Claude Code pattern) ──────────────────────────

    def get_escalation_decision(self) -> RouteDecision | None:
        """Return an escalation RouteDecision if enabled and configured.

        Phase 3 public API: the agent_loop's ESCALATE handler calls this when
        RetryEngine returns RetryAction.ESCALATE so it can reach the
        pre-instantiated escalation provider + adapter without poking at
        private internals.
        """
        if not self.config.escalation_enabled:
            return None
        if not self.config.escalation_provider:
            return None
        return self._route_escalation()

    def _route_escalation(self) -> RouteDecision:
        if not self.config.escalation_provider:
            return self._route_primary()
        if self._escalation_provider is None:
            from prometheus.providers.registry import ProviderRegistry
            self._escalation_provider = ProviderRegistry.create(
                self.config.escalation_provider
            )
        pname = self.config.escalation_provider.get("provider", "")
        model = self.config.escalation_provider.get("model", "unknown")
        return RouteDecision(
            provider=self._escalation_provider,
            adapter=_build_adapter_for(pname),
            reason=RouteReason.ESCALATION,
            use_subagent=self.config.escalation_as_subagent,
            model_name=model,
            provider_name=pname,
            cost_warning=f"Escalating to {model} (local retries exhausted)",
        )

    # ── Task-type rules (Phase 1.5: Hermes-style rule-based routing) ──

    def _route_by_task_type(self, message: str) -> RouteDecision | None:
        """Classify the message and search config.task_rules for a match.

        Returns a RouteDecision with reason=TASK_RULE if a rule matches (and
        classification.confidence >= rule.min_confidence); otherwise returns
        None so the caller falls through to the next routing branch.
        """
        if not self.config.task_rules:
            return None

        classification = self.classifier.classify(message)
        for rule in self.config.task_rules:
            if rule.task_type != classification.task_type:
                continue
            if classification.confidence < rule.min_confidence:
                continue

            cache_key = f"{rule.provider}:{rule.model}"
            if cache_key not in self._task_rule_providers:
                from prometheus.providers.registry import ProviderRegistry
                provider_cfg: dict[str, Any] = {
                    "provider": rule.provider,
                    "model": rule.model,
                }
                if rule.base_url:
                    provider_cfg["base_url"] = rule.base_url
                try:
                    self._task_rule_providers[cache_key] = ProviderRegistry.create(
                        provider_cfg
                    )
                except Exception:
                    log.debug(
                        "Failed to create task-rule provider %s",
                        provider_cfg,
                        exc_info=True,
                    )
                    return None

            return RouteDecision(
                provider=self._task_rule_providers[cache_key],
                adapter=_build_adapter_for(rule.provider),
                reason=RouteReason.TASK_RULE,
                model_name=rule.model,
                provider_name=rule.provider,
            )
        return None

    # ── Primary + fallback (OpenClaw pattern) ─────────────────────

    def _route_primary(self) -> RouteDecision:
        return RouteDecision(
            provider=self.primary_provider,
            adapter=self.primary_adapter,
            reason=RouteReason.PRIMARY,
            model_name=self.primary_model,
        )

    def get_fallback(self, failed_provider_name: str = "") -> RouteDecision | None:
        """Get next available fallback after a provider failure."""
        from prometheus.providers.registry import ProviderRegistry

        for i, (cfg, cached) in enumerate(self._fallback_cache):
            if cached is None:
                try:
                    cached = ProviderRegistry.create(cfg)
                    self._fallback_cache[i] = (cfg, cached)
                except Exception:
                    log.debug("Failed to create fallback provider %s", cfg, exc_info=True)
                    continue
            pname = cfg.get("provider", "unknown")
            return RouteDecision(
                provider=cached,
                adapter=_build_adapter_for(pname),
                reason=RouteReason.FALLBACK,
                model_name=cfg.get("model", "unknown"),
                provider_name=pname,
                cost_warning=f"Primary unavailable — using fallback: {cfg.get('model', 'unknown')}",
            )
        return None

    # ── Auxiliary ──────────────────────────────────────────────────

    def _get_or_create_auxiliary(self, task: str, cfg: dict) -> Any:
        if task not in self._auxiliary_cache:
            from prometheus.providers.registry import ProviderRegistry
            self._auxiliary_cache[task] = ProviderRegistry.create(cfg)
        return self._auxiliary_cache[task]

    # ── Status ────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Current router state for /route command."""
        return {
            "primary": self.primary_model,
            "override": self._override_config.get("model") if self._override_config else None,
            "smart_routing": self.config.smart_routing_enabled,
            "escalation": (
                self.config.escalation_provider.get("model")
                if self.config.escalation_provider
                else None
            ),
            "fallback_count": len(self.config.fallback_chain),
        }


# ── Adapter factory (Prometheus novel) ────────────────────────────

def _build_adapter_for(provider_name: str) -> Any:
    """Build the right ModelAdapter for a given provider name.

    When switching from Gemma to Claude mid-session:
    - Formatter: GemmaFormatter → PassthroughFormatter
    - Strictness: MEDIUM → NONE
    All automatic based on provider name.
    """
    from prometheus.adapter import ModelAdapter
    from prometheus.adapter.formatter import (
        AnthropicFormatter,
        PassthroughFormatter,
        QwenFormatter,
    )

    if provider_name == "anthropic":
        return ModelAdapter(formatter=AnthropicFormatter(), strictness="NONE")
    if provider_name in ("openai", "gemini", "xai"):
        return ModelAdapter(formatter=PassthroughFormatter(), strictness="NONE")
    # Local providers default to QwenFormatter (daemon overrides for gemma)
    return ModelAdapter(formatter=QwenFormatter(), strictness="MEDIUM")


def _parse_task_rules(rules_config: list) -> list[RoutingRule]:
    """Parse task-type routing rules from config (Phase 1.5).

    Invalid rule entries are logged and skipped rather than raising, so a
    typo in one rule doesn't break the whole daemon.
    """
    rules: list[RoutingRule] = []
    for r in rules_config:
        try:
            rules.append(
                RoutingRule(
                    task_type=TaskType(r["task_type"]),
                    provider=r["provider"],
                    model=r["model"],
                    base_url=r.get("base_url"),
                    min_confidence=r.get("min_confidence", 0.0),
                )
            )
        except (KeyError, ValueError) as e:
            log.warning("Invalid routing rule %r: %s", r, e)
    return rules


def load_router_config(config: dict) -> RouterConfig:
    """Parse the router: section from prometheus.yaml."""
    rc = config.get("router", {})
    smart = rc.get("smart_routing", {})
    esc = rc.get("escalation", {})
    aux = rc.get("auxiliary", {})

    return RouterConfig(
        fallback_chain=rc.get("fallback", []),
        task_rules=_parse_task_rules(rc.get("rules", [])),
        smart_routing_enabled=smart.get("enabled", False),
        max_simple_chars=smart.get("max_simple_chars", 160),
        max_simple_words=smart.get("max_simple_words", 28),
        simple_provider=smart.get("simple_provider"),
        escalation_enabled=esc.get("enabled", False),
        escalation_provider=esc.get("provider"),
        escalation_as_subagent=esc.get("as_subagent", True),
        escalation_budget_usd=esc.get("budget_usd", 1.00),
        auxiliary_vision=aux.get("vision"),
        auxiliary_compression=aux.get("compression"),
        auxiliary_summarization=aux.get("summarization"),
    )
