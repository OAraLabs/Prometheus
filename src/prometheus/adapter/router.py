"""
Model Router — Task-based provider selection with fallback chain.

Donor patterns:
- leaky/src/runtime.py: PortRuntime.route_prompt() — token scoring
- hermes-agent/agent/auxiliary_client.py — secondary LLM client
- hermes-agent/hermes_cli/main.py — profile/provider selection

Source: Prometheus (OAra AI Lab)
License: MIT
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


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


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    provider: str       # "llama_cpp", "ollama", "anthropic"
    model: str          # "qwen3.5-32b", "gemma4-26b", "claude-3-haiku"
    base_url: Optional[str] = None
    reason: str = ""


@dataclass
class RoutingRule:
    """A rule mapping task type to provider."""
    task_type: TaskType
    provider: str
    model: str
    base_url: Optional[str] = None
    min_confidence: float = 0.0


class ModelRouter:
    """
    Route tasks to appropriate model providers.

    Donor patterns:
    - Hermes profiles for multi-provider selection
    - Hermes fallback chains for resilience
    """

    def __init__(self, config: dict):
        router_config = config.get("model_router", {})
        self.enabled = router_config.get("enabled", False)
        self.rules = self._parse_rules(router_config.get("rules", []))
        self.fallback_chain = self._parse_fallback(
            router_config.get("fallback_chain", [])
        )
        self.classifier = TaskClassifier()

        # Defaults from main config
        model_cfg = config.get("model", {})
        self.default_provider = model_cfg.get("provider", "llama_cpp")
        self.default_model = model_cfg.get("model", "qwen3.5-32b")
        self.default_base_url = model_cfg.get("base_url", "http://localhost:8080")

    def _parse_rules(self, rules_config: list) -> list[RoutingRule]:
        """Parse routing rules from config."""
        rules = []
        for r in rules_config:
            try:
                task_type = TaskType(r["task_type"])
                rules.append(RoutingRule(
                    task_type=task_type,
                    provider=r["provider"],
                    model=r["model"],
                    base_url=r.get("base_url"),
                    min_confidence=r.get("min_confidence", 0.0),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid routing rule: {r}, error: {e}")
        return rules

    def _parse_fallback(self, fallback_config: list) -> list[ProviderConfig]:
        """Parse fallback chain from config."""
        return [
            ProviderConfig(
                provider=f["provider"],
                model=f.get("model", ""),
                base_url=f.get("base_url"),
                reason="fallback",
            )
            for f in fallback_config
        ]

    def route(
        self,
        message: str,
        tool_mentions: Optional[list[str]] = None,
        force_provider: Optional[str] = None,
    ) -> ProviderConfig:
        """
        Route a message to the appropriate provider.

        Priority:
        1. Forced provider (if specified)
        2. Routing disabled -> default
        3. Rule match -> matched provider
        4. No match -> default
        """
        # Forced provider
        if force_provider:
            return ProviderConfig(
                provider=force_provider,
                model=self.default_model,
                base_url=self.default_base_url,
                reason=f"forced:{force_provider}",
            )

        # Routing disabled
        if not self.enabled:
            return ProviderConfig(
                provider=self.default_provider,
                model=self.default_model,
                base_url=self.default_base_url,
                reason="routing_disabled",
            )

        # Classify the task
        classification = self.classifier.classify(message, tool_mentions)
        logger.debug(
            f"Task classified as {classification.task_type.value} "
            f"(conf={classification.confidence:.2f})"
        )

        # Find matching rule
        for rule in self.rules:
            if rule.task_type == classification.task_type:
                if classification.confidence >= rule.min_confidence:
                    return ProviderConfig(
                        provider=rule.provider,
                        model=rule.model,
                        base_url=rule.base_url or self.default_base_url,
                        reason=f"rule:{classification.task_type.value}",
                    )

        # No match, use default
        return ProviderConfig(
            provider=self.default_provider,
            model=self.default_model,
            base_url=self.default_base_url,
            reason=f"default:{classification.task_type.value}",
        )

    def get_fallback(self, failed_provider: str) -> Optional[ProviderConfig]:
        """Get next provider in fallback chain after a failure."""
        found_failed = False
        for provider in self.fallback_chain:
            if found_failed:
                logger.info(f"Falling back from {failed_provider} to {provider.provider}")
                return provider
            if provider.provider == failed_provider:
                found_failed = True

        logger.warning(f"No fallback available after {failed_provider}")
        return None
