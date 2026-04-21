"""Tests for Model Router — TaskClassifier + ModelRouter."""

import pytest

from prometheus.adapter.router import TaskClassifier, TaskType, ModelRouter


class TestTaskClassifier:
    """Test token-based task classification."""

    def test_code_generation(self):
        c = TaskClassifier()
        result = c.classify("Write a Python function to parse JSON")
        assert result.task_type == TaskType.CODE_GENERATION
        assert result.confidence > 0.3
        assert "python" in result.matched_tokens or "write" in result.matched_tokens

    def test_quick_answer(self):
        c = TaskClassifier()
        result = c.classify("What is the capital of France?")
        assert result.task_type == TaskType.QUICK_ANSWER

    def test_reasoning(self):
        c = TaskClassifier()
        result = c.classify(
            "Explain the trade-offs between microservices and monoliths"
        )
        assert result.task_type == TaskType.REASONING
        assert "tradeoffs" in result.matched_tokens or "explain" in result.matched_tokens

    def test_tool_heavy(self):
        c = TaskClassifier()
        result = c.classify(
            "Search for recent news about AI", tool_mentions=["web_search"]
        )
        assert result.task_type == TaskType.TOOL_HEAVY

    def test_creative(self):
        c = TaskClassifier()
        result = c.classify("Write a short story about a robot learning to love")
        # "write" overlaps CODE_GENERATION and "story" overlaps CREATIVE
        assert result.task_type in (TaskType.CREATIVE, TaskType.CODE_GENERATION)

    def test_short_message_boosts_quick(self):
        c = TaskClassifier()
        result = c.classify("hi")
        # Short messages should boost QUICK_ANSWER
        assert result.task_type == TaskType.QUICK_ANSWER or result.confidence < 0.5

    def test_code_block_boosts_code(self):
        c = TaskClassifier()
        result = c.classify("Fix this: ```python\nprint('hello')\n```")
        assert result.task_type == TaskType.CODE_GENERATION

    def test_empty_message(self):
        c = TaskClassifier()
        result = c.classify("")
        # Empty message triggers short-message boost → QUICK_ANSWER
        assert result.task_type == TaskType.QUICK_ANSWER

    def test_classification_returns_reason(self):
        c = TaskClassifier()
        result = c.classify("Explain why Python is popular")
        assert "tokens=" in result.reason
        assert "len=" in result.reason
        assert "conf=" in result.reason


class TestModelRouter:
    """Test routing and fallback logic."""

    def test_routing_disabled(self):
        config = {
            "model_router": {"enabled": False},
            "model": {"provider": "llama_cpp"},
        }
        router = ModelRouter(config)
        result = router.route("any message")
        assert result.reason == "routing_disabled"
        assert result.provider == "llama_cpp"

    def test_routing_with_rules(self):
        config = {
            "model_router": {
                "enabled": True,
                "rules": [
                    {
                        "task_type": "code_generation",
                        "provider": "llama_cpp",
                        "model": "qwen",
                    }
                ],
            },
            "model": {},
        }
        router = ModelRouter(config)
        result = router.route("Write a Python script")
        assert result.provider == "llama_cpp"
        assert "rule" in result.reason

    def test_forced_provider(self):
        config = {"model_router": {"enabled": True}, "model": {}}
        router = ModelRouter(config)
        result = router.route("anything", force_provider="anthropic")
        assert result.provider == "anthropic"
        assert "forced" in result.reason

    def test_fallback_chain(self):
        config = {
            "model_router": {
                "enabled": True,
                "fallback_chain": [
                    {"provider": "llama_cpp"},
                    {"provider": "ollama"},
                    {"provider": "anthropic"},
                ],
            },
            "model": {},
        }
        router = ModelRouter(config)

        fb1 = router.get_fallback("llama_cpp")
        assert fb1 is not None
        assert fb1.provider == "ollama"

        fb2 = router.get_fallback("ollama")
        assert fb2 is not None
        assert fb2.provider == "anthropic"

        fb3 = router.get_fallback("anthropic")
        assert fb3 is None  # End of chain

    def test_no_matching_rule_uses_default(self):
        config = {
            "model_router": {
                "enabled": True,
                "rules": [],  # No rules
            },
            "model": {
                "provider": "default_provider",
                "model": "default_model",
            },
        }
        router = ModelRouter(config)
        result = router.route("some random message")
        assert result.provider == "default_provider"
        assert "default" in result.reason

    def test_min_confidence_filtering(self):
        config = {
            "model_router": {
                "enabled": True,
                "rules": [
                    {
                        "task_type": "code_generation",
                        "provider": "llama_cpp",
                        "model": "qwen",
                        "min_confidence": 0.99,  # Very high threshold
                    }
                ],
            },
            "model": {"provider": "default_prov", "model": "default_mod"},
        }
        router = ModelRouter(config)
        # Even a code-like message shouldn't meet 0.99 confidence
        result = router.route("Write a script")
        assert "default" in result.reason

    def test_empty_config(self):
        router = ModelRouter({})
        result = router.route("hello")
        assert result.reason == "routing_disabled"
        assert result.provider == "llama_cpp"  # hardcoded default
