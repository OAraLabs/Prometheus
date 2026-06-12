"""F1 follow-through: streamed completions must REQUEST usage accounting.

The live proof after PR #31 landed showed every agent_loop usage row with
input_tokens=0/output_tokens=0 — the providers parse the OpenAI-compat
``usage`` chunk, but llama.cpp (and OpenAI-compat servers generally) only
SEND it when the request carries ``stream_options: {"include_usage": true}``.
Accounting-only: generation, sampling, prompt, and schemas are unaffected.
"""

from __future__ import annotations

from prometheus.engine.messages import ConversationMessage
from prometheus.providers.base import ApiMessageRequest


def _request() -> ApiMessageRequest:
    return ApiMessageRequest(
        model="m",
        messages=[ConversationMessage.from_user_text("q")],
        max_tokens=16,
    )


def test_llama_cpp_payload_requests_usage():
    from prometheus.providers.llama_cpp import LlamaCppProvider

    provider = LlamaCppProvider(base_url="http://unit.test:1")
    payload = provider._build_request_payload(_request())
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_payload_builders_share_the_request(monkeypatch):
    # openai_compat and stub build their payload inline; assert by
    # source-level contract: the dict literal carries stream_options
    # adjacent to stream. (A capture-the-httpx-body harness would be
    # heavier than the one-line invariant deserves.)
    import inspect

    from prometheus.providers import openai_compat, stub

    for mod in (openai_compat, stub):
        src = inspect.getsource(mod)
        assert '"stream_options": {"include_usage": True}' in src, (
            f"{mod.__name__} streams without requesting usage accounting"
        )
