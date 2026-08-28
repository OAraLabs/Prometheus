"""stream_round_with_fallback — the loop-facing wrapper (SPRINT-provider-fallback, Phase 3)."""

from __future__ import annotations

import pytest

from prometheus.engine.fallback import (
    FallbackRefused,
    FallbackTarget,
    stream_round_with_fallback,
)


class Boom(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")

        class R:
            status_code = status

        self.response = R()


class FakeEnvelope:
    """Records which provider served each attempt and what model was in the request."""

    def __init__(self, script):
        self.script = list(script)
        self.attempts: list[tuple[str, str]] = []

    def stream(self, *, provider, request, **_):
        self.attempts.append((provider, request))
        behaviour = self.script.pop(0)

        async def gen():
            for step in behaviour:
                if isinstance(step, Exception):
                    raise step
                yield step

        return gen()


LOCAL = FallbackTarget(model="Qwen3.8-27B", provider_name="llama_cpp",
                       provider="LOCAL_PROVIDER", is_local_backend=True)


async def drive(envelope, *, target=LOCAL, enabled=True, window=32_768, needed=8_000):
    degrades = []
    seen = []
    agen = stream_round_with_fallback(
        envelope=envelope,
        provider="CLOUD_PROVIDER",
        model="qwen3.8-max",
        build_request=lambda m: m,          # the request IS the model name here
        target=target,
        enabled=enabled,
        window_for=lambda m: (window, True),
        estimate_tokens=lambda: needed,
        on_degrade=degrades.append,
        operation="loop_round",
        round_index=0,
        session_id="s1",
    )
    async for ev in agen:
        seen.append(ev)
    return seen, degrades


@pytest.mark.asyncio
async def test_a_healthy_round_passes_through_untouched():
    env = FakeEnvelope([["a", "b"]])
    seen, degrades = await drive(env)
    assert seen == ["a", "b"]
    assert degrades == [], "no degrade on a healthy round"
    assert len(env.attempts) == 1


@pytest.mark.asyncio
async def test_a_terminal_failure_before_output_degrades_and_says_so():
    env = FakeEnvelope([[Boom(401)], ["recovered"]])
    seen, degrades = await drive(env)
    # The notice precedes the fallback's content — see the dedicated test below.
    assert seen[-1] == "recovered"
    assert len(degrades) == 1
    assert "Qwen3.8-27B" in degrades[0].message
    assert degrades[0].serve


@pytest.mark.asyncio
async def test_the_retry_is_rebuilt_for_the_FALLBACK_model():
    """Reusing the failed request would ask the local backend to serve `qwen3.8-max`."""
    env = FakeEnvelope([[Boom(401)], ["ok"]])
    await drive(env)
    assert env.attempts[0] == ("CLOUD_PROVIDER", "qwen3.8-max")
    assert env.attempts[1] == ("LOCAL_PROVIDER", "Qwen3.8-27B"), "request rebuilt for the target"


@pytest.mark.asyncio
async def test_output_already_sent_refuses_instead_of_re_answering():
    env = FakeEnvelope([["half an answer", Boom(401)], ["a different answer"]])
    with pytest.raises(FallbackRefused) as ei:
        await drive(env)
    assert "already been sent" in str(ei.value)
    assert len(env.attempts) == 1, "the fallback must not have been attempted"


@pytest.mark.asyncio
async def test_a_turn_too_large_for_the_fallback_refuses_with_numbers():
    env = FakeEnvelope([[Boom(401)]])
    with pytest.raises(FallbackRefused) as ei:
        await drive(env, needed=118_000, window=32_768)
    assert "118,000" in str(ei.value) and "32,768" in str(ei.value)
    assert len(env.attempts) == 1


@pytest.mark.asyncio
async def test_a_rate_limit_is_re_raised_unchanged():
    """429 is retried a layer down; degrading here abandons a provider about to succeed."""
    env = FakeEnvelope([[Boom(429)]])
    with pytest.raises(Boom):
        await drive(env)
    assert len(env.attempts) == 1


@pytest.mark.asyncio
async def test_no_target_configured_re_raises_the_providers_own_error():
    env = FakeEnvelope([[Boom(401)]])
    with pytest.raises(Boom):
        await drive(env, target=None)


@pytest.mark.asyncio
async def test_disabled_re_raises():
    env = FakeEnvelope([[Boom(401)]])
    with pytest.raises(Boom):
        await drive(env, enabled=False)


@pytest.mark.asyncio
async def test_the_fallback_gets_no_fallback_of_its_own():
    """A chain turns "which model answered?" into archaeology, and the second hop would degrade
    away from a model that was itself the degrade."""
    env = FakeEnvelope([[Boom(401)], [Boom(401)]])
    with pytest.raises(Boom):
        await drive(env)
    assert len(env.attempts) == 2, "exactly one degrade attempted, then the failure stands"


@pytest.mark.asyncio
async def test_the_fallbacks_own_failure_is_not_chained_to_the_first():
    """Raised outside the except block, so it reports as itself rather than as "during handling
    of the above exception"."""
    env = FakeEnvelope([[Boom(401)], [Boom(503)]])
    with pytest.raises(Boom) as ei:
        await drive(env)
    assert "503" in str(ei.value), "the surfaced error is the fallback's, not the original"


@pytest.mark.asyncio
async def test_the_degrade_notice_is_in_the_REPLY_not_only_a_callback():
    """A notice the reader has to go looking for is a silent degrade."""
    from prometheus.providers.base import ApiTextDeltaEvent

    env = FakeEnvelope([[Boom(401)], ["the answer"]])
    seen, _ = await drive(env)
    notices = [e for e in seen if isinstance(e, ApiTextDeltaEvent)]
    assert len(notices) == 1, "exactly one notice, ahead of the answer"
    assert "Qwen3.8-27B" in notices[0].text and "qwen3.8-max" in notices[0].text
    assert seen.index(notices[0]) == 0, "it precedes the fallback's content"


@pytest.mark.asyncio
async def test_no_notice_when_nothing_degraded():
    from prometheus.providers.base import ApiTextDeltaEvent

    env = FakeEnvelope([["a", "b"]])
    seen, _ = await drive(env)
    assert not any(isinstance(e, ApiTextDeltaEvent) for e in seen)
