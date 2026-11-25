"""Origin-scoped toolsets — the credential decides, the gate enforces (Phase 2).

The far side is the provider request and the tool call, through the REAL
run_loop against the REAL registry (the test_profile_wiring tradition):

* a voice_box turn advertises only the voice-box set — no bash, no
  write_file, no computer use — and a call naming one is refused even
  though the tool is registered;
* a carried PTT origin adds messaging, nothing else;
* the terminal origin is unchanged (the full kit);
* origin comes from the credential: a forged ``origin`` in a request body
  changes nothing, and the envelope route refuses the field by name;
* an unmarked scoped device keeps today's typed-chat behaviour; on the
  voice event route it fails closed to voice_box.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.__main__ import create_tool_registry
from prometheus.config.api_token import DeviceIdentity
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.permissions.origins import (
    capability_profile,
    origin_for_identity,
    voice_origin_for,
)
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)


class _RecordingProvider(ModelProvider):
    def __init__(self) -> None:
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest):
        self.requests.append(request)
        msg = ConversationMessage(role="assistant", content=[TextBlock(text="ok")])
        yield ApiTextDeltaEvent(text="ok")
        yield ApiMessageCompleteEvent(message=msg, usage=UsageSnapshot(), stop_reason="stop")


def _drive(context: LoopContext) -> None:
    async def _run():
        async for _event, _usage in run_loop(context, [ConversationMessage.from_user_text("go")]):
            pass

    asyncio.run(_run())


def _context(provider, origin: str | None) -> LoopContext:
    resolver = None
    if origin:
        profile = capability_profile(origin)
        resolver = lambda *a, _p=profile: _p
    return LoopContext(
        provider=provider, model="test", system_prompt="BASE", max_tokens=64,
        tool_registry=create_tool_registry({}), session_id="voice:dev:box",
        profile_resolver=resolver,
    )


def _advertised(provider) -> set[str]:
    return {t["name"] for t in provider.requests[0].tools}


# ── the credential decides ───────────────────────────────────────────────────

def test_global_and_owner_are_terminal():
    assert origin_for_identity(DeviceIdentity(id="global")) == "terminal"
    assert origin_for_identity(DeviceIdentity(id="d1", owner=True)) == "terminal"


def test_unmarked_scoped_device_is_unchanged_on_typed_routes():
    assert origin_for_identity(DeviceIdentity(id="d1")) is None


def test_the_mark_decides_and_an_unknown_mark_grants_nothing():
    assert origin_for_identity(DeviceIdentity(id="d1"), marked_origin="android_ptt") == "android_ptt"
    assert origin_for_identity(DeviceIdentity(id="d1"), marked_origin="root") is None


def test_voice_events_fail_closed_to_voice_box():
    assert voice_origin_for(DeviceIdentity(id="d1")) == "voice_box"
    assert voice_origin_for(DeviceIdentity(id="d1"), marked_origin="glasses") == "glasses"
    assert voice_origin_for(DeviceIdentity(id="global")) == "terminal"


def test_capability_profiles_are_held_and_copied():
    p1 = capability_profile("voice_box")
    p2 = capability_profile("voice_box")
    assert p1.held and p1 is not p2
    assert capability_profile("terminal") is None


# ── the far side: what the provider sees ─────────────────────────────────────

def test_voice_box_advertises_only_the_voice_set():
    provider = _RecordingProvider()
    _drive(_context(provider, "voice_box"))
    advertised = _advertised(provider)
    registry = {t.name for t in create_tool_registry({}).list_tools()}
    expected = set(capability_profile("voice_box").tools) & registry
    assert advertised == expected
    for forbidden in ("bash", "write_file", "edit_file", "download_file", "agent",
                      "sessions_spawn", "task_create", "cron_delete", "message"):
        assert forbidden not in advertised, forbidden


def test_ptt_origins_add_messaging_only():
    for origin in ("android_ptt", "glasses", "ios_ptt"):
        provider = _RecordingProvider()
        _drive(_context(provider, origin))
        advertised = _advertised(provider)
        box_provider = _RecordingProvider()
        _drive(_context(box_provider, "voice_box"))
        assert advertised - _advertised(box_provider) == {"message"}, origin
        assert "bash" not in advertised and "write_file" not in advertised


def test_terminal_is_the_full_kit():
    provider = _RecordingProvider()
    _drive(_context(provider, None))
    advertised = _advertised(provider)
    registry = {t.name for t in create_tool_registry({}).list_tools()}
    assert advertised == registry


# ── the call path: a named-but-withheld tool is refused ──────────────────────

class _ToolCallingProvider(ModelProvider):
    """One round naming bash, then a text round — the call path must refuse it."""

    def __init__(self) -> None:
        self.calls = 0
        self.results: list = []

    async def stream_message(self, request):
        from prometheus.engine.messages import ToolUseBlock
        self.calls += 1
        if self.calls == 1:
            msg = ConversationMessage(role="assistant", content=[
                ToolUseBlock(id="t1", name="bash", input={"command": "echo hi"})])
            yield ApiMessageCompleteEvent(message=msg, usage=UsageSnapshot(), stop_reason="tool_use")
        else:
            msg = ConversationMessage(role="assistant", content=[TextBlock(text="done")])
            yield ApiTextDeltaEvent(text="done")
            yield ApiMessageCompleteEvent(message=msg, usage=UsageSnapshot(), stop_reason="stop")


def test_a_voice_turn_cannot_call_bash_even_when_the_model_names_it():
    from prometheus.engine.messages import ToolResultBlock

    provider = _ToolCallingProvider()
    context = _context(provider, "voice_box")
    messages = [ConversationMessage.from_user_text("go")]

    async def _run():
        async for _e, _u in run_loop(context, messages):
            pass

    asyncio.run(_run())
    refusals = [b for m in messages for b in (m.content if isinstance(m.content, list) else [])
                if isinstance(b, ToolResultBlock)]
    assert refusals, "the bash call must produce a tool result"
    assert all(b.is_error for b in refusals)
    assert any("not available" in str(b.content) for b in refusals)
