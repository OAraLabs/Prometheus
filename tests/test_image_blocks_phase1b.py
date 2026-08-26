"""Phase 1b: the gate, and the caption that must not become a command.

Spec: docs/sprints/SPRINT-image-blocks.md

Three properties, each of which fails silently if it regresses — which is why
each has a test rather than a comment:

  * the gate asks the provider THIS TURN will use, not the process primary
    (the #74 shape: the override decides, and reading the primary answers for a
    model that is not going to run)
  * asking must not ANSWER — no provider is built, and a one-shot override is
    not consumed by a capability check
  * a caption starting with "/" is not a slash command. Before blocks existed
    it could not lead, because "[Image: …]" was prepended; carrying the picture
    in a block makes the caption the whole message.
"""

from __future__ import annotations

import sys
import types

# Same circular-import shim the sibling web tests use.
if "prometheus.memory" not in sys.modules:  # pragma: no cover - import plumbing
    try:
        import prometheus.memory  # noqa: F401
    except Exception:
        sys.modules["prometheus.memory"] = types.ModuleType("prometheus.memory")

import pytest  # noqa: E402

from prometheus.engine.messages import ImageBlock  # noqa: E402
from prometheus.providers.registry import provider_class_supports_vision  # noqa: E402
from prometheus.web.ws_server import WebSocketBridge  # noqa: E402


class _Override:
    def __init__(self, cfg: dict) -> None:
        self.provider_config = cfg


class _Router:
    """Records every question asked, so 'no side effects' is checkable."""

    def __init__(self, override: _Override | None) -> None:
        self._override = override
        self.route_calls = 0
        self.lookups: list[str] = []

    def get_override_for_session(self, session_id):
        self.lookups.append(session_id)
        return self._override

    def route(self, message, context=None):  # pragma: no cover - must not run
        self.route_calls += 1
        raise AssertionError("the capability check must not route — routing consumes a one-shot override")


def _bridge(override: _Override | None) -> tuple[WebSocketBridge, _Router]:
    router = _Router(override)
    ctx = types.SimpleNamespace(model_router=router, provider=object())
    return WebSocketBridge(session_mgr=None, loop_context=ctx, config={}), router


ANTHROPIC_VISION = {"provider": "anthropic", "model": "claude-opus-5", "vision": True}


# ── the gate ────────────────────────────────────────────────────────────────

def test_gate_says_yes_for_a_declared_vision_override():
    bridge, router = _bridge(_Override(ANTHROPIC_VISION))
    assert bridge._turn_supports_vision("beacon:s1") is True
    assert router.lookups == ["beacon:s1"], "the gate must ask about THIS session"


def test_gate_reads_the_session_override_not_the_primary():
    """#74's shape. With no override the primary answers, and in Phase 1 the
    primary keeps the description path — so a session that never switched must
    not get the image path just because the process model can see."""
    bridge, _ = _bridge(None)
    assert bridge._turn_supports_vision("beacon:s1") is False


def test_gate_requires_the_declared_flag():
    """A vision-capable PROVIDER is not the same claim as a vision-capable
    MODEL. Absence-is-not-permission (spec Q2)."""
    bridge, _ = _bridge(_Override({"provider": "anthropic", "model": "claude-opus-5"}))
    assert bridge._turn_supports_vision("beacon:s1") is False


def test_gate_requires_the_provider_to_be_able_to_express_an_image():
    """A preset could declare vision for a provider whose serialiser cannot
    carry one — a user's slash_commands block can override `provider` while the
    preset's `vision` stays. Then the block would reach the OpenAI builder and
    raise. The gate refuses first."""
    # The example moved in Phase 2. It used to be `qwen`, because the OpenAI builder
    # raised on every image; that builder can now express one, so the old fixture
    # asserted a fact that had changed. `llama_cpp` still exhibits the property:
    # declared on the model, but not wired into the image-block path. Property kept,
    # example moved.
    bridge, _ = _bridge(_Override({"provider": "llama_cpp", "model": "qwen3.8-27b", "vision": True}))
    assert bridge._turn_supports_vision("beacon:s1") is False


def test_asking_does_not_answer():
    """route() builds and caches a provider, and under overrides.sticky=false it
    CONSUMES the override. A capability check that routed would spend the very
    thing it was asking about."""
    bridge, router = _bridge(_Override(ANTHROPIC_VISION))
    bridge._turn_supports_vision("beacon:s1")
    assert router.route_calls == 0


def test_no_router_means_no():
    bridge = WebSocketBridge(session_mgr=None, loop_context=None, config={})
    assert bridge._turn_supports_vision("beacon:s1") is False


def test_provider_capability_comes_from_the_class():
    assert provider_class_supports_vision("anthropic") is True
    # qwen flipped to True in Phase 2, when the OpenAI builder learned the `image_url`
    # form. That is CAPABILITY — "can our code put a picture on this wire" — not
    # permission; the declared per-model flag still decides (see the gate tests).
    assert provider_class_supports_vision("qwen") is True
    assert provider_class_supports_vision("openai") is True
    # Not wired into the image-block path: these probe their own endpoint instead.
    assert provider_class_supports_vision("llama_cpp") is False
    assert provider_class_supports_vision("ollama") is False
    assert provider_class_supports_vision("nonesuch") is False


# ── the caption is not a command ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_caption_starting_with_slash_is_not_run_as_a_command():
    """The regression this guard exists for: caption a screenshot "/status" and
    the picture disappears into a command reply. Green without the guard only
    because nothing sent a block before."""
    bridge, _ = _bridge(_Override(ANTHROPIC_VISION))
    routed: list[str] = []

    import prometheus.web.slash_router as sr

    async def _boom(content, ctx):
        routed.append(content)
        raise AssertionError("an upload caption must never reach the slash router")

    original = sr.route_slash
    sr.route_slash = _boom
    try:
        # session_mgr is None, so this returns right after the slash decision —
        # which is the decision under test.
        await bridge._handle_send_message("beacon:s1", "/status", blocks=[
            ImageBlock(media_type="image/png", data="AAAA", source_path="/cache/x.png")
        ])
    finally:
        sr.route_slash = original
    assert routed == []


@pytest.mark.asyncio
async def test_a_slash_message_with_no_blocks_still_routes_as_a_command():
    """The guard must be scoped to uploads — plain /status keeps working."""
    bridge, _ = _bridge(None)
    seen: list[str] = []

    import prometheus.web.slash_router as sr

    class _Outcome:
        handled = False
        reply = None

    async def _spy(content, ctx):
        seen.append(content)
        return _Outcome()

    original = sr.route_slash
    sr.route_slash = _spy
    try:
        await bridge._handle_send_message("beacon:s1", "/status")
    finally:
        sr.route_slash = original
    assert seen == ["/status"]


# ── the block has to land on the TURN ───────────────────────────────────────
#
# Every test above returns before the session is touched, so all nine passed
# while the attach lived in the WRONG METHOD entirely — the patch anchor matched
# run_turn_awaited (Paperclip), which contains the same three lines. The full
# suite caught it via a NameError there; nothing here would have. This is that
# test: it drives the real path far enough to see the block land.


class _FakeSession:
    def __init__(self) -> None:
        self.messages: list = []
        self.persisted: list[str] = []

    def add_user_message(self, text, **kw):
        from prometheus.engine.messages import ConversationMessage

        self.persisted.append(text)          # what LCM would store
        self.messages.append(ConversationMessage.from_user_text(text))
        return len(self.messages) - 1

    def last_persisted_row_id(self):
        return 1


class _FakeSessionMgr:
    def __init__(self, session) -> None:
        self._s = session

    def get(self, session_id):
        return self._s

    def get_or_create(self, session_id):
        return self._s


@pytest.mark.asyncio
async def test_blocks_land_on_the_turn_but_not_in_the_persisted_text(monkeypatch):
    session = _FakeSession()
    router = _Router(_Override(ANTHROPIC_VISION))
    ctx = types.SimpleNamespace(model_router=router, provider=object())
    bridge = WebSocketBridge(session_mgr=_FakeSessionMgr(session), loop_context=ctx, config={})

    async def _no_broadcast(payload):
        return None

    async def _no_agent(session_id, session_obj, **kw):
        return None

    monkeypatch.setattr(bridge, "broadcast", _no_broadcast)
    monkeypatch.setattr(bridge, "_run_agent", _no_agent)

    block = ImageBlock(media_type="image/png", data="AAAA", source_path="/cache/x.png")
    await bridge._handle_send_message("beacon:s1", "[Image: shot.png]", blocks=[block])

    assert session.persisted == ["[Image: shot.png]"], (
        "history must keep the marker text — base64 does not belong in LCM"
    )
    kinds = [b.type for b in session.messages[-1].content]
    assert kinds == ["text", "image"], (
        f"the picture did not reach the turn (content was {kinds})"
    )
    assert session.messages[-1].content[1].source_path == "/cache/x.png"


@pytest.mark.asyncio
async def test_a_send_with_no_blocks_leaves_the_turn_text_only(monkeypatch):
    """The additive default: no blocks means byte-identical to before."""
    session = _FakeSession()
    bridge, _ = _bridge(None)
    bridge.session_mgr = _FakeSessionMgr(session)

    async def _no_broadcast(payload):
        return None

    async def _no_agent(session_id, session_obj, **kw):
        return None

    monkeypatch.setattr(bridge, "broadcast", _no_broadcast)
    monkeypatch.setattr(bridge, "_run_agent", _no_agent)

    await bridge._handle_send_message("beacon:s1", "just text")
    assert [b.type for b in session.messages[-1].content] == ["text"]


@pytest.mark.asyncio
async def test_the_marker_survives_a_caption(monkeypatch, tmp_path):
    """History must still say a picture was here.

    Found by the WIRE TEST, not by these tests: with a caption present the first
    implementation persisted the caption ALONE, so the transcript read "What is
    in this screenshot?" with nothing naming a screenshot — a later turn, or
    search, had no idea an image was ever attached.

    This drives _handle_file_upload, which is where the text is COMPOSED. The first
    version of this test called _handle_send_message with a hand-built string
    and asserted on the string it had just passed in — it could not fail, and
    the mutation proved it: restoring `caption or "[Image: …]"` left it green.
    """
    import base64

    import prometheus.gateway.media_cache as media_cache
    import prometheus.gateway.image_prep as image_prep

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    cached = tmp_path / "img_test.png"
    cached.write_bytes(png)
    monkeypatch.setattr(media_cache, "cache_image_from_bytes", lambda data, ext=None: str(cached))
    monkeypatch.setattr(
        image_prep, "prepare_image_block",
        lambda path: ImageBlock(media_type="image/png", data="AAAA", source_path=str(path)),
    )

    router = _Router(_Override(ANTHROPIC_VISION))
    ctx = types.SimpleNamespace(model_router=router, provider=object())
    bridge = WebSocketBridge(session_mgr=_FakeSessionMgr(_FakeSession()), loop_context=ctx, config={})

    sent: list[tuple[str, list]] = []

    async def _capture(session_id, content, **kw):
        sent.append((content, kw.get("blocks") or []))

    monkeypatch.setattr(bridge, "_handle_send_message", _capture)

    await bridge._handle_file_upload(
        "beacon:s1", "shot.png", base64.b64encode(png).decode(), "image/png",
        "what is this?",
    )

    assert sent, "the upload never dispatched"
    content, blocks = sent[0]
    assert "[Image: shot.png]" in content, (
        f"the transcript lost the picture — composed text was {content!r}"
    )
    assert "what is this?" in content, "the caption must survive too"
    assert len(blocks) == 1 and blocks[0].type == "image"


@pytest.mark.asyncio
async def test_an_uncaptioned_upload_still_names_the_file(monkeypatch, tmp_path):
    import base64

    import prometheus.gateway.media_cache as media_cache
    import prometheus.gateway.image_prep as image_prep

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    cached = tmp_path / "img_test.png"
    cached.write_bytes(png)
    monkeypatch.setattr(media_cache, "cache_image_from_bytes", lambda data, ext=None: str(cached))
    monkeypatch.setattr(
        image_prep, "prepare_image_block",
        lambda path: ImageBlock(media_type="image/png", data="AAAA", source_path=str(path)),
    )

    router = _Router(_Override(ANTHROPIC_VISION))
    ctx = types.SimpleNamespace(model_router=router, provider=object())
    bridge = WebSocketBridge(session_mgr=_FakeSessionMgr(_FakeSession()), loop_context=ctx, config={})
    sent: list[str] = []

    async def _capture(session_id, content, **kw):
        sent.append(content)

    monkeypatch.setattr(bridge, "_handle_send_message", _capture)
    await bridge._handle_file_upload(
        "beacon:s1", "shot.png", base64.b64encode(png).decode(), "image/png", "",
    )
    assert sent == ["[Image: shot.png]"]
