"""Phase 1 of the image-block sprint: a picture can be CARRIED and SERIALISED,
and a provider that cannot express one says so instead of mangling it.

Spec: docs/sprints/SPRINT-image-blocks.md

Phase 1 changes no behaviour on its own — nothing constructs an ImageBlock yet.
What it establishes is the two properties every later phase leans on:

  * anthropic renders the block in Anthropic's own shape (NOT the OpenAI
    `image_url` form — that difference is why VisionTool's hand-built dict
    cannot work on this provider)
  * every other provider RAISES. The bare `else` this replaced treated any
    unrecognised block as a ToolResultBlock, so an ImageBlock would have gone
    out as a tool result with garbage fields: a corrupted message, delivered,
    with no error anywhere. That is the failure shape the sprint exists to
    remove, so it is the one pinned hardest here.
"""

from __future__ import annotations

import pytest

from prometheus.engine.messages import (
    ConversationMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
)
from prometheus.providers.base import ApiMessageRequest
from prometheus.providers.stub import UnsupportedContentBlock, _build_openai_messages

# 1x1 transparent PNG.
PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _image_message() -> ConversationMessage:
    return ConversationMessage(
        role="user",
        content=[
            ImageBlock(media_type="image/png", data=PNG, source_path="/cache/x.png"),
            TextBlock(text="what is in this screenshot?"),
        ],
    )


# ── the union carries it ────────────────────────────────────────────────────

def test_a_conversation_message_can_hold_an_image():
    msg = _image_message()
    assert [b.type for b in msg.content] == ["image", "text"]


def test_it_round_trips_through_the_discriminated_union():
    """History persists blocks as JSON; a block that cannot be re-parsed is a
    picture that survives the request and dies in the transcript."""
    restored = ConversationMessage.model_validate_json(_image_message().model_dump_json())
    block = restored.content[0]
    assert isinstance(block, ImageBlock)
    assert (block.media_type, block.data, block.source_path) == (
        "image/png", PNG, "/cache/x.png",
    )


# ── anthropic renders Anthropic's shape ─────────────────────────────────────

def test_anthropic_emits_the_documented_image_shape():
    from prometheus.providers.anthropic import _build_anthropic_messages

    out = _build_anthropic_messages([_image_message()])
    blocks = out[0]["content"]
    assert blocks[0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG},
    }
    # The text that came with it still rides along, after the image.
    assert blocks[1] == {"type": "text", "text": "what is in this screenshot?"}


def test_anthropic_does_not_emit_the_openai_shape():
    """The two cloud forms are not interchangeable. Emitting `image_url` here is
    the exact mistake VisionTool makes today."""
    from prometheus.providers.anthropic import _build_anthropic_messages

    rendered = repr(_build_anthropic_messages([_image_message()]))
    assert "image_url" not in rendered


def test_anthropic_declares_vision():
    from prometheus.providers.anthropic import AnthropicProvider

    assert AnthropicProvider.supports_vision is True


# ── everything else refuses, loudly ─────────────────────────────────────────

def test_the_openai_builder_raises_on_an_image():
    req = ApiMessageRequest(model="m", messages=[_image_message()])
    with pytest.raises(UnsupportedContentBlock) as exc:
        _build_openai_messages(req)
    # The operator has to be able to tell WHICH picture and WHY.
    assert "image/png" in str(exc.value)
    assert "/cache/x.png" in str(exc.value)


def test_the_image_is_never_mislabelled_as_a_tool_result():
    """The regression that motivated the raise: the old bare `else` would have
    read .tool_use_id / .content off an ImageBlock and shipped it as a tool
    message. Assert the wrong-shaped output cannot be produced at all."""
    req = ApiMessageRequest(model="m", messages=[_image_message()])
    try:
        out = _build_openai_messages(req)
    except UnsupportedContentBlock:
        return
    assert not any(m.get("role") == "tool" for m in out), (
        f"an ImageBlock was serialised as a tool message: {out}"
    )


def test_ordinary_messages_are_untouched():
    """Phase 1 must be byte-identical for everything that is not an image."""
    plain = ConversationMessage(role="user", content=[TextBlock(text="hello")])
    result = ConversationMessage(
        role="user",
        content=[ToolResultBlock(tool_use_id="toolu_1", content="42")],
    )
    out = _build_openai_messages(ApiMessageRequest(model="m", messages=[plain, result]))
    assert out[0] == {"role": "user", "content": "hello"}
    assert out[1] == {"role": "tool", "tool_call_id": "toolu_1", "content": "42"}


# ── the catalog declares capability rather than implying it ─────────────────

def test_the_preset_declares_vision_and_absence_means_false():
    from prometheus.router.model_router import OVERRIDE_PRESETS

    assert OVERRIDE_PRESETS["claude"]["vision"] is True
    # Every other preset is silent, and silence must read as no.
    for key, preset in OVERRIDE_PRESETS.items():
        if key == "claude":
            continue
        assert bool(preset.get("vision", False)) is False, key


def test_no_preset_infers_vision_from_its_model_name():
    """A `*-vl-*` heuristic is the design this sprint rejected: it gets the next
    model that breaks the convention wrong, and the failure is a dropped
    picture. If a vision-named model ever appears without a declared flag, this
    goes red rather than quietly working until it doesn't."""
    from prometheus.router.model_router import OVERRIDE_PRESETS

    for key, preset in OVERRIDE_PRESETS.items():
        looks_visual = "-vl" in str(preset.get("model", "")).lower()
        if looks_visual:
            assert "vision" in preset, (
                f"{key} looks multimodal by name but declares nothing — declare it"
            )
