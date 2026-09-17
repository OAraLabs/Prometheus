"""Phase 2: the shrink re-tune, and images on the OpenAI-compatible wire.

Spec: docs/sprints/SPRINT-image-blocks.md (#268).

Two changes, tested separately because they fail separately:

  2a — shrink only ABOVE 2x the cap. PyMuPDF halves, so shrinking anything merely
       over the cap overshoots: the wire test's 2638x1646 screenshot (62px past
       2576) came out 1319x823, and Claude then misread "Chats 118" as "Chats 115"
       and "LVX" as "LYX". Both directions are asserted — pass-through near the cap
       AND still-shrinking well above it — because a rule with only one side tested
       is half a rule.

  2b — `image_url` on the OpenAI-compatible path, gated on the provider's DECLARED
       capability. The refusal is tested hardest: an image reaching a provider that
       did not declare vision must RAISE, naming what and where, never degrade to
       text and never silently drop.
"""

from __future__ import annotations

import base64
import re

import pytest

from prometheus.engine.messages import ConversationMessage, ImageBlock, TextBlock, ToolResultBlock
from prometheus.gateway import image_prep
from prometheus.providers.base import ApiMessageRequest
from prometheus.providers.stub import UnsupportedContentBlock, _build_openai_messages


def _png(width: int, height: int, path) -> str:
    """A REAL png of exactly these dimensions, written to disk.

    Generated rather than hand-rolled: the code under test decodes the file and reads
    its true pixel size, so a fixture that only claims a size would test nothing.
    """
    import fitz

    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), False)
    pix.clear_with(255)
    p = path / f"{width}x{height}.png"
    p.write_bytes(pix.tobytes(output="png"))
    return str(p)


def _dims(data_b64: str) -> tuple[int, int]:
    import fitz

    pix = fitz.Pixmap(base64.b64decode(data_b64))
    return pix.width, pix.height


# ── 2a: the shrink re-tune ──────────────────────────────────────────────────

def test_an_image_just_over_the_cap_is_sent_as_captured(tmp_path):
    """THE REGRESSION. 2638 is 62px past the 2576 cap; halving it to 1319 threw away
    half the resolution and cost legibility on small UI text."""
    src = _png(2638, 1646, tmp_path)
    block = image_prep.prepare_image_block(src)
    assert block is not None
    assert _dims(block.data) == (2638, 1646), "a 2% overshoot must not cost 50% of the pixels"


def test_an_image_at_the_trigger_is_still_sent_as_captured(tmp_path):
    src = _png(image_prep.SHRINK_ABOVE, 100, tmp_path)
    block = image_prep.prepare_image_block(src)
    assert _dims(block.data)[0] == image_prep.SHRINK_ABOVE


def test_an_image_well_over_the_trigger_still_shrinks(tmp_path):
    """The other half of the rule. A 4x-oversized image genuinely needs bounding
    before it crosses the socket — dropping the shrink entirely is not the fix."""
    src = _png(image_prep.SHRINK_ABOVE * 2 + 10, 200, tmp_path)
    block = image_prep.prepare_image_block(src)
    w, _ = _dims(block.data)
    assert w < image_prep.SHRINK_ABOVE * 2, f"not shrunk at all (w={w})"
    assert w <= image_prep.SHRINK_ABOVE, f"shrunk but still over the trigger (w={w})"


def test_a_small_image_is_byte_identical(tmp_path):
    """No needless re-encode: JPEG artifacts on small text are what this avoids."""
    src = _png(800, 600, tmp_path)
    block = image_prep.prepare_image_block(src)
    assert base64.b64decode(block.data) == open(src, "rb").read()


# ── 2b: the OpenAI-compatible wire ──────────────────────────────────────────

PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _image_turn() -> ConversationMessage:
    return ConversationMessage(
        role="user",
        content=[
            TextBlock(text="what is in this screenshot?"),
            ImageBlock(media_type="image/png", data=PNG_B64, source_path="/cache/x.png"),
        ],
    )


def test_a_declared_provider_gets_the_image_url_shape():
    out = _build_openai_messages(
        ApiMessageRequest(model="m", messages=[_image_turn()]), allow_images=True
    )
    content = out[0]["content"]
    assert isinstance(content, list), "an image forces the array content form"
    assert content[0] == {"type": "text", "text": "what is in this screenshot?"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{PNG_B64}"


def test_it_does_not_emit_anthropics_shape():
    """The two cloud forms are not interchangeable; sending either to the other's
    endpoint is a 400 that reads like an outage."""
    out = _build_openai_messages(
        ApiMessageRequest(model="m", messages=[_image_turn()]), allow_images=True
    )
    assert '"source"' not in repr(out) and "'source'" not in repr(out)


def test_an_undeclared_provider_RAISES_and_says_what_and_where():
    """The direction to hold hardest. Never degrade to text, never drop."""
    with pytest.raises(UnsupportedContentBlock) as exc:
        _build_openai_messages(ApiMessageRequest(model="m", messages=[_image_turn()]), allow_images=False)
    assert "image/png" in str(exc.value)
    assert "/cache/x.png" in str(exc.value)


def test_the_default_is_refusal():
    """Absence is not permission: a caller that forgets the flag gets the raise."""
    with pytest.raises(UnsupportedContentBlock):
        _build_openai_messages(ApiMessageRequest(model="m", messages=[_image_turn()]))


def test_text_only_turns_keep_the_string_content_form():
    """Byte-compatibility for every request that has no picture in it."""
    plain = ConversationMessage(role="user", content=[TextBlock(text="hello")])
    result = ConversationMessage(role="user", content=[ToolResultBlock(tool_use_id="t1", content="42")])
    out = _build_openai_messages(
        ApiMessageRequest(model="m", messages=[plain, result]), allow_images=True
    )
    assert out[0] == {"role": "user", "content": "hello"}
    assert out[1] == {"role": "tool", "tool_call_id": "t1", "content": "42"}


def test_the_provider_carries_its_declared_flag_not_a_guess():
    from prometheus.providers.openai_compat import OpenAICompatProvider

    declared = OpenAICompatProvider(base_url="http://x", api_key="k", model="qwen-vl-max", vision=True)
    silent = OpenAICompatProvider(base_url="http://x", api_key="k", model="qwen-vl-max")
    assert declared.supports_vision is True
    assert silent.supports_vision is False, (
        "a model whose NAME looks multimodal but declares nothing is text-only — "
        "the *-vl-* heuristic is the design this sprint rejected"
    )


# ── 2c: what is declared, and on what evidence ──────────────────────────────

def test_no_openai_compat_preset_declares_vision_without_evidence():
    """Phase 2 ships the MECHANISM; it declares no new capability.

    Nothing in this repo's configured catalog evidences that any openai_compat model
    accepts images, and the one claim that existed was a confabulation: asked which
    vision model was in use, the daemon cited a llama.cpp /props probe — which reads
    the LOCAL endpoint — while naming the qwen cloud provider (see #268's field note).
    False until evidenced, per that note.
    """
    from prometheus.router.model_router import OVERRIDE_PRESETS

    declared = {k for k, v in OVERRIDE_PRESETS.items() if v.get("vision")}
    assert declared == {"claude"}, f"undocumented vision declarations: {declared - {'claude'}}"


# ── 2d: the assistant branch, which read neither image_parts nor raised ─────
#
# Found while reading this builder for #355 (an unrelated media-marker 400). Not
# that bug — noted here so the provenance is not mistaken for a diagnosis.

def _assistant_turn_with_a_picture() -> ApiMessageRequest:
    return ApiMessageRequest(
        model="m",
        messages=[
            ConversationMessage(
                role="assistant",
                content=[
                    TextBlock(text="here is what I found"),
                    ImageBlock(
                        media_type="image/png",
                        data=PNG_B64,
                        source_path="/cache/from-an-assistant.png",
                    ),
                ],
            )
        ],
    )


def test_an_image_on_an_assistant_turn_raises_instead_of_vanishing():
    """The assistant branch read text_parts and tool_calls and nothing else.

    So an ImageBlock on an assistant message was collected into `image_parts` and
    then never read — dropped on the floor, and ONLY on this path: the user and
    tool-result branches both hand `image_parts` to `_user_content`.

    Latent today, because uploads attach to a USER turn (`ws_server.py`), which is
    why nothing caught it. It is still the exact shape 2b tests hardest everywhere
    else, and the reason given there applies unchanged.
    """
    with pytest.raises(UnsupportedContentBlock) as exc:
        _build_openai_messages(_assistant_turn_with_a_picture(), allow_images=True)

    # The operator has to be able to act on it: which turn, and what to do.
    assert "ASSISTANT" in str(exc.value)
    assert "user messages only" in str(exc.value)


def test_the_assistant_image_can_never_become_a_clean_looking_output():
    """Pin the FAILURE the raise replaces, not just the raise.

    Written the way 2b's mislabelling test is: if a future change stops raising,
    the wrong-shaped output must still be impossible to produce. A message that
    serialises without error and without the picture is the silent drop — it looks
    entirely healthy, which is what makes it expensive.
    """
    try:
        out = _build_openai_messages(_assistant_turn_with_a_picture(), allow_images=True)
    except UnsupportedContentBlock:
        return

    serialised = repr(out)
    assert PNG_B64 in serialised, (
        f"the assistant turn serialised cleanly and the picture is gone: {out}"
    )


def test_a_user_turn_with_the_same_picture_still_carries_it():
    """The other direction — the guard must not become a ban on images.

    Without this, 'raise on every ImageBlock' would satisfy the two tests above
    and break the capability 2b exists to deliver.
    """
    req = ApiMessageRequest(
        model="m",
        messages=[
            ConversationMessage(
                role="user",
                content=[
                    TextBlock(text="what is this?"),
                    ImageBlock(
                        media_type="image/png",
                        data=PNG_B64,
                        source_path="/cache/from-a-user.png",
                    ),
                ],
            )
        ],
    )
    out = _build_openai_messages(req, allow_images=True)
    assert len(out) == 1 and out[0]["role"] == "user"
    kinds = [part["type"] for part in out[0]["content"]]
    assert kinds == ["text", "image_url"]
    assert PNG_B64 in out[0]["content"][1]["image_url"]["url"]
