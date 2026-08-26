"""Phase 3a: an image survives storage without putting its bytes in the transcript.

Spec: docs/sprints/SPRINT-image-blocks.md §3.4.

`content_json` is written to LCM, returned by every history fetch, and pushed over
the wire to every client. A base64 payload in there is the same compounding cost the
12k-char description had, only larger — so the payload is dropped whenever
`source_path` can bring it back, and refilled on the way to a model.

The degradation order is the part to hold: bytes → description → marker. A picture
the cache evicted must not take a turn down, and must not silently become an EMPTY
image block either, which reaches a provider as a malformed request.
"""

from __future__ import annotations

import base64
import json

from prometheus.engine.messages import ConversationMessage, ImageBlock, TextBlock

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 64).decode("ascii")


def _msg(**kw) -> ConversationMessage:
    return ConversationMessage(
        role="user",
        content=[TextBlock(text="what is this?"), ImageBlock(media_type="image/png", **kw)],
    )


def test_the_payload_is_not_written_to_the_transcript(tmp_path):
    src = tmp_path / "shot.png"
    src.write_bytes(base64.b64decode(PNG))
    cj = _msg(data=PNG, source_path=str(src)).content_json
    assert PNG not in cj, "the base64 payload reached content_json — it will be in LCM, every history fetch, and every later turn"
    assert str(src) in cj, "the reference that makes dropping the payload safe is missing"


def test_a_block_with_no_source_path_keeps_its_payload():
    """Dropping the bytes is only safe because they can be recovered. With nowhere
    to recover them from, dropping is data loss."""
    cj = _msg(data=PNG).content_json
    assert PNG in cj


def test_it_round_trips_back_into_a_message(tmp_path):
    src = tmp_path / "shot.png"
    src.write_bytes(base64.b64decode(PNG))
    cj = _msg(data=PNG, source_path=str(src)).content_json
    back = ConversationMessage(role="user", content=json.loads(cj))
    block = back.content[1]
    assert isinstance(block, ImageBlock)
    assert block.data == "" and block.source_path == str(src)


def test_rehydrate_refills_the_bytes_from_disk(tmp_path):
    src = tmp_path / "shot.png"
    raw = base64.b64decode(PNG)
    src.write_bytes(raw)
    stored = ConversationMessage(
        role="user", content=json.loads(_msg(data=PNG, source_path=str(src)).content_json)
    ).content[1]
    live = stored.rehydrate()
    assert base64.b64decode(live.data) == raw, "the picture did not come back"


def test_rehydrate_on_a_missing_file_does_not_raise(tmp_path):
    """A cache eviction must not take a turn down."""
    stored = ImageBlock(media_type="image/png", source_path=str(tmp_path / "gone.png"))
    out = stored.rehydrate()
    assert out.data == ""


def test_a_lost_picture_degrades_to_its_description_then_to_a_marker(tmp_path):
    with_desc = ImageBlock(
        media_type="image/png", source_path=str(tmp_path / "gone.png"),
        description="a screenshot of the Beacon sessions list",
    )
    assert with_desc.placeholder_text() == "[Image: a screenshot of the Beacon sessions list]"
    bare = ImageBlock(media_type="image/png", source_path=str(tmp_path / "gone.png"))
    assert bare.placeholder_text() == "[Image: unavailable]"


def test_rehydrate_is_a_no_op_when_the_bytes_are_already_there(tmp_path):
    live = ImageBlock(media_type="image/png", data=PNG, source_path=str(tmp_path / "whatever.png"))
    assert live.rehydrate() is live, "re-reading disk for a block that already has its bytes"


def test_other_blocks_are_untouched_by_the_storage_form():
    """Byte-compatibility: only images change shape."""
    plain = ConversationMessage(role="user", content=[TextBlock(text="hello")])
    assert json.loads(plain.content_json) == [{"type": "text", "text": "hello"}]
