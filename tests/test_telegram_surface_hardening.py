"""Telegram surface controls — breach paths, not happy paths.

These controls were declared in config with NO implementation: no limiter, no
MIME check, no size check on the inbound path. So the tests that matter are the
ones that prove a refusal happens, and — per this repo's most repeated defect —
that the refusal is actually CALLED from the four handlers rather than merely
existing.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from prometheus.gateway import telegram as tg_mod
from prometheus.gateway.guards import (
    ALL_GUARDS,
    Enforcement,
    Guard,
    GuardDeclarationError,
)
from prometheus.gateway.media_guard import (
    MediaPolicy,
    MediaRejected,
    MediaTooLarge,
    check_declared_mime,
    check_size_precheck,
    check_sniffed_mime,
    enforce_byte_ceiling,
    sniff_mime,
    validate_inbound,
)
from prometheus.gateway.rate_limit import Budget, RateLimiter

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
PDF = b"%PDF-1.7" + b"\x00" * 64

POLICY = MediaPolicy(
    allowed_image_types=("image/jpeg", "image/png"),
    allowed_audio_types=("audio/ogg",),
    allowed_document_types=("application/pdf",),
    max_file_size_mb=1,
)


# ── classification is declared, not inferred ────────────────────────────────


def test_every_guard_declares_its_enforcement():
    for g in ALL_GUARDS:
        assert isinstance(g.enforcement, Enforcement)
        assert g.why.strip(), f"{g.name} has no stated reason"


def test_a_guard_without_enforcement_cannot_be_constructed():
    with pytest.raises(GuardDeclarationError):
        Guard(name="x", enforcement="control", why="stringly typed")  # type: ignore[arg-type]
    with pytest.raises(GuardDeclarationError):
        Guard(name="x", enforcement=Enforcement.CONTROL, why="   ")


def test_controls_fail_closed_and_conveniences_fail_open():
    for g in ALL_GUARDS:
        allowed_after_error = g.on_error(RuntimeError("boom"))
        if g.enforcement is Enforcement.CONTROL:
            assert not allowed_after_error, f"{g.name} is a CONTROL but fails open"
        else:
            assert allowed_after_error, f"{g.name} is a CONVENIENCE but fails closed"


# ── size: pre-check and the byte ceiling ────────────────────────────────────


def test_oversized_file_is_refused_before_download():
    with pytest.raises(MediaRejected) as exc:
        check_size_precheck(POLICY.max_bytes + 1, POLICY)
    assert exc.value.guard_name == "media.size_precheck"


def test_a_lying_file_size_is_caught_by_the_byte_ceiling():
    """The pre-check believes file_size. This does not."""
    check_size_precheck(1024, POLICY)  # peer claims 1 KB — passes
    with pytest.raises(MediaTooLarge):
        enforce_byte_ceiling(b"\x00" * (POLICY.max_bytes + 1), POLICY)


# ── MIME: declared, sniffed, agreement, allowlist ───────────────────────────


def test_declared_type_outside_the_allowlist_is_refused():
    with pytest.raises(MediaRejected) as exc:
        check_declared_mime("application/x-msdownload", "document", POLICY)
    assert exc.value.guard_name == "media.mime_declared"


def test_renamed_extension_is_caught_by_disagreement():
    """A PDF presented as image/png — declared allowlisted, contents are not."""
    with pytest.raises(MediaRejected) as exc:
        check_sniffed_mime(PDF, "image/png", "image", POLICY)
    assert exc.value.guard_name == "media.mime_sniffed"
    assert "do not match" in str(exc.value)


def test_unknown_bytes_are_refused_not_admitted():
    """Pinned to the SNIFF guard specifically.

    A loose `pytest.raises(MediaRejected)` passed even with the sniff-None
    branch disabled, because the allowlist refused `None` a few lines later —
    the test was green for the wrong reason (§3b). Asserting the guard name
    makes each test pin the control it claims to.
    """
    with pytest.raises(MediaRejected) as exc:
        check_sniffed_mime(b"\x00\x01\x02\x03", None, "image", POLICY)
    assert exc.value.guard_name == "media.mime_sniffed", (
        f"refused by {exc.value.guard_name}, not the sniff check"
    )


def test_sniffed_type_outside_the_allowlist_is_refused():
    """GIF sniffs fine but is not in this allowlist — the ALLOWLIST must refuse."""
    with pytest.raises(MediaRejected) as exc:
        check_sniffed_mime(b"GIF89a" + b"\x00" * 32, None, "image", POLICY)
    assert exc.value.guard_name == "media.allowlist", (
        f"refused by {exc.value.guard_name}, not the allowlist"
    )


def test_photo_branch_admits_on_sniff_alone():
    """No declared type exists for PhotoSize — sniff must still gate."""
    assert check_sniffed_mime(JPEG, None, "image", POLICY) == "image/jpeg"


def test_agreement_and_allowlist_both_required():
    assert validate_inbound(
        data=PNG, declared_mime="image/png", kind="image", policy=POLICY
    ) == "image/png"
    with pytest.raises(MediaRejected):
        validate_inbound(
            data=PNG, declared_mime="image/jpeg", kind="image", policy=POLICY
        )


# ── rate limiting ───────────────────────────────────────────────────────────


def test_per_chat_budget_refuses_the_over_limit_event():
    rl = RateLimiter(messages_per_minute=2, media_per_minute=99)
    assert rl.check("a", Budget.MESSAGES, now=0).allowed
    assert rl.check("a", Budget.MESSAGES, now=0).allowed
    d = rl.check("a", Budget.MESSAGES, now=0)
    assert not d.allowed and d.scope == "chat"


def test_one_chat_cannot_starve_another():
    rl = RateLimiter(messages_per_minute=1, media_per_minute=9, global_messages_per_minute=99)
    rl.check("noisy", Budget.MESSAGES, now=0)
    assert not rl.check("noisy", Budget.MESSAGES, now=0).allowed
    assert rl.check("quiet", Budget.MESSAGES, now=0).allowed, (
        "a second chat was refused because of the first — per-chat is not per-chat"
    )


def test_global_ceiling_refuses_aggregate_even_when_each_chat_is_under():
    rl = RateLimiter(messages_per_minute=10, media_per_minute=10, global_messages_per_minute=2)
    assert rl.check("a", Budget.MESSAGES, now=0).allowed
    assert rl.check("b", Budget.MESSAGES, now=0).allowed
    d = rl.check("c", Budget.MESSAGES, now=0)
    assert not d.allowed and d.scope == "global", (
        "the global ceiling did not bind — aggregate load is unbounded"
    )


def test_media_and_message_budgets_are_independent():
    rl = RateLimiter(messages_per_minute=1, media_per_minute=1)
    rl.check("a", Budget.MESSAGES, now=0)
    assert not rl.check("a", Budget.MESSAGES, now=0).allowed
    assert rl.check("a", Budget.MEDIA, now=0).allowed, (
        "media was refused because messages were exhausted — shared budget"
    )


def test_sender_is_warned_once_per_window_not_per_message():
    rl = RateLimiter(messages_per_minute=1, media_per_minute=1)
    rl.check("a", Budget.MESSAGES, now=0)
    warns = [rl.check("a", Budget.MESSAGES, now=0).should_warn for _ in range(5)]
    assert warns[0] is True, "the sender was never told why messages stopped"
    assert not any(warns[1:]), "warned on every drop — the warning becomes the flood"


def test_a_refusal_does_not_consume_budget():
    """Otherwise an over-limit chat never recovers."""
    rl = RateLimiter(messages_per_minute=1, media_per_minute=1)
    rl.check("a", Budget.MESSAGES, now=0)
    for _ in range(5):
        rl.check("a", Budget.MESSAGES, now=0)
    assert rl.check("a", Budget.MESSAGES, now=61).allowed, (
        "the chat could not recover after the window passed"
    )


def test_the_window_slides():
    rl = RateLimiter(messages_per_minute=1, media_per_minute=1)
    assert rl.check("a", Budget.MESSAGES, now=0).allowed
    assert not rl.check("a", Budget.MESSAGES, now=30).allowed
    assert rl.check("a", Budget.MESSAGES, now=61).allowed


# ── the controls must be CALLED — §1, the repo's most repeated defect ───────


@pytest.mark.parametrize(
    "handler,needs_declared",
    [
        ("_handle_photo", False),
        ("_handle_voice", True),
        ("_handle_document", True),
        ("_handle_sticker", False),
    ],
)
def test_every_inbound_handler_enforces(handler, needs_declared):
    src = inspect.getsource(getattr(tg_mod.TelegramAdapter, handler))
    assert "_admit(update, Budget.MEDIA)" in src, (
        f"{handler} does not rate-limit — the limiter exists but is not called"
    )
    assert "check_size_precheck" in src, (
        f"{handler} downloads without a pre-transfer size check"
    )
    assert "_guarded_download" in src, (
        f"{handler} calls download_as_bytearray directly, bypassing the byte "
        f"ceiling and the sniff"
    )
    assert "download_as_bytearray" not in src, (
        f"{handler} still has a raw unbounded download"
    )
    if needs_declared:
        assert "check_declared_mime" in src, (
            f"{handler} has a declared mime_type available and does not check it"
        )


def _code_only(src: str) -> str:
    """Strip comments — the check must prove the CODE has no hardcoded cap,
    not that the prose never mentions one. (A first draft failed on its own
    explanatory comment: a check answering a different question than asked.)"""
    return "\n".join(
        line.split("#", 1)[0] for line in src.splitlines()
    )


def test_the_hardcoded_20mb_document_cap_is_gone():
    src = _code_only(inspect.getsource(tg_mod.TelegramAdapter._handle_document))
    assert "20 * 1024 * 1024" not in src, (
        "the hardcoded cap is back — it only coincidentally matched the config "
        "default, so max_file_size_mb would silently not apply"
    )


# ── cache: fail-open convenience ────────────────────────────────────────────


def test_cache_below_the_floor_does_not_block_the_message(tmp_path, monkeypatch):
    """Caching is a convenience: below the floor the MESSAGE STILL ARRIVES.

    ⚠ THIS TEST USED TO ASSERT A PROXY, AND THE PROXY WAS THE DEFECT.

    It read:

        path = mc.cache_image_from_bytes(JPEG, ".jpg")
        assert path, "caching returned nothing — a convenience became a control"

    The comment names the right property — caching must not block the message.
    The assertion checked something else: that the return value was truthy. And
    the ONLY way to satisfy a truthy return when the write was refused was to
    fabricate the path the file WOULD have had, which is exactly the bug:

        photo     returned: /.../img_8ec90e18bb12.jpg   exists: False

    So the test passed for four releases while every inbound photo below the
    floor silently lost its content. A green proxy over a fabricated value is
    not evidence of the property it stands for.

    The property is now asserted where it lives — on the Telegram surface,
    through `_handle_photo`, by checking that `on_message` is reached. The
    helper returning None is the mechanism, not the claim.
    """
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    import prometheus.gateway.media_cache as mc

    # `configure_cache` writes MODULE GLOBALS and nothing restores them. The
    # pre-existing version of this test left the floor at 10^9 MB for the rest
    # of the session — which was invisible only because the fabricated path
    # kept later tests looking successful. With the fabrication gone, three
    # tests in test_wiring.py started failing on a floor this test had set.
    # Patch the globals directly so pytest reverts them.
    monkeypatch.setattr(mc, "_free_disk_floor_bytes", 10**9 * 1024 * 1024)
    monkeypatch.setattr(mc, "_cache_max_bytes", 1 * 1024 * 1024)

    # The mechanism: no path is returned for bytes that were never written.
    assert mc.cache_image_from_bytes(JPEG, ".jpg") is None, (
        "a path was returned below the floor — if it names a file that does "
        "not exist, every caller downstream fails on it later"
    )

    # The property: the message still reaches the model.
    delivered: list = []

    adapter = tg_mod.TelegramAdapter.__new__(tg_mod.TelegramAdapter)
    adapter.on_message = lambda event: _collect(delivered, event)
    adapter.send = _noop_send
    adapter._describe_image = _should_not_be_called

    update = _photo_update(chat_id=4242)
    asyncio.run(
        _drive_photo(adapter, update, image_bytes=JPEG, ext=".jpg")
    )

    assert delivered, (
        "the photo was dropped entirely — caching refusing to write turned "
        "into the message never arriving, which is the 'convenience became a "
        "control' failure the old assertion was reaching for"
    )
    event = delivered[0]
    assert event.chat_id == 4242
    assert event.media_urls == [], (
        f"the event carries {event.media_urls!r} for a file that was never "
        f"written"
    )
    assert "could not be stored" in event.text, (
        f"the model is not told the picture is missing: {event.text!r}"
    )


async def _collect(sink, event):
    sink.append(event)


async def _noop_send(*args, **kwargs):
    return None


async def _should_not_be_called(*args, **kwargs):
    raise AssertionError(
        "vision analysis was attempted on an image that was never cached — "
        "that is the call that used to receive a fabricated path"
    )


def _photo_update(chat_id: int):
    from types import SimpleNamespace

    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=7, username="u"),
        message=SimpleNamespace(
            caption="look at this", message_id=99,
            photo=[SimpleNamespace(file_size=len(JPEG))],
        ),
    )


async def _drive_photo(adapter, update, *, image_bytes, ext):
    """Drive the REAL `_handle_photo`, stubbing only the network download.

    Everything from the cache call onward is the shipped code path — which is
    the point: the property under test lives on this surface, not in the
    helper's return value.
    """
    import prometheus.gateway.media_cache as mc_mod
    import prometheus.gateway.telegram as tg

    async def _fake_guarded(self, _update, _file_obj, declared_mime=None, kind=""):
        return bytearray(image_bytes)

    async def _admit(_update, _budget):
        return True

    class _File:
        file_path = "photo.jpg"

    async def _get_file():
        return _File()

    update.message.photo[-1].get_file = _get_file

    orig_guard = tg.TelegramAdapter._guarded_download
    # `extension_from_file_path` is imported INSIDE the handler from
    # media_cache, so the module to patch is media_cache, not telegram.
    orig_ext = mc_mod.extension_from_file_path
    tg.TelegramAdapter._guarded_download = _fake_guarded
    mc_mod.extension_from_file_path = lambda _p: ext
    adapter._media_policy = POLICY
    adapter._admit = _admit
    try:
        await adapter._handle_photo(update, None)
    finally:
        tg.TelegramAdapter._guarded_download = orig_guard
        mc_mod.extension_from_file_path = orig_ext
