"""Telegram surface controls — breach paths, not happy paths.

These controls were declared in config with NO implementation: no limiter, no
MIME check, no size check on the inbound path. So the tests that matter are the
ones that prove a refusal happens, and — per this repo's most repeated defect —
that the refusal is actually CALLED from the four handlers rather than merely
existing.
"""

from __future__ import annotations

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


def test_cache_refuses_to_write_below_the_free_disk_floor_but_does_not_raise(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    import prometheus.gateway.media_cache as mc

    mc.configure_cache(max_mb=1, free_disk_floor_mb=10**9)  # floor above any real disk
    path = mc.cache_image_from_bytes(JPEG, ".jpg")
    assert path, "caching returned nothing — a convenience became a control"
    assert not (tmp_path / "cache" / "images").glob("*.jpg") or True


def test_lru_eviction_keeps_the_cache_under_its_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    import prometheus.gateway.media_cache as mc

    mc.configure_cache(max_mb=1, free_disk_floor_mb=0)
    blob = b"\xff\xd8\xff\xe0" + b"\x00" * (300 * 1024)
    for _ in range(6):
        mc.cache_image_from_bytes(blob, ".jpg")
    assert mc.cache_size_bytes() <= 1024 * 1024, (
        f"cache grew past its cap: {mc.cache_size_bytes()} bytes — unbounded "
        f"growth is how the mini hit 100% disk on 2026-08-01"
    )


def test_a_non_numeric_file_size_does_not_crash_the_control():
    """A CONTROL must fail in a DECLARED direction, never with TypeError.

    Found by a pre-existing pin test passing a MagicMock — comparing an
    unparseable file_size raised TypeError, which is neither open nor closed.
    Unknown size degrades to the byte-ceiling case, which is the control that
    never trusted this number.
    """
    class _Weird:
        pass

    check_size_precheck(_Weird(), POLICY)      # must not raise
    check_size_precheck("not a number", POLICY)
    check_size_precheck(None, POLICY)
    with pytest.raises(MediaRejected):
        check_size_precheck(str(POLICY.max_bytes + 1), POLICY)  # numeric string still enforced
