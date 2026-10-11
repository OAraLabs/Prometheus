"""PR 6b — the pure half of cockpit level 2 (design §5.2.5).

Every rule here is CI-coverable because it touches no driver, bus or socket:
the skip taxonomy, the frame's exact key set, the size floor, the persist
path. The driver leg that produces the pixels is covered by the on-box check,
not here (see ``computer/cua.py``'s docstring) — and a green suite below must
not read as "thumbnails work", only as "the decisions are right".

The subject under test is a ``WindowCapture`` — OUR type, not the SDK's. That
is deliberate and it is the point of the shape: ``cua.py`` is the only module
that knows what a 0.28.2 ``WindowStateOutput`` looks like, so these rules can
be tested with no SDK installed and no SDK-mirroring fakes to drift. The
translation half is covered in ``test_cua_adapter.py``.
"""

from __future__ import annotations

import base64

import pytest

from prometheus.computer import thumbnails as T
from prometheus.computer.types import WindowCapture


# ── fixtures: a WindowCapture, our own type ────────────────────────────────

def _img(mime_type="image/png", data_base64=None):
    """The inline image half of a capture. Mirrors what cua.py fills in from
    ``SnapshotImage{mime_type, data_base64}``."""
    if data_base64 is None:
        data_base64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 100).decode()
    return {"mime": mime_type, "data": data_base64}


def _el(role="push button", **_over):
    """Kept for call sites that spell a walk as element-ish objects; only the
    role is read, which is what ``WindowCapture.roles`` carries."""
    return role


def _out(*, elements=None, images=(), app_name="Scratch", degraded=False,
         truncated=False, screenshot_width=480, screenshot_height=300,
         screenshot_frame_valid=None, **_ignored):
    """A ``WindowCapture``. ``elements`` is a list of role strings and
    ``images`` a list of _img dicts, matching the call sites below.

    Any SDK-only field a test still passes (``screenshot_file_path``,
    ``elements_complete``) is accepted and ignored on purpose: ``WindowCapture``
    has no such field, which is exactly the property worth having — nothing
    here can read a path off disk or lean on a field 0.28.2 hard-codes False.
    """
    first = images[0] if images else None
    return WindowCapture(
        target="local", app="Scratch", pid=1, window_id=2,
        app_name=app_name,
        roles=tuple(e if isinstance(e, str) else getattr(e, "role", "")
                    for e in (elements if elements is not None
                              else [_el()])),
        degraded=degraded, truncated=truncated,
        frame_valid=screenshot_frame_valid,
        image_mime=(first or {}).get("mime"),
        image_base64=(first or {}).get("data"),
        image_width=screenshot_width if first else None,
        image_height=screenshot_height if first else None,
    )


# ── capture_plan: only an executed step, only with a viewer ────────────────

@pytest.mark.parametrize("status", [
    "executed", "abstained", "refused", "blocked", "reobserve",
    "in_flight_at_stop",
])
def test_only_an_executed_step_is_captured(status):
    """An executed step with a viewer is CAPTURE. Every other status is NONE
    — no decision was made, so the frame says thumbnail:null, not a skip. The
    in-flight-at-stop call is NONE too: its outcome is unknown and the person
    asked us to stop."""
    action, reason = T.capture_plan(status=status, enabled=True, viewers=1)
    if status == "executed":
        assert action == T.CAPTURE and reason is None
    else:
        assert action == T.NONE and reason is None


def test_no_viewer_is_a_skip_that_says_why():
    """The feature is on and the step ran, we just did not capture because
    nobody eligible is watching — that is a decision, so SKIP + reason."""
    action, reason = T.capture_plan(status="executed", enabled=True, viewers=0)
    assert action == T.SKIP and reason == "no_viewer"


def test_disabled_is_none_not_a_skip():
    """enabled=False means the feature is off, not that a capture was decided
    against — the step frame carries thumbnail:null, not a skip."""
    action, reason = T.capture_plan(status="executed", enabled=False, viewers=5)
    assert action == T.NONE and reason is None


def test_the_tri_state_is_three_distinct_outcomes():
    """The whole point of capture_plan over a bare Optional: CAPTURE, SKIP and
    NONE are not confusable. A None return could not carry all three."""
    assert len({T.CAPTURE, T.SKIP, T.NONE}) == 3
    cap, _ = T.capture_plan(status="executed", enabled=True, viewers=1)
    skp, _ = T.capture_plan(status="executed", enabled=True, viewers=0)
    non, _ = T.capture_plan(status="executed", enabled=False, viewers=1)
    assert {cap, skp, non} == {T.CAPTURE, T.SKIP, T.NONE}


# ── the password skip ──────────────────────────────────────────────────────

def test_a_password_role_anywhere_in_the_walk_skips():
    walk = [_el(role="push button"), _el(role="password text"),
            _el(role="fill in")]
    assert T.walk_hides_a_password(walk) is True


def test_a_tokenless_password_node_still_skips():
    """The whole walk, including nodes the chooser never sees (no token)."""
    walk = [_el(role="password text", element_token=None)]
    assert T.walk_hides_a_password(walk) is True


def test_an_ordinary_walk_does_not_skip():
    assert T.walk_hides_a_password([_el(role="push button"),
                                    _el(role="text")]) is False


def test_password_match_is_case_insensitive_and_substring():
    """AT-SPI spells it 'password text'; the check must not depend on the
    exact spelling."""
    assert T.walk_hides_a_password([_el(role="PASSWORD")]) is True


def test_an_empty_walk_does_not_skip_on_password():
    assert T.walk_hides_a_password([]) is False


# ── the incomplete-walk skip: positive evidence required ───────────────────

def test_a_degraded_walk_cannot_vouch_for_itself():
    """A degraded walk may be hiding a password field in the part we did not
    get, so its silence is not evidence."""
    assert T.walk_is_positive_evidence(_out(degraded=True)) is False


def test_a_truncated_walk_cannot_vouch_for_itself():
    assert T.walk_is_positive_evidence(_out(truncated=True)) is False


def test_a_complete_walk_is_positive_evidence():
    assert T.walk_is_positive_evidence(_out()) is True


def test_elements_complete_false_on_linux_does_not_skip():
    """0.28.2 hard-codes elements_complete False on Linux. Gating on it would
    skip every thumbnail on the only platform this runs on — so it must not."""
    assert T.walk_is_positive_evidence(_out(elements_complete=False)) is True


# ── the inline image, as decide_capture sees it ────────────────────────────
# There is no separate inline_image() to test: extraction is one branch of
# decide_capture, so these assert the outcome (an image dict, or a skip
# reason) rather than an intermediate object. The SDK→WindowCapture
# translation itself is covered in test_cua_adapter.py.

def _decide(out, names=("Scratch",)):
    return T.decide_capture(out, app_names=list(names))


def test_a_clean_capture_yields_the_image_with_its_size():
    out = _out(images=[_img()], screenshot_width=480, screenshot_height=300)
    reason, img = _decide(out)
    assert reason is None and img is not None
    assert img["mime_type"] == "image/png"
    assert img["width"] == 480 and img["height"] == 300
    assert img["data_base64"]


def test_no_inline_image_is_capture_failed_not_a_file_read():
    """images empty → capture_failed. We do NOT fall back to a screenshot file
    path: a screenshot on disk is a second copy of the desktop we do not own.
    (WindowCapture has no such field at all, which is the stronger guarantee.)"""
    out = _out(images=[], screenshot_file_path="/tmp/leak.png")
    reason, img = _decide(out)
    assert reason == "capture_failed" and img is None


def test_an_unknown_mime_is_capture_failed():
    reason, img = _decide(_out(images=[_img(mime_type="image/x-evil")]))
    assert reason == "capture_failed" and img is None


def test_empty_data_is_capture_failed():
    reason, img = _decide(_out(images=[_img(data_base64="")]))
    assert reason == "capture_failed" and img is None


# ── frame validity ─────────────────────────────────────────────────────────

def test_an_explicit_false_frame_is_invalid():
    assert T.frame_valid(_out(screenshot_frame_valid=False)) is False


def test_a_none_frame_validity_does_not_refuse():
    """None means 'the driver did not report either way'. Refusing on None
    would skip every thumbnail on a driver that does not populate it."""
    assert T.frame_valid(_out(screenshot_frame_valid=None)) is True


def test_a_true_frame_is_valid():
    assert T.frame_valid(_out(screenshot_frame_valid=True)) is True


# ── the size floor ─────────────────────────────────────────────────────────

def test_the_cap_measures_decoded_bytes_not_the_base64_string():
    """base64 inflates ~4/3, so capping the encoded string would let a
    >256 KB image through. Measure what the picture actually is."""
    raw = b"x" * (T.MAX_THUMBNAIL_BYTES + 10)
    encoded = base64.b64encode(raw).decode()
    assert T.decoded_size(encoded) == len(raw)
    assert len(encoded) > T.MAX_THUMBNAIL_BYTES  # the string is bigger still


def test_malformed_base64_has_no_size():
    assert T.decoded_size("!!!not base64!!!") is None


def test_the_floor_is_256kb():
    """A floor, not a config key — a misconfiguration must not lift it."""
    assert T.MAX_THUMBNAIL_BYTES == 256 * 1024


# ── the frame's exact key set ──────────────────────────────────────────────

def test_the_frame_carries_exactly_the_declared_keys():
    _reason, img = T.decide_capture(_out(images=[_img()]), app_names=["Scratch"])
    frame = T.build_frame(session_id="s", task_id="abc123", seq=7,
                          app="gedit", image=img, captured_at="2026-10-04T00:00:00Z")
    assert set(frame) == T.FRAME_KEYS


def test_the_frame_excludes_every_identifier_the_log_withholds():
    """No pid, window id, title, snapshot id or element data — the title is
    page-authored text in a browser and would cross a content boundary."""
    _reason, img = T.decide_capture(_out(images=[_img()]), app_names=["Scratch"])
    frame = T.build_frame(session_id="s", task_id="abc123", seq=7,
                          app="gedit", image=img, captured_at="t")
    forbidden = {"pid", "window_id", "window_title", "snapshot_id",
                 "elements", "title"}
    assert not (forbidden & set(frame))
    assert "t" not in str(frame.get("window_title", ""))  # never present


# ── the opt-in persist path ────────────────────────────────────────────────

def test_the_persist_path_is_under_the_thumbnails_tree():
    p = T.persist_path("/data", "a1b2c3", 7, "image/png")
    assert p == "/data/computer/thumbnails/a1b2c3/7.png"


def test_a_task_id_cannot_escape_the_tree():
    """task_id is ours, but it becomes a path component — validate, don't
    trust. A traversal here writes a screenshot outside the tree."""
    assert T.persist_path("/data", "../../etc", 7, "image/png") is None
    assert T.persist_path("/data", "a1b2/../x", 7, "image/png") is None


def test_an_unknown_mime_gets_no_extension_and_no_path():
    """The extension comes from the closed map, never from the driver's string."""
    assert T.persist_path("/data", "a1b2c3", 7, "image/x-evil") is None


def test_the_persist_extension_follows_the_mime():
    assert T.persist_path("/d", "abc", 1, "image/jpeg").endswith("1.jpg")
    assert T.persist_path("/d", "abc", 1, "image/webp").endswith("1.webp")


# ── the skip taxonomy is a closed set ──────────────────────────────────────

def test_every_reason_is_in_the_declared_set():
    """A new skip reason is a new thing a person can be told, so the set is
    closed and pinned. If this fails, SKIP_REASONS and the design both change."""
    assert T.SKIP_REASONS == {
        "password_field", "incomplete_walk", "window_changed",
        "capture_failed", "frame_invalid", "too_large", "no_viewer",
    }


# ── the post-capture rules, and their ORDER ────────────────────────────────

def test_a_clean_capture_yields_the_image():
    out = _out(app_name="gedit", images=[_img()],
               screenshot_frame_valid=True)
    reason, img = T.decide_capture(out, app_names=["gedit"])
    assert reason is None and img is not None and img["mime_type"] == "image/png"


def test_an_app_the_driver_reports_as_none_is_a_change():
    """A window the driver cannot attribute is one we did not prove is ours."""
    out = _out(app_name=None, images=[_img()])
    reason, img = T.decide_capture(out, app_names=["gedit"])
    assert reason == "window_changed" and img is None


def test_an_app_that_is_not_the_bound_one_is_a_change():
    """D19: another app came to the front mid-task. Never show it."""
    out = _out(app_name="firefox", images=[_img()])
    reason, img = T.decide_capture(out, app_names=["gedit", "GNOME Text Editor"])
    assert reason == "window_changed" and img is None


def test_every_name_the_door_resolved_the_app_by_is_accepted():
    """The binding may know the app by several names; any of them counts."""
    out = _out(app_name="GNOME Text Editor", images=[_img()])
    reason, _ = T.decide_capture(out, app_names=["gedit", "GNOME Text Editor"])
    assert reason is None


def test_no_names_to_compare_fails_closed():
    """Fail CLOSED: with nothing to compare against we cannot prove the
    window is the person's pick, so it skips rather than guessing."""
    out = _out(app_name="gedit", images=[_img()])
    reason, img = T.decide_capture(out, app_names=[])
    assert reason == "window_changed" and img is None


def test_a_degraded_walk_skips_before_anything_else_about_the_pixels():
    out = _out(app_name="gedit", images=[_img()], degraded=True)
    reason, _ = T.decide_capture(out, app_names=["gedit"])
    assert reason == "incomplete_walk"


def test_a_password_field_skips_the_capture():
    out = _out(app_name="gedit", images=[_img()],
               elements=[_el(role="password text")])
    reason, img = T.decide_capture(out, app_names=["gedit"])
    assert reason == "password_field" and img is None


def test_an_untrustworthy_frame_skips():
    out = _out(app_name="gedit", images=[_img()], screenshot_frame_valid=False)
    reason, _ = T.decide_capture(out, app_names=["gedit"])
    assert reason == "frame_invalid"


def test_no_inline_image_is_capture_failed():
    out = _out(app_name="gedit", images=[])
    reason, _ = T.decide_capture(out, app_names=["gedit"])
    assert reason == "capture_failed"


def test_an_oversized_frame_is_skipped_whole_never_sent_in_pieces():
    big = base64.b64encode(b"x" * (T.MAX_THUMBNAIL_BYTES + 1)).decode()
    out = _out(app_name="gedit", images=[_img(data_base64=big)])
    reason, img = T.decide_capture(out, app_names=["gedit"])
    assert reason == "too_large" and img is None


def test_exactly_the_floor_is_allowed():
    """One byte over skips; exactly at it does not."""
    exact = base64.b64encode(b"x" * T.MAX_THUMBNAIL_BYTES).decode()
    out = _out(app_name="gedit", images=[_img(data_base64=exact)])
    reason, img = T.decide_capture(out, app_names=["gedit"])
    assert reason is None and img is not None


def test_privacy_gates_run_before_the_mechanical_ones():
    """THE ordering invariant. A frame that is both oversized AND shows a
    password field is refused for the password — the size answer is about
    transport, and telling a person 'too large' would be a lie about a
    picture we refused for a different reason."""
    big = base64.b64encode(b"x" * (T.MAX_THUMBNAIL_BYTES + 1)).decode()
    out = _out(app_name="gedit", images=[_img(data_base64=big)],
               elements=[_el(role="password text")])
    reason, _ = T.decide_capture(out, app_names=["gedit"])
    assert reason == "password_field"


def test_a_wrong_app_is_reported_before_a_degraded_walk():
    """Window identity is the outermost gate: if this is not our window, the
    rest of the walk describes someone else's screen and must not be read."""
    out = _out(app_name="firefox", images=[_img()], degraded=True)
    reason, _ = T.decide_capture(out, app_names=["gedit"])
    assert reason == "window_changed"


# ── the frame kind is direct-only: outside the promoted tuple ──────────────

def test_the_thumbnail_kind_is_not_a_promoted_frame_kind():
    """THE load-bearing invariant of PR 6b. SignalBus persists everything it
    is given; a screenshot must never be given to it. ws_server promotes from
    COMPUTER_FRAME_KINDS, so keeping the thumbnail kind out of that tuple is
    what makes it direct-only — never in signal_events, never backfilled."""
    from prometheus.computer.livestream import COMPUTER_FRAME_KINDS
    assert "computer_step_thumbnail" not in COMPUTER_FRAME_KINDS


# ── configuration ──────────────────────────────────────────────────────────

def test_the_config_defaults_are_enabled_and_not_persisted():
    c = T.ThumbnailConfig.from_config({})
    assert c.enabled is True and c.persist is False
    assert c.max_dimension == T.DEFAULT_MAX_DIMENSION


def test_max_dimension_out_of_range_falls_back():
    errs: list[str] = []
    c = T.ThumbnailConfig.from_config(
        {"computer_use": {"thumbnails": {"max_dimension": 100000}}}, errs)
    assert c.max_dimension == T.DEFAULT_MAX_DIMENSION and errs


def test_a_non_integer_dimension_is_rejected_not_crashed():
    errs: list[str] = []
    c = T.ThumbnailConfig.from_config(
        {"computer_use": {"thumbnails": {"max_dimension": "big"}}}, errs)
    assert c.max_dimension == T.DEFAULT_MAX_DIMENSION and errs


def test_a_bool_dimension_is_rejected():
    """bool is an int subclass; True must not sneak in as dimension 1."""
    errs: list[str] = []
    c = T.ThumbnailConfig.from_config(
        {"computer_use": {"thumbnails": {"max_dimension": True}}}, errs)
    assert c.max_dimension == T.DEFAULT_MAX_DIMENSION and errs


def test_persist_that_does_not_parse_fails_to_off():
    """The one key whose wrong answer writes desktop pixels to disk. A
    malformed value must not silently become the default-and-keep-going."""
    errs: list[str] = []
    c = T.ThumbnailConfig.from_config(
        {"computer_use": {"thumbnails": {"persist": "yes"}}}, errs)
    assert c.persist is False and errs


def test_enabled_off_is_honoured():
    c = T.ThumbnailConfig.from_config(
        {"computer_use": {"thumbnails": {"enabled": False}}})
    assert c.enabled is False


def test_a_garbage_config_block_reads_as_defaults():
    errs: list[str] = []
    c = T.ThumbnailConfig.from_config({"computer_use": {"thumbnails": "x"}}, errs)
    assert (c.enabled, c.persist, c.max_dimension) == (True, False,
                                                       T.DEFAULT_MAX_DIMENSION)


# ── the no-sink default ────────────────────────────────────────────────────

def test_no_sink_means_no_viewers_and_a_silent_send():
    """With no bridge wired, nobody is watching, so nothing is captured and a
    send is a no-op rather than a crash."""
    import asyncio
    assert T.NO_SINK.viewer_count("s") == 0
    asyncio.run(T.NO_SINK.send_thumbnail("s", {}))  # must not raise
