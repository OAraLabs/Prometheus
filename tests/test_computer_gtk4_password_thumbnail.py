"""No picture of a window that may show a GTK4 password field (found on the computer-use rig).

GTK4 reports a PasswordEntry as an unnamed "text box", exactly like an empty entry. The password
floor looked only for a role containing "password", so a GTK4 sign-in window was captured and its
thumbnail sent to Beacon ("thumbnail": "sent"). Any unnamed text field now skips the picture.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

from prometheus.computer import thumbnails as T
from prometheus.computer.cua import _window_capture
from prometheus.computer.types import WindowCapture

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()


def _cap(**kw) -> WindowCapture:
    base = dict(target="local", app="rig-login", pid=1, window_id=2, app_name="rig-login",
                roles=("application", "text box", "text box", "button"),
                image_mime="image/png", image_base64=PNG, image_width=10, image_height=10)
    base.update(kw)
    return WindowCapture(**base)


def test_an_unnamed_text_field_may_be_a_password_and_skips_the_picture():
    reason, image = T.decide_capture(_cap(unnamed_text_fields=2), app_names=("rig-login",))
    assert (reason, image) == ("password_field", None)


def test_named_text_fields_alone_do_not_skip():
    reason, image = T.decide_capture(_cap(unnamed_text_fields=0), app_names=("rig-login",))
    assert reason is None and image is not None


def test_the_adapter_counts_unnamed_gtk4_text_boxes_from_the_whole_walk():
    """As cua-driver 0.28.2 reported the rig's GTK4 sign-in window."""
    els = [SimpleNamespace(role="application", label="Rig Login"),
           SimpleNamespace(role="text box", label=None),      # the user name entry
           SimpleNamespace(role="text box", label=None),      # the PasswordEntry
           SimpleNamespace(role="text box", label="A document's text"),
           SimpleNamespace(role="button", label="Sign in")]
    out = SimpleNamespace(elements=els, images=[], app_name="rig-login", degraded=False,
                          truncated=False, screenshot_frame_valid=None, screenshot_width=None,
                          screenshot_height=None, snapshot_id=None)
    built = _window_capture(out, "local", "rig-login", 1, 2)
    assert built.unnamed_text_fields == 2
