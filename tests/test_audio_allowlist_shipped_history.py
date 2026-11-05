"""``gateway.media.allowed_audio_types``: the list Prometheus shipped before #902 is an old default, not an override.

Setup copies the template, so every install from v0.9.0 to 0.9.20 holds ``[audio/ogg, audio/mpeg, audio/wav]``
written out. #902 added ``audio/mp4`` and ``audio/webm`` to the shipped list, but the resolver used any list in the
config verbatim: each of those configs kept refusing m4a and WebM voice recordings, and logged
``config OVERRIDES a shipped default`` at every boot, although nobody chose the old list. The same rule as
``security.denied_paths`` (Will, 2026-10-13): the exact earlier shipped list follows the current default, with one
INFO line; a list the operator really changed keeps the WARNING and is used exactly as written.
"""

from __future__ import annotations

import logging

import pytest

from prometheus.config import divergence
from prometheus.config.shipped_defaults import (
    ALLOWED_AUDIO_TYPES_DEFAULTS_SHIPPED,
    SHIPPED_ALLOWED_AUDIO_TYPES,
    resolve_media_allowlist,
)
from prometheus.config.template import load_template

KEY = "gateway.media.allowed_audio_types"
BEFORE_902 = ("audio/ogg", "audio/mpeg", "audio/wav")


def _resolve(config_list) -> list[str]:
    return resolve_media_allowlist({"allowed_audio_types": config_list}, "allowed_audio_types")


def _audio(config_list) -> divergence.ConfigValue:
    values = divergence.describe({"gateway": {"media": {"allowed_audio_types": config_list}}})
    return next(v for v in values if v.key == KEY)


def _boot_lines(config_list, caplog) -> list[logging.LogRecord]:
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=divergence.logger.name):
        divergence.warn_on_divergence({"gateway": {"media": {"allowed_audio_types": config_list}}})
    return [r for r in caplog.records if KEY in r.getMessage()]


def test_the_history_is_the_two_lists_shipped_and_the_template_carries_the_newest():
    assert ALLOWED_AUDIO_TYPES_DEFAULTS_SHIPPED == (BEFORE_902, SHIPPED_ALLOWED_AUDIO_TYPES)
    assert tuple(load_template()["gateway"]["media"]["allowed_audio_types"]) == SHIPPED_ALLOWED_AUDIO_TYPES


# ── the three cases ──────────────────────────────────────────────────────────

def test_the_list_shipped_before_902_follows_the_current_default_with_one_info_line(caplog):
    assert _resolve(list(BEFORE_902)) == list(SHIPPED_ALLOWED_AUDIO_TYPES)
    assert _audio(list(BEFORE_902)).source == "shipped_value"
    lines = _boot_lines(list(BEFORE_902), caplog)
    assert [r.levelno for r in lines] == [logging.INFO], [r.getMessage() for r in lines]
    assert "audio/webm" in lines[0].getMessage(), "the INFO line should show the list now in force"


def test_an_old_config_admits_an_m4a_voice_memo():
    from prometheus.gateway.media_guard import check_declared_mime, policy_from_config

    policy = policy_from_config({"gateway": {"media": {"allowed_audio_types": list(BEFORE_902)}}})
    check_declared_mime("audio/x-m4a", "audio", policy)
    check_declared_mime("audio/webm;codecs=opus", "audio", policy)


def test_the_current_default_written_out_is_silent(caplog):
    assert _resolve(list(SHIPPED_ALLOWED_AUDIO_TYPES)) == list(SHIPPED_ALLOWED_AUDIO_TYPES)
    assert _audio(list(SHIPPED_ALLOWED_AUDIO_TYPES)).source == "config"
    assert _boot_lines(list(SHIPPED_ALLOWED_AUDIO_TYPES), caplog) == []


@pytest.mark.parametrize("changed", [
    ["audio/ogg", "audio/mpeg"],                 # one taken out
    [*BEFORE_902, "audio/flac"],                 # one added
    list(reversed(BEFORE_902)),                  # same entries, the operator's order
    [],                                          # the documented "no restriction"
], ids=["trimmed", "added", "reordered", "empty"])
def test_anything_but_an_exact_shipped_list_is_the_operators(changed, caplog):
    assert _resolve(changed) == changed
    assert _audio(changed).source == "config_override"
    assert [r.levelno for r in _boot_lines(changed, caplog)] == [logging.WARNING]


def test_absence_still_means_the_shipped_list(caplog):
    assert resolve_media_allowlist({}, "allowed_audio_types") == list(SHIPPED_ALLOWED_AUDIO_TYPES)
    assert _boot_lines(None, caplog) == []
