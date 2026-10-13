"""``security.denied_paths``: a list Prometheus shipped is an old default, not an override.

Setup copies the template, so every install holds the ``denied_paths`` list of its day written out. When the
shipped list grew (the same-Mac pairing directories, 0.9.9), each of those configs would have logged
``config OVERRIDES a shipped default`` at every boot and kept the old list, although nobody chose it. Will,
2026-10-13: recognise the exact previous shipped lists as old defaults and upgrade them quietly, with one INFO
line; a list the operator really changed keeps the WARNING and is used exactly as written.
"""

from __future__ import annotations

import logging

import pytest

from prometheus.config import divergence
from prometheus.config.shipped_defaults import (
    DENIED_PATHS_DEFAULTS_SHIPPED,
    SHIPPED_DENIED_PATHS,
    resolve_denied_paths,
)
from prometheus.config.template import load_template

PREVIOUS = DENIED_PATHS_DEFAULTS_SHIPPED[-2]


def _denied(config_list) -> divergence.ConfigValue:
    values = divergence.describe({"security": {"denied_paths": config_list}})
    return next(v for v in values if v.key == "security.denied_paths")


def _boot_lines(config_list, caplog) -> list[logging.LogRecord]:
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=divergence.logger.name):
        divergence.warn_on_divergence({"security": {"denied_paths": config_list}})
    return [r for r in caplog.records if "security.denied_paths" in r.getMessage()]


def test_the_newest_shipped_list_is_the_default_and_the_template_carries_it():
    assert SHIPPED_DENIED_PATHS == DENIED_PATHS_DEFAULTS_SHIPPED[-1]
    assert tuple(load_template()["security"]["denied_paths"]) == SHIPPED_DENIED_PATHS


# ── the three cases ──────────────────────────────────────────────────────────

def test_an_old_default_is_upgraded_to_the_new_list_with_one_info_line(caplog):
    assert resolve_denied_paths({"denied_paths": list(PREVIOUS)}) == list(SHIPPED_DENIED_PATHS)
    assert _denied(list(PREVIOUS)).source == "shipped_value"
    lines = _boot_lines(list(PREVIOUS), caplog)
    assert [r.levelno for r in lines] == [logging.INFO], [r.getMessage() for r in lines]
    assert "pairing" in lines[0].getMessage(), "the INFO line should show the list now in force"


def test_the_new_default_written_out_is_silent(caplog):
    assert resolve_denied_paths({"denied_paths": list(SHIPPED_DENIED_PATHS)}) == list(SHIPPED_DENIED_PATHS)
    assert _denied(list(SHIPPED_DENIED_PATHS)).source == "config"
    assert _boot_lines(list(SHIPPED_DENIED_PATHS), caplog) == []


def test_a_real_override_is_used_as_written_and_still_warns(caplog):
    mine = [*PREVIOUS, "/srv/private"]
    assert resolve_denied_paths({"denied_paths": mine}) == mine
    assert _denied(mine).source == "config_override"
    lines = _boot_lines(mine, caplog)
    assert [r.levelno for r in lines] == [logging.WARNING], [r.getMessage() for r in lines]


# ── what counts as an old default ────────────────────────────────────────────

@pytest.mark.parametrize("earlier", DENIED_PATHS_DEFAULTS_SHIPPED[:-1], ids=lambda t: str(len(t)))
def test_every_earlier_shipped_list_is_an_old_default(earlier):
    assert resolve_denied_paths({"denied_paths": list(earlier)}) == list(SHIPPED_DENIED_PATHS)
    assert _denied(list(earlier)).source == "shipped_value"


@pytest.mark.parametrize("changed", [
    list(reversed(PREVIOUS)),           # same entries, the operator's order
    list(PREVIOUS[:-1]),                # one taken out
    [],                                 # the documented "no policy denials"
], ids=["reordered", "trimmed", "empty"])
def test_anything_but_an_exact_shipped_list_is_the_operators(changed, caplog):
    assert resolve_denied_paths({"denied_paths": changed}) == changed
    assert _denied(changed).source == "config_override"
    assert [r.levelno for r in _boot_lines(changed, caplog)] == [logging.WARNING]


def test_absence_still_means_the_shipped_list(caplog):
    assert resolve_denied_paths({}) == list(SHIPPED_DENIED_PATHS)
    assert _boot_lines(None, caplog) == []
