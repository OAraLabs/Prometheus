"""What the daemon's ``daemon_start`` archive event records, and why an unset ``--bind`` is not in it.

The daemon archives ``vars(args)`` at start. The 12 parity traces (tests/fixtures/parity) record that dict, and
the CI replay compares the daemon's whole archive against them. Adding the ``--bind`` flag added ``bind: null``
to every run, so every golden "changed" while nothing the daemon does had: the replay failed on all 12 with the
one line ``args.bind: recorded <absent>, replayed null``.

An option that was not given has nothing to record, so an unset ``bind`` is left out. An explicit one is a fact
about the run and is kept. Every other argument is recorded exactly as before, null or not, so the goldens
stay valid, and a test ties the real parser's default invocation to what every golden holds: the next flag
added to the daemon fails here, with the reason, instead of in a CI job with a diff nobody asked for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from prometheus import daemon

GOLDENS = sorted((Path(__file__).parent / "fixtures" / "parity").glob("*.expected.json"))


def _recorded_args(node):
    """Every ``daemon_start`` event's args dict in a parsed golden, wherever it is nested."""
    if isinstance(node, dict):
        if node.get("type") == "daemon_start" and isinstance(node.get("data", {}).get("args"), dict):
            yield node["data"]["args"]
        for value in node.values():
            yield from _recorded_args(value)
    elif isinstance(node, list):
        for value in node:
            yield from _recorded_args(value)


def test_an_unset_bind_is_not_recorded():
    namespace = argparse.Namespace(config=None, debug=False, telegram_only=False, bind=None)
    assert daemon.daemon_start_args(namespace) == {"config": None, "debug": False, "telegram_only": False}


def test_a_bind_that_was_given_is_recorded():
    namespace = argparse.Namespace(config=None, debug=False, telegram_only=False, bind="127.0.0.1")
    assert daemon.daemon_start_args(namespace)["bind"] == "127.0.0.1"


def test_every_other_unset_argument_is_still_recorded_as_null():
    """Only ``bind`` is special. ``config: null`` has always been in the record and the goldens depend on it."""
    namespace = argparse.Namespace(config=None, debug=False, telegram_only=False, bind=None)
    assert "config" in daemon.daemon_start_args(namespace)
    assert daemon.daemon_start_args(namespace)["config"] is None


def test_recording_does_not_change_the_namespace():
    namespace = argparse.Namespace(config=None, debug=False, telegram_only=False, bind=None)
    daemon.daemon_start_args(namespace)
    assert namespace.bind is None and "bind" in vars(namespace)


def test_the_goldens_are_there_to_be_compared():
    assert len(GOLDENS) >= 12, f"expected the parity goldens, found {len(GOLDENS)}"


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda p: p.name.removesuffix(".expected.json"))
def test_a_default_start_records_the_keys_every_golden_recorded(golden):
    """The real parser, no flags: the recorded keys must be the goldens' keys. A new daemon flag that is not
    handled here shows up as a key the goldens do not have, and the message names it."""
    recorded = list(_recorded_args(json.loads(golden.read_text())))
    assert recorded, f"{golden.name} has no daemon_start event"
    now = daemon.daemon_start_args(daemon.build_parser().parse_args([]))
    for args in recorded:
        assert set(now) == set(args), (
            f"a default start now records {sorted(set(now) - set(args))} that {golden.name} does not, and lacks "
            f"{sorted(set(args) - set(now))}. A new flag must not change what every run records: leave it out "
            "when unset (see daemon_start_args), or rebaseline the parity traces on purpose."
        )
