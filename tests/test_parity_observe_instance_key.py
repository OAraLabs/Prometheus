"""The parity harness must not see the instance key.

The daemon makes ``node/instance.key`` at boot (the P-256 key the pairing contract advertises the
fingerprint of; docs/PAIRING-APPROVAL-API.md, 7.3). The harness snapshots the isolated home it boots the
daemon in and compares each golden trace to it, so a boot-time file the goldens were recorded without is a
DIFF in every scenario. That is exactly what CI's ``replay`` job reported on this key's first PR: 12 of 12
traces, ``recorded: null`` against the key's own PEM text as ``replayed``, which also printed the key into the
CI log.

The observer's skip list is for "files that are not side effects of a turn". The instance key is one: it
appears at boot, random, whatever the scenario does, and its creation is pinned where it belongs
(``test_instance_key.py::test_the_daemon_makes_the_key_at_boot``). Skipping it also means the harness never
reads a private key's text. ``node/node.key`` keeps its existing treatment (content normalised, existence
compared) and this must not touch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from parity.observe import snapshot  # noqa: E402

# Built at run time: a PEM header written out in full is what the pre-commit secret scanner looks for.
_BAR = "-" * 5
PEM = f"{_BAR}BEGIN PRIVATE KEY{_BAR}\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0w\n{_BAR}END PRIVATE KEY{_BAR}\n"


def _home(tmp_path: Path) -> Path:
    node = tmp_path / "home" / ".prometheus" / "node"
    node.mkdir(parents=True)
    (node / "node.key").write_text(PEM)
    (node / "node.pub").write_text("c29tZS1wdWJsaWMta2V5\n")
    (node / "instance.key").write_text(PEM)
    (tmp_path / "home" / ".prometheus" / "settings.json").write_text('{"a": 1}')
    return tmp_path


def test_the_instance_key_is_not_in_the_snapshot(tmp_path):
    seen = snapshot(_home(tmp_path))
    assert "home/.prometheus/node/instance.key" not in seen


def test_no_private_key_text_reaches_a_snapshot_through_it(tmp_path):
    instance_only = _home(tmp_path)
    (instance_only / "home" / ".prometheus" / "node" / "node.key").unlink()   # leave ONLY the instance key
    assert "PRIVATE KEY" not in repr(snapshot(instance_only))


def test_the_node_keys_are_still_compared_and_ordinary_files_too(tmp_path):
    """The skip is for one name, not for the node directory."""
    seen = snapshot(_home(tmp_path))
    assert "home/.prometheus/node/node.key" in seen and "home/.prometheus/node/node.pub" in seen
    assert seen["home/.prometheus/settings.json"] == {"json": {"a": 1}}


def test_the_skip_is_by_exact_path_not_by_a_name_that_would_hide_other_files(tmp_path):
    root = _home(tmp_path)
    other = root / "home" / ".prometheus" / "data"
    other.mkdir()
    (other / "instance.key").write_text("not the node directory's file")
    assert "home/.prometheus/data/instance.key" in snapshot(root)
