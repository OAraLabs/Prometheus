"""``~/.config/prometheus`` is the credentials folder: the whole of it is out of reach of a model-written shell and
of the file tools, not just the file named ``env``.

The floor used to be one pattern, ``~/.config/*/*env`` (names ending in "env"). That kept ``env`` itself out and let
``env.bak-<date>`` (a copy of it, made by a person before an edit), the APNs signing key
(``AuthKey_<id>.p8``, named in ``push.apns.key_path``) and any token file beside them be read: measured on a real
install, a confined shell read all three while ``env`` was refused. Nothing a shell or a file tool legitimately does
needs this folder (the configuration lives in ``~/.prometheus``), so the rule is the folder, as ``.ssh`` and
``.gnupg`` are.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from prometheus.config import shipped_defaults
from prometheus.permissions import checker
from prometheus.permissions.checker import SecurityGate

PROFILE = Path(__file__).resolve().parent.parent / "deploy" / "apparmor" / "prometheus-bash"
TEMPLATE = Path(__file__).resolve().parent.parent / "config" / "prometheus.yaml.default"
FOLDER_RULE = re.compile(r"^\s*deny @\{HOME\}/\.config/prometheus/\{,\*\*\}\s+rwlkmx,\s*$", re.M)

CREDENTIAL_FILES = [
    "env",
    "env.bak-2026-11-16",
    "AuthKey_386G3GT393.p8",
    "device-tokens/phone.id",
    "anything-else.token",
]


def test_the_apparmor_profile_denies_the_whole_credentials_folder():
    assert len(FOLDER_RULE.findall(PROFILE.read_text(encoding="utf-8"))) == 1


@pytest.mark.parametrize("name", CREDENTIAL_FILES)
@pytest.mark.parametrize("tool, read_only", [("read_file", True), ("write_file", False)])
def test_the_gate_refuses_every_file_in_the_credentials_folder(name, tool, read_only):
    decision = SecurityGate().evaluate(tool, is_read_only=read_only,
                                       file_path=f"/home/someone/.config/prometheus/{name}")
    assert not decision.allowed, f"{tool} of {name} was not refused"


def test_the_floor_holds_whatever_the_config_says():
    # A config that lists its own denied_paths (the mini's does) is the operator's list; the floor beneath it is not.
    gate = SecurityGate(denied_paths=["/etc", "/home/someone/backups"])
    decision = gate.evaluate("read_file", is_read_only=True,
                             file_path="/home/someone/.config/prometheus/env.bak-2026-11-16")
    assert not decision.allowed


def test_another_apps_config_folder_is_left_alone():
    decision = SecurityGate().evaluate("read_file", is_read_only=True,
                                       file_path="/home/someone/.config/other-app/settings.yaml")
    assert decision.allowed


def test_the_shipped_default_list_names_the_folder_and_older_lists_still_follow_it():
    assert "/*/.config/prometheus" in shipped_defaults.SHIPPED_DENIED_PATHS
    previous = shipped_defaults.DENIED_PATHS_DEFAULTS_SHIPPED[-2]
    assert "/*/.config/prometheus" not in previous
    # A config holding the previous shipped list exactly did not choose it: it gets the new one.
    assert shipped_defaults.resolve_denied_paths({"denied_paths": list(previous)}) == list(shipped_defaults.SHIPPED_DENIED_PATHS)


def test_the_template_default_equals_the_shipped_default():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert '"/*/.config/prometheus"' in text


def test_the_floor_is_in_the_structural_list():
    assert "/*/.config/prometheus" in checker._ALWAYS_DENIED_PATHS
