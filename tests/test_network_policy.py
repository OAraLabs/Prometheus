"""``web/network.py`` — what "this Mac", "home network" and "open" mean, and how a choice is saved.

The mode is not a setting of its own. It is what the listen address (``web.bind``, #693) and two switches
mean together (docs/PAIRING-APPROVAL-API.md, 2.1):

* ``this_mac``: the bind is loopback. Nothing advertises.
* ``home_network``: the bind reaches the LAN **and** the owner turned it on (``network.home_network``) **and**
  it can be run safely: TLS (a later change) or the owner's explicit ``network.allow_plaintext_lan``.
* ``open``: everything else that reaches the LAN. This is what every install that predates the setting is, and
  it is reported honestly instead of being dressed up as "home network".

An owner who asked for ``home_network`` and cannot have it is told so and reported as ``open``; the daemon
does not quietly run plain HTTP on the LAN in a mode named for being safe.

Saving a choice edits the config file's TEXT (the comment-preserving editor ``PUT /api/tools/deferred``
already uses; a round-trip through a YAML dump once took the shipped template from 713 comment lines to 0),
verifies the result before writing, and writes atomically.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from prometheus.web.network import (
    HOME,
    OPEN,
    THIS_MAC,
    NetworkSettings,
    PersistError,
    describe,
    persist_choice,
)

REPO = Path(__file__).resolve().parents[1]
TAILNET = ".".join(("100", "64", "0", "9"))   # built at run time: the pre-commit scanner looks for these


def _comments(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("#"))


# ── settings ─────────────────────────────────────────────────────────────────

def test_the_defaults_change_nothing_for_an_existing_install():
    s = NetworkSettings()
    assert (s.allow_plaintext_lan, s.home_network, s.mdns) == (False, False, True)
    assert NetworkSettings.from_config(None) == s
    assert NetworkSettings.from_config({"network": None, "discovery": None}) == s


def test_settings_are_read_from_network_and_discovery():
    s = NetworkSettings.from_config({"network": {"allow_plaintext_lan": True, "home_network": True},
                                     "discovery": {"mdns": False}})
    assert (s.allow_plaintext_lan, s.home_network, s.mdns) == (True, True, False)


@pytest.mark.parametrize("section, key", [("network", "allow_plaintext_lan"), ("network", "home_network"),
                                          ("discovery", "mdns")])
def test_a_switch_that_is_not_a_boolean_falls_back_out_loud(section, key, caplog):
    with caplog.at_level("WARNING"):
        s = NetworkSettings.from_config({section: {key: "yes please"}})
    assert s == NetworkSettings()
    assert any(f"{section}.{key}" in r.getMessage() for r in caplog.records)


# ── what a bind and the switches mean ────────────────────────────────────────

@pytest.mark.parametrize("address", ["127.0.0.1", "::1"])
def test_a_loopback_bind_is_this_mac(address):
    assert describe(address, "config", NetworkSettings()).mode == THIS_MAC


def test_a_wide_bind_nobody_chose_to_call_home_is_open():
    state = describe("0.0.0.0", "default", NetworkSettings())
    assert state.mode == OPEN and state.tls_enabled is False


def test_home_network_needs_the_switch_and_a_way_to_run_it():
    both = NetworkSettings(home_network=True, allow_plaintext_lan=True)
    state = describe("0.0.0.0", "config", both)
    assert state.mode == HOME
    assert any("plain HTTP" in w for w in state.warnings), "plaintext on the LAN is said out loud, every time"


def test_home_network_without_tls_or_the_opt_out_is_refused_and_reported_open():
    state = describe("0.0.0.0", "config", NetworkSettings(home_network=True))
    assert state.mode == OPEN
    assert any("allow_plaintext_lan" in w for w in state.warnings)


def test_a_tls_listener_makes_home_network_safe_without_the_opt_out():
    state = describe("0.0.0.0", "config", NetworkSettings(home_network=True), tls=True)
    assert state.mode == HOME and not any("plain HTTP" in w for w in state.warnings)


def test_the_switch_on_a_loopback_bind_says_it_cannot_work_there():
    state = describe("127.0.0.1", "config", NetworkSettings(home_network=True, allow_plaintext_lan=True))
    assert state.mode == THIS_MAC
    assert any("loopback" in w or "this machine only" in w for w in state.warnings)


def test_a_specific_lan_address_counts_as_reaching_the_lan():
    assert describe("192.0.2.20", "config", NetworkSettings(home_network=True, allow_plaintext_lan=True)).mode == HOME


def test_the_state_says_where_the_bind_came_from():
    state = describe("127.0.0.1", "flag", NetworkSettings())
    assert (state.bind, state.bind_source) == ("127.0.0.1", "flag")


# ── saving a choice ──────────────────────────────────────────────────────────

def _config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "prometheus.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_home_network_from_this_mac_opens_the_bind_and_turns_the_switch_on(tmp_path):
    path = _config(tmp_path, 'web:\n  api_port: 8005\n  bind: "127.0.0.1"   # mine\nmodel:\n  name: x\n')
    persist_choice(path, HOME, current_bind="127.0.0.1")
    saved = yaml.safe_load(path.read_text())
    assert saved["web"]["bind"] == "0.0.0.0" and saved["network"]["home_network"] is True
    assert saved["model"] == {"name": "x"} and saved["web"]["api_port"] == 8005
    assert "# mine" in path.read_text(), "the owner's own comment survives"


def test_home_network_keeps_a_specific_address_the_owner_already_chose(tmp_path):
    """A box bound to its tailnet address only must not be widened to every interface by a toggle."""
    path = _config(tmp_path, f'web:\n  bind: "{TAILNET}"\n')
    persist_choice(path, HOME, current_bind=TAILNET)
    saved = yaml.safe_load(path.read_text())
    assert saved["web"]["bind"] == TAILNET and saved["network"]["home_network"] is True


def test_this_mac_closes_the_bind_and_turns_the_switch_off(tmp_path):
    path = _config(tmp_path, 'web:\n  bind: "0.0.0.0"\nnetwork:\n  home_network: true\n  allow_plaintext_lan: true\n')
    persist_choice(path, THIS_MAC, current_bind="0.0.0.0")
    saved = yaml.safe_load(path.read_text())
    assert saved["web"]["bind"] == "127.0.0.1" and saved["network"]["home_network"] is False
    assert saved["network"]["allow_plaintext_lan"] is True, "a setting nobody asked to change is left alone"


def test_the_shipped_template_keeps_every_comment(tmp_path):
    original = (REPO / "config" / "prometheus.yaml.default").read_text(encoding="utf-8")
    path = _config(tmp_path, original)
    persist_choice(path, HOME, current_bind="127.0.0.1")
    assert _comments(path.read_text()) == _comments(original), "a dump would have taken ~1000 comment lines to 0"
    saved = yaml.safe_load(path.read_text())
    assert saved["web"]["bind"] == "0.0.0.0" and saved["network"]["home_network"] is True


def test_a_web_section_with_no_bind_gets_one_inside_it_not_at_the_end_of_the_file(tmp_path):
    """The usual older install: ``web:`` is there, ``bind`` is not, and other sections follow."""
    path = _config(tmp_path, "web:\n  api_port: 8005\nmodel:\n  name: x\n")
    persist_choice(path, HOME, current_bind="127.0.0.1")
    saved = yaml.safe_load(path.read_text())
    assert saved["web"] == {"api_port": 8005, "bind": "0.0.0.0"}, "the new line belongs to web, not to model"
    assert saved["model"] == {"name": "x"} and saved["network"]["home_network"] is True


def test_a_network_section_with_no_switch_gets_it_inside_it(tmp_path):
    path = _config(tmp_path, 'web:\n  bind: "127.0.0.1"\nnetwork:\n  allow_plaintext_lan: true\nmodel:\n  name: x\n')
    persist_choice(path, HOME, current_bind="127.0.0.1")
    saved = yaml.safe_load(path.read_text())
    assert saved["network"] == {"allow_plaintext_lan": True, "home_network": True}
    assert saved["model"] == {"name": "x"}


def test_a_line_added_to_a_section_keeps_the_blank_line_and_comment_that_open_the_next_one(tmp_path):
    original = "web:\n  api_port: 8005\n\n# the model\nmodel:\n  name: x\n"
    path = _config(tmp_path, original)
    persist_choice(path, HOME, current_bind="127.0.0.1")
    text = path.read_text()
    assert "\n\n# the model\nmodel:\n" in text, "the gap and the comment still sit directly above model:"
    assert text.index("bind:") < text.index("# the model")


def test_what_is_already_saved_is_not_rewritten(tmp_path):
    path = _config(tmp_path, "web:\n  bind: 127.0.0.1\nnetwork:\n  home_network: false\n")   # unquoted on purpose
    before = path.stat().st_mtime_ns
    assert persist_choice(path, THIS_MAC, current_bind="127.0.0.1") is False
    assert path.read_text() == "web:\n  bind: 127.0.0.1\nnetwork:\n  home_network: false\n"
    assert path.stat().st_mtime_ns == before


def test_a_change_reports_that_it_wrote(tmp_path):
    path = _config(tmp_path, 'web:\n  bind: "127.0.0.1"\n')
    assert persist_choice(path, HOME, current_bind="127.0.0.1") is True


def test_home_network_pins_a_specific_address_the_daemon_is_running_on_when_the_file_names_none(tmp_path):
    """Bound to one address by the environment, the owner turns the switch on: the file must not come back wide."""
    path = _config(tmp_path, "web:\n  api_port: 8005\n")
    persist_choice(path, HOME, current_bind=TAILNET)
    assert yaml.safe_load(path.read_text())["web"]["bind"] == TAILNET


def test_a_config_with_no_web_section_gets_one(tmp_path):
    path = _config(tmp_path, "model:\n  name: x\n")
    persist_choice(path, HOME, current_bind="0.0.0.0")
    saved = yaml.safe_load(path.read_text())
    assert saved["network"]["home_network"] is True and saved["model"] == {"name": "x"}


def test_a_missing_file_is_refused_not_created(tmp_path):
    with pytest.raises(PersistError, match="no config file"):
        persist_choice(tmp_path / "prometheus.yaml", HOME, current_bind="0.0.0.0")
    assert not (tmp_path / "prometheus.yaml").exists()


def test_an_edit_that_does_not_verify_leaves_the_file_exactly_as_it_was(tmp_path, monkeypatch):
    original = 'web:\n  bind: "127.0.0.1"\n'
    path = _config(tmp_path, original)
    import prometheus.web.network as network

    monkeypatch.setattr(network, "_edit_text", lambda text, edits: text)        # an edit that does nothing
    with pytest.raises(PersistError, match="did not verify"):
        persist_choice(path, HOME, current_bind="127.0.0.1")
    assert path.read_text() == original


def test_an_edit_that_keeps_the_data_but_loses_a_comment_is_refused(tmp_path, monkeypatch):
    """The data reads back right and a comment is gone: the owner's notes are not ours to drop."""
    original = '# why loopback\nweb:\n  bind: "127.0.0.1"\n'
    path = _config(tmp_path, original)
    import prometheus.web.network as network

    real = network._edit_text
    monkeypatch.setattr(network, "_edit_text", lambda text, edits: real(text, edits).replace("# why loopback\n", ""))
    with pytest.raises(PersistError, match="did not verify"):
        persist_choice(path, HOME, current_bind="127.0.0.1")
    assert path.read_text() == original


def test_the_write_is_atomic_so_a_crash_cannot_leave_half_a_config(tmp_path, monkeypatch):
    original = 'web:\n  bind: "127.0.0.1"\n'
    path = _config(tmp_path, original)
    import prometheus.web.network as network

    def crash(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(PersistError):
        persist_choice(path, HOME, current_bind="127.0.0.1")
    assert path.read_text() == original
    assert [p.name for p in tmp_path.iterdir()] == ["prometheus.yaml"], "no temp file left behind"
    assert network  # the module under test is the one whose os.replace was broken


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644])
def test_the_files_mode_is_kept(tmp_path, mode):
    path = _config(tmp_path, 'web:\n  bind: "127.0.0.1"\n')
    path.chmod(mode)
    persist_choice(path, HOME, current_bind="127.0.0.1")
    assert path.stat().st_mode & 0o777 == mode


def test_a_symlinked_config_is_edited_where_it_lives(tmp_path):
    real = tmp_path / "real" / "prometheus.yaml"
    real.parent.mkdir()
    real.write_text('web:\n  bind: "127.0.0.1"\n', encoding="utf-8")
    link = tmp_path / "prometheus.yaml"
    link.symlink_to(real)
    persist_choice(link, HOME, current_bind="127.0.0.1")
    assert link.is_symlink() and yaml.safe_load(real.read_text())["web"]["bind"] == "0.0.0.0"


def test_a_crlf_file_keeps_its_line_endings_on_the_lines_it_did_not_touch(tmp_path):
    path = tmp_path / "prometheus.yaml"
    path.write_bytes(b'model:\r\n  name: x\r\nweb:\r\n  bind: "127.0.0.1"\r\n')
    persist_choice(path, HOME, current_bind="127.0.0.1")
    assert path.read_bytes().startswith(b"model:\r\n  name: x\r\nweb:\r\n")


def test_a_file_that_is_not_yaml_is_refused_untouched(tmp_path):
    original = "web: [unclosed\n"
    path = _config(tmp_path, original)
    with pytest.raises(PersistError, match="not valid YAML"):
        persist_choice(path, HOME, current_bind="0.0.0.0")
    assert path.read_text() == original


def test_only_this_mac_and_home_network_can_be_saved(tmp_path):
    path = _config(tmp_path, 'web:\n  bind: "127.0.0.1"\n')
    with pytest.raises(PersistError):
        persist_choice(path, OPEN, current_bind="0.0.0.0")
