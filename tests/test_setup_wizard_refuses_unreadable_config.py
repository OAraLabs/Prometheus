"""THE INVARIANT: setup never overwrites a config file it could not parse.

WHY THIS IS THE ONE THAT MATTERS
--------------------------------
Every other site in the config-honesty sweep degrades to defaults. This one
destroys the operator's configuration, and it needs nothing to be wrong with
DEFAULTS_PATH to do it — the wizard writes to its own explicitly-resolved path.

The path, before this change:

  1. ``run()`` calls ``_load_existing_config()``, which returned None on a
     parse failure — the same value it returns for "no file".
  2. ``if existing and not self._gateway_only:`` is therefore False, so
     ``_ask_rerun()`` — the "Start fresh (overwrite everything)" prompt — is
     NEVER SHOWN. The operator is not asked.
  3. ``_write_config()`` reads the file again, gets ``{}``, and
     ``_save_config`` writes a fresh config over it.

A single YAML typo in a hand-edited config therefore cost the whole file, on a
run that printed nothing alarming and exited 0. Adding a log line does not fix
that: a non-interactive run has nobody to read it, and an interactive operator
sees it scroll past above a successful-looking setup. So the wizard refuses.

BOTH DIRECTIONS
---------------
Every test that asserts a refusal also asserts the file is byte-identical
afterwards, and the suite includes the positive control — a VALID config still
reaches the normal rerun prompt. A wizard that refused everything would satisfy
"did not overwrite" while being completely broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from prometheus import setup_wizard as sw
from prometheus.setup_wizard import SetupWizard

BROKEN = "model:\n  provider: llama_cpp\n  base_url: [unclosed\n"
VALID = "model:\n  provider: llama_cpp\n  base_url: http://localhost:8080\n"


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    """Point the wizard at a temp config path."""
    path = tmp_path / "prometheus.yaml"
    monkeypatch.setattr(sw, "_config_target", lambda: path)
    return path


def _wizard(interactive: bool, monkeypatch, answers=None):
    monkeypatch.setattr(sw.sys.stdin, "isatty", lambda: interactive)
    if answers is not None:
        it = iter(answers)
        monkeypatch.setattr(sw, "_input", lambda *a, **k: next(it))
    return SetupWizard()


# ── non-interactive: refuse, never write, exit non-zero ───────────────────

def test_non_interactive_refuses_and_leaves_the_file_untouched(cfg, monkeypatch, capsys):
    cfg.write_text(BROKEN)
    before = cfg.read_bytes()

    assert _wizard(False, monkeypatch).run() is False, (
        "run() must return False so cli/setup.py exits non-zero"
    )
    assert cfg.read_bytes() == before, "the unparseable config was modified"
    assert not list(cfg.parent.glob("*.bak*")), (
        "a non-interactive run must not create a backup either — it does "
        "nothing to that path at all"
    )
    err = capsys.readouterr().err
    assert "could not be parsed" in err
    assert "Refusing to overwrite" in err
    assert "run setup again" in err, "must tell the operator how to proceed"


def test_cli_setup_maps_the_refusal_to_exit_status_1(cfg, monkeypatch):
    """The refusal is only worth anything if the process exits non-zero."""
    from prometheus.cli import setup as cli_setup

    cfg.write_text(BROKEN)
    monkeypatch.setattr(sw.sys.stdin, "isatty", lambda: False)
    rc = cli_setup.run_setup(type("A", (), {"gateway_only": False})())
    assert rc == 1
    assert cfg.read_text() == BROKEN


# ── interactive: offered a backup, defaulting to cancel ───────────────────

def test_interactive_cancel_is_the_default_and_preserves_the_file(cfg, monkeypatch, capsys):
    cfg.write_text(BROKEN)
    # Empty input => _input returns the default => choice 2 => cancel.
    assert _wizard(True, monkeypatch, answers=[""]).run() is False
    assert cfg.read_text() == BROKEN
    assert "unchanged" in capsys.readouterr().err


def test_interactive_backup_copies_the_file_and_says_where(cfg, monkeypatch, capsys):
    cfg.write_text(BROKEN)
    wiz = _wizard(True, monkeypatch, answers=["1"])
    wiz._load_existing_config()
    assert wiz._handle_unreadable_config() is True

    backup = cfg.with_suffix(cfg.suffix + ".bak")
    assert backup.read_text() == BROKEN, "the backup must hold the ORIGINAL bytes"
    out = capsys.readouterr().out
    assert str(backup) in out, "the operator must be told where it went"


def test_backup_never_clobbers_an_earlier_backup(cfg, monkeypatch):
    cfg.write_text(BROKEN)
    first = cfg.with_suffix(cfg.suffix + ".bak")
    first.write_text("AN EARLIER BACKUP")

    wiz = _wizard(True, monkeypatch, answers=["1"])
    wiz._load_existing_config()
    assert wiz._handle_unreadable_config() is True

    assert first.read_text() == "AN EARLIER BACKUP", "clobbered an older backup"
    assert Path(f"{first}.1").read_text() == BROKEN


def test_a_failed_backup_refuses_rather_than_overwriting(cfg, monkeypatch, capsys):
    """Failing to protect the file is not a licence to destroy it."""
    cfg.write_text(BROKEN)
    wiz = _wizard(True, monkeypatch, answers=["1"])
    wiz._load_existing_config()

    def _boom(self, data):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "write_bytes", _boom)
    assert wiz._handle_unreadable_config() is False
    assert cfg.read_text() == BROKEN
    assert "Refusing to continue" in capsys.readouterr().err


# ── the positive control ──────────────────────────────────────────────────

def test_a_valid_existing_config_still_reaches_the_rerun_prompt(cfg, monkeypatch):
    """A wizard that refused everything would pass every test above."""
    cfg.write_text(VALID)
    wiz = _wizard(True, monkeypatch)

    asked: list[str] = []
    monkeypatch.setattr(sw, "_ask_choice",
                        lambda prompt, *a, **k: (asked.append(prompt), 4)[1])
    assert wiz.run() is False           # 4 == Cancel
    assert wiz._unreadable_config is None, "a valid config is not 'unreadable'"
    assert asked and "Existing configuration found" in asked[0], (
        "the normal overwrite prompt must still be reached"
    )


def test_an_absent_config_is_not_treated_as_unreadable(cfg, monkeypatch):
    """The distinction the old code could not make."""
    assert not cfg.exists()
    wiz = _wizard(True, monkeypatch)
    assert wiz._load_existing_config() is None
    assert wiz._unreadable_config is None


def test_an_empty_config_file_is_not_a_refusal(cfg, monkeypatch):
    """safe_load returns None on an empty file, but it PARSED.

    `or {}` already handles it, and refusing here would block a legitimate
    `touch prometheus.yaml && prometheus setup`.
    """
    cfg.write_text("# nothing yet\n")
    wiz = _wizard(True, monkeypatch)
    assert wiz._load_existing_config() == {}
    assert wiz._unreadable_config is None
