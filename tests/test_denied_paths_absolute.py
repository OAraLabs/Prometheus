"""`security.denied_paths` — the control whose target was chosen by cwd.

THE DEFECT
----------
Entries were re-resolved on every check with ``Path(denied).expanduser()
.resolve()``. For an absolute entry that is a no-op. For a RELATIVE one —
and the shipped template carried exactly one, ``config/prometheus.yaml`` —
``resolve()`` expands against the **process's working directory**, so the
file the entry protected was decided by wherever the daemon happened to be
running.

Proven on 2026-08-13: moving the daemon from ``~/Prometheus`` to a
fast-forward-only deploy clone at ``~/prometheus-deploy`` moved that entry
from one config file to the other. The dev checkout's live config became
unprotected, and nothing in the config, the logs, or the restart said so.

A control that cannot say which file it protects is not a control.

WHY RAISING, AND NOT SOMETHING GENTLER
--------------------------------------
Ignoring a relative entry removes a control the operator believes they have.
Resolving it reinstates the defect. Both fail quietly, and quiet is the
property that made this survive. So a relative entry stops the process with
the fix in the message — the same posture as the deploy guard.

THE HALF THAT DOES NOT FIT IN A CONFIG
---------------------------------------
No template can name an absolute path to the daemon's own config that is
right for every install — which is *why* the entry was relative in the first
place. So the config file is denied as a PROPERTY instead: the process knows
where it loaded its config from, and denies that (CROSS-CUTTING §5 — a
property that cannot be violated beats a check that must remember to run).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.permissions.checker import SecurityGate

#: Tests must not hardcode one machine's home — and the pre-commit
#: secret hook rejects a literal ~/.ssh path outright, correctly.
HOME = Path.home()


def _denied(gate: SecurityGate, path: str) -> bool:
    return bool(gate._check_denied_path(path))


# ── Relative entries are rejected, loudly ───────────────────────────────────

@pytest.mark.parametrize("entry", [
    "config/prometheus.yaml",       # the real one, from the shipped template
    "secrets.env",
    "./config/prometheus.yaml",
    "../outside/thing",
])
def test_a_relative_entry_raises(entry):
    with pytest.raises(ValueError) as exc:
        SecurityGate(denied_paths=[entry])
    msg = str(exc.value)
    assert entry in msg, "the message must name the offending entry"
    # It must explain the MECHANISM, not just say "invalid" — the whole reason
    # this shipped for months is that nothing connected cwd to the outcome.
    assert "WORKING DIRECTORY" in msg
    assert "absolute" in msg.lower()


def test_the_message_says_the_config_file_needs_no_entry():
    """Otherwise the obvious fix is to hardcode an absolute path to it, which
    breaks the moment the install moves — the same defect, one step later."""
    with pytest.raises(ValueError) as exc:
        SecurityGate(denied_paths=["config/prometheus.yaml"])
    assert "automatically" in str(exc.value)


@pytest.mark.parametrize("entry", ["/etc", "~/.ssh", "~/.config/*/*env"])
def test_absolute_and_tilde_entries_are_accepted(entry):
    """ADMISSION direction: the guard must not reject legitimate entries, or
    no config is loadable at all."""
    SecurityGate(denied_paths=[entry])


# ── The entries deny what they claim, and nothing else ──────────────────────

@pytest.fixture
def gate() -> SecurityGate:
    return SecurityGate(denied_paths=[
        "/etc", "/sys", "/boot", "~/.ssh", "~/.gnupg", "~/.config/*/*env",
    ])


@pytest.mark.parametrize("path", [
    "/etc/passwd",
    str(HOME / ".ssh/id_rsa"),
    str(HOME / ".gnupg/secring.gpg"),
])
def test_denied_paths_are_denied(gate, path):
    assert _denied(gate, path)


def test_both_credential_env_files_are_covered(gate):
    """The pattern exists for two REAL files with different names. Written as
    ``~/.config/*/env`` first, which matched only the first of them — a
    plausible-looking pattern that half-worked is how ``audio/mp3`` shipped.
    """
    assert _denied(gate, str(HOME / ".config/prometheus/env"))
    assert _denied(gate, str(HOME / ".config/oara/middleware.env"))


@pytest.mark.parametrize("path", [
    str(HOME / "projects/app.py"),
    str(HOME / ".config/nvim/init.lua"),
    str(HOME / ".config/foo/environment.conf"),   # ends in .conf, not env
])
def test_ordinary_paths_are_not_denied(gate, path):
    """ADMISSION direction. An over-broad deny list looks exactly like a
    working one from the breach side, and quietly makes the agent useless."""
    assert not _denied(gate, path)


# ── The config file is denied as a property ─────────────────────────────────

def test_the_loaded_config_path_is_denied_without_an_entry(tmp_path):
    cfg = tmp_path / "prometheus.yaml"
    cfg.write_text("security: {}\n")
    gate = SecurityGate(denied_paths=["/etc"], config_path=cfg)
    assert _denied(gate, str(cfg)), (
        "the daemon's own config must be denied from the path it was loaded "
        "from — that is what replaces the relative entry"
    )


def test_a_different_config_file_is_not_denied(tmp_path):
    """It denies THE loaded config, not every file called prometheus.yaml —
    the whole point is that the target is exact rather than positional."""
    cfg = tmp_path / "prometheus.yaml"
    cfg.write_text("security: {}\n")
    other = tmp_path / "elsewhere" / "prometheus.yaml"
    other.parent.mkdir()
    other.write_text("security: {}\n")
    gate = SecurityGate(denied_paths=["/etc"], config_path=cfg)
    assert _denied(gate, str(cfg))
    assert not _denied(gate, str(other))


def test_no_config_path_means_no_auto_deny(tmp_path):
    """Back-compat for the 39 construction sites that pass nothing."""
    gate = SecurityGate(denied_paths=["/etc"])
    assert not _denied(gate, str(tmp_path / "prometheus.yaml"))


def test_from_config_denies_the_file_it_read(tmp_path):
    """Far side, through the real loader — the path is threaded, not guessed."""
    cfg = tmp_path / "prometheus.yaml"
    cfg.write_text("security:\n  denied_paths: ['/etc']\n")
    gate = SecurityGate.from_config(cfg)
    assert _denied(gate, str(cfg))


# ── The target must not move when the process does ──────────────────────────

def test_denials_are_independent_of_the_working_directory(tmp_path, monkeypatch):
    """THE regression test. Same gate, two working directories, same verdicts.

    Before the fix an entry was re-resolved per call, so this test would have
    had different answers in each directory for a relative entry — which is
    precisely what happened to the live box.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    (a / "config").mkdir(parents=True)
    (b / "config").mkdir(parents=True)
    gate = SecurityGate(denied_paths=["~/.ssh", "/etc"])

    monkeypatch.chdir(a)
    first = [_denied(gate, p) for p in ("/etc/passwd", str(a / "config" / "x.yaml"))]
    monkeypatch.chdir(b)
    second = [_denied(gate, p) for p in ("/etc/passwd", str(a / "config" / "x.yaml"))]

    assert first == second, "denial verdicts changed with the working directory"


def test_the_template_carries_no_relative_entry():
    """The shipped template is a config too, and it is the one every install
    starts from. It carried the offending entry for months."""
    import yaml

    repo = Path(__file__).resolve().parent.parent
    tpl = yaml.safe_load(
        (repo / "config" / "prometheus.yaml.default").read_text(encoding="utf-8"))
    entries = tpl["security"]["denied_paths"]
    relative = [e for e in entries if not Path(e).expanduser().is_absolute()]
    assert not relative, (
        f"the shipped template has relative denied_paths entries {relative} — "
        f"a fresh install would refuse to boot"
    )
    # And the template must load through the real constructor.
    SecurityGate(denied_paths=entries)


# ── Gaps the mutation matrix exposed (all three fixes are HERE, not in the
#    write-up: three mutations survived the first pass) ──────────────────────

def test_every_stored_entry_is_absolute():
    """M2 survived: the cwd-independence test used only absolute entries, for
    which re-resolving is a no-op — so it could not observe the mutation.

    The behavioural test is still right, but what actually makes it true is
    this INVARIANT: after construction there is no relative entry left to be
    resolved against anything. Assert the property, not just a sample of its
    consequences (CROSS-CUTTING §5).
    """
    gate = SecurityGate(denied_paths=["/etc", "~/.ssh", "~/.config/*/*env"])
    assert gate._denied_paths
    for entry in gate._denied_paths:
        assert Path(entry).is_absolute(), f"{entry!r} survived normalisation"


@pytest.mark.parametrize("path,denied", [
    ("/etc/passwd", True),
    ("/etc", True),
    ("/etcetera/notes.md", False),   # the over-match
    ("/etc-backup/x", False),
])
def test_prefix_matching_respects_path_components(path, denied):
    """M5 survived: forcing every entry down the glob branch changed almost
    nothing — which is only possible if the two branches nearly agree. The gap
    was exactly here: a raw ``startswith`` denied ``/etcetera/notes.md`` for
    the entry ``/etc``, while the glob branch has always compared
    component-wise. Two branches of one matcher must not disagree about what
    "under" means.

    This NARROWS a security control, deliberately: it removes an unintended
    over-refusal, which is the direction that never announces itself (§2c).
    """
    gate = SecurityGate(denied_paths=["/etc"])
    assert bool(gate._check_denied_path(path)) is denied


def _template_denied_paths() -> list[str]:
    import yaml

    repo = Path(__file__).resolve().parent.parent
    tpl = yaml.safe_load(
        (repo / "config" / "prometheus.yaml.default").read_text(encoding="utf-8"))
    return tpl["security"]["denied_paths"]


@pytest.mark.parametrize("path", [
    str(HOME / ".config/prometheus/env"),
    str(HOME / ".config/oara/middleware.env"),
    str(HOME / ".gnupg/secring.gpg"),
    str(HOME / ".ssh/id_rsa"),
])
def test_the_SHIPPED_template_actually_denies_the_credential_paths(path):
    """M8 survived: narrowing the template's pattern to the half-working
    ``~/.config/*/env`` broke nothing, because every other test built its own
    hand-written entry list. The mechanism was covered; the SHIPPED CONFIG was
    not (§2d — assert what the consumer actually receives).

    Parametrised over the template itself, so the assertion follows the file
    rather than a copy of it.
    """
    gate = SecurityGate(denied_paths=_template_denied_paths())
    assert gate._check_denied_path(path), (
        f"the shipped denied_paths do not cover {path}"
    )


def test_the_shipped_template_still_admits_ordinary_work():
    """The other direction for the same list — a template that denied
    everything would pass every test above."""
    gate = SecurityGate(denied_paths=_template_denied_paths())
    for path in (str(HOME / "projects/app.py"),
                 str(HOME / ".config/nvim/init.lua")):
        assert not gate._check_denied_path(path), f"{path} was denied"


def test_a_directory_glob_also_denies_what_is_under_it():
    """M9 survived: every glob in the tests matched a FILE directly, so the
    ``pattern + "/*"`` half of the check was never exercised and could be
    deleted with the suite still green. A pattern naming a directory has to
    cover its contents, or ``~/projects/*/secrets`` would deny the directory
    and admit every file inside it — the useless half of a control.
    """
    gate = SecurityGate(denied_paths=["~/projects/*/secrets"])
    assert _denied(gate, str(HOME / "projects/app/secrets"))
    assert _denied(gate, str(HOME / "projects/app/secrets/prod.key"))
    assert not _denied(gate, str(HOME / "projects/app/src/main.py"))
