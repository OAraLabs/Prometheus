"""Path guards compare what a path IS, not how it is spelled (WP-X.27).

WHAT WAS WRONG
--------------
Each guard below resolved the path it was asked about and compared it with
entries it had NOT resolved, so wherever the two spellings differ the guard
never fired:

- on macOS ``/etc``, ``/tmp`` and ``/var`` are symlinks into ``/private``;
- on any host, a ``~`` entry, or any configured path that runs through a
  symlink (``$HOME`` behind one, a workspace under a linked directory).

The download tool's guard was fixed this way first (#574, WP-X.23). These
are the other readers of the same lists, fixed with #574's helpers, which
now live in ``security/path_guard.py`` and are shared, not copied:

- A1  workspace binding (``context/workspace.py``): entries not even
      ``~``-expanded; ``/workspace /etc`` accepted on macOS.
- A2  the grep/glob prune layer (``tools/denied_prune.py``): entries only
      ``~``-expanded; a denied directory's files returned in results.
- A3  the gate's ``rm -r`` protected roots (``permissions/checker.py``):
      ``/tmp/ws`` never equalled the stored ``/private/tmp/ws``.
- A4  the gate's glob ``denied_paths``: ``/tmp/*.secret`` never matched
      ``/private/tmp/x.secret``.
- A5/A6  the coding sandboxes compared glob entries as directory names.

Every test marked FAILS ON MAIN fails on origin/main (e854a1f) on macOS;
the symlink ones fail there on Linux too.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prometheus.coding.sandbox import DockerSandbox, ProcessSandbox, SandboxViolation
from prometheus.context.workspace import validate_workspace_path
from prometheus.permissions.checker import SecurityGate
from prometheus.security import path_guard
from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin import GrepTool
from prometheus.tools.denied_prune import is_denied, resolve_denied


@pytest.fixture
def linked(tmp_path):
    """``real/secret`` and ``link -> real``: one directory, two spellings."""
    real = tmp_path / "real"
    (real / "secret").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    return real, link


def _case_insensitive(directory: Path) -> bool:
    probe = directory / "CaseProbe"
    probe.mkdir()
    try:
        return (directory / "caseprobe").exists()
    finally:
        probe.rmdir()


# ── A1: workspace binding ───────────────────────────────────────────────────

def test_a_workspace_under_a_denied_dir_named_through_a_symlink_is_refused(linked):
    """FAILS ON MAIN."""
    real, link = linked
    (real / "secret" / "proj").mkdir()

    ws, why = validate_workspace_path(
        str(real / "secret" / "proj"), {"denied_paths": [str(link / "secret")]})

    assert ws is None, "a workspace inside a denied directory was accepted"
    assert str(link / "secret") in why  # named as configured


def test_a_workspace_under_a_tilde_entry_is_refused(tmp_path, monkeypatch):
    """FAILS ON MAIN. The entries came from the config as written, and a
    ``~/...`` entry never matched a resolved path."""
    home = tmp_path / "home"
    (home / "vault" / "notes").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    ws, why = validate_workspace_path(str(home / "vault" / "notes"), {"denied_paths": ["~/vault"]})

    assert ws is None
    assert "~/vault" in why


def test_the_shipped_etc_entry_refuses_etc_as_a_workspace():
    """FAILS ON MAIN on macOS, where ``/etc`` resolves to ``/private/etc``."""
    ws, why = validate_workspace_path("/etc", None)

    assert ws is None, "/etc was accepted as a workspace"
    assert "/etc" in why


def test_a_workspace_outside_every_denied_entry_is_still_accepted(linked, tmp_path):
    real, link = linked
    (tmp_path / "project").mkdir()

    ws, why = validate_workspace_path(str(tmp_path / "project"), {"denied_paths": [str(link / "secret")]})

    assert why is None
    assert ws == (tmp_path / "project").resolve()


def test_a_case_variant_of_a_denied_dir_is_the_same_dir_only_where_the_volume_says_so(tmp_path):
    """By identity: on a case-insensitive volume (APFS) ``VAULT`` IS
    ``vault`` and is refused; on a case-sensitive one it is another directory
    and is accepted. No string comparison can tell these apart."""
    (tmp_path / "vault").mkdir()
    insensitive = _case_insensitive(tmp_path)
    if not insensitive:
        (tmp_path / "VAULT").mkdir()

    ws, _ = validate_workspace_path(str(tmp_path / "VAULT"), {"denied_paths": [str(tmp_path / "vault")]})

    assert (ws is None) is insensitive


# ── A2: the grep/glob prune layer ───────────────────────────────────────────

def test_the_prune_layer_denies_a_file_under_a_denied_dir_named_through_a_symlink(linked):
    """FAILS ON MAIN."""
    real, link = linked
    key = real / "secret" / "key"
    key.write_text("x")

    assert is_denied(key, resolve_denied([str(link / "secret")]))


def test_the_prune_layer_honours_a_glob_whose_prefix_is_a_symlink(linked):
    """FAILS ON MAIN."""
    real, link = linked
    key = real / "a.key"
    key.write_text("x")

    assert is_denied(key, resolve_denied([f"{link}/*.key"]))
    assert not is_denied(real / "secret", resolve_denied([f"{link}/*.key"]))


def test_the_prune_layer_denies_etc_by_its_shipped_entry():
    """FAILS ON MAIN on macOS: ``/etc/hosts`` resolves to ``/private/etc/hosts``."""
    assert is_denied(Path("/etc/hosts"), resolve_denied(["/etc"]))


def test_grep_withholds_a_denied_file_it_reaches_by_its_real_path(linked, tmp_path):
    """FAILS ON MAIN. End to end: the key's line came back in the results."""
    real, link = linked
    (real / "secret" / "id_rsa").write_text("NEEDLE in a private key\n")
    (tmp_path / "notes.txt").write_text("NEEDLE in the open\n")

    out = asyncio.run(GrepTool(denied_paths=[str(link / "secret")]).execute(
        GrepTool.input_model(pattern="NEEDLE"), ToolExecutionContext(cwd=tmp_path))).output

    assert "private key" not in out
    assert "notes.txt" in out
    assert "withheld" in out


# ── A3: rm -r aimed at a protected root ─────────────────────────────────────

def test_rm_of_a_workspace_named_through_a_symlinked_parent_is_blocked(tmp_path):
    """FAILS ON MAIN. The workspace root is stored resolved; the operand was
    compared as typed. On macOS the same happens to any workspace under /tmp."""
    (tmp_path / "realp" / "ws").mkdir(parents=True)
    (tmp_path / "linkp").symlink_to(tmp_path / "realp", target_is_directory=True)
    gate = SecurityGate(workspace_root=str(tmp_path / "linkp" / "ws"))

    assert gate._rm_targets_a_protected_root(f"rm -rf {tmp_path / 'linkp' / 'ws'}")
    assert gate.evaluate("bash", command=f"rm -rf {tmp_path / 'linkp' / 'ws'}").action == "DENY"


def test_rm_of_home_by_its_real_path_is_blocked(tmp_path, monkeypatch):
    """FAILS ON MAIN. ``~`` was only expanded, so with $HOME behind a symlink
    the directory's real path was not a protected root."""
    real = tmp_path / "home-real"
    real.mkdir()
    (tmp_path / "home").symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert SecurityGate()._rm_targets_a_protected_root(f"rm -rf {real}")


def test_rm_through_a_symlink_with_a_trailing_slash_is_blocked(tmp_path):
    """FAILS ON MAIN. ``alias/`` names the directory the link points at."""
    (tmp_path / "ws").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "ws", target_is_directory=True)
    gate = SecurityGate(workspace_root=str(tmp_path / "ws"))

    assert gate._rm_targets_a_protected_root(f"rm -rf {tmp_path / 'alias'}/")


def test_rm_of_a_symlink_to_the_workspace_removes_only_the_link_and_is_allowed(tmp_path):
    """rm removes a symlink operand, not what it points at: not a root."""
    (tmp_path / "ws").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "ws", target_is_directory=True)
    gate = SecurityGate(workspace_root=str(tmp_path / "ws"))

    assert gate._rm_targets_a_protected_root(f"rm -rf {tmp_path / 'alias'}") == ""


def test_rm_inside_a_protected_root_is_still_allowed(tmp_path):
    """The roots themselves, not their contents."""
    (tmp_path / "ws" / "build").mkdir(parents=True)
    gate = SecurityGate(workspace_root=str(tmp_path / "ws"))

    assert gate._rm_targets_a_protected_root(f"rm -rf {tmp_path / 'ws' / 'build'}") == ""
    assert gate._rm_targets_a_protected_root("rm -rf /") != ""


# ── A4: the gate's glob denied_paths ────────────────────────────────────────

def test_a_glob_entry_with_a_symlinked_prefix_denies_the_resolved_path(linked):
    """FAILS ON MAIN."""
    real, link = linked
    gate = SecurityGate(denied_paths=[f"{link}/*.secret"])

    reason = gate._check_denied_path(str(real / "x.secret"))

    assert reason == f"Path {str(real / 'x.secret')!r} matches denied pattern {f'{link}/*.secret'!r}"
    assert gate._check_denied_path(str(real / "x.txt")) == ""


def test_a_tmp_glob_entry_denies_a_file_in_tmp():
    """FAILS ON MAIN on macOS, where /tmp is /private/tmp."""
    gate = SecurityGate(denied_paths=["/tmp/*.prometheus-wpx27"])

    assert "matches denied pattern" in gate._check_denied_path("/tmp/probe.prometheus-wpx27")


def test_a_glob_case_variant_follows_the_volume(tmp_path):
    """The glob's literal prefix is compared by identity, so ``KEYS/`` is
    ``keys/`` exactly where the volume says it is."""
    (tmp_path / "keys").mkdir()
    insensitive = _case_insensitive(tmp_path)
    if not insensitive:
        (tmp_path / "KEYS").mkdir()
    gate = SecurityGate(denied_paths=[f"{tmp_path}/keys/*.pem"])

    reason = gate._check_denied_path(str(tmp_path / "KEYS" / "a.pem"))

    assert bool(reason) is insensitive


def test_a_refusal_that_matched_before_names_the_same_entry(linked):
    """The first pass is main's comparison, so whatever was refused before
    is refused by the same entry: here the second one, which matched as
    written, not the first, which matches only once resolved."""
    real, link = linked
    (real / "secret" / "proj").mkdir()

    _, why = validate_workspace_path(
        str(real / "secret" / "proj"),
        {"denied_paths": [str(link / "secret"), str(real / "secret")]})

    assert why == f"{real / 'secret' / 'proj'} is under denied path {real / 'secret'}"


# ── A5/A6: the coding sandboxes ─────────────────────────────────────────────

def test_the_process_sandbox_honours_a_glob_entry(tmp_path):
    """FAILS ON MAIN: the glob was compared as a directory named ``*.secret``."""
    box = ProcessSandbox(root=tmp_path, denied_paths=(f"{tmp_path}/*.secret",))

    with pytest.raises(SandboxViolation, match="denied by policy"):
        box.resolve("a.secret")
    assert box.resolve("a.txt") == (tmp_path / "a.txt").resolve()


def test_the_docker_sandbox_honours_a_tilde_entry(tmp_path, monkeypatch):
    """FAILS ON MAIN: ``~/vault`` became ``<cwd>/~/vault``. (No caller passes
    denied_paths to the docker sandbox today; this is the latent case.)"""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "vault").mkdir()
    # The constructor talks to docker; no container is needed to resolve a path.
    with patch("prometheus.coding.sandbox.subprocess.run") as run:
        run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
        box = DockerSandbox(tmp_path, task_id="wpx27", denied_paths=["~/vault"])

    with pytest.raises(SandboxViolation, match="denied by policy"):
        box.resolve("vault/x")
    assert box.resolve("notes.txt") == (tmp_path / "notes.txt").resolve()


# ── the helpers are #574's, shared ──────────────────────────────────────────

def test_the_download_guard_uses_the_shared_helper_not_a_copy():
    """FAILS ON MAIN (the helper was private to the download tool)."""
    from prometheus.tools.builtin import download_file

    assert download_file.inside_by_identity is path_guard.inside_by_identity


def test_a_relative_entry_is_never_resolved_against_the_working_directory(tmp_path, monkeypatch):
    """Resolving a relative entry against the cwd is the defect
    ``_normalise_denied_path`` refuses to start on; the spellings must not
    reintroduce it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()

    assert path_guard.entry_spellings("config") == ("config",)
    assert path_guard.denying_entry(tmp_path / "config" / "x", ["config"]) is None


# ── from review: what an earlier version of this change got wrong ──────────

def test_a_denied_directory_whose_name_looks_like_a_glob_is_still_denied(tmp_path):
    """The sandboxes compared entries literally; ``[old]`` read as a pattern
    is a character class that matches neither itself nor anything under it."""
    (tmp_path / "[old]").mkdir()
    box = ProcessSandbox(root=tmp_path, denied_paths=(str(tmp_path / "[old]"),))

    with pytest.raises(SandboxViolation, match="denied by policy"):
        box.resolve("[old]/f.md")
    with pytest.raises(SandboxViolation, match="denied by policy"):
        box.resolve("[old]")


def test_an_absent_entry_denies_a_case_variant_only_where_the_volume_folds_case(tmp_path):
    """On a case-sensitive volume ``Secrets`` (absent) and ``secrets`` (there)
    are different directories, and the operator denied only the first."""
    (tmp_path / "secrets" / "proj").mkdir(parents=True)
    insensitive = _case_insensitive(tmp_path)
    entry = str(tmp_path / "Secrets")  # on a folding volume this IS secrets

    denied = path_guard.denying_entry(tmp_path / "secrets" / "proj", [entry])

    assert (denied == entry) is insensitive


def test_the_rm_guard_names_the_root_main_named(tmp_path, monkeypatch):
    """An identity match on an earlier root must not pre-empt the string
    match main made on a later one: the message is the same."""
    (tmp_path / "realp" / "home").mkdir(parents=True)
    (tmp_path / "linkp").symlink_to(tmp_path / "realp", target_is_directory=True)
    monkeypatch.setenv("HOME", str(tmp_path / "linkp" / "home"))
    gate = SecurityGate(workspace_root="~")

    reason = gate._rm_targets_a_protected_root("rm -rf ~")

    assert reason.endswith(f"resolves to {tmp_path / 'linkp' / 'home'}"), reason


# ── the credential floor by case (macOS) ────────────────────────────────────

@pytest.fixture
def home_with_ssh(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("not a real key\n")
    monkeypatch.setenv("HOME", str(home))
    return home


def test_the_ssh_floor_denies_a_case_variant_where_it_is_the_same_file(home_with_ssh):
    """FAILS ON MAIN on macOS: ``~/.SSH/id_rsa`` IS ``~/.ssh/id_rsa`` there,
    and the floor ``/*/.ssh`` was matched as text, case and all."""
    insensitive = _case_insensitive(home_with_ssh)
    gate = SecurityGate()  # the always-denied floor alone

    assert gate._check_denied_path(str(home_with_ssh / ".ssh" / "id_rsa"))
    reason = gate._check_denied_path(str(home_with_ssh / ".SSH" / "id_rsa"))

    assert bool(reason) is insensitive
    if insensitive:
        assert reason.endswith("matches denied pattern '/*/.ssh'")


def test_creating_a_case_variant_of_an_absent_ssh_dir_is_refused_where_it_would_be_ssh(
        tmp_path, monkeypatch):
    """FAILS ON MAIN on macOS: with no ``~/.ssh`` yet, writing
    ``~/.SSH/authorized_keys`` creates the directory sshd reads as ``~/.ssh``."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    insensitive = _case_insensitive(home)

    reason = SecurityGate()._check_denied_path(str(home / ".SSH" / "authorized_keys"))

    assert bool(reason) is insensitive


# ── from the second review: the prefilter, the floor's wildcard part, and
#    keeping main's words where main already refused ──────────────────────

def test_a_firmlink_spelling_in_any_case_is_still_the_denied_directory():
    """On macOS ``/system/volumes/data/private/etc`` (any case) IS
    ``/private/etc``. Where there is no Data volume the path is simply absent."""
    firmlinked = Path("/System/Volumes/Data/private/etc").exists()
    gate = SecurityGate(denied_paths=["/etc"])

    for spelling in ("/System/Volumes/Data/private/etc/hosts",
                     "/system/volumes/data/private/etc/hosts",
                     "/SYSTEM/Volumes/Data/private/etc/hosts"):
        assert bool(gate._check_denied_path(spelling)) is firmlinked, spelling


def test_the_env_floor_denies_a_case_variant_of_its_wildcard_part(tmp_path, monkeypatch):
    """FAILS ON MAIN on macOS: ``~/.config/app/.ENV`` IS ``.env`` there, and
    ``*env`` in the floor ``/*/.config/*/*env`` was matched as text."""
    home = tmp_path / "home"
    (home / ".config" / "app").mkdir(parents=True)
    (home / ".config" / "app" / ".env").write_text("TOKEN=not-real\n")
    monkeypatch.setenv("HOME", str(home))
    insensitive = _case_insensitive(home)

    reason = SecurityGate()._check_denied_path(str(home / ".config" / "app" / ".ENV"))

    assert bool(reason) is insensitive


def test_a_link_to_a_glob_named_directory_is_compared_as_a_name(tmp_path):
    (tmp_path / "[old]").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "[old]", target_is_directory=True)
    insensitive = _case_insensitive(tmp_path)

    denied = path_guard.denying_entry(tmp_path / "[OLD]" / "f", [str(tmp_path / "link")])

    assert (denied is not None) is insensitive


def test_documents_keep_the_gates_words_for_what_the_gate_refused(tmp_path, monkeypatch):
    """The documents API ran the sandbox, then the gate. The sandbox now
    refuses glob entries too; the 403 must still carry the gate's reason, as
    it did when the gate was the only layer to refuse them."""
    from prometheus.config.shipped_defaults import SHIPPED_DENIED_PATHS
    from prometheus.documents import DocumentsError, DocumentsService

    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".config" / "app").mkdir(parents=True)
    (tmp_path / ".config" / "app" / ".env").write_text("TOKEN=not-real\n")
    denied = list(SHIPPED_DENIED_PATHS)
    svc = DocumentsService(tmp_path, denied_paths=denied, gate=SecurityGate(denied_paths=denied))

    with pytest.raises(DocumentsError) as exc:
        svc.read(".config/app/.env")

    assert exc.value.status == 403
    assert exc.value.message == (
        f"denied by SecurityGate: Path {str(tmp_path / '.config' / 'app' / '.env')!r} "
        "matches denied pattern '/*/.config/*/*env'")


def test_the_sandbox_names_the_entry_its_literal_comparison_named(tmp_path):
    """A glob listed first must not take the message from the literal entry
    main's sandbox matched."""
    (tmp_path / "data").mkdir()
    box = ProcessSandbox(root=tmp_path, denied_paths=(f"{tmp_path}/*", str(tmp_path / "data")))

    with pytest.raises(SandboxViolation) as exc:
        box.resolve("data/a.key")

    assert str(exc.value).endswith(f"(denied root: {tmp_path / 'data'})")


def test_the_rm_guard_names_the_operand_main_named(tmp_path, monkeypatch):
    """Two operands: main blocked on the second, by its spelling; the first
    is a protected root only by identity (``~/ws/..`` IS ``~``)."""
    home = tmp_path / "home"
    (home / "ws").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    gate = SecurityGate(workspace_root=str(home / "ws"))

    reason = gate._rm_targets_a_protected_root("rm -rf ~/ws/.. ~/.prometheus")

    assert "'~/.prometheus' resolves to" in reason, reason
