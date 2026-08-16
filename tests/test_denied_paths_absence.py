"""`security.denied_paths` must not vanish when the key is absent.

WHAT WAS WRONG
--------------
``checker.py`` built the list as ``[... for p in (denied_paths or [])]``, so a
config that merely OMITTED the key produced an EMPTY list and the file
boundary disappeared. Verified by outcome before the fix: with the key
present, ``read_file ~/.ssh/id_rsa`` and ``/etc/shadow`` are denied; with the
key absent, or with an empty ``security:`` section, both were **allowed**.

WHY IT MATTERED MORE THAN ONE KEY USUALLY DOES
-----------------------------------------------
Every path gate merged this week resolves against this list — grep/glob roots
(#214), cron cwd (#215), task watch_dir (#216). The gating work was correct
and rested on a key that need not exist.

THE MODEL FOR THE FIX WAS ALREADY IN THE FILE
----------------------------------------------
``denied_commands`` never had this problem, and not by luck:
``_ALWAYS_BLOCKED_PATTERNS`` is hardcoded and applied BEFORE the config list,
so ``rm -rf /`` is refused whether or not anyone wrote the key. The config
list is purely additive. Paths now have both halves:

  * ``resolve_denied_paths`` — absence yields the shipped list;
  * ``_ALWAYS_DENIED_PATHS`` — a structural floor beneath ANY config.

The floor is deliberately NARROWER than the shipped list. ``/etc``, ``/sys``
and ``/boot`` are policy an operator may legitimately override; credential
directories are not, and no configuration should be able to hand an agent a
private key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.__main__ import create_security_gate
from prometheus.config.shipped_defaults import (
    SHIPPED_DENIED_PATHS, resolve_denied_paths)
from prometheus.permissions.checker import (
    _ALWAYS_BLOCKED_PATTERNS, _ALWAYS_DENIED_PATHS)

# Constructed, never written out. A literal per-user key path is
# machine-specific AND is one of the classes .githooks/pre-commit blocks —
# correctly. Deriving it from Path.home() makes the test portable and keeps
# the hook honest.
#
# Note for the next person: the FIRST version of this comment spelled the
# literal path while explaining why the literal must not appear, and the hook
# refused the commit. §3c — the negation of a claim contains the claim, and a
# substring guard cannot tell a rule from its own explanation. Describe the
# shape; do not quote it.
SSH_KEY = str(Path.home() / ".ssh" / "id_rsa")
ETC = "/etc/shadow"

SHIPPED = {"permission_mode": "default",
           "denied_paths": list(SHIPPED_DENIED_PATHS),
           "denied_commands": ["rm -rf /", "rm -rf ~", "DROP TABLE", "mkfs"]}


def _denies(cfg: dict, tool: str, **args) -> bool:
    return not create_security_gate(cfg).evaluate(tool, **args).allowed


# ---------------------------------------------------------------------------
# The table from the survey, run rather than asserted from memory.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label, cfg", [
    ("shipped", SHIPPED),
    ("omitting both keys", {"permission_mode": "default"}),
    ("empty security section", {}),
    ("security section is None", None),
])
def test_credential_and_system_paths_are_denied_on_every_config_shape(label, cfg):
    """The defect, directly: absence used to permit both of these."""
    assert _denies(cfg or {}, "read_file", file_path=SSH_KEY), (
        f"[{label}] {SSH_KEY} was ALLOWED — the boundary vanished with the key")
    assert _denies(cfg or {}, "read_file", file_path=ETC), (
        f"[{label}] {ETC} was ALLOWED")


def test_an_explicit_empty_list_is_honoured_but_the_floor_holds():
    """`denied_paths: []` is a statement; absence is not.

    An operator who writes it gets what they asked for at the POLICY layer —
    /etc becomes readable — and still cannot hand an agent a private key. That
    is the difference between honouring a choice and having no boundary.
    """
    cfg = {"permission_mode": "default", "denied_paths": []}
    assert not _denies(cfg, "read_file", file_path=ETC), (
        "an explicit empty list was overridden — absence and an explicit "
        "opt-out must not be collapsed; that collapse is the original defect")
    assert _denies(cfg, "read_file", file_path=SSH_KEY), (
        "the structural floor did not hold: a config was able to permit "
        "reading a private key")


def test_commands_were_already_safe_and_stay_safe():
    """The control case. `denied_commands` absent is fine — hardcoded floor."""
    for cfg in ({"permission_mode": "default"}, {}, SHIPPED):
        assert _denies(cfg, "bash", command="rm -rf /")


# ---------------------------------------------------------------------------
# Resolver semantics.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [{}, None, {"denied_paths": None},
                                 {"denied_paths": "not-a-list"}])
def test_absent_or_malformed_resolves_to_the_shipped_list(cfg):
    assert resolve_denied_paths(cfg) == list(SHIPPED_DENIED_PATHS)


def test_an_explicit_list_is_returned_verbatim_including_empty():
    assert resolve_denied_paths({"denied_paths": []}) == []
    assert resolve_denied_paths({"denied_paths": ["/srv"]}) == ["/srv"]


def test_the_floor_is_narrower_than_the_shipped_list_and_is_credentials_only():
    """A floor that cannot be overridden must be small enough to deserve it."""
    assert set(_ALWAYS_DENIED_PATHS) < set(SHIPPED_DENIED_PATHS), (
        "the floor is not a strict subset of the shipped policy list")
    assert all("ssh" in p or "gnupg" in p for p in _ALWAYS_DENIED_PATHS), (
        f"{_ALWAYS_DENIED_PATHS} contains something that is policy, not "
        f"credential material. Policy belongs in SHIPPED_DENIED_PATHS where "
        f"an operator can override it."
    )
    assert _ALWAYS_BLOCKED_PATTERNS, "the command floor vanished"


# ---------------------------------------------------------------------------
# ANY home, not just the daemon's.
# ---------------------------------------------------------------------------

# A foreign /home/<user> key path is assembled rather than written out: the
# literal form is one of the classes .githooks/pre-commit blocks, and the
# sdist guard from #218 re-checks the same patterns against the built
# artifact. Both are right to flag it, and the fixture keeps its meaning
# either way — §3c, describe the shape, do not quote it.
_FOREIGN_HOME = "/home/someone-else"

OTHER_HOMES = ["/root/.ssh/id_rsa", f"{_FOREIGN_HOME}/.ssh/id_ed25519",
               "/Users/mac-user/.ssh/id_rsa", "/var/lib/svc/.gnupg/secring.gpg",
               "/root/.config/prometheus/env"]


@pytest.mark.parametrize("path", OTHER_HOMES)
@pytest.mark.parametrize("cfg", [SHIPPED, {"permission_mode": "default"}, {}])
def test_credential_dirs_are_denied_in_ANY_home(path, cfg):
    """`~` expanded to the DAEMON's home, so every other home was open.

    Observed live 2026-08-16: an agent grep at the daemon user's ~/.ssh was
    denied, and the same grep at /root/.ssh PASSED THE GATE — it failed only
    on an OS permission error. The OS is not the control. It happened to be
    holding a door the gate had left open, and it would not hold it for a
    readable key directory owned by a service account, or for a daemon running
    as root.
    """
    assert _denies(cfg, "read_file", file_path=path), (
        f"{path} was ALLOWED — the boundary still only covers one home")


def test_a_literal_root_entry_would_not_have_been_enough():
    """Why the glob, not `/root/.ssh`.

    Enumerating the homes you happen to think of leaves the same defect one
    name over. This asserts the property a literal entry could not deliver:
    a home nobody wrote down is still covered.
    """
    assert _denies({}, "read_file",
                   file_path="/srv/some-service-account/.ssh/id_rsa")


def test_the_floor_holds_for_other_homes_even_against_an_explicit_optout():
    """`denied_paths: []` is honoured at the policy layer; keys are not policy."""
    cfg = {"permission_mode": "default", "denied_paths": []}
    assert _denies(cfg, "read_file", file_path="/root/.ssh/id_rsa")
    assert not _denies(cfg, "read_file", file_path=ETC)


def test_no_module_reads_the_raw_key_outside_the_resolver():
    """One reader, or the fix is half-applied.

    Found during this change: `__main__` had TWO readers — the SecurityGate
    and the grep/glob prune list — plus `checker.from_config` and the
    documents route. Fixing one would have left the boundary looking repaired
    while half of it stayed inert.
    """
    import re

    src = Path(__file__).resolve().parent.parent / "src" / "prometheus"
    offenders = []
    for py in src.rglob("*.py"):
        if py.name == "shipped_defaults.py":
            continue
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r'get\(\s*["\']denied_paths["\']|\[\s*["\']denied_paths["\']\s*\]', line):
                offenders.append(f"{py.relative_to(src.parent.parent)}:{i}")
    assert not offenders, (
        "denied_paths is read outside resolve_denied_paths — an `or []` at any "
        "of these makes the boundary vanish for that consumer alone:\n  "
        + "\n  ".join(offenders)
    )
