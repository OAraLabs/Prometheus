"""Every role in the offered set must be a name AT-SPI can actually emit.

`_CLICKABLE_ROLES` contained "tab". AT-SPI's role enum has "page tab" and never
"tab", so that entry never matched anything — tab switching was unreachable
behind an allowlist entry that made the set look like it covered it. "button"
is the same: AT-SPI emits "push button", "toggle button", "radio button", never
bare "button".

A dead entry is worse than a missing one: a missing role is visibly absent,
while a dead one reads as coverage. Same shape as the a11y-bus constant that
did not match what the system emitted.

⚠ This test reads the HOST's AT-SPI enum, so it is skipped where the bindings
are absent rather than asserting from a hardcoded copy of the names — a second
copy of the vocabulary is how the original defect was built.
"""

from __future__ import annotations

import pytest

from prometheus.computer.candidates import _CLICKABLE_ROLES, _EDITABLE_ROLES

atspi = pytest.importorskip(
    "gi.repository.Atspi",
    reason="AT-SPI bindings absent; the role vocabulary cannot be interrogated",
)

#: Dead entries that are KNOWN and not yet corrected, each with its reason and
#: its fix. A RATCHET, like mypy-debt.txt: the guard stops NEW dead entries
#: appearing, and fixing one means deleting its line here. Empty is the goal.
#:
#: ⚠ Both are held because correcting them changes what `build_candidates`
#: offers, and the extent carries NO role-set term — so any change to the
#: offered set retroactively widens every stored click grant. That decision is
#: Will's and is open. See docs/computer-use-corpus.md §7.
KNOWN_DEAD: dict[str, str] = {
    "tab": (
        "AT-SPI emits 'page tab'. Correcting it is a dead-string fix rather "
        "than a widening — the entry was always meant to cover tabs — but it "
        "still changes the offered set, so it waits on the grant-versioning "
        "decision."
    ),
    "button": (
        "AT-SPI emits 'push button' / 'toggle button' / 'radio button', all "
        "three already in the set. So this entry is redundant as well as dead "
        "and the fix is deletion, not correction. Removing it offers strictly "
        "LESS, so it does not widen any grant — but it is kept here until the "
        "set is touched deliberately rather than in passing."
    ),
}


def _valid_role_names() -> set[str]:
    import gi

    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi

    names = {
        Atspi.role_get_name(getattr(Atspi.Role, a))
        for a in dir(Atspi.Role)
        if a.isupper()
    }
    names.discard(None)
    return names


@pytest.mark.parametrize("role_set_name", ["_CLICKABLE_ROLES", "_EDITABLE_ROLES"])
def test_no_role_in_the_offered_set_is_dead(role_set_name):
    roles = {"_CLICKABLE_ROLES": _CLICKABLE_ROLES,
             "_EDITABLE_ROLES": _EDITABLE_ROLES}[role_set_name]
    valid = _valid_role_names()
    dead = sorted(r for r in roles if r not in valid and r not in KNOWN_DEAD)
    assert not dead, (
        f"{role_set_name} contains {dead}, which AT-SPI never emits. Such an "
        f"entry matches nothing while making the set look like it covers that "
        f"class of control. Either correct it to the real name (AT-SPI has "
        f"'page tab', not 'tab'; 'push button', not 'button') or record it in "
        f"KNOWN_DEAD with the reason."
    )


def test_the_atspi_vocabulary_is_readable_at_all():
    """If this ever returns nothing the test above passes vacuously."""
    valid = _valid_role_names()
    assert len(valid) > 50, f"only {len(valid)} role names — enum not readable"
    assert "push button" in valid and "page tab" in valid
