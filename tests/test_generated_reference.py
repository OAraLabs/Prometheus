"""The generated reference must be CURRENT — audit P11.3.

WHAT THIS REPLACES
------------------
Three inventories were maintained by hand, and all three had drifted,
because nothing compared them to the code. Measured the day this landed:

* ``docs/guide/api.md`` documented 89 of 104 real paths — **22 product
  endpoints undocumented**, including the entire ``/api/devices`` and
  ``/api/tasks`` families and all of ``/api/wiki/*``.
* ``features.md``'s Telegram table was missing six registered commands.
* No config-key reference existed at all, for a 396-key template.

WHY A GENERATOR IS NOT ENOUGH, AND THIS FILE IS THE POINT
----------------------------------------------------------
``scripts/gen_reference.py`` on its own would rot exactly like the hand-
written tables did: it only helps on the day someone remembers to run it.
The guard below regenerates IN MEMORY and compares against what is checked
in, so moving a route or renaming a command fails CI until the doc is
regenerated. The ratchet is the deliverable; the script is plumbing.

WHY THE GUIDES ARE NOT GENERATED
--------------------------------
``api.md``'s Purpose column carries explanation no generator can produce —
*"a dead subprocess reads unhealthy, never as empty success"*. Overwriting
that with a mechanical table would trade a drift problem for a worse
information problem. So: generated files are the COMPLETE inventory, the
guides stay the CURATED explanation, and only completeness — the half that
was actually failing — is taken away from human memory.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "docs" / "reference"

sys.path.insert(0, str(REPO / "scripts"))


def _gen():
    import gen_reference

    return gen_reference


@pytest.fixture(scope="module")
def rendered() -> dict[str, str]:
    return _gen().render_all()


def test_every_generated_file_is_current(rendered: dict[str, str]) -> None:
    """The ratchet. Source moves, the doc follows, or CI stops you."""
    stale = []
    for name, expected in rendered.items():
        path = REFERENCE / name
        if not path.exists():
            stale.append(f"{name} (missing)")
        elif path.read_text(encoding="utf-8") != expected:
            stale.append(name)
    assert not stale, (
        "generated reference is out of date: " + ", ".join(stale) + ".\n\n"
        "Something in the source moved and the doc did not follow — a route, "
        "a slash command, or a config key. Regenerate:\n"
        "    uv run python scripts/gen_reference.py"
    )


def test_the_files_say_they_are_generated(rendered: dict[str, str]) -> None:
    """A generated file a human edits by hand is worse than no file: the
    edit survives until the next regeneration silently discards it."""
    for name in rendered:
        head = (REFERENCE / name).read_text(encoding="utf-8")[:200]
        assert "DO NOT EDIT" in head, f"{name} lacks the generated banner"
        assert "gen_reference.py" in head, f"{name} does not name its generator"


def test_check_mode_agrees_with_the_files_on_disk() -> None:
    """``--check`` is what CI would call; it must pass on a clean tree.

    Asserted through the real entry point, because a --check that always
    returned 0 would make a CI wiring look green while proving nothing.
    """
    assert _gen().main.__module__  # imported, not shadowed
    import contextlib
    import io

    argv = sys.argv
    sys.argv = ["gen_reference.py", "--check"]
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = _gen().main()
    finally:
        sys.argv = argv
    assert code == 0, f"--check failed on a clean tree:\n{buf.getvalue()}"


def test_check_mode_actually_fails_when_a_file_is_stale(tmp_path) -> None:
    """The other direction — a --check that cannot fail gates nothing."""
    gen = _gen()
    name = "routes.md"
    path = REFERENCE / name
    original = path.read_text(encoding="utf-8")
    import contextlib
    import io

    argv = sys.argv
    try:
        path.write_text(original + "\n<!-- drift -->\n", encoding="utf-8")
        sys.argv = ["gen_reference.py", "--check"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = gen.main()
        assert code == 1, "--check passed on a file that had drifted"
        assert name in buf.getvalue()
    finally:
        sys.argv = argv
        path.write_text(original, encoding="utf-8")


# ── the inventories are REAL, not a plausible-looking table ────────────

def test_routes_come_from_the_live_app_not_a_regex() -> None:
    """Every path in the generated table must exist on the real app.

    A table built by grepping decorators would pass the currency check
    above while being wrong about what is actually mounted — the exact
    failure the hand-written doc had.
    """
    import yaml

    from prometheus.web.server import create_app

    app = create_app(yaml.safe_load(
        (REPO / "config" / "prometheus.yaml.default").read_text(encoding="utf-8")))
    live = {
        getattr(r, "path") for r in app.routes
        if getattr(r, "path", None) and getattr(r, "methods", None)
    }

    text = (REFERENCE / "routes.md").read_text(encoding="utf-8")
    daemon_section = text.split("## Daemon app")[1].split("## Setup app")[0]
    listed = set(re.findall(r"^\| `([^`]+)` \|", daemon_section, re.M))

    assert listed <= live, (
        "routes.md lists paths the app does not mount: "
        + ", ".join(sorted(listed - live))
    )
    # And nothing real is silently dropped (built-ins have their own section).
    builtins = set(_gen()._FASTAPI_BUILTINS)
    assert (live - builtins) <= listed, (
        "routes.md is missing live paths: "
        + ", ".join(sorted((live - builtins) - listed))
    )


def test_discord_commands_are_the_registered_ones_not_the_group_names() -> None:
    """Pins the bug this generator shipped with, and its fix.

    The first version matched ``name="..."`` anywhere in discord.py. It
    returned fifteen strings of which ZERO were commands — four group
    names, the ``/prometheus`` root, the provider overrides, and the
    logger name ``discord_gateway``. Discord registers into nested groups,
    so the invocation is ``/prometheus core help`` and a grep for the
    expected flat shape finds nothing real.
    """
    discord = _gen()._discord_commands()

    assert len(discord) > 40, (
        f"only {len(discord)} Discord commands found; discord.py's own "
        f"docstring says 43 families. The extractor has probably stopped "
        f"matching the registration call."
    )
    assert "discord_gateway" not in discord, (
        "the logger name is back in the command list — the extractor is "
        "matching `name=` kwargs again instead of `self._register(...)`"
    )
    for group in ("core", "session", "ops", "provider"):
        assert discord.get(group) != f"/prometheus {group}", (
            f"the GROUP {group!r} is being listed as a command"
        )
    # Every value is a real nested invocation, never a bare slash name.
    bad = [n for n, inv in discord.items() if not inv.startswith("/prometheus ")]
    assert not bad, f"Discord invocations must be nested: {bad}"


def test_config_key_table_covers_the_whole_template() -> None:
    """The count in the prose is parsed back out and re-derived, so the
    sentence cannot drift from the table it introduces."""
    import yaml

    template = yaml.safe_load(
        (REPO / "config" / "prometheus.yaml.default").read_text(encoding="utf-8"))
    expected = len(_gen()._flatten(template))

    text = (REFERENCE / "config-keys.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*(\d+)\*\*", text)
    assert m, "config-keys.md no longer states its key count"
    assert int(m.group(1)) == expected
    assert len(re.findall(r"^\| `", text, re.M)) == expected
