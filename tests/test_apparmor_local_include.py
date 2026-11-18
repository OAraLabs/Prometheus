"""The shipped bash profile reads machine-local rules from /etc/apparmor.d/local/prometheus-bash.

An owner's own folders (a backup folder, say) are not the project's to name in a public profile, and a rule added
by hand to the installed profile is lost the next time a release installs the shipped one. The local file is the
place for them: ``include if exists`` means a machine without it is unaffected, and a deploy that replaces the
profile leaves the local file alone. ``deny`` beats any allow whatever its position, so where the include sits
does not weaken a rule above it, but it must sit INSIDE the profile's braces (a rule outside them is a parse error).
"""

from __future__ import annotations

import re
from pathlib import Path

PROFILE = Path(__file__).resolve().parent.parent / "deploy" / "apparmor" / "prometheus-bash"
INCLUDE = re.compile(r"^\s*include if exists <local/prometheus-bash>\s*$", re.M)


def _text() -> str:
    return PROFILE.read_text(encoding="utf-8")


def test_the_profile_includes_the_local_file_if_it_exists():
    assert len(INCLUDE.findall(_text())) == 1


def test_the_include_is_inside_the_profile_block():
    text = _text()
    match = INCLUDE.search(text)
    assert match is not None
    opening = text.index("{", text.index("profile prometheus-bash"))
    closing = text.rindex("}")
    assert opening < match.start() < closing


def test_the_profile_names_no_owner_folder():
    # The shipped profile is generic: a machine's own folders belong in the local file.
    for line in _text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        assert "backups" not in line
