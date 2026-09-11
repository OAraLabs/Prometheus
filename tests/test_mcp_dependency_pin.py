"""The declared `mcp` requirement must not admit 2.x.

WHY THIS EXISTS
---------------
`mcp = ["mcp>=1.0"]` is an unbounded floor. On any clean install after mcp
2.0.0 shipped, the resolver takes the newest — 2.2.0 at the time of writing —
and MCP tool discovery dies outright:

    mcp 1.9.4   list_tools(self, cursor: str | None = None)
    mcp 1.27.1  list_tools(self, cursor: str | None)              [+ a params overload]
    mcp 1.30.0  list_tools(self, cursor: str | None)              [+ a params overload]
    mcp 2.0.0   list_tools(self, *, params: PaginatedRequestParams | None = None)
    mcp 2.2.0   list_tools(self, *, params: PaginatedRequestParams | None = None)

The boundary is exactly 2.0.0, verified against the published wheels: every
1.x up to and including the newest still accepts `cursor=`, and 2.0.0 is the
first release that does not. `<2` is therefore the precise cap, not a
conservative guess — it excludes what breaks and nothing else.

`prometheus.mcp.runtime._list_all_tools` calls `list_tools(cursor=...)`, and it
passes the keyword UNCONDITIONALLY — on the first iteration too, when the
cursor is still None. So under 2.x it is not a pagination edge case: it is a
TypeError on the very first discovery call, and every configured server fails
to connect. Nothing downgrades gracefully, because nothing catches it as a
version problem.

WHAT THIS GUARDS, AND WHAT IT DOES NOT
--------------------------------------
This asserts the SPECIFIER's behaviour against real version strings, not the
presence of a `<` character in the text. `"mcp>=1.0,<2"` and `"mcp>=1.0"` both
contain the substring "mcp>=1.0"; only one of them excludes 2.2.0.

`test_the_unbounded_floor_would_still_admit_2x` is the mutation check: it
replays the exact known-bad shape and proves this assertion has power. Without
it, a specifier parser that silently returned "contains nothing" would make
every other test here pass green.

WHEN THE CAP COMES OFF
----------------------
The cap is honest about being an immediate fix, not a verdict on mcp 2.x.
Supporting 2.x means porting the call site to `params=`; when that lands,
`test_the_cap_and_its_reason_stay_together` is the test that fails and says so.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# The first mcp release carrying the keyword-only `params=` signature, and the
# newest one at the time this guard was written. Both must be excluded.
BREAKING = [Version("2.0.0"), Version("2.2.0")]
# Versions that DO accept `cursor=`. The cap must not be so tight it excludes
# what actually works. 1.27.1 is what uv.lock resolves to; 1.30.0 is the
# newest 1.x. Both are verified against the published wheels.
WORKING = [Version("1.9.4"), Version("1.27.1"), Version("1.30.0")]


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def _mcp_requirements() -> dict[str, Requirement]:
    """Every declared requirement on the `mcp` distribution, keyed by extra."""
    found: dict[str, Requirement] = {}
    for extra, specs in _extras().items():
        for spec in specs:
            req = Requirement(spec)
            if req.name == "mcp":
                found[extra] = req
    return found


def test_both_declaration_sites_are_present():
    """`full` duplicates the `mcp` extra by hand; a cap on one is half a fix."""
    declared = _mcp_requirements()
    assert set(declared) == {"mcp", "full"}, (
        f"expected the mcp requirement in exactly the 'mcp' and 'full' extras, "
        f"found it in: {sorted(declared)}"
    )


@pytest.mark.parametrize("extra", ["mcp", "full"])
@pytest.mark.parametrize("bad", BREAKING, ids=str)
def test_declared_specifier_excludes_2x(extra: str, bad: Version):
    req = _mcp_requirements()[extra]
    assert not req.specifier.contains(bad), (
        f"[{extra}] declared {str(req)!r} admits mcp {bad}, whose "
        f"ClientSession.list_tools is keyword-only `params=`. "
        f"MCP tool discovery raises TypeError on the first call."
    )


@pytest.mark.parametrize("extra", ["mcp", "full"])
@pytest.mark.parametrize("good", WORKING, ids=str)
def test_declared_specifier_still_admits_working_versions(extra: str, good: Version):
    req = _mcp_requirements()[extra]
    assert req.specifier.contains(good), (
        f"[{extra}] declared {str(req)!r} excludes mcp {good}, which accepts "
        f"`cursor=` and works. The cap is meant to exclude 2.x, not to strand "
        f"the extra on an old release."
    )


def test_the_locked_version_satisfies_the_declared_cap():
    """uv.lock and pyproject must not disagree about what is installable.

    A cap that excludes the pinned resolution is a lock that cannot install.
    """
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    locked: str | None = None
    lines = lock.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == 'name = "mcp"' and lines[i + 1].startswith("version = "):
            locked = lines[i + 1].split("=", 1)[1].strip().strip('"')
            break
    assert locked is not None, "no [[package]] entry for mcp found in uv.lock"

    for extra, req in _mcp_requirements().items():
        assert req.specifier.contains(Version(locked)), (
            f"[{extra}] declared {str(req)!r} excludes mcp {locked}, which is "
            f"the version uv.lock pins."
        )


@pytest.mark.parametrize("bad", BREAKING, ids=str)
def test_the_unbounded_floor_would_still_admit_2x(bad: Version):
    """Mutation check — replay the known-bad shape against these assertions.

    If this fails, the exclusion tests above are measuring nothing.
    """
    known_bad = Requirement("mcp>=1.0")
    assert known_bad.specifier.contains(bad), (
        "the original unbounded shape 'mcp>=1.0' no longer admits "
        f"{bad} — this test's premise is stale, and the exclusion tests "
        "above may be passing vacuously."
    )


def test_the_cap_and_its_reason_stay_together():
    """The cap exists because the call site is 1.x-shaped. Keep them in sync.

    When `_list_all_tools` is ported to mcp 2.x's `params=`, this fails —
    which is the intended prompt to revisit `<2` in the same change, rather
    than leaving a cap whose stated reason has quietly stopped being true.
    """
    source = (REPO_ROOT / "src" / "prometheus" / "mcp" / "runtime.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    keywords: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "list_tools":
            keywords = [kw.arg for kw in node.keywords if kw.arg is not None]
            break
    else:
        pytest.fail(
            "no `*.list_tools(...)` call found in mcp/runtime.py — the reason "
            "recorded for the `<2` cap no longer matches the code. Re-derive "
            "the cap rather than trusting this file's comment."
        )

    assert "cursor" in keywords, (
        f"mcp/runtime.py calls list_tools with {keywords!r}, not `cursor=`. "
        f"If the call site moved to mcp 2.x's keyword-only `params=`, the "
        f"`<2` cap in pyproject.toml should be re-evaluated in this change."
    )
