"""Security floors on packages the extras bring in — each extra that declares
one must refuse the last vulnerable release and admit the first fixed one.

WHY THESE ARE FLOORS AND NOT JUST LOCK BUMPS
-------------------------------------------
uv.lock (and pip-audit over it) governs the deploy venv and CI. It does not
govern someone who runs `pip install -U 'oara-prometheus[full]'`: pip keeps
an installed dependency that still satisfies every requirement, so a
vulnerable transitive (aiohttp under discord.py, pydantic-settings under
mcp, strawberry-graphql under arize-phoenix) survives the upgrade unless a
requirement of OURS excludes it. Same reasoning as the base floors on
anyio and starlette (tests/test_web_is_base.py).

Each (last_bad, first_good) pair was checked with pip-audit: the first is
flagged, the second is clean. mcp's own floor lives in
tests/test_mcp_dependency_pin.py beside its upper bound.

Deliberately NOT floored (lock only): setuptools (voice/evals — its advisory
is in sdist building, which the daemon never does, and a floor would force
a setuptools upgrade into users' environments), and joserfc, pyasn1 and
pydantic-ai-slim (evals-only developer tooling the daemon never loads).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# (extra, distribution, last vulnerable, first fixed)
FLOORS = [
    ("mcp", "pydantic-settings", "2.14.1", "2.14.2"),   # CVE-2026-58203
    ("full", "pydantic-settings", "2.14.1", "2.14.2"),
    ("discord", "aiohttp", "3.14.2", "3.14.3"),         # CVE-2026-69244, the last of 3.13.x's
    ("full", "aiohttp", "3.14.2", "3.14.3"),
    # phoenix pins strawberry-graphql exactly: <16.4.0 pins 0.314.3 (3
    # advisories), 16.4.0 pins 0.316.0
    ("evals", "arize-phoenix", "16.3.0", "16.4.0"),
]


def _extra(name: str) -> dict[str, Requirement]:
    specs = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))[
        "project"]["optional-dependencies"][name]
    return {canonicalize_name(Requirement(s).name): Requirement(s) for s in specs}


@pytest.mark.parametrize(("extra", "dist", "last_bad", "first_good"), FLOORS,
                         ids=[f"{e}-{d}" for e, d, _, _ in FLOORS])
def test_the_extra_floors_its_vulnerable_dependency(extra, dist, last_bad, first_good):
    reqs = _extra(extra)
    assert dist in reqs, (
        f"`{extra}` no longer declares {dist} — an upgraded install keeps "
        f"whatever vulnerable version it already has")
    spec = reqs[dist].specifier
    assert not spec.contains(last_bad), (
        f"`{extra}`: {dist}{spec} admits {last_bad}, which has known advisories")
    assert spec.contains(first_good), f"`{extra}`: {dist}{spec} excludes {first_good}"
