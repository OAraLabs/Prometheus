"""Root test conftest — TRIPWIRE enforcement (Piece 1).

This file exists for exactly one purpose: make ``@pytest.mark.acceptance``
mean something structural. An acceptance-marked test that PASSES while any
registered double (tests/support/doubles.py) was touched is turned into a
FAILURE with an explicit message — "green over doubles" can no longer merge
quietly (#73/#74 failure class).

Escape hatch: ``@pytest.mark.acceptance(allow_doubles=["name", ...])`` for
genuinely-unreachable externals (e.g. the real Anthropic API). Names must be
listed individually; wildcards are rejected loudly. The allowance is per-test
and visible at the test site — never global, never implicit.
"""

from __future__ import annotations

import pytest

from tests.support.doubles import registry

_VIOLATION_TEMPLATE = (
    "ACCEPTANCE TEST TERMINATED IN DOUBLE: {name} (replaces {replaces}). "
    "An acceptance test must exercise the real path. Either un-double it or "
    "demote the test (remove the acceptance mark) — do not suppress this check."
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "acceptance(allow_doubles=[...]): end-to-end test entering at a real "
        "entry point; FAILS if any registered double is touched on its path. "
        "allow_doubles lists individually-named permitted doubles (no wildcards).",
    )


@pytest.fixture(autouse=True)
def _tripwire_reset():
    """Zero every double's touch counter so enforcement reads only the current test."""
    registry.reset()
    yield


def _validated_allowance(marker: pytest.Mark) -> set[str]:
    allow = marker.kwargs.get("allow_doubles", [])
    if isinstance(allow, str) or not isinstance(allow, (list, tuple, set)):
        pytest.fail(
            "acceptance(allow_doubles=...) must be a list/tuple/set of individual "
            f"double names, got {allow!r} — no bare strings, no blanket allows.",
            pytrace=False,
        )
    names = set()
    for entry in allow:
        if not isinstance(entry, str) or not entry or "*" in entry:
            pytest.fail(
                f"acceptance(allow_doubles=...) entries must be explicit double names, "
                f"got {entry!r} — wildcards are not permitted, name each double.",
                pytrace=False,
            )
        names.add(entry)
    unknown = names - registry.known_names()
    if unknown:
        pytest.fail(
            f"acceptance(allow_doubles=...) names unknown double(s) {sorted(unknown)} — "
            f"known: {sorted(registry.known_names())}. Register the double or fix the name.",
            pytrace=False,
        )
    return names


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    try:
        result = yield
    except BaseException:
        raise  # the test failed on its own — report that, don't mask it

    marker = item.get_closest_marker("acceptance")
    if marker is None:
        return result

    allowed = _validated_allowance(marker)
    violations = [r for r in registry.touched_records() if r.name not in allowed]
    if violations:
        pytest.fail(
            "\n".join(
                _VIOLATION_TEMPLATE.format(name=r.name, replaces=r.replaces)
                for r in violations
            ),
            pytrace=False,
        )
    return result
