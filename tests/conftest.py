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

import warnings

import pytest

from tests.support.doubles import registry
from tests.support.plumbing_allowlist import PLUMBING_TARGETS

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


@pytest.fixture(autouse=True)
def _isolated_state_dirs(tmp_path, monkeypatch):
    """Point ALL live-state roots at per-test tmp dirs.

    Found live 2026-07-05: a test_wiring MessageEvent(chat_id=99) flowed
    through TelegramAdapter.on_message → _save_chat_id → the REAL
    ``~/.prometheus/last_telegram_chat_id`` — so every suite run broke the
    daemon's startup greeting (400 "Chat not found" on chat 99). Tests
    must never touch ``~/.prometheus`` or ``~/.config/prometheus``; this
    kills the whole leak class. Tests that need a specific dir still win —
    their own monkeypatch.setenv overrides this default.
    """
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "prom-config"))
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(tmp_path / "prom-env"))


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


def _sentinel_target_name(target: object, attr: str) -> str:
    """Render a monkeypatched target as '<owner>.<attr>' for allowlist matching."""
    owner = getattr(target, "__module__", None)
    qual = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if owner and qual and owner != qual:
        return f"{owner}.{qual}.{attr}"
    return f"{qual or type(target).__name__}.{attr}"


def _warn_unregistered_substitutes(item: pytest.Item) -> None:
    """Coverage sentinel (checkpoint-1 condition): an acceptance test that
    monkeypatches a behavior substitute NOT in the double registry gets a loud
    WARNING — the gap self-announces until TRIPWIRE-2 registers the long tail.

    Heuristic by design: scans the function-scoped ``monkeypatch`` fixture's
    setattr log; a replacement that is callable (or a class) and carries no
    ``__tripwire_record__`` and is not in the plumbing allowlist is flagged.
    unittest.mock.patch and fixture-private MonkeyPatch instances are outside
    this net (documented limitation).
    """
    mp = item.funcargs.get("monkeypatch") if hasattr(item, "funcargs") else None
    setattr_log = getattr(mp, "_setattr", None) if mp is not None else None
    if not setattr_log:
        return
    for target, attr, _old in setattr_log:
        current = getattr(target, attr, None)
        if not callable(current):
            continue  # value/config swap, not a behavior substitute
        if getattr(current, "__tripwire_record__", None) is not None:
            continue  # registered double — the tripwire proper covers it
        if isinstance(current, type) and getattr(current, "__tripwire_record__", None) is not None:
            continue
        name = _sentinel_target_name(target, attr)
        if name in PLUMBING_TARGETS:
            continue
        warnings.warn(
            f"ACCEPTANCE TEST USED UNREGISTERED SUBSTITUTE: {name} — tripwire "
            "cannot verify this path until it is registered (TRIPWIRE-2).",
            UserWarning,
            stacklevel=2,
        )


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
    _warn_unregistered_substitutes(item)
    return result
