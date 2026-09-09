"""Phase C — the config analogue of deploy_guard.sh's staleness check.

The failure this exists for, measured on 2026-09-07: the live config pinned
``model.max_tool_iterations_cloud: 100`` over a shipped default of 500, and
nothing said so. ``/health`` was green, the tracked tree was clean, the deploy
guard exited 0, and every cloud turn ran at a fifth of the intended ceiling.

``test_config_drift.py`` catches the PRESENCE half of this class and documents
that it checks one direction only. These tests cover the VALUE half.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from prometheus.config import shipped_defaults as sd
from prometheus.config.divergence import (
    CONFIG_SOURCES,
    GOVERNED,
    ConfigValue,
    _same,
    describe,
    warn_on_divergence,
)

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "prometheus"


class _RecordingLog:
    """Captures lazy %-formatted warnings the way logging would render them."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, msg: str, *args: object) -> None:
        self.warnings.append(msg % args if args else msg)


def _source_for(config: dict, key: str) -> str:
    return _value_for(config, key).source


def _value_for(config: dict, key: str) -> ConfigValue:
    match = [v for v in describe(config) if v.key == key]
    assert len(match) == 1, f"{key} is not governed"
    return match[0]


# ---------------------------------------------------------------------------
# The four states, each produced by a real config
# ---------------------------------------------------------------------------


def test_absent_key_is_shipped_default_not_config():
    """THE POINT. 'no value' and 'value equals default' must not render alike.

    This is Will's live config after 2026-09-08: the cloud cap line was
    DELETED so it inherits the shipped value. Reporting that as `config`
    would say the operator pinned 500, which is exactly the fact that must
    survive the next raise.
    """
    agreeing = {"model": {"max_tool_iterations": sd.SHIPPED_MAX_TOOL_ITERATIONS}}
    value = _value_for(agreeing, "model.max_tool_iterations_cloud")
    assert value.source == "shipped_default"
    assert value.resolved == sd.SHIPPED_MAX_TOOL_ITERATIONS_CLOUD


def test_present_and_equal_is_config():
    agreeing = {"model": {"max_tool_iterations": sd.SHIPPED_MAX_TOOL_ITERATIONS}}
    value = _value_for(agreeing, "model.max_tool_iterations")
    assert value.source == "config"
    assert value.resolved == sd.SHIPPED_MAX_TOOL_ITERATIONS


def test_present_and_different_is_an_override():
    """The 2026-09-07 defect itself, as it was on disk."""
    pinned = sd.SHIPPED_MAX_TOOL_ITERATIONS_CLOUD // 5
    assert pinned != sd.SHIPPED_MAX_TOOL_ITERATIONS_CLOUD, "pick a real override"
    value = _value_for({"model": {"max_tool_iterations_cloud": pinned}},
                       "model.max_tool_iterations_cloud")
    assert value.source == "config_override"
    assert value.resolved == pinned
    assert value.shipped == sd.SHIPPED_MAX_TOOL_ITERATIONS_CLOUD


@pytest.mark.parametrize("bad", [0, -5, "abc", [], {}])
def test_present_but_unusable_is_rejected_not_config(bad):
    """A value the resolver refuses must not read as 'the operator chose 500'."""
    value = _value_for({"model": {"max_tool_iterations": bad}},
                       "model.max_tool_iterations")
    assert value.source == "config_rejected", (
        f"{bad!r} resolved to {value.resolved!r} tagged {value.source!r}"
    )
    assert value.resolved == sd.SHIPPED_MAX_TOOL_ITERATIONS


def test_every_source_is_declared():
    for value in describe({}):
        assert value.source in CONFIG_SOURCES


# ---------------------------------------------------------------------------
# The list/tuple trap that produced a wrong finding in the Phase B sweep
# ---------------------------------------------------------------------------


def test_a_list_equal_to_a_shipped_tuple_is_not_a_divergence():
    """SHIPPED_DENIED_PATHS is a tuple; YAML yields a list. `!=` lies."""
    config = {"security": {"denied_paths": list(sd.SHIPPED_DENIED_PATHS)}}
    assert _source_for(config, "security.denied_paths") == "config"


def test_a_genuinely_different_list_still_diverges():
    """Guard the guard: normalising must not swallow a real difference."""
    config = {"security": {"denied_paths": ["/etc"]}}
    assert _source_for(config, "security.denied_paths") == "config_override"


def test_same_does_not_equate_a_scalar_with_a_one_element_sequence():
    assert _same(("a",), ["a"]) is True
    assert _same("a", ["a"]) is False
    assert _same(["a"], "a") is False


def test_an_explicitly_empty_list_is_an_override_not_an_absence():
    """`denied_paths: []` is a deliberate statement the resolver honours."""
    value = _value_for({"security": {"denied_paths": []}}, "security.denied_paths")
    assert value.source == "config_override"
    assert value.resolved == []


# ---------------------------------------------------------------------------
# Warn, do not refuse — and say which value wins
# ---------------------------------------------------------------------------


def test_the_warning_names_the_key_both_values_and_the_winner():
    pinned = sd.SHIPPED_MAX_TOOL_ITERATIONS_CLOUD // 5
    log = _RecordingLog()
    warn_on_divergence({"model": {"max_tool_iterations_cloud": pinned}}, log=log)

    hits = [w for w in log.warnings if "max_tool_iterations_cloud" in w]
    assert len(hits) == 1, log.warnings
    line = hits[0]
    assert str(pinned) in line, "the configured value is not named"
    assert str(sd.SHIPPED_MAX_TOOL_ITERATIONS_CLOUD) in line, (
        "the shipped value is not named"
    )
    assert "CONFIG WINS" in line, "the line does not say which value is in force"


def test_a_rejected_value_says_it_does_nothing():
    log = _RecordingLog()
    warn_on_divergence({"model": {"max_tool_iterations": 0}}, log=log)
    hits = [w for w in log.warnings if "max_tool_iterations" in w and "cloud" not in w]
    assert len(hits) == 1, log.warnings
    assert "does nothing" in hits[0]
    assert str(sd.SHIPPED_MAX_TOOL_ITERATIONS) in hits[0], (
        "the value actually in force is not named"
    )


def test_agreeing_and_absent_keys_stay_silent():
    """Two quiet states. A warning per boot for a correct config is noise."""
    log = _RecordingLog()
    warn_on_divergence({}, log=log)
    assert log.warnings == []


def test_divergence_warns_but_never_refuses():
    """No raise, and the resolved values are returned unchanged."""
    log = _RecordingLog()
    values = warn_on_divergence(
        {"gateway": {"telegram_enabled": True}, "model": {"max_tool_iterations": 0}},
        log=log,
    )
    assert len(values) == len(GOVERNED)
    assert log.warnings, "a diverging config produced no warning at all"


def test_it_reaches_the_real_logger(caplog):
    pinned = sd.SHIPPED_MAX_TOOL_ITERATIONS_CLOUD // 5
    with caplog.at_level("WARNING", logger="prometheus.config.divergence"):
        warn_on_divergence({"model": {"max_tool_iterations_cloud": pinned}})
    assert any("max_tool_iterations_cloud" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Robustness: a hand-mangled config must not crash the boot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", [
    None, {}, {"model": None}, {"model": "not-a-dict"},
    {"gateway": {"media": None}}, {"gateway": {"media": "nope"}},
    {"tools": {"deferred_loading": None}},
])
def test_a_malformed_config_still_describes_every_key(config):
    values = describe(config)
    assert len(values) == len(GOVERNED)
    assert all(v.source == "shipped_default" for v in values)


# ---------------------------------------------------------------------------
# Anti-rot: the registry cannot silently fall behind
# ---------------------------------------------------------------------------


def _public_shipped_constants() -> dict[str, object]:
    return {
        name: getattr(sd, name)
        for name in dir(sd)
        if name.startswith("SHIPPED_")
    }


def test_every_shipped_constant_is_governed():
    """Add a SHIPPED_* default without governing it and this goes red."""
    constants = _public_shipped_constants()
    governed_values = [g.shipped for g in GOVERNED]

    assert len(GOVERNED) == len(constants), (
        f"{len(constants)} SHIPPED_* constants but {len(GOVERNED)} governed "
        f"keys: {sorted(constants)} vs {sorted(g.key for g in GOVERNED)}. A "
        "shipped default nothing governs can be overridden in silence — the "
        "exact failure Phase C exists to end. Govern it, or say here why it "
        "cannot be."
    )
    for name, value in constants.items():
        assert any(_same(value, gv) for gv in governed_values), (
            f"{name} is not reachable from any governed key"
        )


def test_governed_keys_use_the_daemons_own_resolvers():
    """A reported number and an enforced number must not drift.

    resolve_effective_limit's docstring names that drift as having shipped
    twice. Reimplementing precedence here would be the third, so the entries
    must hold the REAL resolver objects, not copies of their logic.
    """
    by_key = {g.key: g.resolve for g in GOVERNED}
    assert by_key["model.max_tool_iterations"] is sd.resolve_max_tool_iterations
    assert by_key["model.max_tool_iterations_cloud"] is sd.resolve_max_tool_iterations_cloud
    assert by_key["gateway.telegram_enabled"] is sd.resolve_telegram_enabled
    assert by_key["security.workspace_root"] is sd.resolve_workspace_root
    assert by_key["security.denied_paths"] is sd.resolve_denied_paths


def test_always_loaded_mirror_agrees_with_the_real_loader():
    """The one duplicated rule, pinned to its original.

    `always_loaded` resolves inline in DynamicToolLoader.__init__ rather than
    behind a resolve_*, so divergence.py mirrors it. If the loader's rule
    changes, this fails instead of the report quietly describing a value
    nothing enforces.
    """
    from prometheus.__main__ import create_tool_registry
    from prometheus.context.dynamic_tools import DynamicToolLoader

    for deferred in ({}, {"always_loaded": []}, {"always_loaded": ["bash", "grep"]}):
        loader = DynamicToolLoader(create_tool_registry({}), dict(deferred) or None)
        mirrored = _value_for(
            {"tools": {"deferred_loading": dict(deferred)}},
            "tools.deferred_loading.always_loaded",
        ).resolved
        assert set(mirrored) == set(loader._always_loaded), (
            f"mirror disagrees with DynamicToolLoader for {deferred!r}"
        )


def test_the_daemon_actually_calls_it_at_boot():
    """Structural: a reporter nothing invokes is the silence it replaces."""
    tree = ast.parse((SRC / "daemon.py").read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "warn_on_divergence" in called, (
        "daemon.py imports the divergence reporter but never calls it"
    )


# ---------------------------------------------------------------------------
# Rejection is decided by the resolver's ADMISSION rule, not by the outcome
# ---------------------------------------------------------------------------


def test_a_coerced_string_is_honoured_not_rejected():
    """MEASURED: ``_positive_int`` obeys "500" — ``int("500")`` parses.

    Its docstring claims a quoted number is 'treated as "not configured"
    rather than obeyed', which is false. Classifying it as rejected would
    tell an operator their working line does nothing.
    """
    quoted = str(sd.SHIPPED_MAX_TOOL_ITERATIONS)
    value = _value_for({"model": {"max_tool_iterations": quoted}},
                       "model.max_tool_iterations")
    assert value.source == "config"
    assert value.resolved == sd.SHIPPED_MAX_TOOL_ITERATIONS


def test_a_coerced_string_that_differs_is_an_override():
    """The same coercion, landing on a different number, must still show."""
    other = str(sd.SHIPPED_MAX_TOOL_ITERATIONS_CLOUD // 5)
    value = _value_for({"model": {"max_tool_iterations_cloud": other}},
                       "model.max_tool_iterations_cloud")
    assert value.source == "config_override"
    assert value.resolved == int(other)


@pytest.mark.parametrize("key, raw, source", [
    ("security.denied_paths", "/etc", "config_rejected"),      # scalar, not a list
    ("security.denied_paths", ["/etc"], "config_override"),
    ("security.workspace_root", "", "config_rejected"),        # blank
    ("security.workspace_root", [""], "config_rejected"),      # list of blanks
    ("security.workspace_root", "~/x", "config_override"),
    ("gateway.telegram_enabled", True, "config_override"),
])
def test_admission_shapes_across_the_registry(key, raw, source):
    section, name = key.rsplit(".", 1)
    config: dict = {}
    node = config
    parts = section.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = {name: raw}
    assert _source_for(config, key) == source


_REJECTION_CORPUS = [0, -1, "", "abc", "/etc", [], [""], {}, 3.7, True, ["a"]]


@pytest.mark.parametrize("entry", GOVERNED, ids=lambda e: e.key)
def test_a_rejected_value_really_did_change_nothing(entry):
    """THE PROPERTY. ``admits`` may never claim rejection for a honoured value.

    A false 'your value does nothing' is worse than silence: it tells the
    operator to delete a line that is working. So for every raw value the
    admission rule refuses, the REAL resolver must in fact return the shipped
    default — asserted against the resolver, not against a belief about it.
    """
    for raw in _REJECTION_CORPUS:
        if entry.admits(raw):
            continue
        resolved = entry.resolve({entry.name: raw})
        assert _same(resolved, entry.shipped), (
            f"{entry.key}: admits({raw!r}) is False but the resolver returned "
            f"{resolved!r}, not the shipped {entry.shipped!r} — the report "
            f"would say a working value does nothing"
        )
