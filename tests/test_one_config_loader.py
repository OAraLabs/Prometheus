"""One config loader, and the three silences it removes.

The daemon and the CLI each had their own loader, and each had what the other
lacked: the daemon searched correctly and reported all four read states to the
ledger but never applied env overrides; the CLI applied them but read with
``yaml.safe_load(fh) or {}``, which renders "your config is empty" and "you
have no config" as the same value and logs neither.

Two more reads were wrong in the same family:
  * config pins could not correct an ABSENT or FALSY value — the two cases
    most worth pinning — while /health reported the pin as active;
  * LCM compaction resolved its config by parent count to a path that has
    never existed on the deploy clone, so it ran on hard-coded defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from prometheus.daemon import apply_config_pins, load_config as daemon_load
from prometheus.__main__ import load_config as cli_load


CONFIG = {"model": {"model": "from-yaml", "max_tool_iterations": 500},
          "gateway": {"telegram_enabled": False}}


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "prometheus.yaml"
    p.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# The asymmetry the audit named
# --------------------------------------------------------------------------- #


def test_the_daemon_honours_env_overrides(config_file, monkeypatch):
    """THE DEFECT. PROMETHEUS_MODEL worked under `oara` and was silently dead
    under `oara daemon` — a security-adjacent knob dark on the surface that
    actually runs the system."""
    monkeypatch.setenv("PROMETHEUS_MODEL", "from-env")
    assert daemon_load(str(config_file))["model"]["model"] == "from-env"


def test_both_entry_points_resolve_identically(config_file, monkeypatch):
    """THE POINT OF ONE LOADER. Same file, same environment, same dict."""
    monkeypatch.setenv("PROMETHEUS_MODEL", "from-env")
    assert daemon_load(str(config_file)) == cli_load(str(config_file))


def test_without_the_env_var_the_file_still_wins(config_file, monkeypatch):
    monkeypatch.delenv("PROMETHEUS_MODEL", raising=False)
    assert daemon_load(str(config_file))["model"]["model"] == "from-yaml"


def test_the_daemon_still_refuses_an_unusable_config(tmp_path):
    """strict=True is the ONLY difference between the two callers, and it must
    survive the unification: booting the whole system on substituted defaults
    is not the same situation as one CLI command tolerating a missing file."""
    from prometheus.config.load import ConfigReadError

    bad = tmp_path / "prometheus.yaml"
    bad.write_text("model: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigReadError):
        daemon_load(str(bad))


def test_the_cli_still_tolerates_what_it_always_did(tmp_path):
    """Non-strict, deliberately — the same dict the old body produced."""
    empty = tmp_path / "prometheus.yaml"
    empty.write_text("", encoding="utf-8")
    assert cli_load(str(empty)) == {}


# --------------------------------------------------------------------------- #
# Config pins: absent and falsy are exactly what pins are for
# --------------------------------------------------------------------------- #


@pytest.fixture()
def pins(tmp_path: Path) -> Path:
    p = tmp_path / "config_pins.json"
    p.write_text(json.dumps({"gateway.telegram_enabled": True}), encoding="utf-8")
    return p


def test_a_pin_corrects_an_absent_key(pins):
    cfg: dict = {"model": {}}
    drifted = apply_config_pins(cfg, pins)
    assert drifted == ["gateway.telegram_enabled"]
    assert cfg["gateway"]["telegram_enabled"] is True


def test_a_pin_corrects_a_falsy_value(pins):
    """`if val and ...` skipped False, 0, "" and [] — so pinning a flag ON
    against a config that had turned it OFF did nothing at all."""
    cfg = {"gateway": {"telegram_enabled": False}}
    assert apply_config_pins(cfg, pins) == ["gateway.telegram_enabled"]
    assert cfg["gateway"]["telegram_enabled"] is True


def test_a_matching_value_is_not_reported_as_drift(pins):
    """Guard the guard: a pin that always fires would be equally useless."""
    cfg = {"gateway": {"telegram_enabled": True}}
    assert apply_config_pins(cfg, pins) == []


# --------------------------------------------------------------------------- #
# LCM compaction reads the operator's config
# --------------------------------------------------------------------------- #


def test_lcm_resolves_config_through_the_shared_search_order(monkeypatch, tmp_path):
    """It used `Path(__file__).parents[4]`, which on the deploy clone points
    one level ABOVE the repo — a path that has never existed. A parent count
    cannot be right in any case: a wheel ships only src/prometheus, so no
    config/ exists at ANY level above that module."""
    from prometheus.memory import lcm_engine

    cfg_dir = tmp_path / ".prometheus"
    cfg_dir.mkdir()
    (cfg_dir / "prometheus.yaml").write_text(
        yaml.safe_dump({"context": {"effective_limit": 4242, "fresh_tail_count": 7}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(cfg_dir))
    monkeypatch.chdir(tmp_path)

    resolved = lcm_engine._prometheus_yaml()
    assert resolved is not None and resolved.is_file()

    loaded = lcm_engine._load_config_from_yaml()
    assert loaded.context_threshold == 4242, (
        "compaction is still running on hard-coded defaults"
    )
    assert loaded.fresh_tail_count == 7


def test_lcm_no_longer_computes_a_config_path_by_parent_count():
    """Structural. The behavioural test above passes with ANY working path;
    this pins the reason it works, because a reintroduced `parents[N]` would
    keep passing on a checkout and silently break every wheel install."""
    import ast
    import inspect

    from prometheus.memory import lcm_engine

    # AST, not a substring: this module's own docstrings QUOTE the old
    # `parents[4]` to explain the defect, and a text match cannot tell an
    # explanation from a reintroduction.
    tree = ast.parse(inspect.getsource(lcm_engine))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "parents"
    ]
    assert not offenders, (
        f"lcm_engine computes a path by parent count again (line(s) {offenders}). "
        "Use config_search_paths — the answer the CLI, the daemon and doctor "
        "already share, which knows about $PROMETHEUS_CONFIG_DIR and does not "
        "assume a checkout layout."
    )
