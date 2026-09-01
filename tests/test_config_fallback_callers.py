"""What changes for the eight subsystems now that their fallback config LOADS.

Every one of these ``from_config`` classmethods has the same shape::

    if config_path is None:
        config_path = <the fallback>
    try:
        ... open(config_path) ...
    except OSError:
        <defaults>

While the fallback named a path one directory above the repo root, the
``except`` arm ran EVERY time and all eight resolved against an empty config.
These tests are the other half of that: with the fallback fixed, each caller is
pinned to the values it now reads, and to the values it still falls back to when
no config exists anywhere (which must be unchanged — a fix that also moves the
no-config behaviour would be two changes wearing one coat).

``permissions/checker.py`` gets the most coverage because it is the only one
where the delta is a GATE DECISION rather than a number.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.config import defaults  # noqa: E402


@pytest.fixture()
def repo_config(tmp_path, monkeypatch):
    """Write a repo-local ``config/prometheus.yaml`` and point the resolver at it.

    Returns the writer so each test declares only the section it cares about.
    """
    cfg_dir = tmp_path / "repo" / "config"
    cfg_dir.mkdir(parents=True)
    path = cfg_dir / "prometheus.yaml"

    def _write(data: dict) -> Path:
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", path)
        return path

    return _write


@pytest.fixture()
def no_config(tmp_path, monkeypatch):
    """No config on ANY candidate — the pre-fix world, which must be preserved."""
    monkeypatch.setattr(defaults, "REPO_CONFIG_PATH", tmp_path / "absent.yaml")
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "prom-config"))


# ---------------------------------------------------------------------------
# 1. context/budget.py — TokenBudget
#    Read by /context on Telegram and in gateway.commands, neither of which
#    passes a config_path. Report-only: no enforcement path consumes it.
# ---------------------------------------------------------------------------

class TestTokenBudget:
    def test_reads_effective_limit_instead_of_the_24000_fallback(self, repo_config):
        from prometheus.context.budget import TokenBudget

        repo_config({"context": {"effective_limit": 72000, "reserved_output": 3000}})
        budget = TokenBudget.from_config()
        assert budget.effective_limit == 72000
        assert budget.reserved_output == 3000

    def test_model_overrides_become_reachable(self, repo_config):
        """The whole precedence ladder was dead: with ``ctx = {}`` there were no
        overrides to match, so every model reported the same 24000."""
        from prometheus.context.budget import TokenBudget

        repo_config({
            "context": {
                "effective_limit": 72000,
                "model_overrides": {"gemma4-31b": {"effective_limit": 48000}},
            }
        })
        assert TokenBudget.from_config(model="gemma4-31b").effective_limit == 48000
        assert TokenBudget.from_config(model="unlisted").effective_limit == 72000

    def test_cloud_default_limit_becomes_reachable(self, repo_config):
        from prometheus.context.budget import TokenBudget

        repo_config({"context": {"effective_limit": 72000, "cloud_default_limit": 1000000}})
        budget = TokenBudget.from_config(
            model="claude-opus-5", local_model="gemma4-31b"
        )
        assert budget.effective_limit == 1000000

    def test_absent_config_still_yields_24000(self, no_config):
        from prometheus.context.budget import TokenBudget

        budget = TokenBudget.from_config(model="anything")
        assert budget.effective_limit == 24000
        assert budget.reserved_output == 2000


# ---------------------------------------------------------------------------
# 2. context/compression.py — ContextCompressor
#    No production caller uses the fallback today; the shipped template's
#    fresh_tail_count (32) equals the hardcoded default, so a live config
#    changes nothing. Pinned anyway: the key is now READ, and a config that
#    sets it differently must take effect.
# ---------------------------------------------------------------------------

class TestContextCompressor:
    def test_fresh_tail_count_is_read_from_config(self, repo_config):
        from prometheus.context.budget import TokenBudget
        from prometheus.context.compression import ContextCompressor

        repo_config({"context": {"fresh_tail_count": 8}})
        compressor = ContextCompressor.from_config(TokenBudget(effective_limit=1000))
        assert compressor._fresh_tail_count == 8

    def test_absent_config_still_yields_32(self, no_config):
        from prometheus.context.budget import TokenBudget
        from prometheus.context.compression import ContextCompressor

        compressor = ContextCompressor.from_config(TokenBudget(effective_limit=1000))
        assert compressor._fresh_tail_count == 32


# ---------------------------------------------------------------------------
# 3. context/truncation.py — ToolResultTruncator
#    Same story as ContextCompressor: live tool_result_max (4000) already
#    equals _DEFAULT_MAX_TOKENS, so the live delta is zero.
# ---------------------------------------------------------------------------

class TestToolResultTruncator:
    def test_tool_result_max_is_read_from_config(self, repo_config):
        from prometheus.context.truncation import ToolResultTruncator

        repo_config({"context": {"tool_result_max": 25}})
        truncator = ToolResultTruncator.from_config()
        assert truncator._max_tokens == 25
        # bash keeps the last 100 lines, so the cut needs more than 100 of them.
        long_output = "\n".join(f"line {i}" for i in range(500))
        assert len(truncator.truncate("bash", long_output)) < len(long_output)

    def test_absent_config_still_yields_4000(self, no_config):
        from prometheus.context.truncation import ToolResultTruncator

        assert ToolResultTruncator.from_config()._max_tokens == 4000


# ---------------------------------------------------------------------------
# 4. learning/skill_refiner.py — SkillRefiner
#    THE LOUDEST DELTA of the seven non-security callers: from_config returns
#    None unless learning.skill_refinement_enabled is true, and with an empty
#    config it could never be true. A daemon started without --config wired NO
#    refiner; with a config that enables it, it now wires a post-task hook that
#    calls the model after every turn.
# ---------------------------------------------------------------------------

class TestSkillRefiner:
    def test_enabled_in_config_now_produces_a_refiner(self, repo_config):
        from prometheus.learning.skill_refiner import SkillRefiner

        repo_config({
            "learning": {
                "skill_refinement_enabled": True,
                "skill_refiner_model": "pinned-model",
            }
        })
        refiner = SkillRefiner.from_config(MagicMock())
        assert refiner is not None
        assert refiner._model == "pinned-model"

    def test_disabled_in_config_still_produces_nothing(self, repo_config):
        from prometheus.learning.skill_refiner import SkillRefiner

        repo_config({"learning": {"skill_refinement_enabled": False}})
        assert SkillRefiner.from_config(MagicMock()) is None

    def test_absent_config_still_produces_nothing(self, no_config):
        """Absence stays OFF. An autonomous subsystem is opted INTO — the same
        ruling ``_sentinel_enabled`` and ``resolve_telegram_enabled`` encode."""
        from prometheus.learning.skill_refiner import SkillRefiner

        assert SkillRefiner.from_config(MagicMock()) is None


# ---------------------------------------------------------------------------
# 5. learning/gepa.py — GEPAOptimizer
#    Same enable-gate shape. gepa_enabled is false in the live config, so the
#    live delta is zero — but the gate is now decided by the file rather than
#    by the file being unreadable.
# ---------------------------------------------------------------------------

class TestGEPAOptimizer:
    def test_enabled_in_config_now_produces_an_optimizer(self, repo_config):
        from prometheus.learning.gepa import GEPAOptimizer

        repo_config({
            "learning": {"gepa_enabled": True, "gepa_min_traces_required": 4},
            "evals": {"judge_model": "pinned-judge"},
        })
        gepa = GEPAOptimizer.from_config(MagicMock())
        assert gepa is not None
        assert gepa._min_traces == 4

    def test_evals_judge_pins_are_now_reachable(self, repo_config):
        """``evals.judge_model`` exists to stop the optimizer self-judging. It
        was unreachable through this path: ``data = {}`` meant no evals section
        and the judge fell through to whatever the endpoint had loaded."""
        from prometheus.learning.gepa import GEPAOptimizer

        repo_config({
            "learning": {"gepa_enabled": True},
            "evals": {"judge_model": "pinned-judge", "judge_base_url": "http://j"},
        })
        gepa = GEPAOptimizer.from_config(MagicMock())
        assert gepa is not None
        assert gepa._judge_model == "pinned-judge"

    def test_absent_config_still_produces_nothing(self, no_config):
        from prometheus.learning.gepa import GEPAOptimizer

        assert GEPAOptimizer.from_config(MagicMock()) is None


# ---------------------------------------------------------------------------
# 6. learning/nudge.py — PeriodicNudge
#    daemon.py passes args.config, which the systemd unit sets — so the live
#    daemon never used this fallback. It is reachable for any `prometheus
#    daemon` started without --config.
# ---------------------------------------------------------------------------

class TestPeriodicNudge:
    def test_interval_and_enable_flag_are_read_from_config(self, repo_config):
        from prometheus.learning.nudge import PeriodicNudge

        repo_config({"learning": {"nudge_interval": 3, "nudge_enabled": True}})
        nudge = PeriodicNudge.from_config()
        assert nudge.interval == 3 and nudge.enabled is True
        assert nudge.maybe_inject(3) is not None
        assert nudge.maybe_inject(2) is None

    def test_config_can_now_turn_the_nudge_OFF(self, repo_config):
        """The direction that was unreachable. ``nudge_enabled`` defaults to
        True in the except-arm, so an operator who wrote ``false`` got nudges
        anyway — the key could only ever be read as absent."""
        from prometheus.learning.nudge import PeriodicNudge

        repo_config({"learning": {"nudge_enabled": False}})
        nudge = PeriodicNudge.from_config()
        assert nudge.enabled is False
        assert nudge.maybe_inject(15) is None

    def test_absent_config_still_yields_interval_15_enabled(self, no_config):
        from prometheus.learning.nudge import PeriodicNudge

        nudge = PeriodicNudge.from_config()
        assert nudge.interval == 15 and nudge.enabled is True


# ---------------------------------------------------------------------------
# 7. learning/skill_creator.py — SkillCreator
#    No production caller reaches this fallback (daemon.py, teacher.py and
#    web/server.py all construct SkillCreator directly). Pinned so the key is
#    known to be live if one ever does.
# ---------------------------------------------------------------------------

class TestSkillCreator:
    def test_min_tool_calls_is_read_from_config(self, repo_config):
        from prometheus.learning.skill_creator import SkillCreator

        repo_config({"learning": {"skill_min_tool_calls": 9}})
        assert SkillCreator.from_config(MagicMock())._min_tool_calls == 9

    def test_absent_config_still_yields_3(self, no_config):
        from prometheus.learning.skill_creator import SkillCreator

        assert SkillCreator.from_config(MagicMock())._min_tool_calls == 3


# ---------------------------------------------------------------------------
# 8. permissions/checker.py — SecurityGate
#
# The one where the fallback decides whether a tool call runs. Reached by
# ``cron_scheduler._get_security_gate()``, the lazily-built gate that keeps
# unattended cron gated when the daemon never wired one.
#
# The delta runs in BOTH directions, which is why each is pinned separately:
# configured denied_commands and permission_mode make it stricter, configured
# workspace_root and allowed_commands make it looser than the shipped default.
# ---------------------------------------------------------------------------

class TestSecurityGateNowReadsItsConfig:
    def test_denied_commands_now_deny(self, repo_config):
        """``sec = {}`` meant ``denied_commands=[]`` — the operator's list was
        never consulted. Only ``_ALWAYS_BLOCKED_PATTERNS`` held."""
        from prometheus.permissions.checker import SecurityGate

        repo_config({"security": {"denied_commands": ["DROP TABLE"]}})
        gate = SecurityGate.from_config()
        assert gate.evaluate("bash", command="psql -c 'DROP TABLE users'").action == "DENY"

    def test_configured_workspace_root_replaces_the_shipped_default(
        self, repo_config, tmp_path
    ):
        """The LOOSENING delta, and the one worth reading twice.

        With no config the gate fell back to ``SHIPPED_WORKSPACE_ROOT``
        (``~/.prometheus/workspace``), so a write anywhere else was APPROVE. An
        operator's multi-root ``workspace_root`` now takes effect — writes under
        those roots are ALLOW without a prompt.
        """
        from prometheus.permissions.checker import SecurityGate

        allowed = tmp_path / "allowed"
        allowed.mkdir()
        repo_config({"security": {"workspace_root": [str(allowed)]}})
        gate = SecurityGate.from_config()

        inside = gate.evaluate("write_file", file_path=str(allowed / "note.txt"))
        outside = gate.evaluate("write_file", file_path=str(tmp_path / "elsewhere.txt"))
        assert inside.action == "ALLOW"
        assert outside.action == "APPROVE"

    def test_allowed_commands_now_auto_allow(self, repo_config):
        """The other loosening: the trusted-command allowlist was always empty,
        so a vetted background command still needed an approver that unattended
        cron does not have."""
        from prometheus.permissions.checker import SecurityGate

        repo_config({"security": {"allowed_commands": [r"^curl -sS https://ok\.test/x$"]}})
        gate = SecurityGate.from_config()
        assert gate.evaluate("bash", command="curl -sS https://ok.test/x").action == "ALLOW"
        assert gate.evaluate("bash", command="curl -sS https://other.test/x").action == "APPROVE"

    def test_permission_mode_is_now_honoured(self, repo_config):
        from prometheus.permissions.checker import PermissionMode, SecurityGate

        repo_config({"security": {"permission_mode": "strict"}})
        assert SecurityGate.from_config().current_mode() == PermissionMode.STRICT

    def test_persistent_grants_are_now_loaded(self, repo_config, tmp_path):
        """``security.grants`` is written by ``persist_grant`` and re-read at
        construction. Through this path it was write-only in principle and
        unreachable in practice: the file it read was never there."""
        from prometheus.permissions.checker import SecurityGate

        target = tmp_path / "granted"
        target.mkdir()
        repo_config({
            "security": {
                "grants": [{
                    "kind": "path_prefix", "value": str(target),
                    "tool": "write_file", "id": "g-1",
                }],
            }
        })
        gate = SecurityGate.from_config()
        assert len(gate._grants) == 1
        assert gate.evaluate(
            "write_file", file_path=str(target / "x.txt")
        ).action == "ALLOW"


class TestSecurityGateSelfProtection:
    def test_the_config_file_it_ACTUALLY_read_is_the_one_it_denies(self, repo_config):
        """The constructor denies ``config_path`` on the stated principle that
        "the process knows where it read its config from, so it can deny that
        file without anyone writing it down".

        ``from_config`` broke that property: it recorded a path it had never
        read. The gate self-denied ``~/config/prometheus.yaml`` — a file that
        does not exist — while the config it was actually configured from (had
        it been able to read one) stayed readable and writable by the agent.
        """
        from prometheus.permissions.checker import SecurityGate

        path = repo_config({"security": {}})
        gate = SecurityGate.from_config()

        assert Path(gate._config_path).resolve() == path.resolve()
        assert gate.evaluate("read_file", file_path=str(path)).action == "DENY"
        assert gate.evaluate("write_file", file_path=str(path)).action == "DENY"

    def test_persist_grant_writes_back_to_that_same_file(self, repo_config, tmp_path):
        """``_config_path`` is a WRITE target as well as a read one.

        ``_rewrite_config_grants`` bails with ``return False`` on a path that
        does not exist, so through this constructor ``persist_grant`` was a
        silent no-op: ``/approve always`` reported success on the surface and
        nothing reached disk. It now round-trips into the file the gate was
        built from — and only into ``security.grants``.
        """
        from prometheus.permissions.checker import Grant, SecurityGate

        path = repo_config({
            "security": {"permission_mode": "default"},
            "context": {"effective_limit": 72000},
        })
        gate = SecurityGate.from_config()
        target = tmp_path / "granted"

        assert gate.persist_grant(
            Grant(kind="path_prefix", value=str(target), tool_name="write_file")
        ) is True

        on_disk = yaml.safe_load(path.read_text())
        assert [g["value"] for g in on_disk["security"]["grants"]] == [str(target)]
        # Untouched neighbours — the splice is surgical, not a whole-file dump.
        assert on_disk["security"]["permission_mode"] == "default"
        assert on_disk["context"]["effective_limit"] == 72000

    def test_the_floor_holds_whatever_the_config_says(self, repo_config):
        """``_ALWAYS_DENIED_PATHS`` is structural, not policy. A config that
        clears denied_paths cannot hand the agent private keys — the property
        that made ``denied_paths: []`` safe to honour verbatim."""
        from prometheus.permissions.checker import SecurityGate

        repo_config({"security": {"denied_paths": []}})
        gate = SecurityGate.from_config()
        assert gate.evaluate(
            "read_file", file_path=str(Path.home() / ".ssh" / "id_rsa")
        ).action == "DENY"


class TestSecurityGateAbsenceIsUnchanged:
    """No config anywhere must behave EXACTLY as the broken constant did."""

    def test_shipped_defaults_still_apply(self, no_config):
        from prometheus.config.shipped_defaults import (
            SHIPPED_DENIED_PATHS,
            SHIPPED_WORKSPACE_ROOT,
        )
        from prometheus.permissions.checker import PermissionMode, SecurityGate

        gate = SecurityGate.from_config()
        assert gate._workspaces == (Path(SHIPPED_WORKSPACE_ROOT).expanduser().resolve(),)
        assert gate._denied_commands == []
        assert gate._grants == []
        assert gate.current_mode() == PermissionMode.DEFAULT
        for denied in SHIPPED_DENIED_PATHS:
            assert any(denied.lstrip("/") in p for p in gate._denied_paths)

    def test_denied_paths_floor_still_applies(self, no_config):
        from prometheus.permissions.checker import SecurityGate

        gate = SecurityGate.from_config()
        assert gate.evaluate("read_file", file_path="/etc/shadow").action == "DENY"


class TestCronUsesTheSameResolution:
    """``cron_scheduler`` is the production caller of the lazily-built gate."""

    def test_lazy_cron_gate_picks_up_configured_denied_commands(self, repo_config):
        from prometheus.gateway import cron_scheduler

        repo_config({"security": {"denied_commands": ["forbidden-cron-verb"]}})
        cron_scheduler.set_cron_security_gate(None)
        try:
            gate = cron_scheduler._get_security_gate()
            assert gate is not None
            assert gate.evaluate(
                "bash", command="forbidden-cron-verb --now"
            ).action == "DENY"
        finally:
            cron_scheduler.set_cron_security_gate(None)
