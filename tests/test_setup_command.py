"""``prometheus setup`` routing + the forwarding aliases (Phase 0, item 1)."""

from __future__ import annotations

import argparse

import pytest

from prometheus.cli import setup as setup_mod


class _WizardSpy:
    """Stands in for SetupWizard: records construction, run() returns True."""

    instances: list["_WizardSpy"] = []

    def __init__(self, gateway_only: bool = False) -> None:
        self.gateway_only = gateway_only
        _WizardSpy.instances.append(self)

    def run(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _reset_spy():
    _WizardSpy.instances = []
    yield


class TestRunSetupRouting:
    def test_default_routes_to_rich_wizard(self, monkeypatch):
        import prometheus.setup_wizard as wizard_mod
        monkeypatch.setattr(wizard_mod, "SetupWizard", _WizardSpy)
        rc = setup_mod.run_setup(argparse.Namespace())
        assert rc == 0
        assert len(_WizardSpy.instances) == 1
        assert _WizardSpy.instances[0].gateway_only is False

    def test_gateway_only_threads_through(self, monkeypatch):
        import prometheus.setup_wizard as wizard_mod
        monkeypatch.setattr(wizard_mod, "SetupWizard", _WizardSpy)
        rc = setup_mod.run_setup(argparse.Namespace(gateway_only=True))
        assert rc == 0
        assert _WizardSpy.instances[0].gateway_only is True

    def test_fast_routes_to_run_init(self, monkeypatch, tmp_path):
        calls: list[dict] = []

        def fake_run_init(**kwargs):
            calls.append(kwargs)
            return {"web": {"enabled": True}}

        import prometheus.cli.init as init_mod
        monkeypatch.setattr(init_mod, "run_init", fake_run_init)
        rc = setup_mod.run_setup(argparse.Namespace(
            fast=True, noninteractive=False,
            target_dir=str(tmp_path), timeout=0.2,
        ))
        assert rc == 0
        assert calls[0]["noninteractive"] is False
        assert calls[0]["target_dir"] == tmp_path
        assert calls[0]["timeout"] == 0.2
        assert len(_WizardSpy.instances) == 0

    def test_noninteractive_implies_fast(self, monkeypatch):
        calls: list[dict] = []
        import prometheus.cli.init as init_mod
        monkeypatch.setattr(
            init_mod, "run_init", lambda **kw: calls.append(kw) or {},
        )
        rc = setup_mod.run_setup(argparse.Namespace(noninteractive=True))
        assert rc == 0
        assert calls[0]["noninteractive"] is True

    def test_fast_clean_no_write_exit_is_2(self, monkeypatch):
        # run_init returning None = exited cleanly WITHOUT writing config.
        import prometheus.cli.init as init_mod
        monkeypatch.setattr(init_mod, "run_init", lambda **kw: None)
        rc = setup_mod.run_setup(argparse.Namespace(fast=True))
        assert rc == 2


class TestMainCliAliases:
    """`prometheus setup`, `prometheus --setup`, and `prometheus-init`."""

    def _patch_run_setup(self, monkeypatch):
        seen: list[argparse.Namespace] = []

        def fake(args):
            seen.append(args)
            return 0

        monkeypatch.setattr(setup_mod, "run_setup", fake)
        return seen

    def test_setup_subcommand(self, monkeypatch):
        import sys
        from prometheus.__main__ import main as cli_main
        seen = self._patch_run_setup(monkeypatch)
        monkeypatch.setattr(
            sys, "argv", ["prometheus", "setup", "--fast", "--noninteractive"],
        )
        with pytest.raises(SystemExit) as exc:
            cli_main()
        assert exc.value.code == 0
        assert seen[0].fast is True
        assert seen[0].noninteractive is True

    def test_dashdash_setup_alias_forwards(self, monkeypatch, capsys):
        import sys
        from prometheus.__main__ import main as cli_main
        seen = self._patch_run_setup(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["prometheus", "--setup"])
        with pytest.raises(SystemExit) as exc:
            cli_main()
        assert exc.value.code == 0
        assert len(seen) == 1
        assert getattr(seen[0], "gateway_only") is False
        assert "prometheus setup" in capsys.readouterr().out

    def test_setup_gateway_only_alias(self, monkeypatch):
        import sys
        from prometheus.__main__ import main as cli_main
        seen = self._patch_run_setup(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["prometheus", "--setup-gateway-only"])
        with pytest.raises(SystemExit) as exc:
            cli_main()
        assert exc.value.code == 0
        assert seen[0].gateway_only is True

    def test_prometheus_init_console_script_forwards(self, monkeypatch, tmp_path, capsys):
        from prometheus.cli.init import main as init_main
        seen: list[list[str]] = []
        monkeypatch.setattr(
            "prometheus.cli.setup.main", lambda argv: seen.append(argv) or 0,
        )
        rc = init_main(["--noninteractive", "--target-dir", str(tmp_path)])
        assert rc == 0
        argv = seen[0]
        assert "--fast" in argv
        assert "--noninteractive" in argv
        assert str(tmp_path) in argv
        assert "prometheus setup --fast" in capsys.readouterr().out
