"""D10: a fresh install listens on this machine only; an existing install is never flipped.

``web.bind`` still defaults to every interface when unset, on purpose: a Mac mini reached over Tailscale
has no ``web.bind`` and must keep working. So the narrowing happens where a config is WRITTEN, and only for a
config that is NEW:

* ``oara setup`` (the fast path), the interactive wizard and setup mode's ``configure`` write
  ``web.bind: 127.0.0.1`` into a new ``prometheus.yaml``;
* a rerun on an existing config leaves its ``web.bind`` exactly as it was, absent included. This is the case
  that would silently take a working remote deployment offline, so it is tested for every writer that can
  touch an existing file (the fast path replaces one with a backup; the wizard merges into it);
* setup mode that was reached on a bind someone CHOSE (``--bind`` or ``PROMETHEUS_WEB_BIND``) keeps that bind,
  even a wide one; only the unspecified default becomes loopback;
* ``oara doctor`` warns only about an UNSET bind that resolves to every interface. A deliberate ``0.0.0.0`` is
  an owner's decision and is not nagged about (it still says plainly that it is plain HTTP).

A headless box set up from another machine therefore ends up loopback-only until its owner sets ``web.bind``
on purpose. That is the documented consequence of making the safe choice the default.
"""

from __future__ import annotations

import yaml

import prometheus.setup_wizard as wizard_mod
from prometheus.cli.init import run_init

CANDIDATE = [{"name": "llama.cpp", "url": "http://127.0.0.1:1", "models_path": "/v1/models",
              "provider": "llama_cpp"}]
MINI = {"web": {"enabled": True, "api_port": 8005, "ws_port": 8010}}   # a deployment that predates D10


def _fast_setup(target):
    run_init(noninteractive=True, target_dir=target, candidates=CANDIDATE, probe_url=None)
    return yaml.safe_load((target / "prometheus.yaml").read_text(encoding="utf-8"))


def _existing(target, web):
    target.mkdir(parents=True, exist_ok=True)
    (target / "prometheus.yaml").write_text(yaml.safe_dump({"web": web}), encoding="utf-8")


# ── oara setup (the fast path) ───────────────────────────────────────────────

def test_a_new_config_from_oara_setup_listens_on_this_machine_only(tmp_path):
    assert _fast_setup(tmp_path / "new")["web"]["bind"] == "127.0.0.1"


def test_rerunning_oara_setup_on_a_config_with_no_bind_does_not_add_one(tmp_path):
    _existing(tmp_path, MINI["web"])
    cfg = _fast_setup(tmp_path)
    assert "bind" not in cfg["web"], "the rerun flipped a deployment that had never set web.bind"


def test_rerunning_oara_setup_keeps_a_bind_that_was_chosen(tmp_path):
    for chosen in ("0.0.0.0", "192.0.2.10", "127.0.0.1", "::1"):
        target = tmp_path / chosen.replace(":", "_")
        _existing(target, {**MINI["web"], "bind": chosen})
        assert _fast_setup(target)["web"]["bind"] == chosen


def test_the_fast_path_still_writes_the_ports(tmp_path):
    cfg = _fast_setup(tmp_path / "new")
    assert (cfg["web"]["enabled"], cfg["web"]["api_port"], cfg["web"]["ws_port"]) == (True, 8005, 8010)


# ── the interactive wizard ───────────────────────────────────────────────────

def _wizard(monkeypatch, path):
    monkeypatch.setattr(wizard_mod, "_config_target", lambda: path)
    return wizard_mod.SetupWizard()


def _read(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_a_new_config_from_the_wizard_listens_on_this_machine_only(tmp_path, monkeypatch):
    path = tmp_path / "prometheus.yaml"
    _wizard(monkeypatch, path)._write_config()
    assert _read(path)["web"]["bind"] == "127.0.0.1"


def test_the_wizard_rewriting_an_existing_config_does_not_add_a_bind(tmp_path, monkeypatch):
    path = tmp_path / "prometheus.yaml"
    path.write_text(yaml.safe_dump(MINI), encoding="utf-8")
    _wizard(monkeypatch, path)._write_config()                  # "run the full wizard again"
    assert "bind" not in _read(path)["web"]


def test_the_wizard_merging_into_an_existing_config_does_not_add_a_bind(tmp_path, monkeypatch):
    path = tmp_path / "prometheus.yaml"
    path.write_text(yaml.safe_dump(MINI), encoding="utf-8")
    _wizard(monkeypatch, path)._merge_and_write(_read(path))    # "change provider" / "add a gateway"
    assert "bind" not in _read(path)["web"]


def test_the_wizard_keeps_a_bind_that_was_chosen(tmp_path, monkeypatch):
    path = tmp_path / "prometheus.yaml"
    path.write_text(yaml.safe_dump({"web": {**MINI["web"], "bind": "0.0.0.0"}}), encoding="utf-8")
    _wizard(monkeypatch, path)._write_config()
    assert _read(path)["web"]["bind"] == "0.0.0.0"
