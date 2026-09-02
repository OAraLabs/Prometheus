"""A config_pin must be visible without grepping a log.

The mechanism silently restored `model.model: gemma4-26b` on every boot for six
weeks while the backend served Qwen. Its only evidence was one WARNING, and
that WARNING fires only WHEN a pin acts — so absence of a warning was
indistinguishable from absence of the mechanism. An operator asking "is
anything pinned?" had to already know the file existed in order to look.

Three surfaces, one vocabulary:

* per-boot INFO — fires EVERY boot, drift or not. This is the requirement that
  closes the absence-of-evidence gap, not a nice-to-have.
* ``/health`` — COUNTS ONLY. It is unauthenticated and the pinned values carry
  a backend URL and model names; the same reason no SHA is exposed there.
* ``/api/status`` and ``oara doctor`` — full detail, bearer-gated or
  local.

Always ``config_pins``, never bare ``pins``: Beacon uses "pin" for SKILL pins
on an adjacent screen, and one word meaning two things in one UI gets misread.
"""

from __future__ import annotations

import logging

import pytest

from prometheus.daemon import (
    CONFIG_PIN_EFFECT,
    CONFIG_PINS_FILENAME,
    CONFIG_PINS_STATE,
    apply_config_pins,
    read_config_pins,
)

pytestmark = pytest.mark.integration

CFG = """\
model:
  provider: llama_cpp
  model: gemma4-26b
  base_url: http://gpu:8080
"""


@pytest.fixture()
def paths(tmp_path):
    cfg = tmp_path / "prometheus.yaml"
    cfg.write_text(CFG)
    pin = tmp_path / CONFIG_PINS_FILENAME
    return cfg, pin


def _load(cfg):
    import yaml
    with cfg.open() as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# The per-boot INFO — the requirement, not a nicety
# --------------------------------------------------------------------------- #


class TestEveryBootSaysWhatIsPinned:
    def test_info_fires_when_nothing_drifted(self, paths, caplog):
        """The case the old code was silent for, and the reason six weeks passed."""
        cfg, pin = paths
        pin.write_text("model.model: gemma4-26b\n")   # matches; no drift
        with caplog.at_level(logging.INFO, logger="prometheus.daemon"):
            drifted = apply_config_pins(_load(cfg), pin)
        assert drifted == []
        assert "config_pins:" in caplog.text
        assert "1 active" in caplog.text
        assert "model.model" in caplog.text
        assert "drift corrected this boot: none" in caplog.text

    def test_info_fires_when_no_pin_file_exists(self, paths, caplog):
        """Absence must be stated, not inferred from silence."""
        cfg, pin = paths
        with caplog.at_level(logging.INFO, logger="prometheus.daemon"):
            apply_config_pins(_load(cfg), pin)
        assert "config_pins: none active" in caplog.text

    def test_info_names_the_drifted_keys_when_it_acts(self, paths, caplog):
        cfg, pin = paths
        pin.write_text("model.model: Qwen3.8-27B-UD-Q4_K_XL.gguf\n")
        with caplog.at_level(logging.INFO, logger="prometheus.daemon"):
            apply_config_pins(_load(cfg), pin)
        assert "drift corrected this boot: model.model" in caplog.text


# --------------------------------------------------------------------------- #
# One mechanism, one sentence
# --------------------------------------------------------------------------- #


class TestOneVocabulary:
    def test_the_warning_renders_the_shared_sentence(self, paths, caplog):
        cfg, pin = paths
        pin.write_text("model.model: Qwen3.8-27B-UD-Q4_K_XL.gguf\n")
        with caplog.at_level(logging.WARNING, logger="prometheus.daemon"):
            apply_config_pins(_load(cfg), pin)
        assert CONFIG_PIN_EFFECT in caplog.text

    def test_the_sentence_is_defined_once(self):
        """A surface with its own copy of the wording is a surface that drifts."""
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "src"
        literal = "Corrected in memory; the config file is left as written."
        hits = [
            p for p in src.rglob("*.py")
            if literal in p.read_text(encoding="utf-8", errors="ignore")
        ]
        assert len(hits) == 1, (
            f"the effect sentence is written out in {len(hits)} files "
            f"({[h.name for h in hits]}); it must live only where "
            "CONFIG_PIN_EFFECT is defined"
        )

    def test_doctor_and_the_rest_surface_use_the_constant(self):
        from pathlib import Path
        src = Path(__file__).resolve().parent.parent / "src"
        for rel in ("prometheus/cli/doctor.py", "prometheus/web/server.py"):
            assert "CONFIG_PIN_EFFECT" in (src / rel).read_text()


class TestNamingAvoidsBeaconsCollision:
    @pytest.mark.parametrize("rel", [
        "prometheus/cli/doctor.py", "prometheus/web/server.py",
    ])
    def test_the_surface_key_is_config_pins_not_pins(self, rel):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "src" / rel).read_text()
        assert '"config_pins"' in src
        assert '"pins":' not in src.replace('"pins": dict(pins)', ""), (
            'a bare "pins" key collides with Beacon\'s skill pins'
        )


# --------------------------------------------------------------------------- #
# /health leaks nothing; /api/status carries the detail
# --------------------------------------------------------------------------- #


class TestHealthIsCountsOnly:
    @pytest.fixture()
    def client(self):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from prometheus.web.server import create_app
        return TestClient(create_app({}))

    def test_health_reports_whether_anything_is_pinned(self, client):
        body = client.get("/health").json()
        assert "config_pins" in body
        assert set(body["config_pins"]) == {"active", "drift_corrected_at_boot"}

    def test_health_leaks_no_keys_or_values(self, client, paths):
        """Unauthenticated. The pinned values include a backend URL."""
        cfg, pin = paths
        pin.write_text("model.base_url: http://secret-host:8080\n")
        apply_config_pins(_load(cfg), pin)
        try:
            body = client.get("/health").json()
            blob = str(body)
            assert "secret-host" not in blob, "a pinned VALUE reached /health"
            assert "model.base_url" not in blob, "a pinned KEY reached /health"
            assert body["config_pins"]["active"] == 1
        finally:
            CONFIG_PINS_STATE.update({"path": None, "pins": {}, "drifted": []})

    def test_counts_reflect_the_boot_outcome(self, client, paths):
        cfg, pin = paths
        pin.write_text("model.model: Qwen3.8-27B-UD-Q4_K_XL.gguf\n")
        apply_config_pins(_load(cfg), pin)
        try:
            cp = client.get("/health").json()["config_pins"]
            assert cp == {"active": 1, "drift_corrected_at_boot": 1}
        finally:
            CONFIG_PINS_STATE.update({"path": None, "pins": {}, "drifted": []})


class TestReadConfigPins:
    def test_absent_file_is_empty_not_an_error(self, tmp_path):
        assert read_config_pins(tmp_path / "nope.yaml") == {}

    def test_unparseable_file_is_empty_not_an_error(self, tmp_path):
        p = tmp_path / CONFIG_PINS_FILENAME
        p.write_text("{{{ not yaml")
        assert read_config_pins(p) == {}


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


class TestDoctorCheck:
    def _check(self, monkeypatch, tmp_path, pins: str | None, cfg_text: str = CFG):
        from prometheus.cli import doctor as D
        cfg = tmp_path / "prometheus.yaml"
        cfg.write_text(cfg_text)
        if pins is not None:
            (tmp_path / CONFIG_PINS_FILENAME).write_text(pins)
        monkeypatch.setattr("prometheus.config.paths.get_config_dir", lambda: tmp_path)
        monkeypatch.setattr(D, "resolve_config_path", lambda *a, **k: (cfg, [cfg]))
        return D.check_config_pins()

    def test_no_pin_file_is_ok_and_says_so(self, monkeypatch, tmp_path):
        c = self._check(monkeypatch, tmp_path, None)
        assert c.status == "ok" and "none active" in c.message

    def test_agreeing_pins_are_ok_and_listed(self, monkeypatch, tmp_path):
        c = self._check(monkeypatch, tmp_path, "model.model: gemma4-26b\n")
        assert c.status == "ok"
        assert "1 active" in c.message and "model.model" in c.message
        assert "agrees" in c.message

    def test_an_overriding_pin_is_a_WARNING_naming_both_values(self, monkeypatch, tmp_path):
        """The state that silently overrides an operator's edit is the loud one."""
        c = self._check(monkeypatch, tmp_path,
                        "model.model: Qwen3.8-27B-UD-Q4_K_XL.gguf\n")
        assert c.status == "warning"
        assert "gemma4-26b" in c.message           # what the file says
        assert "Qwen3.8-27B-UD-Q4_K_XL.gguf" in c.message   # what is enforced
        assert CONFIG_PIN_EFFECT in c.message      # the shared sentence
        assert c.fix and CONFIG_PINS_FILENAME in c.fix

    def test_the_check_is_named_config_pins(self, monkeypatch, tmp_path):
        assert self._check(monkeypatch, tmp_path, None).name == "config_pins"
