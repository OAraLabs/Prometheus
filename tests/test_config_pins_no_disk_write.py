"""A config pin corrects the running config. It must not rewrite the file.

Correcting one pinned key used to `yaml.dump` the WHOLE config, dropping every
comment and reformatting the rest. That already fired on the live box: the
shipped template carries 430 comment lines and the deployed config carried 0,
including the blocks explaining that `denied_paths` does not cover bash and
that `bash_confinement` defaults to `off`. The file an operator opens to learn
what a key means had been emptied of meaning by a routine correction.

The load-bearing test is `test_the_file_is_byte_identical_after_a_correction`.
The in-memory tests all pass against the old code too — the disk write was
never what made the pin work.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest

from prometheus.daemon import apply_config_pins

pytestmark = pytest.mark.integration

CONFIG_WITH_COMMENTS = """\
# Prometheus config — the comments here are the point of this test.
model:
  provider: llama_cpp
  # This is the model the daemon talks to. A drift correction must not
  # delete this explanation.
  model: gemma4-26b
  base_url: http://gpu:8080

security:
  # denied_paths does NOT cover bash — the gate is handed a command
  # string and never sees the paths inside it.
  denied_paths:
    - /etc
    - "/*/.ssh"
  # bash_confinement defaults to off, so the floor is not in force
  # until an operator turns it on.
  bash_confinement: "off"
"""


def _write(tmp_path: Path, pins: str) -> tuple[Path, Path]:
    cfg = tmp_path / "prometheus.yaml"
    cfg.write_text(CONFIG_WITH_COMMENTS)
    pin = tmp_path / "config_pins.yaml"
    pin.write_text(pins)
    return cfg, pin


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _loaded(cfg: Path) -> dict:
    import yaml
    with cfg.open() as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# The correction still happens — in memory
# --------------------------------------------------------------------------- #


class TestThePinStillTakesEffect:
    def test_a_drifted_key_is_corrected_in_the_loaded_config(self, tmp_path):
        """Asserted on the loaded dict, NOT the file. The dict is what the
        daemon runs on; the file was only ever made to agree afterwards."""
        cfg, pin = _write(tmp_path, "model.model: Qwen3.8-27B-UD-Q4_K_XL.gguf\n")
        config = _loaded(cfg)
        assert config["model"]["model"] == "gemma4-26b"   # precondition

        drifted = apply_config_pins(config, pin)

        assert config["model"]["model"] == "Qwen3.8-27B-UD-Q4_K_XL.gguf"
        assert drifted == ["model.model"]

    def test_a_matching_key_is_not_reported_as_drift(self, tmp_path):
        cfg, pin = _write(tmp_path, "model.model: gemma4-26b\n")
        assert apply_config_pins(_loaded(cfg), pin) == []

    def test_a_nested_key_resolves_through_the_dotted_path(self, tmp_path):
        cfg, pin = _write(tmp_path, "model.base_url: http://other:9090\n")
        config = _loaded(cfg)
        apply_config_pins(config, pin)
        assert config["model"]["base_url"] == "http://other:9090"

    def test_no_pin_file_is_a_no_op(self, tmp_path):
        cfg, _ = _write(tmp_path, "")
        assert apply_config_pins(_loaded(cfg), tmp_path / "absent.yaml") == []

    def test_the_boot_warning_still_fires(self, tmp_path, caplog):
        """The warning is the only in-band evidence a pin acted. It stays."""
        cfg, pin = _write(tmp_path, "model.model: Qwen3.8-27B-UD-Q4_K_XL.gguf\n")
        with caplog.at_level(logging.WARNING, logger="prometheus.daemon"):
            apply_config_pins(_loaded(cfg), pin)
        assert "CONFIG DRIFT DETECTED" in caplog.text
        assert "model.model" in caplog.text
        assert "gemma4-26b" in caplog.text


# --------------------------------------------------------------------------- #
# THE test: the file is not touched
# --------------------------------------------------------------------------- #


class TestTheFileIsNeverRewritten:
    def test_the_file_is_byte_identical_after_a_correction(self, tmp_path):
        """Byte-for-byte, comments included. This is the whole change."""
        cfg, pin = _write(tmp_path, "model.model: Qwen3.8-27B-UD-Q4_K_XL.gguf\n")
        before_bytes = cfg.read_bytes()
        before_sha = _sha(cfg)
        before_mtime = cfg.stat().st_mtime_ns

        config = _loaded(cfg)
        assert apply_config_pins(config, pin) == ["model.model"], (
            "precondition: this case must actually drift, or the byte-identity "
            "assertion below proves nothing"
        )

        assert _sha(cfg) == before_sha, "the config file was rewritten"
        assert cfg.read_bytes() == before_bytes
        assert cfg.stat().st_mtime_ns == before_mtime, "the file was reopened for write"

    def test_every_comment_survives(self, tmp_path):
        cfg, pin = _write(tmp_path, "model.model: Qwen3.8-27B-UD-Q4_K_XL.gguf\n")
        before = [ln for ln in cfg.read_text().splitlines() if ln.strip().startswith("#")]
        assert len(before) == 7, f"fixture should carry 7 comment lines, got {len(before)}"

        apply_config_pins(_loaded(cfg), pin)

        after = [ln for ln in cfg.read_text().splitlines() if ln.strip().startswith("#")]
        assert after == before, (
            "comment lines changed — this is the 430-to-0 loss reproducing"
        )
        assert any("does NOT cover bash" in ln for ln in after)
        assert any("defaults to off" in ln for ln in after)

    def test_the_stale_on_disk_value_is_left_alone(self, tmp_path):
        """The disagreement is the intended state, not a bug.

        The file shows what the operator wrote; the pin surfaces show what is
        enforced. Previously the file silently agreed, which is how a pin
        restored the wrong model on every boot for six weeks invisibly.
        """
        cfg, pin = _write(tmp_path, "model.model: Qwen3.8-27B-UD-Q4_K_XL.gguf\n")
        config = _loaded(cfg)
        apply_config_pins(config, pin)

        assert config["model"]["model"] == "Qwen3.8-27B-UD-Q4_K_XL.gguf"   # running
        assert _loaded(cfg)["model"]["model"] == "gemma4-26b"              # on disk
