"""Tests for GRAFT-CLEAN-SLATE: identity templates and generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.cli.generate_identity import (
    TEMPLATES_DIR,
    detect_hardware,
    generate_identity_files,
    render_agents_md,
    render_soul_md,
)

_HW = {"hostname": "h", "os": "Linux", "arch": "x86_64",
       "cpu": "C", "ram_gb": 8, "gpu": None, "has_gpu": False}


# ── Templates exist ─────────────────────────────────────────────────

def test_soul_template_exists():
    assert (TEMPLATES_DIR / "SOUL.md.template").is_file()

def test_agents_template_exists():
    assert (TEMPLATES_DIR / "AGENTS.md.template").is_file()


# ── Hardware detection ──────────────────────────────────────────────

def test_detect_hardware_returns_expected_keys():
    hw = detect_hardware()
    for key in ("hostname", "os", "arch", "cpu", "ram_gb", "gpu", "has_gpu"):
        assert key in hw

def test_detect_hardware_hostname_nonempty():
    assert len(detect_hardware()["hostname"]) > 0

def test_detect_hardware_ram_positive():
    assert detect_hardware()["ram_gb"] > 0


# ── Template rendering ──────────────────────────────────────────────

def test_render_soul_md_contains_owner():
    result = render_soul_md("Alice", _HW)
    assert "Alice" in result
    assert "{{" not in result

def test_render_soul_md_single_machine():
    hw = {**_HW, "gpu": "RTX 4090 (24GB)", "has_gpu": True}
    result = render_soul_md("Bob", hw)
    assert "RTX 4090" in result

def test_render_soul_md_split_machines():
    hw = {**_HW, "gpu": "Apple Silicon (unified memory)", "has_gpu": True}
    result = render_soul_md("Charlie", hw, hardware_layout="split",
                            brain_machine_name="Mini", gpu_machine_name="Beefy")
    assert "Mini" in result
    assert "Beefy" in result

def test_render_soul_md_no_placeholders():
    result = render_soul_md("Test", _HW)
    assert "{{" not in result and "}}" not in result

def test_render_soul_md_vision_available_true():
    result = render_soul_md("Test", _HW, vision_available=True)
    assert "confirmed available" in result

def test_render_soul_md_vision_available_false():
    result = render_soul_md("Test", _HW, vision_available=False)
    assert "not available" in result

def test_render_agents_md_has_specializations():
    result = render_agents_md()
    for name in ("general-purpose", "explorer", "planner", "worker", "verification"):
        assert name in result


# ── File generation ─────────────────────────────────────────────────

def test_generate_creates_all_files(tmp_path):
    results = generate_identity_files("Tester", _HW, dest=tmp_path)
    assert (tmp_path / "SOUL.md").is_file()
    assert (tmp_path / "AGENTS.md").is_file()
    assert (tmp_path / "MEMORY.md").is_file()
    assert (tmp_path / "USER.md").is_file()

def test_generate_skips_existing(tmp_path):
    generate_identity_files("First", _HW, dest=tmp_path)
    results = generate_identity_files("Second", _HW, dest=tmp_path, overwrite=False)
    assert results["SOUL.md"] == "exists (skipped)"
    assert "First" in (tmp_path / "SOUL.md").read_text()

def test_generate_overwrites_when_requested(tmp_path):
    generate_identity_files("First", _HW, dest=tmp_path)
    generate_identity_files("Second", _HW, dest=tmp_path, overwrite=True)
    assert "Second" in (tmp_path / "SOUL.md").read_text()

def test_memory_never_overwritten(tmp_path):
    generate_identity_files("First", _HW, dest=tmp_path)
    (tmp_path / "MEMORY.md").write_text("important")
    generate_identity_files("Second", _HW, dest=tmp_path, overwrite=True)
    assert (tmp_path / "MEMORY.md").read_text() == "important"

def test_user_md_never_overwritten(tmp_path):
    generate_identity_files("First", _HW, dest=tmp_path)
    (tmp_path / "USER.md").write_text("user data")
    generate_identity_files("Second", _HW, dest=tmp_path, overwrite=True)
    assert (tmp_path / "USER.md").read_text() == "user data"

def test_no_personal_names_in_templates():
    for name in ("SOUL.md.template", "AGENTS.md.template"):
        content = (TEMPLATES_DIR / name).read_text()
        assert "Will" not in content
        assert "OAra" not in content
        assert "100.110" not in content
        assert "100.104" not in content
