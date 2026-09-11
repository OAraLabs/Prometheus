"""Image generation must not be able to overwrite the daemon's own state.

THE DEFECT
----------
`_allowed_image_roots` returned `[get_config_dir(), get_workspace_dir()]`. The
config dir is `~/.prometheus/` — which is where the daemon keeps its STATE, not
a safe floor. An `output_path` of `data/lcm.db`, `data/tasks.db`,
`data/devices.json` or `telemetry.db` sits inside the allow-list.

Reproduced against the real `_save_image_bytes`; all four were overwritten and
began `b'\\xff\\xd8\\xff-PNG-BYTE'`:

    LCM store          -> overwritten: True
    task database      -> overwritten: True
    device enrolments  -> overwritten: True
    telemetry          -> overwritten: True

The function's own docstring already said "writing image bytes to an arbitrary
path is never a legitimate need" — the roots just were not narrow enough to
express it.

THE FIX
-------
The allow-list is the image directories and nothing else: `cache/images/` and
`<workspace>/images/`. An operator who wants images elsewhere relocates the
workspace (`PROMETHEUS_WORKSPACE_DIR`), which moves the allowed root with it —
operator-chosen, not model-chosen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("PROMETHEUS_WORKSPACE_DIR", raising=False)
    return tmp_path


def _save(target: Path | None, data: bytes = b"\xff\xd8\xff-image"):
    from prometheus.tools.builtin.image_generate import _save_image_bytes

    return _save_image_bytes(
        data, ext=".png", override_path=str(target) if target else None
    )


#: The daemon's real state, all of it under the OLD allow-list.
DAEMON_STATE = {
    "lcm store": "data/lcm.db",
    "task database": "data/tasks.db",
    "device enrolments": "data/devices.json",
    "telemetry": "telemetry.db",
    "audit": "data/audit.db",
}


@pytest.mark.parametrize("label", sorted(DAEMON_STATE))
def test_daemon_state_cannot_be_overwritten(label, state_dir):
    target = state_dir / DAEMON_STATE[label]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"REAL DAEMON STATE")

    with pytest.raises(ValueError, match="not under any allowed root"):
        _save(target)

    assert target.read_bytes() == b"REAL DAEMON STATE", (
        f"{label} was modified despite the refusal"
    )


def test_a_traversal_out_of_an_allowed_root_is_refused(state_dir):
    """resolve-then-check: a path that STARTS legal must not end illegal."""
    target = state_dir / "cache" / "images" / ".." / ".." / "data" / "lcm.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="not under any allowed root"):
        _save(target)


def test_a_path_outside_the_state_dir_entirely_is_refused(state_dir):
    with pytest.raises(ValueError, match="not under any allowed root"):
        _save(state_dir / ".." / ".." / "etc" / "passwd")


# ── the fix must not break image generation ─────────────────────────────────

def test_the_image_cache_is_still_writable(state_dir):
    """Without this, an empty allow-list would pass every test above."""
    out = Path(_save(state_dir / "cache" / "images" / "out.png"))
    assert out.exists() and out.stat().st_size > 0


def test_the_workspace_images_dir_is_writable_including_subdirs(state_dir):
    """A run keeping its pictures beside its work is a legitimate need."""
    out = Path(_save(state_dir / "workspace" / "images" / "run1" / "frame.png"))
    assert out.exists()


def test_no_override_still_writes_to_the_cache(state_dir):
    """The default path is untouched by this change."""
    out = Path(_save(None))
    assert out.exists()
    assert "images" in out.parts


def test_the_allowed_roots_are_only_image_directories(state_dir):
    """The property, asserted on the roots rather than only on their effects.

    A future edit that re-adds `get_config_dir()` "for convenience" fails here
    with the reason attached, rather than silently re-opening the hole.
    """
    from prometheus.tools.builtin.image_generate import _allowed_image_roots

    roots = _allowed_image_roots()
    assert roots, "the allow-list is empty — image output is disabled entirely"
    for root in roots:
        assert root.name == "images", (
            f"{root} is not an image directory. The daemon keeps its state "
            f"under ~/.prometheus/, so any broader root lets an output_path "
            f"reach data/lcm.db, data/tasks.db or telemetry.db."
        )
    assert state_dir not in [Path(r) for r in roots]
