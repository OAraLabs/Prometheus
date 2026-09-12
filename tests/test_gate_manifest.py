"""The gate manifest must make an incomparable comparison REFUSE (#479).

Two runs on one tree gave different pass/skip splits and nothing said so. The
manifest records what gated each run; these tests pin that the record is real
(probes the same functions the skipifs consult), that it is honest about what it
contains (no infrastructure identifier, because stdout is persisted state), and
above all that the comparison tool returns VOID — not passed, not failed — when
the manifests differ.

A manifest that always reported the same gates would look green and prove
nothing, so the "differs → void" direction is pinned as hard as the
"matches → admissible" one.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import gate_manifest as gm

REPO = Path(__file__).resolve().parent.parent
COMPARE = REPO / "scripts" / "compare_gate_manifests.py"


# ── the manifest is real, not a plausible-looking block of text ─────────────


def test_manifest_reports_the_gates_that_actually_decide_skips() -> None:
    """The gates that moved 39 tests between runs must all be present."""
    names = {g.name for g in gm.collect_gates()}
    for required in (
        "bwrap_write_floor",
        "apparmor_read_floor",
        "root_mount_options",
        "root_writable",
        "network_dns",
        "symlinks",
        "read_permission_revocable",
        "live_repo_config",
    ):
        assert required in names, f"gate {required!r} missing from the manifest"


def test_write_floor_gate_agrees_with_the_skipif_it_explains() -> None:
    """The manifest must report the SAME verdict the skipif consulted.

    A parallel reimplementation would drift and describe a gate that decided
    nothing. This calls the very function ``_floor_available`` in
    test_bash_write_floor.py calls, and asserts the two agree — so the manifest
    cannot start lying while the skip keeps working.
    """
    from prometheus.permissions import confinement as C

    ok, _detail = C.write_preflight(force=True)
    C.reset_write_cache()
    expected = "available" if ok else "unavailable"

    value, _ = gm._write_floor_gate()
    assert value == expected


def test_write_floor_probe_leaves_no_cached_verdict_behind() -> None:
    """The manifest runs at session start; it must not poison later probes.

    ``write_preflight`` caches on its argument. If the manifest left its verdict
    cached, a later probe composed with a different prefix could inherit it —
    the exact hazard the cache's own docstring warns about.
    """
    from prometheus.permissions import confinement as C

    gm._write_floor_gate()
    assert C._write_preflight_cache == {}, (
        "the manifest left a cached write-floor verdict behind: "
        f"{C._write_preflight_cache!r}"
    )


def test_gates_are_sorted_so_two_identical_hosts_hash_equal() -> None:
    """Reproducible hashing is the whole comparison mechanism."""
    first = gm.collect_gates()
    assert [g.name for g in first] == sorted(g.name for g in first)
    assert gm.manifest_hash(first) == gm.manifest_hash(gm.collect_gates())


def test_manifest_hash_ignores_the_reason_prose() -> None:
    """Details carry probe output (paths, errno); the verdict is what gates.

    Two runs that gated identically must compare equal even when the prose
    differs — otherwise every run is "incomparable" and the tool says nothing.
    """
    gates = gm.collect_gates()
    noisy = tuple(
        gm.Gate(g.name, g.value, g.detail + " some varying text") for g in gates
    )
    assert gm.manifest_hash(gates) == gm.manifest_hash(noisy)


def test_manifest_hash_changes_when_a_verdict_changes() -> None:
    """The other direction: a hash that cannot change gates nothing."""
    gates = gm.collect_gates()
    flipped = tuple(
        gm.Gate(g.name, g.value + "-flipped", g.detail) for g in gates
    )
    assert gm.manifest_hash(gates) != gm.manifest_hash(flipped)


def test_manifest_hash_folded_in_the_schema() -> None:
    """Different schema versions must never hash-equal."""
    gates = gm.collect_gates()
    current = gm.manifest_hash(gates)
    saved = gm.SCHEMA
    try:
        gm.SCHEMA = saved + 1
        assert gm.manifest_hash(gates) != current
    finally:
        gm.SCHEMA = saved


# ── no infrastructure identifier: stdout is persisted state (#474's lesson) ──


def test_manifest_carries_no_hostname_or_address() -> None:
    """The manifest is printed and pasted into issues; it must not leak a host.

    No tailnet/LAN address, no *.ts.net name, no machine hostname. Binary gates
    record PRESENCE, not a resolved path — a path carries a username and names
    a machine.
    """
    blob = json.dumps(gm.as_dict())
    assert not re.search(r"\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", blob), (
        "manifest carries a CGNAT/tailnet address"
    )
    assert not re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", blob), (
        "manifest carries a dotted-quad address"
    )
    assert ".ts.net" not in blob, "manifest carries a tailnet host name"
    assert "/home/" not in blob, (
        "manifest carries an absolute path (a username is an identifier)"
    )


def test_binary_gates_record_presence_not_location() -> None:
    for g in gm.collect_gates():
        if g.name.startswith("bin:"):
            assert g.value in ("present", "absent"), (
                f"{g.name} recorded {g.value!r}, not a presence verdict"
            )
            assert not g.detail, f"{g.name} should not carry a path: {g.detail!r}"


def test_the_probe_host_is_the_only_network_name_and_it_is_public() -> None:
    assert gm.NETWORK_PROBE_HOST == "duckduckgo.com"


# ── the pytest wiring ───────────────────────────────────────────────────────


def _run_pytest(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd or REPO),
        timeout=300,
    )


def test_sessionstart_writes_a_parseable_manifest_file(tmp_path: Path) -> None:
    out = tmp_path / "manifest.json"
    proc = _run_pytest(
        ["tests/test_generated_reference.py", "-q", "-p", "no:randomly",
         f"--gate-manifest={out}"]
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == gm.SCHEMA
    assert re.fullmatch(r"[0-9a-f]{16}", data["hash"]), data["hash"]
    assert "bwrap_write_floor" in data["gates"]
    assert "collected_at" in data


def test_manifest_is_repeated_in_the_terminal_summary(tmp_path: Path) -> None:
    """The summary is what gets quoted into an issue — the state must be in it.

    A verdict without the gates that produced it is how an unearned "no
    regression" gets written down.
    """
    proc = _run_pytest(
        ["tests/test_generated_reference.py", "-q", "-p", "no:randomly"]
    )
    assert "GATE MANIFEST" in proc.stdout, proc.stdout[-1500:]
    assert "bwrap_write_floor" in proc.stdout


def test_two_runs_on_one_tree_agree(tmp_path: Path) -> None:
    """The property the whole thing exists to make visible, asserted not assumed.

    Same tree, same context, twice: identical hashes. If this fails the
    manifest is recording something that moves within a single environment, and
    every comparison built on it would be void by default — useless.
    """
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    args = ["tests/test_generated_reference.py", "-q", "-p", "no:randomly"]
    assert _run_pytest([*args, f"--gate-manifest={a}"]).returncode == 0
    assert _run_pytest([*args, f"--gate-manifest={b}"]).returncode == 0
    ha = json.loads(a.read_text())["hash"]
    hb = json.loads(b.read_text())["hash"]
    assert ha == hb, f"the manifest is not reproducible on one host: {ha} != {hb}"


# ── the comparison tool: void, not passed ───────────────────────────────────


def _manifest(**gate_overrides: str) -> dict:
    """A real manifest with selected gate values overridden."""
    data = gm.as_dict()
    for name, value in gate_overrides.items():
        data["gates"][name]["value"] = value
    data["hash"] = "overridden" + str(len(gate_overrides))
    return data


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_comparison_of_identical_manifests_is_admissible(tmp_path: Path) -> None:
    m = gm.as_dict()
    proc = subprocess.run(
        [sys.executable, str(COMPARE),
         str(_write(tmp_path / "a.json", m)),
         str(_write(tmp_path / "b.json", m))],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "MATCH" in proc.stdout
    assert "comparable" in proc.stdout


def test_comparison_of_differing_gates_is_VOID_not_passed(tmp_path: Path) -> None:
    """THE load-bearing assertion. The real observed flip must refuse.

    Mutating the gate to "available" simulates the other host state (root
    mounted rw) — the exact pair of runs that produced 7781/404 and 7742/441.
    A tool that reported "no new failures" here would be the unearned claim.
    """
    before = _manifest()
    after = _manifest(bwrap_write_floor="available")
    proc = subprocess.run(
        [sys.executable, str(COMPARE),
         str(_write(tmp_path / "a.json", before)),
         str(_write(tmp_path / "b.json", after))],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1, f"VOID must exit 1, got {proc.returncode}"
    assert "VOID" in proc.stderr
    assert "UNMEASURED" in proc.stderr, (
        "void must say unmeasured — not passed, not failed"
    )
    assert "bwrap_write_floor" in proc.stderr, "it must name the gate that flipped"
    assert "no new failures" not in proc.stdout.lower()


def test_comparison_refuses_across_schema_versions(tmp_path: Path) -> None:
    """Different schema = different SET of gates measured, not different state."""
    a = gm.as_dict()
    b = gm.as_dict()
    b["schema"] = a["schema"] + 1
    proc = subprocess.run(
        [sys.executable, str(COMPARE),
         str(_write(tmp_path / "a.json", a)), str(_write(tmp_path / "b.json", b))],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1
    assert "schema" in proc.stderr.lower()


def test_comparison_refuses_a_hand_edited_gate_set(tmp_path: Path) -> None:
    """A manifest someone edited must not be silently trusted."""
    a = gm.as_dict()
    b = gm.as_dict()
    del b["gates"]["network_dns"]
    b["gates"]["invented_gate"] = {"value": "x", "detail": ""}
    proc = subprocess.run(
        [sys.executable, str(COMPARE),
         str(_write(tmp_path / "a.json", a)), str(_write(tmp_path / "b.json", b))],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 1
    assert "gate SETS differ" in proc.stderr
    assert "invented_gate" in proc.stderr


def test_comparison_refuses_a_file_that_is_not_a_manifest(tmp_path: Path) -> None:
    bad = tmp_path / "notamanifest.json"
    bad.write_text('{"hello": "world"}', encoding="utf-8")
    good = _write(tmp_path / "good.json", gm.as_dict())
    proc = subprocess.run(
        [sys.executable, str(COMPARE), str(bad), str(good)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2, f"unparseable input must exit 2, got {proc.returncode}"
    assert "--gate-manifest" in proc.stderr, "it must say how to produce one"


def test_comparison_refuses_a_missing_file(tmp_path: Path) -> None:
    good = _write(tmp_path / "good.json", gm.as_dict())
    proc = subprocess.run(
        [sys.executable, str(COMPARE), str(tmp_path / "absent.json"), str(good)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 2
