"""#471 — the nightly backup captured SQLite non-atomically.

Every test here asserts a property of the ARTEFACT the backup will carry, not
that a function ran. The defect being prevented is a copy that looks fine and is
not, so "it returned without raising" proves nothing on its own.

The load-bearing one is `test_wal_content_is_in_the_snapshot`: it is the test
that fails if anyone replaces VACUUM INTO with a file copy, which is exactly the
regression this module exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from prometheus.jobs.db_snapshot import (
    SnapshotError,
    discover_databases,
    main,
    prune_snapshots,
    run_snapshot,
    snapshot_database,
)


def _make_db(path: Path, rows: int = 500, *, wal: bool = True) -> sqlite3.Connection:
    """A WAL database with *rows* rows. Returns the OPEN connection.

    The connection is deliberately left open: closing it checkpoints the WAL into
    the main file, which would erase the very condition these tests are about.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (i INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t (v) VALUES (?)", [(f"r{i}",) for i in range(rows)])
    conn.commit()
    return conn


def _count(path: Path) -> int:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return c.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    finally:
        c.close()


def _count_or_none(path: Path) -> int | None:
    """Rows, or None when the file is not a usable database at all."""
    try:
        return _count(path)
    except sqlite3.Error:
        return None


# ---------------------------------------------------------------------------
# The property the whole module exists for
# ---------------------------------------------------------------------------


def test_wal_content_is_in_the_snapshot(tmp_path):
    """A copy of the main file alone LOSES committed rows. The snapshot must not.

    This is the regression guard. Swap VACUUM INTO for shutil.copy and this test
    is the one that goes red — every other assertion here would still pass.
    """
    src = tmp_path / "live.db"
    conn = _make_db(src, rows=500)
    # Commit more without checkpointing: these rows live in the -wal only.
    conn.executemany("INSERT INTO t (v) VALUES (?)", [(f"wal{i}",) for i in range(2000)])
    conn.commit()

    assert (src.with_name(src.name + "-wal")).exists(), "fixture must have a live WAL"

    naive = tmp_path / "naive_copy.db"          # what a plain file copy captures
    naive.write_bytes(src.read_bytes())

    snap = tmp_path / "out" / "live.db"
    result = snapshot_database(src, snap)
    conn.close()

    live_rows = 2500
    assert _count(snap) == live_rows, "the snapshot must carry the WAL's committed rows"
    assert result.integrity == "ok"

    # The plain copy is not merely lossy. In WAL mode the CREATE TABLE is in the
    # -wal too, so the main file on its own does not even contain the table —
    # `_count_or_none` returns None. Measured, and the stronger version of what
    # this test was written to claim.
    naive_rows = _count_or_none(naive)
    assert naive_rows != live_rows, (
        "fixture is not exercising the defect: the main file already held every "
        "row, so a plain copy would have been correct and this test proves nothing"
    )
    assert naive_rows is None or naive_rows < live_rows


def test_the_snapshot_has_no_sidecars(tmp_path):
    """Self-contained is the point: tar cannot tear a single static file."""
    src = tmp_path / "live.db"
    conn = _make_db(src)
    snap = tmp_path / "out" / "live.db"
    snapshot_database(src, snap)
    conn.close()

    assert snap.exists()
    assert sorted(p.name for p in snap.parent.iterdir()) == ["live.db"]


def test_a_live_writer_does_not_tear_the_snapshot(tmp_path):
    """The condition the nightly backup actually runs under."""
    src = tmp_path / "live.db"
    conn = _make_db(src, rows=2000)

    stop = threading.Event()

    def churn():
        w = sqlite3.connect(src)
        n = 0
        while not stop.is_set():
            w.execute("INSERT INTO t (v) VALUES (?)", (f"live{n}",))
            w.commit()
            n += 1
        w.close()

    writer = threading.Thread(target=churn)
    writer.start()
    time.sleep(0.2)
    try:
        result = snapshot_database(src, tmp_path / "out" / "live.db")
    finally:
        stop.set()
        writer.join()
        conn.close()

    assert result.integrity == "ok"
    # A point in time, not a smear: coherent, and not required to equal "now".
    assert result.tables == 1
    assert _count(Path(result.snapshot)) >= 2000


def test_the_sources_data_files_are_never_modified(tmp_path):
    """mode=ro is a property, not a promise — pin the EXACT claim.

    A snapshot job that mutates a live production database while "backing it up"
    is a worse defect than the one being fixed.

    The claim is deliberately narrow because the broad one is false, and this
    test is where that was discovered rather than asserted: `.db` and `-wal` come
    out byte-identical, and `-shm` does NOT. The wal-index is updated by every
    reader including a read-only connection, SQLite rebuilds it from the `-wal`
    whenever it is missing, and it holds no database content. Writing this test
    as "nothing under the directory changed" would fail for a reason that is not
    a defect — and then get "fixed" by deleting the assertion that matters.
    """
    def digest(paths):
        return {
            p.name: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in paths
        }

    src = tmp_path / "live.db"
    conn = _make_db(src)
    conn.executemany("INSERT INTO t (v) VALUES (?)", [(f"w{i}",) for i in range(500)])
    conn.commit()

    data_files = [src, src.with_name(src.name + "-wal")]
    assert all(p.exists() for p in data_files), "fixture must have a live WAL"

    before, before_shm = digest(data_files), digest([src.with_name(src.name + "-shm")])
    snapshot_database(src, tmp_path / "out" / "live.db")
    after, after_shm = digest(data_files), digest([src.with_name(src.name + "-shm")])
    conn.close()

    assert before == after, "the files holding data must come out byte-identical"
    # Not asserted equal: see the docstring. Named so the difference is a known
    # fact about SQLite rather than an unexplained flake for the next reader.
    assert set(before_shm) == set(after_shm) == {"live.db-shm"}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discovery_skips_sidecars_backups_and_excluded_dirs(tmp_path):
    root = tmp_path / ".prometheus"
    conn = _make_db(root / "data" / "telemetry.db")
    conn2 = _make_db(root / "memory.db")

    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "cache").mkdir(parents=True, exist_ok=True)
    (root / "db-snapshots" / "20260101T000000Z").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "old.db").write_bytes(b"")
    (root / "cache" / "tmp.db").write_bytes(b"")
    (root / "db-snapshots" / "20260101T000000Z" / "memory.db").write_bytes(b"")
    (root / "memory.db.backup-20260601").write_bytes(b"")

    found = {p.relative_to(root).as_posix() for p in discover_databases(root)}
    conn.close()
    conn2.close()

    assert found == {"data/telemetry.db", "memory.db"}
    # -wal/-shm do not end in .db and must never be captured separately; a
    # previous snapshot set must not be re-snapshotted into the next one.
    assert not any("-wal" in f or "-shm" in f for f in found)
    assert not any(f.startswith("db-snapshots") for f in found)


# ---------------------------------------------------------------------------
# Fail loud, and leave a readable trace
# ---------------------------------------------------------------------------


def test_a_corrupt_database_fails_the_run(tmp_path):
    root = tmp_path / ".prometheus"
    conn = _make_db(root / "good.db")
    (root / "broken.db").write_bytes(b"this is not a database" * 100)

    with pytest.raises(SnapshotError):
        run_snapshot(root, keep=3)
    conn.close()


def test_the_outcome_is_readable_after_a_failure(tmp_path):
    """#471's finding is that this failure is silent on BOTH ends.

    An exit code is only loud if something reads it, and the cron that runs this
    reports to nobody. The status file is the channel that does not depend on
    anyone watching, so it must survive the failure path.
    """
    root = tmp_path / ".prometheus"
    (root / "broken.db").parent.mkdir(parents=True, exist_ok=True)
    (root / "broken.db").write_bytes(b"not a database" * 100)

    assert main(["--root", str(root)]) == 1          # non-zero exit

    status = json.loads((root / "db-snapshots" / "LAST_RUN.json").read_text())
    assert status["ok"] is False
    assert "broken.db" in status["error"]
    assert status["finished_at"] > 0


def test_capturing_nothing_is_a_failure_not_a_success(tmp_path):
    """A run that found no databases must not report a successful backup."""
    root = tmp_path / ".prometheus"
    root.mkdir(parents=True)
    with pytest.raises(SnapshotError, match="nothing was captured"):
        run_snapshot(root, keep=3)


def test_a_successful_run_is_also_recorded(tmp_path):
    root = tmp_path / ".prometheus"
    conn = _make_db(root / "memory.db")
    assert main(["--root", str(root)]) == 0
    conn.close()

    status = json.loads((root / "db-snapshots" / "LAST_RUN.json").read_text())
    assert status["ok"] is True and status["error"] is None
    assert [Path(d["source"]).name for d in status["databases"]] == ["memory.db"]


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_prune_keeps_the_newest_complete_sets(tmp_path):
    snaps = tmp_path / "db-snapshots"
    for name in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
        d = snaps / name
        d.mkdir(parents=True)
        (d / "MANIFEST.json").write_text("{}")

    result = prune_snapshots(snaps, keep=2)
    assert result.pruned == ["20260101T000000Z"]
    assert result.set_aside == []
    assert sorted(p.name for p in snaps.iterdir()) == [
        "20260102T000000Z", "20260103T000000Z",
    ]


def test_an_interrupted_set_is_pruned_first_and_costs_no_slot(tmp_path):
    """A half-written set must never be retained in preference to a good one.

    The manifest is written last and only after every database verified, so its
    absence means the run died partway. Keeping that over a complete set would
    make retention actively harmful.
    """
    snaps = tmp_path / "db-snapshots"
    for name, complete in (
        ("20260101T000000Z", True),
        ("20260102T000000Z", True),
        ("20260103T000000Z", False),   # newest, but interrupted
    ):
        d = snaps / name
        d.mkdir(parents=True)
        # The marker is what makes it OURS. A run writes it before copying
        # anything, so even wreckage carries it; that is what separates an
        # interrupted run from a directory this job never touched.
        (d / ".written-by-db-snapshot").write_text("x")
        if complete:
            (d / "MANIFEST.json").write_text("{}")

    result = prune_snapshots(snaps, keep=2)
    assert "20260103T000000Z" in result.pruned
    assert result.set_aside == []
    assert sorted(p.name for p in snaps.iterdir()) == [
        "20260101T000000Z", "20260102T000000Z",
    ]


def test_a_directory_this_job_did_not_write_is_never_deleted(tmp_path):
    """The regression this fix exists for.

    The first version read "no MANIFEST.json" as "interrupted run" and deleted on
    sight, BEFORE --keep was applied — so the very first run on any machine
    destroyed whatever was already in db-snapshots/, at any keep value. On
    2026-09-20 that deleted three hand-made directories holding the only copies
    of some June databases outside the nightly archives.
    """
    snaps = tmp_path / "db-snapshots"
    stranger = snaps / "20260621T044344Z-pre-memhygiene"
    stranger.mkdir(parents=True)
    (stranger / "memory.db").write_bytes(b"irreplaceable")

    result = prune_snapshots(snaps, keep=1)

    assert result.pruned == []
    assert result.set_aside == ["20260621T044344Z-pre-memhygiene"]
    moved = snaps / "unrecognized" / "20260621T044344Z-pre-memhygiene"
    assert moved.is_dir(), "it must be moved, not deleted"
    assert (moved / "memory.db").read_bytes() == b"irreplaceable"


def test_setting_aside_is_reported_separately_from_deleting(tmp_path):
    """One "(pruned N)" covering both is how the deletion went unnoticed."""
    snaps = tmp_path / "db-snapshots"
    for name in ("20260101T000000Z", "20260102T000000Z"):
        d = snaps / name
        d.mkdir(parents=True)
        (d / "MANIFEST.json").write_text("{}")
    (snaps / "hand-made").mkdir(parents=True)

    result = prune_snapshots(snaps, keep=1)

    assert result.pruned == ["20260101T000000Z"]
    assert result.set_aside == ["hand-made"]


def test_the_unrecognized_dir_is_not_itself_treated_as_a_set(tmp_path):
    """Otherwise each run sets aside the previous run's set-aside directory."""
    snaps = tmp_path / "db-snapshots"
    keeper = snaps / "20260102T000000Z"
    keeper.mkdir(parents=True)
    (keeper / "MANIFEST.json").write_text("{}")
    (snaps / "unrecognized" / "moved-earlier").mkdir(parents=True)

    result = prune_snapshots(snaps, keep=1)

    assert result.pruned == []
    assert result.set_aside == []
    assert (snaps / "unrecognized" / "moved-earlier").is_dir()


def test_an_interrupted_run_leaves_a_marker_the_next_run_can_act_on(tmp_path):
    """End to end: the marker must survive the failure path, or it proves nothing.

    A run that dies partway has no manifest. Without the marker written up front,
    the next run cannot distinguish its own wreckage from somebody else's data —
    and would have to choose between leaking wreckage forever and deleting data.
    """
    root = tmp_path / ".prometheus"
    conn = _make_db(root / "good.db")
    (root / "broken.db").write_bytes(b"not a database" * 100)

    with pytest.raises(SnapshotError):
        run_snapshot(root, keep=3)
    conn.close()

    sets = [p for p in (root / "db-snapshots").iterdir() if p.is_dir()]
    assert len(sets) == 1
    wreckage = sets[0]
    assert (wreckage / ".written-by-db-snapshot").is_file()
    assert not (wreckage / "MANIFEST.json").exists()

    # ...and retention then recognises it as ours and clears it.
    result = prune_snapshots(root / "db-snapshots", keep=3)
    assert result.pruned == [wreckage.name]
    assert result.set_aside == []
