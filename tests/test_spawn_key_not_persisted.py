"""A spawned agent task must not write the Anthropic key anywhere durable.

WHAT HAPPENED
-------------
`BackgroundTaskManager.create_agent_task` built the child's command like this::

    cmd = ["python", "-m", "prometheus", "--headless",
           "--api-key", effective_api_key]
    command = " ".join(shlex.quote(part) for part in cmd)

`command` is a PERSISTED field on TaskRecord. One `sessions_spawn` therefore
wrote the key into:

  1. the task database          — TaskStore has a `command TEXT` column
  2. every REST list/get body   — the serialisers emit `command`
  3. the model's context window — `task_get` prints `command:` verbatim
  4. the task's OUTPUT FILE     — see below, and this one is the surprise

`shlex.quote` was doing its job correctly the whole time. Correct escaping is
orthogonal to "must not be written down", and quoting a secret does not stop it
being stored.

THE FOURTH SINK
---------------
Neither `--headless` nor `--api-key` exists on this CLI. argparse took the key
as an invalid positional choice and echoed it back:

    oara: error: argument command: invalid choice: 'sk-ant-...'
      (choose from setup, token, doctor, ...)

That is the child's stderr, which `_watch_process` captures into the task's
output file and `task_get` reads back. So the leak reproduced itself in a
second durable place, through the error message of a command that could never
have worked: every local_agent spawn died at argument parsing.

THE FIX THESE TESTS PIN
-----------------------
The credential travels in the child's ENVIRONMENT, through an in-memory overlay
attached to nothing durable. There is no TaskRecord field for it, so `_persist`
has nothing to write, TaskStore has no column, REST has no key, and `task_get`
has nothing to print. The perimeter is the SHAPE — the secret is never in a
persisted field — not a redaction pass over a field that still holds it.

These tests assert on CONTENT, not on absence of a field: they take the key
they passed in and search every durable surface for that exact string. A
redaction that replaced the key with `***` would pass a "field is not the key"
check and fail these, which is the intended direction — and a value that was
never stored is not the same as one that was stored and masked.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.tasks.manager import BackgroundTaskManager  # noqa: E402
from prometheus.tasks.store import TaskStore  # noqa: E402

# Shaped like a real Anthropic key so the credential-shaped guards in this repo
# treat it the way they would treat one, and obviously synthetic so it is not.
FAKE_KEY = "sk-ant-api03-NOT-A-REAL-KEY-" + "0" * 40


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        "prometheus.tasks.manager.get_tasks_dir", lambda: tmp_path, raising=False
    )
    mgr = BackgroundTaskManager()
    mgr.store = TaskStore(tmp_path / "tasks.db")
    return mgr


def _spawn(mgr, tmp_path, **kwargs):
    return asyncio.run(
        mgr.create_agent_task(
            prompt="do the thing",
            description="spawned agent",
            cwd=str(tmp_path),
            api_key=FAKE_KEY,
            **kwargs,
        )
    )


# --------------------------------------------------------------------------
# The perimeter — every durable surface, checked for the literal key
# --------------------------------------------------------------------------
def test_the_key_is_not_in_the_command(manager, tmp_path):
    record = _spawn(manager, tmp_path)
    assert record.command is not None
    assert FAKE_KEY not in record.command, (
        f"the key is in the persisted command string: {record.command!r}"
    )


def test_the_key_is_not_anywhere_in_the_record(manager, tmp_path):
    """Not just `command` — the whole record, serialised.

    A fix that moved the key from `command` to `metadata` or `spec` would pass
    the narrower test and leak exactly as much.
    """
    record = _spawn(manager, tmp_path)
    blob = json.dumps(asdict(record), default=str)
    assert FAKE_KEY not in blob, (
        "the key appears somewhere in the TaskRecord. Every field on this "
        "record is persisted and served over REST."
    )


def _db_bytes(tmp_path: Path) -> bytes:
    """Every byte sqlite has on disk for this store.

    `tasks.db` alone is NOT enough: TaskStore runs `PRAGMA journal_mode=WAL`,
    so a committed row sits in `tasks.db-wal` until a checkpoint moves it. An
    earlier draft of this test read only the main file, found it full of zeros,
    and would have reported "the key is not on disk" about a database whose
    rows it had not looked at. Globbing `tasks.db*` covers the main file, the
    WAL and the shared-memory index without naming any of them.
    """
    return b"".join(p.read_bytes() for p in sorted(tmp_path.glob("tasks.db*")))


def _db_row_text(tmp_path: Path) -> str:
    """Every value of every column of every row, as text.

    `SELECT *` rather than a column list — a leak into a column this test did
    not think to name is exactly the case worth catching.
    """
    import sqlite3

    conn = sqlite3.connect(tmp_path / "tasks.db")
    try:
        rows = conn.execute("SELECT * FROM tasks").fetchall()
    finally:
        conn.close()
    return "\n".join(str(value) for row in rows for value in row)


def test_the_key_is_not_in_the_database(manager, tmp_path):
    record = _spawn(manager, tmp_path)
    assert manager.store is not None

    row_text = _db_row_text(tmp_path)
    assert record.id in row_text, (
        "the task was not actually persisted, so searching the database for "
        "the key proves nothing"
    )
    assert FAKE_KEY not in row_text, "the key is in a task-table column"

    raw = _db_bytes(tmp_path)
    assert record.id.encode() in raw, (
        "nothing on disk mentions this task — the byte scan is reading the "
        "wrong files"
    )
    assert FAKE_KEY.encode() not in raw, (
        "the key is on disk somewhere under the task store, even if not in a "
        "column this test knows about"
    )


async def _spawn_and_drain(mgr, tmp_path, **kwargs):
    """Spawn, then let the output reader finish — IN THE SAME EVENT LOOP.

    This must be async. `asyncio.run()` tears the loop down the moment it
    returns, and `_watch_process` / `_copy_output` are tasks ON that loop, so
    the output file is still empty when the helper hands back. An earlier draft
    of the test below used `asyncio.run` and read a zero-byte file every time —
    it asserted "the key is not in the output" about output that had never been
    captured, and passed just as happily with the leak restored.
    """
    record = await mgr.create_agent_task(
        prompt="do the thing",
        description="spawned agent",
        cwd=str(tmp_path),
        api_key=FAKE_KEY,
        **kwargs,
    )
    # POLL FOR OUTPUT, do not wait for exit. The fixed command starts an
    # interactive agent that reads stdin and does not terminate on its own, so
    # awaiting the process waiter would block until the timeout — 60 seconds
    # of a test suite, to learn something visible in the first few hundred
    # milliseconds.
    for _ in range(150):
        if record.output_file.exists() and record.output_file.stat().st_size:
            break
        await asyncio.sleep(0.1)
    await mgr.stop_task(record.id)
    return record


async def test_the_key_is_not_in_the_task_output(manager, tmp_path):
    """The sink nobody intended: argparse echoing the key back as an error.

    `oara: error: argument command: invalid choice: 'sk-ant-...'` is the
    child's stderr, and `_copy_output` captures stderr into the task's output
    file, which `task_get` and the REST output endpoint both read back.
    """
    record = await _spawn_and_drain(manager, tmp_path)
    text = record.output_file.read_text(encoding="utf-8", errors="replace")
    assert text.strip(), (
        "the task produced NO output at all — this test cannot distinguish a "
        "clean capture from a capture that never happened, so its verdict is "
        "worthless. Fix the harness before trusting it."
    )
    assert FAKE_KEY not in text, (
        f"the key was echoed into the task's captured output:\n{text[:800]}"
    )


async def test_the_old_flags_would_have_echoed_the_key_into_the_output(
    manager, tmp_path
):
    """Mutation check for the sink above: prove the capture really sees stderr.

    Replays the removed command verbatim. If this does not find the key, the
    test above is measuring nothing.
    """
    import shlex

    old_command = " ".join(
        shlex.quote(p) for p in
        ["python", "-m", "prometheus", "--headless", "--api-key", FAKE_KEY]
    )
    record = await manager.create_agent_task(
        prompt="p", description="d", cwd=str(tmp_path), command=old_command,
    )
    waiter = manager._waiters.get(record.id)
    if waiter is not None:
        try:
            await asyncio.wait_for(waiter, timeout=60)
        except asyncio.TimeoutError:  # pragma: no cover
            pass
    text = record.output_file.read_text(encoding="utf-8", errors="replace")
    assert FAKE_KEY in text, (
        "the pre-fix command did NOT echo the key into the captured output. "
        "Either argparse stopped quoting the offending value, or this harness "
        "is not capturing child stderr — in which case the output-file test "
        "above proves nothing.\n"
        f"captured: {text[:800]!r}"
    )


def test_the_command_no_longer_uses_flags_that_do_not_exist(manager, tmp_path):
    """`--headless` and `--api-key` are not arguments this CLI accepts.

    Keeping them guaranteed the child died at argument parsing — and that the
    death message contained the key.
    """
    record = _spawn(manager, tmp_path)
    assert "--headless" not in record.command
    assert "--api-key" not in record.command


def test_the_model_override_still_reaches_the_command(manager, tmp_path):
    """Neutralising the leak must not silently drop the caller's arguments."""
    record = _spawn(manager, tmp_path, model="some-model")
    assert "--model" in record.command
    assert "some-model" in record.command
    assert FAKE_KEY not in record.command


def test_the_key_does_reach_the_child_environment(manager, tmp_path):
    """Removed from the command is not the same as not delivered.

    A "fix" that merely dropped the credential would pass every test above and
    break the feature. The overlay must actually be attached to the task.
    """
    record = _spawn(manager, tmp_path)
    overlay = manager._task_env.get(record.id)
    assert overlay is not None, "no environment overlay was registered"
    assert overlay.get("ANTHROPIC_API_KEY") == FAKE_KEY


def test_an_explicit_command_override_gets_no_injected_credential(manager, tmp_path):
    """The caller's own string is theirs; nothing is spliced into it."""
    record = asyncio.run(
        manager.create_agent_task(
            prompt="p",
            description="d",
            cwd=str(tmp_path),
            api_key=FAKE_KEY,
            command="echo hello",
        )
    )
    assert record.command == "echo hello"
    assert manager._task_env.get(record.id) is None


# --------------------------------------------------------------------------
# Mutation check
# --------------------------------------------------------------------------
def test_the_old_construction_would_have_leaked_into_all_of_them(manager, tmp_path):
    """Replay the removed line and prove these assertions have power.

    If the harness stopped exercising anything — a task that is never created,
    a store that never writes — every test above would still be green and this
    one would not.
    """
    import shlex

    cmd = ["python", "-m", "prometheus", "--headless", "--api-key", FAKE_KEY]
    old_command = " ".join(shlex.quote(part) for part in cmd)

    assert FAKE_KEY in old_command, (
        "the removed construction no longer embeds the key — this test's "
        "premise is stale and the perimeter tests may be vacuous"
    )

    # And it survives `shlex.quote` intact, which is the point: escaping is not
    # concealment.
    assert shlex.quote(FAKE_KEY) == FAKE_KEY or FAKE_KEY in shlex.quote(FAKE_KEY)

    # The record built from it would carry the key through every surface.
    record = asyncio.run(
        manager.create_agent_task(
            prompt="p",
            description="d",
            cwd=str(tmp_path),
            command=old_command,
        )
    )
    assert FAKE_KEY in record.command
    assert FAKE_KEY in json.dumps(asdict(record), default=str)
    assert FAKE_KEY in _db_row_text(tmp_path), (
        "the store did not persist the command — the database assertion above "
        "is not actually reading what it thinks it is"
    )
    assert FAKE_KEY.encode() in _db_bytes(tmp_path), (
        "the byte scan cannot see a key that IS in the database — it is "
        "reading the wrong files, and its clean verdict above means nothing"
    )
