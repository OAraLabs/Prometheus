"""Item 4 — per-turn file checkpoints for sessions with a workspace.

The three look-alikes each disclaim undo in their own docstrings (the
mutation verifier holds no content, the divergence CheckpointStore is
forensic, BackupVault snapshots Prometheus's own tree). This one holds the
bytes. Under test: the store's create/diff/restore contract including files
the turn CREATED (deleted on restore) and files too large to capture
(reported, never pretended); caps that refuse rather than lie; retention
with blob GC; the loop taking a checkpoint at turn start ONLY when the
session has a workspace; and the REST surface with its confirm-by-name.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from prometheus.checkpoints import FileCheckpointStore, resolve_checkpoints_config
from prometheus.checkpoints.store import CheckpointRefused


def _ws(tmp_path) -> Path:
    ws = tmp_path / "ws"; ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "app.py").write_text("print('v1')\n")
    (ws / "README.md").write_text("readme v1\n")
    (ws / ".git").mkdir(); (ws / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (ws / "node_modules").mkdir(); (ws / "node_modules" / "x.js").write_text("junk\n")
    return ws


class TestConfig:
    def test_defaults_and_overrides(self) -> None:
        d = resolve_checkpoints_config({})
        assert d["enabled"] is True and d["keep_per_session"] == 20 and ".git" in d["skip_dirs"]
        from prometheus.checkpoints.store import DEFAULT_SKIP_DIRS, DEFAULTS
        assert d["skip_dirs"] == list(DEFAULT_SKIP_DIRS) == DEFAULTS["skip_dirs"]   # the literal and the constant agree
        o = resolve_checkpoints_config({"checkpoints": {"enabled": False, "max_files": 3, "skip_dirs": ["out"]}})
        assert o["enabled"] is False and o["max_files"] == 3 and o["skip_dirs"] == ["out"]


class TestStore:
    def test_create_captures_content_and_skips_generated_trees(self, tmp_path) -> None:
        ws = _ws(tmp_path)
        store = FileCheckpointStore(tmp_path / "cp")
        rec = store.create("s1", ws, "first turn")
        assert rec.files == 2 and rec.skipped == 0 and rec.label == "first turn"
        assert {f["rel_path"] for f in rec.files_detail} == {"README.md", "src/app.py"}
        # Content-addressed: the blob is the file's bytes, once.
        blob = store._blob_path(rec.files_detail[0]["sha256"])
        assert blob.exists()
        assert store.list("s1")[0]["id"] == rec.id

    def test_diff_and_restore_changed_deleted_and_added(self, tmp_path) -> None:
        ws = _ws(tmp_path)
        store = FileCheckpointStore(tmp_path / "cp")
        rec = store.create("s1", ws)
        # The turn: modifies one file, deletes one, creates one (as bash would).
        (ws / "src" / "app.py").write_text("print('v2')\n")
        (ws / "README.md").unlink()
        (ws / "src" / "new.py").write_text("created by the turn\n")
        d = store.diff(rec.id)
        assert d["changed"] == ["src/app.py"] and d["deleted"] == ["README.md"] and d["added"] == ["src/new.py"]
        dry = store.restore(rec.id, dry_run=True)
        assert dry["dry_run"] and sorted(dry["restored"]) == ["README.md", "src/app.py"] and dry["deleted"] == ["src/new.py"]
        assert (ws / "src" / "app.py").read_text() == "print('v2')\n"      # dry run touched nothing
        res = store.restore(rec.id)
        assert (ws / "src" / "app.py").read_text() == "print('v1')\n"
        assert (ws / "README.md").read_text() == "readme v1\n"
        assert not (ws / "src" / "new.py").exists()
        assert res["errors"] == []
        assert store.diff(rec.id)["changed"] == [] and store.diff(rec.id)["added"] == []

    def test_too_large_is_recorded_not_restored_and_named(self, tmp_path) -> None:
        ws = _ws(tmp_path)
        (ws / "big.bin").write_bytes(b"x" * 5000)
        store = FileCheckpointStore(tmp_path / "cp", config={"max_file_bytes": 1000})
        rec = store.create("s1", ws)
        assert rec.skipped == 1 and rec.skipped_detail[0]["rel_path"] == "big.bin"
        (ws / "big.bin").write_bytes(b"y" * 5000)
        d = store.diff(rec.id)
        assert d["uncaptured"] == ["big.bin"] and "big.bin" not in d["added"]
        res = store.restore(rec.id)
        assert (ws / "big.bin").read_bytes() == b"y" * 5000               # never touched
        assert res["uncaptured"] == ["big.bin"]

    def test_caps_refuse_and_write_nothing(self, tmp_path) -> None:
        ws = _ws(tmp_path)
        store = FileCheckpointStore(tmp_path / "cp", config={"max_files": 1})
        with pytest.raises(CheckpointRefused, match="max_files"):
            store.create("s1", ws)
        assert store.list("s1") == []
        store2 = FileCheckpointStore(tmp_path / "cp2", config={"max_total_bytes": 5})
        with pytest.raises(CheckpointRefused, match="max_total_bytes"):
            store2.create("s1", ws)

    def test_retention_prunes_and_gcs_blobs(self, tmp_path) -> None:
        ws = _ws(tmp_path)
        store = FileCheckpointStore(tmp_path / "cp", config={"keep_per_session": 2})
        first = store.create("s1", ws)
        (ws / "README.md").write_text("readme v2\n"); store.create("s1", ws)
        (ws / "README.md").write_text("readme v3\n"); store.create("s1", ws)
        ids = [c["id"] for c in store.list("s1")]
        assert len(ids) == 2 and first.id not in ids
        v1_sha = next(f["sha256"] for f in first.files_detail if f["rel_path"] == "README.md")
        assert not store._blob_path(v1_sha).exists()                        # unreferenced → gone
        assert store.get(first.id) is None

    def test_missing_workspace_is_refused(self, tmp_path) -> None:
        store = FileCheckpointStore(tmp_path / "cp")
        with pytest.raises(CheckpointRefused):
            store.create("s1", tmp_path / "nope")


class TestLoop:
    def _drive(self, tmp_path, *, workspace):
        from prometheus.engine.agent_loop import LoopContext, run_loop
        from prometheus.engine.messages import ConversationMessage, TextBlock
        from prometheus.engine.usage import UsageSnapshot
        from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
        from prometheus.tools.base import ToolRegistry

        class _P(ModelProvider):
            async def stream_message(self, request):  # noqa: ANN001
                yield ApiMessageCompleteEvent(
                    message=ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop",
                )

        store = FileCheckpointStore(tmp_path / "cp")
        ctx = LoopContext(
            provider=_P(), model="stub", system_prompt="BASE", max_tokens=32, tool_registry=ToolRegistry(),
            cwd=tmp_path, checkpoint_store=store,
            workspace_resolver=(lambda sid: str(workspace)) if workspace else None,
        )

        async def go():
            async for _ in run_loop(ctx, [ConversationMessage.from_user_text("please refactor the app")], session_id="desktop:s1"):
                pass
        asyncio.run(go())
        return store

    def test_a_turn_with_a_workspace_checkpoints_first(self, tmp_path) -> None:
        ws = _ws(tmp_path)
        store = self._drive(tmp_path, workspace=ws)
        cps = store.list("desktop:s1")
        assert len(cps) == 1 and cps[0]["files"] == 2 and cps[0]["label"] == "please refactor the app"

    def test_label_is_the_human_message_not_an_injected_turn(self, tmp_path) -> None:
        """Live shape 2026-09-01: the label read "[FILE MUTATION VERIFIER]
        Files touched this turn…" because the previous turn's injected report
        is a user-role message. A person reads these labels."""
        from prometheus.engine.agent_loop import LoopContext, run_loop
        from prometheus.engine.messages import ConversationMessage, TextBlock
        from prometheus.engine.usage import UsageSnapshot
        from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
        from prometheus.tools.base import ToolRegistry

        class _P(ModelProvider):
            async def stream_message(self, request):  # noqa: ANN001
                yield ApiMessageCompleteEvent(
                    message=ConversationMessage(role="assistant", content=[TextBlock(text="ok")]),
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop",
                )

        ws = _ws(tmp_path)
        store = FileCheckpointStore(tmp_path / "cp")
        ctx = LoopContext(provider=_P(), model="stub", system_prompt="BASE", max_tokens=32,
                          tool_registry=ToolRegistry(), cwd=tmp_path, checkpoint_store=store,
                          workspace_resolver=lambda sid: str(ws))
        history = [
            ConversationMessage.from_user_text("please add a README"),
            ConversationMessage(role="assistant", content=[TextBlock(text="done")]),
            ConversationMessage(role="user", content=[TextBlock(text="[FILE MUTATION VERIFIER]\nFiles touched this turn: …")],
                                provenance="file_mutation_verifier", is_trusted=False),
        ]

        async def go():
            async for _ in run_loop(ctx, history, session_id="desktop:s1"):
                pass
        asyncio.run(go())
        assert store.list("desktop:s1")[0]["label"] == "please add a README"

    def test_no_workspace_no_checkpoint(self, tmp_path) -> None:
        store = self._drive(tmp_path, workspace=None)
        assert store.list("desktop:s1") == []


class TestRoutes:
    def _client(self, tmp_path, store=None):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from prometheus.web.server import create_app
        app = create_app({})
        app.state.checkpoint_store = store
        return TestClient(app)

    def test_without_a_store_is_503(self, tmp_path) -> None:
        assert self._client(tmp_path).get("/api/sessions/s/checkpoints").status_code == 503

    def test_list_get_restore_with_confirm(self, tmp_path) -> None:
        ws = _ws(tmp_path)
        store = FileCheckpointStore(tmp_path / "cp")
        rec = store.create("desktop:s1", ws, "turn one")
        (ws / "README.md").write_text("changed\n"); (ws / "extra.txt").write_text("new\n")
        client = self._client(tmp_path, store)
        assert client.get("/api/sessions/desktop:s1/checkpoints").json()["checkpoints"][0]["id"] == rec.id
        body = client.get(f"/api/sessions/desktop:s1/checkpoints/{rec.id}").json()
        assert body["diff"]["changed"] == ["README.md"] and body["diff"]["added"] == ["extra.txt"]
        # Another session cannot see or restore it.
        assert client.get(f"/api/sessions/other/checkpoints/{rec.id}").status_code == 404
        # Wrong confirm → 400, nothing touched.
        r = client.post(f"/api/sessions/desktop:s1/checkpoints/{rec.id}/restore", json={"confirm": "nope"})
        assert r.status_code == 400 and (ws / "README.md").read_text() == "changed\n"
        # Dry run needs no confirm and touches nothing.
        r = client.post(f"/api/sessions/desktop:s1/checkpoints/{rec.id}/restore", json={"dry_run": True})
        assert r.status_code == 200 and r.json()["restored"] == ["README.md"] and (ws / "extra.txt").exists()
        # The real thing.
        r = client.post(f"/api/sessions/desktop:s1/checkpoints/{rec.id}/restore", json={"confirm": rec.id})
        assert r.status_code == 200, r.text
        assert (ws / "README.md").read_text() == "readme v1\n" and not (ws / "extra.txt").exists()
