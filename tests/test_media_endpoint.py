"""#339 — GET /api/media + upload turns persist their image block (Phase 3b).

Two halves. The endpoint: stored image bytes served by reference, with the
path-guard doctrine (resolve THEN compare — symlinks and .. included; a
source_path is history-supplied data and must not read arbitrary disk).
The persistence fix: an uploaded image's block now rides the SAME user turn
into the durable store as a reference — before this, blocks were attached
after the persist, so the stored row was a text marker and no thumbnail
could ever come back (the live finding in the issue).
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from prometheus.engine.messages import ImageBlock  # noqa: E402
from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.gateway.media_cache import image_cache_dir  # noqa: E402
from prometheus.memory.lcm_engine import LCMEngine  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"x" * 64


def _cache_png(name: str = "shot.png") -> Path:
    p = image_cache_dir() / name
    p.write_bytes(PNG_BYTES)
    return p


class TestMediaEndpoint:
    @pytest.fixture()
    def client(self) -> TestClient:
        return TestClient(create_app({}))

    def test_serves_cached_bytes_with_sniffed_type(self, client) -> None:
        p = _cache_png()
        resp = client.get("/api/media", params={"path": str(p)})
        assert resp.status_code == 200
        assert resp.content == PNG_BYTES
        assert resp.headers["content-type"].startswith("image/png")

    def test_path_outside_the_media_root_is_refused(
        self, client, tmp_path: Path
    ) -> None:
        outside = tmp_path / "secret.png"
        outside.write_bytes(PNG_BYTES)
        resp = client.get("/api/media", params={"path": str(outside)})
        assert resp.status_code == 403

    def test_traversal_out_of_the_root_is_refused(
        self, client, tmp_path: Path
    ) -> None:
        secret = tmp_path / "env-file"
        secret.write_text("TOKEN=abc")
        dotted = image_cache_dir() / ".." / ".." / ".." / str(secret).lstrip("/")
        resp = client.get("/api/media", params={"path": str(dotted)})
        assert resp.status_code in (403, 404)
        assert b"TOKEN" not in resp.content

    def test_symlink_escaping_the_root_is_refused(
        self, client, tmp_path: Path
    ) -> None:
        # Resolve-then-compare is the doctrine: the literal path is inside
        # the root, the TARGET is not.
        secret = tmp_path / "secret.bin"
        secret.write_bytes(b"private")
        link = image_cache_dir() / "innocent.png"
        os.symlink(secret, link)
        resp = client.get("/api/media", params={"path": str(link)})
        assert resp.status_code == 403
        assert b"private" not in resp.content

    def test_evicted_file_is_a_404_not_an_error(self, client) -> None:
        resp = client.get(
            "/api/media", params={"path": str(image_cache_dir() / "gone.png")}
        )
        assert resp.status_code == 404


class TestUploadTurnPersistsItsBlock:
    def test_stored_row_carries_the_reference_block(self, tmp_path: Path) -> None:
        # The acceptance line: upload → reload → the stored turn carries an
        # image block with a source_path (and NO base64 in history).
        engine = LCMEngine(MagicMock(), db_path=tmp_path / "lcm.db")
        mgr = SessionManager()
        mgr.lcm_engine = engine
        session = mgr.get_or_create("desktop:upload-test")

        cached = _cache_png("upload.png")
        block = ImageBlock(
            media_type="image/png",
            data=base64.b64encode(PNG_BYTES).decode("ascii"),
            source_path=str(cached),
        )
        session.add_user_message("[Image: upload.png]", blocks=[block])

        con = sqlite3.connect(str(tmp_path / "lcm.db"))
        content_json = con.execute(
            "SELECT content_json FROM lcm_messages "
            "WHERE session_id='desktop:upload-test'"
        ).fetchone()[0]
        con.close()
        stored = json.loads(content_json)
        kinds = [b["type"] for b in stored]
        assert kinds == ["text", "image"], kinds
        assert stored[1]["source_path"] == str(cached)
        assert stored[1]["data"] == ""          # reference, not payload
        assert base64.b64encode(PNG_BYTES).decode("ascii") not in content_json

        # The LIVE message kept its bytes — the model still sees the image
        # on this turn.
        assert session.messages[-1].content[1].data != ""

    def test_ws_send_path_passes_blocks_before_the_persist(self) -> None:
        # Wiring assertion in the repo's source-guard style: the old shape
        # (attach after add_user_message) must not come back — it is the
        # exact line that kept image blocks out of history.
        import inspect

        from prometheus.web.ws_server import WebSocketBridge

        src = inspect.getsource(WebSocketBridge._handle_send_message)
        assert "add_user_message(content, blocks=blocks)" in src
        assert "session.messages[-1].content.extend(blocks)" not in src
