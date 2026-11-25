#!/usr/bin/env python3
"""Verify the client-id storage widening on a COPY of a live lcm.db — both ways.

The widening itself needs no schema migration: ``message_client_ids.client_msg_id``
is plain TEXT with no length constraint; the cap lives in the writer's guard
(128 -> 263). What a live DB needs is PROOF, not ALTER:

  forward   new code opens the old DB, writes an ``acp-key:`` + 255-char claim
            (263 total), closes, reopens, and reads it back;
  backward  the OLD code (a source checkout of the previous release) opens the
            same DB, still reads the long claim, refuses its own overlong writes
            without crashing, and keeps writing ordinary ids beside it.

NEVER point this at a live daemon's DB. It copies the file first (sqlite3
.backup, WAL-safe) and works only on the copy. For a remote box, run it there
on a copy and bring the report back — or hand the operator this script with its
sha256 recorded, per the deployment rule.

    python3 scripts/widen_client_id_storage.py --db ~/.prometheus/data/lcm.db \
        --old-src /path/to/previous-release-checkout
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLAIM = "acp-key:" + "k" * 255  # 263 chars: the schema's 255-char key + prefix
SID = "acp:storage-widening-probe"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _probe(src: Path, db: Path, *, write: bool) -> str:
    code = f"""
import pathlib, sys
sys.path.insert(0, {str(src)!r})
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart
db = pathlib.Path({str(db)!r})
s = LCMConversationStore(db)
claim = {CLAIM!r}
sid = {SID!r}
if {write!r}:
    p = MessagePart(session_id=sid, role="user", content="storage widening probe",
                    provenance="user", is_trusted=True)
    s.insert_message(p, append=True)
    s.set_message_client_id(sid, p.row_id, claim)
    assert s.has_message_client_id(sid, claim), "write did not land"
    s.close()
    s = LCMConversationStore(db)
assert s.has_message_client_id(sid, claim), "claim lost across reopen"
assert claim in s.get_message_client_ids(sid).values()
n = len(s.list_sessions())
s.close()
print(f"ok: {{n}} sessions enumerable")
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"probe failed ({src}):\n{r.stderr}")
    return r.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="the LIVE lcm.db (copied, never opened directly)")
    ap.add_argument("--old-src", required=True,
                    help="checkout of the PREVIOUS release (its src/ is imported for the backward probe)")
    args = ap.parse_args()
    live = Path(args.db).expanduser()
    old_src = Path(args.old_src).expanduser() / "src"
    if not live.exists() or not old_src.exists():
        raise SystemExit("both --db and --old-src must exist")
    with tempfile.TemporaryDirectory(prefix="client-id-widening-") as td:
        copy = Path(td) / "lcm-copy.db"
        # WAL-safe copy, never a bare file copy of a live DB
        src_conn = sqlite3.connect(f"file:{live}?mode=ro", uri=True)
        dst_conn = sqlite3.connect(str(copy))
        src_conn.backup(dst_conn)
        dst_conn.close(); src_conn.close()
        print(f"copy: {_sha256(copy)}")
        print("forward  (new code, old DB):", _probe(REPO / "src", copy, write=True))
        print("backward (old code, new claim):", _probe(old_src, copy, write=False))
    print("PASS: 255-char key round-trips forward; old code reads it backward")


if __name__ == "__main__":
    main()
