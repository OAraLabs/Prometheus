"""Capture real wire exchanges from the TCK's conforming v2 fixture agent.

Drives tests/fixtures/agents/v2/conforming.py over stdio (the TCK's own
harness pattern: NDJSON, one frame per line) and records the agent's responses
VERBATIM. The output is a fixture file, not hand-written data.

Usage:
    python capture.py /path/to/acp-tck-checkout [out.json]

The driver requests below are the TCK's own initialize/new/prompt/resume
shapes (read from its _base.py handler), sent with our framing. Only the
AGENT's lines are stored; they are the other side's real code output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    tck = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "tck_exchanges.json"
    agent = tck / "tests" / "fixtures" / "agents" / "v2" / "conforming.py"
    assert agent.exists(), agent

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": 2, "info": {"name": "capture", "version": "0"},
                    "capabilities": {}}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/new",
         "params": {"cwd": str(tck)}},
    ]
    proc = subprocess.Popen(
        [sys.executable, str(agent)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=str(agent.parent),
    )
    captured: list[dict] = []
    session_id = None
    try:
        for req in requests:
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            captured.append({"request": req, "response_raw": line.rstrip("\n")})
            resp = json.loads(line)
            if req["method"] == "session/new":
                session_id = resp["result"]["sessionId"]
        if session_id:
            more = [
                {"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
                 "params": {"sessionId": session_id,
                            "prompt": [{"type": "text", "text": "hello"}],
                            "idempotencyKey": "capture-key-1"}},
                {"jsonrpc": "2.0", "id": 4, "method": "session/prompt",
                 "params": {"sessionId": session_id,
                            "prompt": [{"type": "text", "text": "hello"}],
                            "idempotencyKey": "capture-key-1"}},
                {"jsonrpc": "2.0", "id": 5, "method": "session/resume",
                 "params": {"sessionId": session_id, "cwd": str(tck),
                            "replayFrom": {"type": "start"}}},
                {"jsonrpc": "2.0", "id": 6, "method": "session/list", "params": {}},
            ]
            for req in more:
                proc.stdin.write(json.dumps(req) + "\n")
                proc.stdin.flush()
                # Read until the response with this id arrives; notifications
                # (session/update) in between are captured too, verbatim.
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    msg = json.loads(line)
                    captured.append({"request": req, "response_raw": line.rstrip("\n")})
                    if msg.get("id") == req["id"]:
                        break
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
    out.write_text(json.dumps(captured, indent=2) + "\n")
    print(f"captured {len(captured)} lines -> {out}")


if __name__ == "__main__":
    main()