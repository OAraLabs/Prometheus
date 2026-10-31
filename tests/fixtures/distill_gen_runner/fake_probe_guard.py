"""Stand-in for ~/distill/phase4/probe_guard.py (its CLI), for the generation-runner tests.

Logs its arguments to $FAKE_STATE/guard-calls.log, writes --pidfile, listens on --listen and passes GETs
to --upstream. Once $FAKE_STATE/guard-trip exists it trips the abort switch like the real one: writes
--abort-file and answers 503 from then on. SIGTERM: stops, removes its pidfile, exits 0.
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    for opt in ("--listen", "--upstream", "--probe-unit", "--probe-name", "--llama-unit", "--log", "--pidfile",
                "--deadline", "--abort-file", "--notify-cmd", "--min-tps"):
        ap.add_argument(opt)
    a = ap.parse_args()
    state = Path(os.environ["FAKE_STATE"])
    with open(state / "guard-calls.log", "a") as f:
        f.write(json.dumps(sys.argv[1:]) + "\n")
    aborted = threading.Event()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if aborted.is_set():
                self.send_response(503)
                self.end_headers()
                return
            try:
                with urllib.request.urlopen(f"http://{a.upstream}{self.path}", timeout=5) as r:
                    body, status = r.read(), r.status
            except Exception:  # noqa: BLE001
                body, status = b"{}", 502
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    host, port = a.listen.rsplit(":", 1)
    srv = ThreadingHTTPServer((host, int(port)), H)
    Path(a.pidfile).write_text(f"{os.getpid()}\n")

    def watch():
        while not aborted.is_set():
            if (state / "guard-trip").exists():
                Path(a.abort_file).write_text("probe at 03:10:11: verdict DARK\n")
                aborted.set()
            time.sleep(0.05)

    threading.Thread(target=watch, daemon=True).start()

    def stop(*_):
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    try:
        srv.serve_forever(poll_interval=0.05)
    finally:
        srv.server_close()
        Path(a.pidfile).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
