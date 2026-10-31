"""Stand-in for llama-server, for the tower job runner tests (no model, no GPU).

Flags read: -m, -c, --host, --port (the rest are accepted and ignored). Environment:
FAKE_STATE (dir: gpu.<pid> marker while "holding VRAM", server-starts.log),
FAKE_SERVER_CTX_LIMITS ("<model name part>=<max ctx>,...": a larger -c exits 1 like a
failed KV allocation),
FAKE_SERVER_LOAD_S (503 on /health until loaded), FAKE_SERVER_DIE_AFTER_S (crash).
Answers /health, /props, /v1/models and /v1/chat/completions (content "42").
SIGTERM: drop the GPU marker and exit 0 (as the real one does once loaded).
limits.log: its RLIMIT_CORE (soft hard) per start.
"""

import json
import os
import resource
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def arg(flag, default=None):
    a = sys.argv[1:]
    return a[a.index(flag) + 1] if flag in a else default


def main():
    state = os.environ["FAKE_STATE"]
    ctx = int(arg("-c", "4096"))
    host, port = arg("--host", "127.0.0.1"), int(arg("--port", "8080"))
    model = arg("-m", "model.gguf")
    with open(os.path.join(state, "server-starts.log"), "a") as f:
        f.write(f"{os.getpid()} {ctx} {os.path.basename(model)} {' '.join(sys.argv[1:])}\n")
    with open(os.path.join(state, "limits.log"), "a") as f:
        f.write("core %d %d\n" % resource.getrlimit(resource.RLIMIT_CORE))
    limit = 1 << 30
    for item in filter(None, os.environ.get("FAKE_SERVER_CTX_LIMITS", "").split(",")):
        part, _, mx = item.partition("=")
        if part in os.path.basename(model):
            limit = int(mx)
    if ctx > limit:
        time.sleep(0.2)
        print("llama_init_from_model: failed to allocate buffer for kv cache", flush=True)
        sys.exit(1)
    marker = os.path.join(state, f"gpu.{os.getpid()}")
    open(marker, "w").close()
    t0 = time.time()
    load_s = float(os.environ.get("FAKE_SERVER_LOAD_S", "0.3"))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                ok = time.time() - t0 >= load_s
                self.send(200 if ok else 503, {"status": "ok" if ok else "loading model"})
            elif self.path == "/props":
                self.send(200, {"default_generation_settings": {"n_ctx": ctx}, "total_slots": 1})
            elif self.path == "/v1/models":
                self.send(200, {"object": "list", "data": [{"id": os.path.basename(model)}]})
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(n)
            if self.path == "/v1/chat/completions":
                self.send(200, {"id": "x", "object": "chat.completion", "model": os.path.basename(model),
                                "choices": [{"index": 0, "finish_reason": "stop",
                                             "message": {"role": "assistant", "content": "42"}}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}})
            else:
                self.send(404, {"error": "not found"})

    srv = ThreadingHTTPServer((host, port), H)

    def bye(*_):
        try:
            os.unlink(marker)
        except OSError:
            pass
        os._exit(0)

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    die = os.environ.get("FAKE_SERVER_DIE_AFTER_S")
    if die:
        threading.Timer(float(die), lambda: os._exit(1)).start()
    print(f"main: server is listening on http://{host}:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
