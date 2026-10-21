"""Measure round 0 of a fresh install's first turn, for ``local_probe.MIN_WINDOW`` (scratch only).

A fresh install is what a person gets from Beacon's first run with a local AI: setup mode with no config, paired,
configured for that AI, completed, and then a first message. This drives exactly that, from a given checkout, in a
scratch HOME, and records round 0's prompt in the model server's own tokens (``usage.prompt_tokens``; Ollama's
``prompt_eval_count``), read from the logging proxy (``proxy.py``) the model is reached through.

Configure's re-probe runs against the real server through the proxy, so this also exercises
``providers/local_probe.py`` on a real llama.cpp or Ollama.

Never point this at a live daemon: it starts its own, on spare loopback ports, from ``--checkout``. A fresh install
listens on this machine only, so nothing is advertised. The model server is shared: one turn per run.

Usage:
  python measure_floor.py --checkout DIR --work DIR --upstream-config CFG --provider llama_cpp|ollama
         --model ID --label NAME [--message TEXT] [--results OUT.json]

``--upstream-config`` is a config whose ``model.base_url`` is the real model server (read by the proxy, never printed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import socket
import sys
import time
import uuid
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_gate import _drive, _start, _stop  # noqa: E402

#: Beacon's first suggestion on its "Say hello" screen.
FIRST_MESSAGE = "What can you help me with?"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_code(log: Path, timeout: float = 120.0) -> str:
    end = time.time() + timeout
    while time.time() < end:
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        m = re.search(r"Pairing code \(printed once[^\n]*\n\s*\n\s*(\d{6})\s*\n", text)
        if m:
            return m.group(1)
        time.sleep(1)
    raise SystemExit(f"setup mode printed no pairing code within {timeout:.0f}s (see {log})")


def _round0(capture: Path) -> dict:
    """The turn's first model request: the first captured chat request that offers tools (a side call, such as the
    session title, offers none)."""
    for line in capture.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        req = rec.get("request") or {}
        if req.get("tools"):
            usage = (rec.get("response") or {}).get("usage") or {}
            return {"prompt_tokens": usage.get("prompt_tokens"), "tools_offered": len(req["tools"]),
                    "messages": len(req.get("messages") or []), "path": rec.get("path")}
    raise SystemExit(f"no request with tools was captured in {capture}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--upstream-config", required=True)
    ap.add_argument("--provider", required=True, choices=("llama_cpp", "ollama", "lm_studio", "vllm"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--message", default=FIRST_MESSAGE)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--results")
    args = ap.parse_args()

    checkout, work = Path(args.checkout).resolve(), Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    home = work / f"home-{args.label}"
    if home.exists():
        raise SystemExit(f"{home} exists: a fresh install starts from nothing")
    home.mkdir()
    cwd = work / f"cwd-{args.label}"  # an empty directory: no project files, as an app-launched daemon has none
    cwd.mkdir(exist_ok=True)
    api, ws, proxy_port = _free_port(), _free_port(), _free_port()
    capture, dlog = work / f"capture-{args.label}.jsonl", work / f"daemon-{args.label}.log"
    for stale in (capture, dlog):  # appended to by the processes: an old run's pairing code would be read
        stale.unlink(missing_ok=True)
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home), "LANG": "C.UTF-8",
        "PROMETHEUS_CONFIG_DIR": str(home / ".prometheus"), "XDG_CONFIG_HOME": str(home / ".config"),
        "PYTHONPATH": str(checkout / "src"), "PYTHONSAFEPATH": "1", "PYTHONNOUSERSITE": "1",
        "PROMETHEUS_WEB_API_PORT": str(api), "PROMETHEUS_WEB_WS_PORT": str(ws),
    }
    proxy = daemon = None
    base = f"http://127.0.0.1:{api}"
    try:
        proxy = _start([sys.executable, str(HERE / "proxy.py"), str(proxy_port), str(capture),
                        str(Path(args.upstream_config).expanduser())], work / f"proxy-{args.label}.log", work,
                       {**__import__("os").environ})
        time.sleep(2)
        # --bind 127.0.0.1: setup mode otherwise listens on every interface while it waits to be paired. The config it
        # writes says 127.0.0.1 either way (a fresh install listens on this machine only).
        daemon = _start([str(checkout / ".venv" / "bin" / "python"), str(checkout / "scripts" / "daemon.py"),
                         "--bind", "127.0.0.1"], dlog, cwd, env)
        code = _wait_for_code(dlog)
        with httpx.Client(timeout=60) as http:
            # The banner is printed before setup mode listens.
            for _ in range(60):
                try:
                    if http.get(f"{base}/api/setup/status", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(1)
            token = http.post(f"{base}/api/setup/pair", json={"code": code}).raise_for_status().json()["token"]
            auth = {"Authorization": f"Bearer {token}"}
            conf = http.post(f"{base}/api/setup/configure", headers=auth, json={
                "provider": args.provider, "base_url": f"http://127.0.0.1:{proxy_port}", "model": args.model})
            if conf.status_code != 200:
                raise SystemExit(f"configure refused: {conf.status_code} {conf.text[:400]}")
            http.post(f"{base}/api/setup/complete", headers=auth, json={}).raise_for_status()
            end = time.time() + 240
            while time.time() < end:
                try:
                    if http.get(f"{base}/api/status", headers=auth, timeout=5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise SystemExit("the configured daemon did not come up within 240s")
            effective = (http.get(f"{base}/api/status", headers=auth).json().get("context") or {})
        sid = f"beacon:{uuid.uuid4()}"
        run = asyncio.run(_drive(api, ws, token, sid, args.message, args.timeout))
        time.sleep(3)
        result = {"label": args.label, "provider": args.provider, "model": args.model,
                  "checkout_commit": _commit(checkout), "message": args.message,
                  "round0": _round0(capture), "turn_elapsed_s": run["elapsed_s"], "timed_out": run["timed_out"],
                  "status_context": {k: effective.get(k) for k in ("limit", "source", "fallback") if k in effective}}
        print(json.dumps(result))
        if args.results:
            Path(args.results).write_text(json.dumps(result, indent=1), encoding="utf-8")
    finally:
        _stop(daemon)
        _stop(proxy)


def _commit(checkout: Path) -> str:
    import subprocess

    out = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return out.stdout.strip() + ("+dirty" if dirty else "")


if __name__ == "__main__":
    main()
