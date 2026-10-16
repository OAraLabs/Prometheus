"""Boot the real daemon for the local gate, and talk to it the way Beacon does.

Nothing between the config file and the WebSocket is replaced: ``scripts/
daemon.py`` runs as a child process from the checkout under test, reads a
config built from THAT checkout's shipped template, assembles its prompt,
builds its deferred catalog, dispatches tools, compacts, and streams frames to
the test client. Only the model is a stand-in (tests/support/scripted_llama.py),
reached over HTTP like the real server.

The checkout under test is ``LOCAL_GATE_CHECKOUT`` (a directory with ``src/``,
``scripts/daemon.py`` and ``.venv/``) when set, else this repository with the
running interpreter, so the same gate can be read against an older release.

Operator files are synthetic, at the sizes of a daily-driver install (SOUL,
AGENTS, ANATOMY, MEMORY, USER, 41 saved files), so the fixed overhead is of
the size the live box carries without copying anyone's content.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import signal
import socket
import string
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

REPO = Path(__file__).resolve().parents[2]
STUB_MCP = REPO / "scripts" / "local_gate" / "stub_mcp.py"

# Sizes of the operator files on the box this gate was written against (chars).
# MEMORY.md is padded past the live 10,349 so that round 0 of a turn with the
# SHIPPED tool set (12 tools, no Context7 or vault pair) costs what the live
# box's round 0 costs: about 18.6k server tokens.
OPERATOR_FILE_SIZES = {"SOUL.md": 3724, "AGENTS.md": 2370, "ANATOMY.md": 2592, "MEMORY.md": 24000, "USER.md": 644}
SAVED_FILES = 41

# ~2.35k server tokens each: a fresh window of three results plus the live
# overhead crosses the 26,214 line, while the largest recent result stays
# under the 2,458 at which the projected rule binds with a 4096 reply room.
LEDGER_MODULE_CHARS = 8800

MCP_TOOLS = [{
    "name": "lookup",
    "description": "Look up a library's documentation by library name and topic.",
    "inputSchema": {"type": "object", "properties": {
        "library": {"type": "string"}, "topic": {"type": "string"}}, "required": ["library", "topic"]},
}]


def checkout_under_test() -> tuple[Path, str]:
    raw = os.environ.get("LOCAL_GATE_CHECKOUT")
    if raw:
        root = Path(raw).expanduser().resolve()
        return root, str(root / ".venv" / "bin" / "python")
    return REPO, sys.executable


def _free_ports(n: int) -> list[int]:
    socks, ports = [], []
    for _ in range(n):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        socks.append(s)
        ports.append(s.getsockname()[1])
    for s in socks:
        s.close()
    return ports


def _filler(title: str, size: int) -> str:
    line = f"- {title}: a note kept for the gate's prompt-size fixture.\n"
    body = f"# {title}\n\n" + line * (size // len(line) + 1)
    return body[:size]


def _set(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def ledger_module(name: str) -> str:
    head = f'# FILE ledger/{name}.py\n"""ledger.{name}: part of a small double-entry ledger."""\n\n'
    lines = [head]
    if name == "billing":
        lines.append("def compute_total(entries: list[dict]) -> int:\n    return sum(int(e.get('amount', 0)) for e in entries)\n\n")
    i = 0
    while sum(len(x) for x in lines) < LEDGER_MODULE_CHARS:
        i += 1
        lines.append(
            f"def {name}_step_{i}(entries: list[dict], *, scale: int = {i}) -> int:\n"
            "    total = 0\n    for entry in entries:\n"
            "        amount = int(entry.get('amount', 0)) * scale\n"
            f"        total += amount if entry.get('kind') == '{name}' else -amount\n"
            "    return total\n\n"
        )
    return "".join(lines)


@dataclass
class GateDaemon:
    root: Path
    api: int
    ws: int
    token: str
    proc: subprocess.Popen
    log: Path
    workspace: Path
    home: Path
    #: The operator's own files as the daemon read them (bootstrap files and
    #: the deploy dir's PROMETHEUS.md): what the overhead check does NOT
    #: count as Prometheus's.
    operator_texts: list[str] = field(default_factory=list)

    def log_tail(self, n: int = 60) -> str:
        try:
            return "\n".join(self.log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return "(no log)"

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=10)

    # -- the client ----------------------------------------------------------

    def _h(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def get(self, path: str) -> Any:
        r = httpx.get(f"http://127.0.0.1:{self.api}{path}", headers=self._h(), timeout=30)
        r.raise_for_status()
        return r.json()

    def bind_workspace(self, sid: str) -> None:
        httpx.put(f"http://127.0.0.1:{self.api}/api/sessions/{sid}/workspace", headers=self._h(),
                  json={"path": str(self.workspace)}, timeout=30).raise_for_status()

    def use_model(self, sid: str, key: str) -> dict:
        r = httpx.post(f"http://127.0.0.1:{self.api}/api/sessions/{sid}/model", headers=self._h(),
                       json={"key": key}, timeout=30)
        r.raise_for_status()
        return r.json()

    def turn(self, message: str, *, sid: str | None = None, timeout: float = 240.0) -> "Turn":
        sid = sid or f"beacon:{uuid.uuid4()}"
        frames = asyncio.run(self._drive(sid, message, timeout))
        stored = self.get(f"/api/sessions/{sid}/messages")
        return Turn(sid=sid, frames=frames, stored=stored.get("messages", stored) if isinstance(stored, dict) else stored)

    async def _drive(self, sid: str, message: str, timeout: float) -> list[dict]:
        import websockets

        frames: list[dict] = []
        t0 = time.time()
        async with websockets.connect(f"ws://127.0.0.1:{self.ws}", max_size=None) as ws:
            await ws.send(json.dumps({"type": "auth", "token": self.token}))
            await ws.send(json.dumps({"type": "subscribe", "payload": {"sessions": [sid]}}))
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"http://127.0.0.1:{self.api}/api/chat/send", headers=self._h(),
                                      json={"session_id": sid, "message": message})
                r.raise_for_status()
            done = False
            while time.time() - t0 < timeout:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    if done:
                        break
                    continue
                fr = json.loads(raw)
                if (fr.get("payload") or {}).get("session_id") not in (None, sid):
                    continue
                frames.append(fr)
                if fr.get("type") == "command_done":
                    break  # a slash command answers with chat_message + command_done, no chat_done
                if fr.get("type") in ("chat_done", "error"):
                    done = True
                if done and fr.get("type") == "agent_state" and (fr.get("payload") or {}).get("state") == "idle":
                    break
        return frames


@dataclass
class Turn:
    """What the client received for one turn, and what history now holds."""

    sid: str
    frames: list[dict]
    stored: list[dict] = field(default_factory=list)

    def of(self, kind: str) -> list[dict]:
        return [f.get("payload") or {} for f in self.frames if f.get("type") == kind]

    @property
    def streamed(self) -> str:
        return "".join(p.get("content", "") for p in self.of("chat_delta"))

    @property
    def tools_started(self) -> list[str]:
        return [p.get("tool_name", "") for p in self.of("tool_call_start")]

    def selects(self) -> int:
        return sum(1 for p in self.of("tool_call_start")
                   if p.get("tool_name") == "tool_search" and (p.get("inputs") or {}).get("action") == "select")

    @property
    def finished(self) -> bool:
        return bool(self.of("chat_done"))

    @property
    def final_stored_text(self) -> str:
        for m in reversed(self.stored):
            if m.get("role") != "assistant":
                continue
            c = m.get("content")
            if isinstance(c, list):
                return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            return str(c or m.get("text") or "")
        return ""


#: The cloud route a gate session can switch to (POST /api/sessions/{id}/model),
#: pointed at a cloud stand-in through the operator-facing slash_commands entry.
CLOUD_KEY = "qwen"


def build_and_boot(tmp: Path, model_url: str, *, cloud_url: str | None = None,
                   extra: dict[str, Any] | None = None) -> GateDaemon:
    checkout, python = checkout_under_test()
    home = tmp / "home"
    pdir = home / ".prometheus"
    work = tmp / "workspace"
    for d in (pdir / "files", home / ".config", work / "src" / "ledger"):
        d.mkdir(parents=True, exist_ok=True)
    for name, size in OPERATOR_FILE_SIZES.items():
        (pdir / name).write_text(_filler(name.removesuffix(".md"), size), encoding="utf-8")
    for i in range(SAVED_FILES):
        (pdir / "files" / f"note-{i:02d}.md").write_text("x" * (900 + 37 * i), encoding="utf-8")
    from tests.support.scripted_llama import LEDGER_MODULES

    for m in LEDGER_MODULES:
        (work / "src" / "ledger" / f"{m}.py").write_text(ledger_module(m), encoding="utf-8")
    wiki = pdir / "wiki"
    (wiki / "entities").mkdir(parents=True, exist_ok=True)
    (wiki / "index.md").write_text("# Wiki index\n\n- [[deploy-checklist]]: steps before a restart\n", encoding="utf-8")
    (wiki / "entities" / "deploy-checklist.md").write_text(
        "# Deploy checklist\n\n1. Prepare the release.\n2. Arm the window.\n3. Restart and verify.\n", encoding="utf-8")
    (tmp / "mcp_tools.json").write_text(json.dumps(MCP_TOOLS), encoding="utf-8")

    cfg = yaml.safe_load((checkout / "config" / "prometheus.yaml.default").read_text(encoding="utf-8"))
    api, ws, api_tls, ws_tls, dash = _free_ports(5)
    token = "".join(random.choice(string.ascii_letters) for _ in range(40))
    cfg["web"] = {"enabled": True, "api_port": api, "ws_port": ws, "tls_api_port": api_tls, "tls_ws_port": ws_tls,
                  "bind": "127.0.0.1", "trusted_proxies": [], "api_token": token, "dashboard_port": dash}
    settings: dict[str, Any] = {
        "gateway.telegram_enabled": False, "gateway.cron_enabled": False,
        "sentinel.enabled": False, "pairing.requests_enabled": False, "discovery.mdns": False,
        "push.enabled": False, "tracing.enabled": False, "anatomy.scan_on_startup": False,
        "learning.auto_skill_creation": False, "learning.curator_enabled": False,
        "training.capture_enabled": False, "doctor.startup_check": False,
        "infrastructure.archive_enabled": False,
        "security.workspace_root": [str(work)], "security.bash_write_allow": [str(work)],
        # Local services on this box are reachable over loopback, which the
        # dead proxy below does not cover: the shipped fallback (ollama on
        # :11434) and image backend (ComfyUI on :8188) point at real servers,
        # so they are pointed at a dead port instead.
        "model.fallback.enabled": False, "model.fallback_url": "http://127.0.0.1:9",
        "image_generation.default_backend": "comfyui", "image_generation.comfyui.base_url": "http://127.0.0.1:9",
        "evals.judge_base_url": "http://127.0.0.1:9",
        # The MCP tool is the deferred one gate (b) reaches for.
        "tools.deferred_loading.mcp_always_deferred": True,
        "mcp_servers": {"docs": {"command": python, "args": [str(STUB_MCP), str(tmp / "mcp_tools.json")]}},
    }
    # A release before multi-root workspaces (#176) takes one root as a string.
    if isinstance((cfg.get("security") or {}).get("workspace_root"), str):
        settings["security.workspace_root"] = str(work)
    settings.update({"model.provider": "llama_cpp", "model.base_url": model_url, "model.model": ""})
    if cloud_url:
        settings[f"slash_commands.{CLOUD_KEY}"] = {
            "provider": "qwen", "model": "qwen3.8-max", "api_key_env": "GATE_CLOUD_KEY", "base_url": cloud_url + "/v1"}
    settings.update(extra or {})
    for k, v in settings.items():
        _set(cfg, k, v)
    cfg_path = tmp / "prometheus.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home), "LANG": "C.UTF-8",
           "PROMETHEUS_CONFIG_DIR": str(pdir), "XDG_CONFIG_HOME": str(home / ".config"),
           "PYTHONPATH": str(checkout / "src"), "PYTHONSAFEPATH": "1", "PYTHONNOUSERSITE": "1",
           "GATE_CLOUD_KEY": "gate" + "fake" + "cloudkey",
           # Nothing leaves the machine: every proxy-aware HTTP client goes to a
           # dead local port, while the stand-ins on loopback stay reachable.
           "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "ALL_PROXY": "http://127.0.0.1:9",
           "NO_PROXY": "127.0.0.1,localhost"}
    # The daemon's working directory is a deploy-style copy holding the
    # checkout's PROMETHEUS.md once, as the live daemon runs from its deploy
    # clone. Running it from a checkout nested under another one would stack
    # the project instructions twice (the walk goes up to /).
    deploy = tmp / "deploy"
    deploy.mkdir(exist_ok=True)
    if (checkout / "PROMETHEUS.md").is_file():
        (deploy / "PROMETHEUS.md").write_text((checkout / "PROMETHEUS.md").read_text(encoding="utf-8"), encoding="utf-8")
    log = tmp / "daemon.log"
    proc = subprocess.Popen([python, str(checkout / "scripts" / "daemon.py"), "--config", str(cfg_path)],
                            cwd=str(deploy), env=env, stdout=open(log, "ab"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    d = GateDaemon(root=checkout, api=api, ws=ws, token=token, proc=proc, log=log, workspace=work, home=home,
                   operator_texts=[p.read_text(encoding="utf-8") for p in
                                   [*(pdir / n for n in OPERATOR_FILE_SIZES), deploy / "PROMETHEUS.md"] if p.is_file()])
    t0 = time.time()
    while time.time() - t0 < 180:
        if proc.poll() is not None:
            raise RuntimeError(f"daemon exited during boot ({proc.returncode}):\n{d.log_tail()}")
        try:
            if httpx.get(f"http://127.0.0.1:{api}/api/status", headers=d._h(), timeout=5).status_code == 200:
                return d
        except httpx.HTTPError:
            pass
        time.sleep(1)
    d.stop()
    raise RuntimeError(f"daemon did not come up:\n{d.log_tail()}")
