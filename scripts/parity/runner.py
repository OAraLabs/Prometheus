"""Run one scenario against one isolated daemon, in record or replay mode.

The runner is the only piece that knows the daemon's API. It drives each step
the way a client does (REST to act, the WebSocket to learn when a turn has
ended), then, once the daemon has exited, reads back everything it persisted.

Timing is captured on the monotonic clock at four kinds of instant, all
outside the daemon: the harness's POST, the model server's receipt of each
completion request and its last flushed byte, and the WebSocket frame that
ends the turn. Tool execution time comes from the daemon's own
``tool_calls.latency_ms``, which brackets ``tool.execute`` and nothing else.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from parity.instance import API_TOKEN, HarnessError, Instance, http
from parity.model_server import COMPLETIONS_PATHS, Exchange, ModelServer, ServerState
from parity.observe import snapshot, tree
from parity.scenarios import Evidence, Scenario
from parity.wsclient import EventListener, WSClient

# The harness's deviations from the shipped template. Each one is a thing the
# replay does NOT exercise, so each is here for a stated reason and the PR
# lists them. Scenario overrides (compaction thresholds, coding.enabled) are
# applied on top of these, per scenario.
HARNESS_OVERRIDES: dict[str, Any] = {
    "model": {
        "provider": "llama_cpp",
        "base_url": "{{MODEL_URL}}",
        "model": "",
        # The template points the legacy fallback at a real Ollama on
        # localhost. Aim it at the model server, so any fallback the daemon
        # takes is recorded (or flagged) instead of reaching a real model.
        "fallback_url": "{{ALT_URL}}",
    },
    "web": {"enabled": True, "api_port": "{{API_PORT}}", "ws_port": "{{WS_PORT}}",
            "api_token": "{{API_TOKEN}}"},
    # The second backend, for the model switch. Ollama speaks the same
    # /v1/chat/completions wire, so one model server stands in for both.
    "backends": {"alt": {"provider": "ollama", "base_url": "{{ALT_URL}}",
                         "model": "qwen2.5:7b-instruct"}},
    # Machine identity (GPU, hosts, addresses) scanned into the system prompt:
    # it differs per host by design and could carry real identifiers.
    "anatomy": {"enabled": False, "scan_on_startup": False, "include_in_system_prompt": False},
    # `git fetch` in the SOURCE checkout every 300 s — not the daemon's to do here.
    "deployment": {"origin_fetch": {"enabled": False}},
    # Boot-time Docker cleanup is HOST-GLOBAL: it would remove the live
    # daemon's stale coding containers.
    "coding": {"docker_cleanup_enabled": False},
    # First run 120 s after a fresh boot: a timer, so whether it lands inside
    # a scenario depends on how long the scenario takes.
    "learning": {"curator_enabled": False},
    # The bwrap write floor is available on some hosts and not others (not on
    # a stock CI runner). Its own CI job proves it; here it would make a bash
    # result differ by host.
    "security": {"bash_write_confinement": "off"},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def build_config(src_root: Path, scenario: Scenario) -> str:
    """The scenario's full config text, with {{PLACEHOLDERS}} for ports."""
    base = yaml.safe_load((src_root / "config" / "prometheus.yaml.default").read_text())
    cfg = _merge(_merge(base, HARNESS_OVERRIDES), scenario.config)
    return yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)


def render_config(text: str, subs: dict[str, str]) -> str:
    for key, val in subs.items():
        text = text.replace(f"'{{{{{key}}}}}'", val).replace(f"{{{{{key}}}}}", val)
    return text


@dataclass
class TurnTiming:
    session: str
    t_post: int
    t_done: int
    tools_ms: list[tuple[float, float]] = field(default_factory=list)  # (wall ts, latency_ms)


@dataclass
class RunOutput:
    scenario: str
    mode: str
    steps: list[dict]
    stores: dict[str, Any]
    exchanges: list[Exchange]
    served: list[Any]
    unconsumed: list[int]
    turns: list[TurnTiming]
    wall_offset_ns: int          # time.time_ns() - time.monotonic_ns() at start
    rss: list[dict]
    boot_seconds: float | None
    shutdown: str
    daemon_log: Path
    errors: list[str] = field(default_factory=list)          # the harness could not run
    step_failures: list[str] = field(default_factory=list)   # the daemon misbehaved (diffed)

    def evidence(self) -> Evidence:
        if self.mode == "record":
            reqs = [ex.request for ex in self.exchanges if ex.method == "POST"
                    and ex.path in COMPLETIONS_PATHS]
            labels = [ex.upstream for ex in self.exchanges if ex.method == "POST"
                      and ex.path in COMPLETIONS_PATHS]
        else:
            reqs = [s.request for s in self.served]
            labels = [s.upstream for s in self.served]
        return Evidence(stores=self.stores, steps=self.steps, requests=reqs,
                        upstream_labels=labels)


class Runner:
    def __init__(self, scenario: Scenario, *, mode: str, root: Path, src_root: Path,
                 config_text: str, upstreams: dict[str, str] | None = None,
                 recorded: list[Exchange] | None = None,
                 turn_timeout: float = 900.0) -> None:
        self.scenario = scenario
        self.mode = mode
        self.root = root
        self.src_root = src_root
        self.config_text = config_text
        self.state = ServerState(mode=mode, upstreams=upstreams or {},
                                 recorded=list(recorded or []))
        self.server = ModelServer(self.state)
        self.turn_timeout = turn_timeout
        self.steps: list[dict] = []
        self.step_failures: list[str] = []
        self.turns: list[TurnTiming] = []
        self.rss: list[dict] = []

    # -- setup ------------------------------------------------------------
    def _materialize_files(self) -> None:
        for where, files in self.scenario.files.items():
            base = self.root / where
            for rel, text in files.items():
                p = base / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
        for where in self.scenario.git_repos:
            repo = self.root / where
            env = {"HOME": str(self.root / "home"), "PATH": "/usr/bin:/bin",
                   "GIT_AUTHOR_NAME": "parity", "GIT_AUTHOR_EMAIL": "parity@example.invalid",
                   "GIT_COMMITTER_NAME": "parity", "GIT_COMMITTER_EMAIL": "parity@example.invalid",
                   "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
                   "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
            for cmd in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"],
                        ["git", "commit", "-q", "-m", "parity fixture"]):
                subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True)

    # -- steps ------------------------------------------------------------
    def _tool_rows(self, inst: Instance, session: str) -> list[tuple[float, float]]:
        import sqlite3
        db = inst.cfg_dir / "telemetry.db"
        if not db.exists():
            return []
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT timestamp, latency_ms FROM tool_calls WHERE session_id = ? "
                "AND latency_ms IS NOT NULL ORDER BY timestamp", (session,)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            con.close()
        return [(float(t), float(ms)) for t, ms in rows]

    def _chat(self, inst: Instance, ws: EventListener, step: dict) -> dict:
        session = step["session"]
        code, before = http("GET", f"{inst.base}/api/sessions/{session}/messages")
        seen = {m["message_id"] for m in (before or {}).get("messages", [])} if code == 200 else set()
        tools_before = len(self._tool_rows(inst, session))
        mark = ws.mark()
        t_post = time.monotonic_ns()
        payload = {"session_id": session, "message": step["message"]}
        if step.get("mode"):
            payload["mode"] = step["mode"]
        code, body = http("POST", f"{inst.base}/api/chat/send", payload)
        if code != 200:
            return {"op": "chat", "session": session, "error": f"send -> {code}: {body}"}
        hit = ws.wait_for(
            lambda ev: ev.get("type") in ("chat_done", "error")
            and (ev.get("payload") or {}).get("session_id") == session,
            timeout=self.turn_timeout, start=mark)
        if hit is None:
            return {"op": "chat", "session": session,
                    "error": f"no chat_done within {self.turn_timeout}s ({ws.error or 'timeout'})"}
        t_done, ev = hit
        timing = TurnTiming(session=session, t_post=t_post, t_done=t_done)
        timing.tools_ms = self._tool_rows(inst, session)[tools_before:]
        self.turns.append(timing)
        code, after = http("GET", f"{inst.base}/api/sessions/{session}/messages")
        new = [m for m in (after or {}).get("messages", []) if m["message_id"] not in seen]
        reply = next((m.get("content") for m in reversed(new) if m.get("role") == "assistant"), None)
        return {
            "op": "chat", "session": session, "end": ev.get("type"),
            "error_event": (ev.get("payload") if ev.get("type") == "error" else None),
            "reply": reply,
            "messages": [{k: m.get(k) for k in ("role", "content", "content_json", "provenance",
                                                "is_trusted")} for m in new],
        }

    def _step(self, inst: Instance, ws: EventListener, step: dict) -> dict:
        op = step["op"]
        if op == "chat":
            return self._chat(inst, ws, step)
        if op == "workspace":
            path = str(self.root / "ws" / step["ws"])
            code, body = http("PUT", f"{inst.base}/api/sessions/{step['session']}/workspace",
                              {"path": path})
            return {"op": op, "status": code, "result": body}
        if op == "tree":
            return {"op": op, "label": step.get("label"), "tree": tree(self.root / "ws" / step["ws"])}
        if op == "checkpoints":
            code, body = http("GET", f"{inst.base}/api/sessions/{step['session']}/checkpoints")
            return {"op": op, "status": code, "result": body}
        if op == "restore_latest":
            code, body = http("GET", f"{inst.base}/api/sessions/{step['session']}/checkpoints")
            cps = (body or {}).get("checkpoints") or [] if isinstance(body, dict) else []
            if not cps:
                return {"op": op, "error": f"no checkpoint to restore ({code})"}
            cid = cps[0]["id"]
            code, res = http("POST", f"{inst.base}/api/sessions/{step['session']}/checkpoints/"
                                     f"{cid}/restore", {"confirm": cid, "dry_run": False})
            return {"op": op, "status": code, "result": res}
        if op == "model":
            code, body = http("POST", f"{inst.base}/api/sessions/{step['session']}/model",
                              {"key": step["key"]}, timeout=60)
            return {"op": op, "status": code, "result": body}
        if op == "code":
            repo = str(self.root / "ws" / step["ws"])
            code, body = http("POST", f"{inst.base}/api/code", {
                "repo": repo, "description": step["description"],
                "acceptance_command": step["acceptance"], "task_id": step["task_id"],
                "max_rounds": step.get("max_rounds", 12),
                "max_wall_seconds": step.get("max_wall_seconds", 900)}, timeout=60)
            if code != 200:
                return {"op": op, "error": f"POST /api/code -> {code}: {body}"}
            task = body["task_id"]
            deadline = time.monotonic() + self.turn_timeout
            res: Any = None
            while time.monotonic() < deadline:
                c, res = http("GET", f"{inst.base}/api/code/{task}")
                if c == 200 and res.get("status") in ("completed", "failed", "killed", "blocked"):
                    break
                time.sleep(0.5)
            else:
                return {"op": op, "error": f"coding run {task} did not finish"}
            return {"op": op, "result": {k: res.get(k) for k in
                                         ("status", "return_code", "error", "report")}}
        raise HarnessError(f"unknown step op {op!r}")

    # -- quiescence -------------------------------------------------------
    @staticmethod
    def _quiet_disk(root: Path, quiet_for: float, limit: float) -> None:
        """Wait until nothing under ``root`` has changed for ``quiet_for`` s.

        The writes that follow the last model response (the session-title row,
        a final telemetry row) land milliseconds later on a fast host and
        later still on a loaded CI runner; a fixed sleep would be a race."""
        def newest() -> int:
            latest = 0
            for p in root.rglob("*"):
                try:
                    latest = max(latest, p.stat().st_mtime_ns)
                except OSError:
                    pass
            return latest
        deadline = time.monotonic() + limit
        last, since = newest(), time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.1)
            now = newest()
            if now != last:
                last, since = now, time.monotonic()
            elif time.monotonic() - since >= quiet_for:
                return

    def _settle(self) -> None:
        """Let background model calls (session titles) finish before the stop."""
        if self.mode == "replay":
            want = sum(1 for ex in self.state.recorded
                       if ex.method == "POST" and ex.path in COMPLETIONS_PATHS)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                with self.state.lock:
                    done = sum(1 for i in self.state.consumed
                               if self.state.recorded[i].path in COMPLETIONS_PATHS)
                    busy = self.state.in_flight
                if done >= want and busy == 0:
                    break
                time.sleep(0.05)
            self._quiet_disk(self.root / "home", quiet_for=1.0, limit=10.0)
        else:
            quiet_since = time.monotonic()
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                with self.state.lock:
                    busy = self.state.in_flight
                if busy:
                    quiet_since = time.monotonic()
                elif time.monotonic() - quiet_since > 5:
                    break
                time.sleep(0.1)

    # -- the run ----------------------------------------------------------
    def run(self) -> RunOutput:
        wall_offset = time.time_ns() - time.monotonic_ns()
        ports = self.server.start(["primary", "alt"])
        subs = {"MODEL_URL": f"http://127.0.0.1:{ports['primary']}",
                "ALT_URL": f"http://127.0.0.1:{ports['alt']}",
                "API_TOKEN": API_TOKEN}
        inst = Instance(self.root, self.src_root, render_config(self.config_text, subs))
        errors: list[str] = []
        shutdown = "never started"
        ws: EventListener | None = None
        prepared = False
        try:
            inst.prepare()
            prepared = True
            self._materialize_files()
            inst.start()
            self.rss.append({"at": "boot", **inst.rss_kb()})
            client = WSClient("127.0.0.1", inst.ws_port)
            client.send_json({"type": "auth", "token": API_TOKEN})
            ws = EventListener(client).start()
            if ws.wait_for(lambda ev: ev.get("type") == "connected", timeout=15) is None:
                raise HarnessError(f"WebSocket auth never answered 'connected' ({ws.error})")
            for step in self.scenario.steps:
                result = self._step(inst, ws, step)
                # A step that fails because of what the DAEMON did (no
                # checkpoint to restore, a turn that never finished) is
                # behaviour: it stays in the step result and is DIFFED. Only
                # the harness failing to drive the daemon at all is an error.
                if "error" in result:
                    self.step_failures.append(f"step {step['op']}: {result['error']}")
                self.steps.append(result)
                self.rss.append({"at": f"after {step['op']}", **inst.rss_kb()})
            self._settle()
            self.rss.append({"at": "end", **inst.rss_kb()})
        except HarnessError as exc:
            errors.append(str(exc))
        finally:
            if ws is not None:
                ws.client.close()
            shutdown = inst.stop()
            self.server.stop()
        # Never read a root this run did not prepare (e.g. one prepare refused).
        stores = snapshot(self.root) if prepared and self.root.exists() else {}
        served = list(self.state.served)
        unconsumed = [i for i, ex in enumerate(self.state.recorded)
                      if ex.method == "POST" and ex.path in COMPLETIONS_PATHS
                      and i not in self.state.consumed] if self.mode == "replay" else []
        return RunOutput(
            scenario=self.scenario.name, mode=self.mode, steps=self.steps, stores=stores,
            exchanges=list(self.state.recorded),
            served=served, unconsumed=unconsumed, turns=self.turns,
            wall_offset_ns=wall_offset, rss=self.rss, boot_seconds=inst.boot_seconds,
            shutdown=shutdown, daemon_log=inst.log_path, errors=errors,
            step_failures=self.step_failures)


def clean_root(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
