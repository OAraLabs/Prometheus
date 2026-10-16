"""Real-model regression gate for local tool calling and context (scratch only).

Starts the logging proxy and a scratch daemon from a given checkout, drives
each scenario the way Beacon does (REST ``/api/chat/send`` plus the WebSocket
stream), stops both, then grades the gate items from what the wire, the
frames, the stored transcript and the scratch telemetry show:

  a  sub-agent spawn: ``agent`` is called in the turn and the reply streams
  b  a deferred tool is found and called in the same turn, with at most one
     ``tool_search`` select (cron, wiki, memory search, an MCP tool)
  c  a long multi-file session on the 32k window: no compaction pass while the
     request is under the pre-0.9.12 line, and the turn finishes the task after
     a pass
  d  no silent turn: the stored final text reached Beacon as streamed text,
     and every forced ending says why
  e  fixed overhead, per component, in the server's own tokens
(f, cloud unchanged, is covered by the parity suite and unit tests; it needs
no model.)

Never point this at a live daemon: it starts its own, from ``--checkout``,
with the config ``isolate.py`` wrote. The model server is shared, so each
scenario waits for ``--idle-cmd`` (exit 0 = the shared server is free) and
the suite is kept short.

Usage:
  python run_gate.py --checkout DIR --work DIR --label NAME --api-port N
         --ws-port N --proxy-port N --scenarios a,b_cron,...
         [--idle-cmd CMD] [--results OUT.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import websockets

HERE = Path(__file__).resolve().parent

# The pre-0.9.12 compaction line on a 32,768 window: the 80% threshold, which
# binds while the output room is 4096 and the largest recent result is under
# 2,458 tokens (compactor.py projected rule). A pass below it is "early".
LINE_32K_PRE_0912 = 26_214

SCENARIOS: dict[str, dict] = {
    "a_agent": {
        "gate": "a",
        "message": "Use a sub-agent to list the files in your workspace, then tell me what it found.",
        "want_tool": "agent",
    },
    "b_cron": {
        "gate": "b",
        "message": "Every weekday at 9am, remind me to check the build. Set that up now as a scheduled job.",
        "want_tool": "cron_create",
    },
    "b_wiki": {
        "gate": "b",
        "message": "Look up the deploy checklist in my Prometheus wiki and tell me its steps.",
        "want_tool": "wiki_query",
    },
    "b_memory": {
        "gate": "b",
        "message": "Search our past conversations for anything we said about the release checklist, and tell me what you find.",
        "want_tool": "lcm_grep",
    },
    "b_mcp": {
        "gate": "b",
        "message": "Use the Context7 documentation tools to look up how FastAPI dependency injection works, then summarize it in two sentences.",
        "want_tool_prefix": "mcp__context7__",
    },
    "c_coding": {
        "gate": "c",
        # Not bound to a workspace: a plain Beacon chat, so the daemon's own
        # project instructions stay in the prompt as they do live. The paths
        # are spelled out instead.
        "message": (
            "In the project at {workspace}, read every module in src/ledger/ (there are nine files, read each "
            "one), then add a one-line docstring to the function compute_total in src/ledger/billing.py that says "
            "what the ledger package does overall. Tell me what you changed."
        ),
        "edited_file": "src/ledger/billing.py",
    },
    # The release tier of the reachability suite: plain requests that need a
    # deferred tool, none naming it. Graded found / called / correct.
    "r_cron": {"gate": "r", "message": "Every weekday at 9am, remind me to check the build.",
               "correct": {"cron_create"}},
    "r_agent": {"gate": "r", "message": "Get a helper to look through my workspace and summarize what's there.",
                "correct": {"agent", "sessions_spawn"}},
    "r_history": {"gate": "r", "message": "What did we talk about earlier regarding the release checklist? "
                  "Look back through our past conversations.",
                  "correct": {"lcm_grep", "lcm_expand_query", "lcm_describe", "lcm_expand"}},
    "r_wiki": {"gate": "r", "message": "Check my Prometheus wiki for the deploy checklist.",
               "correct": {"wiki_query"}},
    "r_tasks": {"gate": "r", "message": "What background tasks do I have running right now?",
                "correct": {"task_list", "task_get"}},
    "r_sessions": {"gate": "r", "message": "Show me my recent chat sessions.", "correct": {"sessions_list"}},
    "r_todo": {"gate": "r", "message": "Keep a to-do list for this conversation: buy milk, then call the bank.",
               "correct": {"todo_write"}},
    # Fix 5b (Ollama on its native /api/chat), on an Ollama-served primary:
    # every request names the window it was budgeted for (options.num_ctx), and
    # no prompt fills it (Ollama would cut it from the front, silently).
    "o_tool": {"gate": "o", "message": "Run the shell command `echo ollama-ok` and tell me exactly what it printed.",
               "want_tool": "bash"},
    "o_plain": {"gate": "o", "message": "Say hello in one short sentence."},
    "o_window": {"gate": "o", "message": "Summarise this log in one line:\n" + ("2026-10-16 build step ok; " * 9000)},
    "d_iteration_cap": {
        "gate": "d",
        "message": "Run the shell command `echo one`, then `echo two`, then `echo three`, as three separate tool calls one at a time, then tell me all three outputs.",
        "expect_forced_stop": True,
    },
}


# ---------------------------------------------------------------- processes


def _start(cmd: list[str], log: Path, cwd: Path, env: dict) -> subprocess.Popen:
    fh = open(log, "ab")
    return subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)


def _stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def _wait_up(api: int, token: str, proc: subprocess.Popen, timeout: float = 240.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"daemon exited during boot (code {proc.returncode}); see its log")
        try:
            r = httpx.get(f"http://127.0.0.1:{api}/api/status", headers={"Authorization": f"Bearer {token}"}, timeout=5)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise SystemExit("daemon did not come up")


def _wait_idle(cmd: str | None) -> None:
    if not cmd:
        return
    for _ in range(60):
        if subprocess.run(cmd, shell=True).returncode == 0:
            return
        time.sleep(10)
    raise SystemExit("the shared model server stayed busy; stopping the suite")


# ---------------------------------------------------------------- one turn


async def _drive(api: int, ws_port: int, token: str, sid: str, message: str, timeout: float) -> dict:
    frames: list[dict] = []
    deltas: list[str] = []
    t0 = time.time()
    async with websockets.connect(f"ws://127.0.0.1:{ws_port}", max_size=None) as ws:
        await ws.send(json.dumps({"type": "auth", "token": token}))
        await ws.send(json.dumps({"type": "subscribe", "payload": {"sessions": [sid]}}))
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"http://127.0.0.1:{api}/api/chat/send",
                headers={"Authorization": f"Bearer {token}"},
                json={"session_id": sid, "message": message, "client_msg_id": str(uuid.uuid4())},
            )
            r.raise_for_status()
        done_at = None
        while time.time() - t0 < timeout:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                if done_at is not None:
                    break
                continue
            fr = json.loads(raw)
            p = fr.get("payload") or {}
            if p.get("session_id") not in (None, sid):
                continue
            fr["_t"] = round(time.time() - t0, 2)
            frames.append(fr)
            typ = fr.get("type")
            if typ == "chat_delta":
                deltas.append(p.get("content", ""))
            if typ in ("chat_done", "error") and done_at is None:
                done_at = time.time()
            if done_at is not None and typ == "agent_state" and p.get("state") == "idle":
                break
    return {"frames": frames, "streamed": "".join(deltas), "elapsed_s": round(time.time() - t0, 1),
            "timed_out": done_at is None}


def _stored_messages(api: int, token: str, sid: str) -> list[dict]:
    r = httpx.get(f"http://127.0.0.1:{api}/api/sessions/{sid}/messages",
                  headers={"Authorization": f"Bearer {token}"}, timeout=30)
    r.raise_for_status()
    body = r.json()
    return body.get("messages", body) if isinstance(body, dict) else body


# ---------------------------------------------------------------- grading


def _captures(path: Path, t_from: float, t_to: float) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if t_from <= rec.get("t", 0) <= t_to:
            out.append(rec)
    return out


def _telemetry(db: Path, sid: str, t_from: float) -> dict:
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        turns = con.execute(
            "SELECT rounds, forced_stop_reason, terminal_kind, outcome, total_prompt_tokens FROM turns"
            " WHERE session_id=? ORDER BY started_at DESC LIMIT 1", (sid,)).fetchall()
        comp = con.execute(
            "SELECT COUNT(*) FROM subsystem_runs WHERE subsystem='context_compactor' AND timestamp>=?",
            (t_from,)).fetchone()[0]
        lucky = con.execute(
            "SELECT COUNT(*) FROM subsystem_runs WHERE operation='lucky_guess' AND timestamp>=?",
            (t_from,)).fetchone()[0]
    finally:
        con.close()
    t = turns[0] if turns else (None,) * 5
    return {"rounds": t[0], "forced_stop": t[1], "terminal_kind": t[2], "outcome": t[3],
            "prompt_tokens_total": t[4], "summarize_spans": comp, "lucky_guess_markers": lucky}


_COMPACT_RE = re.compile(r"ContextCompactor: .*tokens (\d+) → (\d+)(?: \(ceiling (\d+))?")


def _compactor_lines(log: Path, start_offset: int) -> list[dict]:
    if not log.exists():
        return []
    with open(log, "rb") as fh:
        fh.seek(start_offset)
        text = fh.read().decode("utf-8", "replace")
    out = []
    for line in text.splitlines():
        m = _COMPACT_RE.search(line)
        if m:
            out.append({"before": int(m.group(1)), "after": int(m.group(2)),
                        "ceiling": int(m.group(3)) if m.group(3) else None})
    return out


def _has_docstring(path: Path, func: str) -> bool:
    """True when ``def func(...)``'s first body line opens a docstring."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith(f"def {func}("):
            body = next((x for x in lines[i + 1:] if x.strip()), "")
            return body.strip().startswith(('"""', "'''", '"', "'"))
    return False


def _last_assistant_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") != "assistant":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        return str(m.get("text") or "")
    return ""


def _grade(name: str, spec: dict, run: dict, caps: list[dict], tel: dict, comp: list[dict], work: Path) -> dict:
    frames = run["frames"]
    starts = [f["payload"] for f in frames if f.get("type") == "tool_call_start"]
    called = [s.get("tool_name") or s.get("name") for s in starts]
    selects = sum(1 for s in starts if (s.get("tool_name") == "tool_search"
                                        and (s.get("inputs") or {}).get("action") == "select"))
    # A turn round carries the system prompt; the compactor's summary calls are
    # a lone user message (compactor.py summarize path), so they are counted apart.
    rounds = [c for c in caps if any(m.get("role") == "system" for m in c.get("request", {}).get("messages", []))]
    summary_calls = len(caps) - len(rounds)
    offered = sorted({t["function"]["name"] for c in rounds[:1] for t in c["request"].get("tools") or []})
    prompt_tokens = [(c["response"].get("usage") or {}).get("prompt_tokens") for c in rounds]
    final_text = run.get("final_text", "")
    streamed = run["streamed"]
    silent = bool(final_text.strip()) and final_text.strip() not in streamed
    done = [f["payload"] for f in frames if f.get("type") == "chat_done"]
    res = {
        "scenario": name, "gate": spec["gate"], "elapsed_s": run["elapsed_s"], "timed_out": run["timed_out"],
        "tools_called": called, "selects": selects, "rounds": len(rounds), "summary_calls": summary_calls,
        "max_tokens": rounds[0]["request"].get("max_tokens") if rounds else None,
        "offered_round0": offered, "prompt_tokens_per_round": prompt_tokens,
        "compaction_passes": comp, "telemetry": tel,
        "final_text_streamed": not silent, "final_text_head": final_text[:240],
        "chat_done_fields": sorted({k for d in done for k in d}),
        "error_frames": [f["payload"].get("message") for f in frames if f.get("type") == "error"],
    }
    checks: dict[str, bool] = {}
    if spec["gate"] == "o":
        native = [c for c in caps if "options" in c.get("request", {})]
        sent = [((c["request"].get("options") or {}).get("num_ctx"),
                 ((c["response"] or {}).get("usage") or {}).get("prompt_tokens")) for c in native]
        res["ollama"] = {"requests": len(native), "num_ctx_and_prompt": sent,
                         "think_sent": [c["request"].get("think") for c in native if "think" in c["request"]]}
        checks["native_route"] = bool(native)
        checks["num_ctx_sent"] = bool(native) and all(n for n, _ in sent)
        checks["never_filled"] = all(p is None or (n and p < n) for n, p in sent)
        checks["reply_streamed"] = bool(run["streamed"].strip())
    if spec["gate"] == "r":
        correct = spec["correct"]
        searched = " ".join(p.get("result") or "" for p in (f.get("payload") or {} for f in frames
                            if f.get("type") == "tool_call_end") if p.get("tool_name") == "tool_search")
        offered_now = set(offered)
        res["reach"] = {
            "found": bool(correct & offered_now) or any(f'"{t}"' in searched for t in correct),
            "called": any(c in correct for c in called),
            "correct": bool(called) and next((c for c in called if c not in ("tool_search",)), None) in correct,
            "first_non_search_call": next((c for c in called if c != "tool_search"), None),
        }
        checks.update({k: v for k, v in res["reach"].items() if k in ("found", "called", "correct")})
    want = spec.get("want_tool")
    prefix = spec.get("want_tool_prefix")
    if want or prefix:
        hit = any((c == want) if want else (c or "").startswith(prefix) for c in called)
        checks["tool_called"] = hit
        checks["at_most_one_select"] = selects <= 1
    # A pass is "early" when it fired under the pre-0.9.12 line, or when the
    # line it aimed at was pulled under it by the reply room (ceiling below
    # 26,214: the projected rule binding). The second catches a session whose
    # request jumps past the line in one round, as a model reading two files
    # per round does.
    res["early_compaction_passes"] = [
        p for p in comp
        if p["before"] < LINE_32K_PRE_0912 or (p["ceiling"] is not None and p["ceiling"] < LINE_32K_PRE_0912)]
    if spec["gate"] == "c":
        checks["no_early_compaction"] = not res["early_compaction_passes"]
        checks["task_finished"] = _has_docstring(work / spec["edited_file"], "compute_total")
        checks["no_forced_stop"] = not tel.get("forced_stop")
        if comp:
            # The turn went on after its first pass: at least one model round
            # came after it and the turn ended on the model's own answer.
            checks["continued_after_compaction"] = len(rounds) > 1 and not tel.get("forced_stop")
    if spec.get("expect_forced_stop"):
        checks["forced_stop_happened"] = bool(tel.get("forced_stop"))
    checks["final_text_reached_beacon"] = not silent
    checks["no_timeout"] = not run["timed_out"]
    res["checks"] = checks
    res["pass"] = all(checks.values())
    return res


# ---------------------------------------------------------------- overhead


def _overhead(proxy_port: int, cap: dict) -> dict:
    """Split round 0's request into components and count each in the server's tokens."""
    req = cap["request"]
    tok = lambda s: len(httpx.post(f"http://127.0.0.1:{proxy_port}/tokenize", json={"content": s}, timeout=60).json()["tokens"])  # noqa: E731
    system = next((m["content"] for m in req["messages"] if m["role"] == "system"), "")
    parts, cur, title = [], [], "(start)"
    for line in system.split("\n"):
        heading = line.startswith("# ") or line.startswith("<!-- Bootstrap:") or line.startswith("You have access to these tools:")
        if heading:
            if cur:
                parts.append((title, "\n".join(cur)))
            title, cur = line.strip()[:60], [line]
        else:
            cur.append(line)
    if cur:
        parts.append((title, "\n".join(cur)))
    rows = sorted(((t, tok(txt)) for t, txt in parts), key=lambda r: -r[1])
    tools = req.get("tools") or []
    per_tool = sorted(((t["function"]["name"], tok(json.dumps(t["function"], ensure_ascii=False))) for t in tools),
                      key=lambda r: -r[1])
    users = sum(tok(m["content"]) for m in req["messages"] if m["role"] == "user" and isinstance(m.get("content"), str))
    total = (cap["response"].get("usage") or {}).get("prompt_tokens")
    sys_tokens = tok(system)
    tools_tokens = tok("\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools))
    return {"server_prompt_tokens": total, "system_prompt_tokens": sys_tokens, "system_sections": rows,
            "tool_schemas_tokens": tools_tokens, "per_tool": per_tool, "user_tokens": users,
            "template_remainder": (total or 0) - sys_tokens - tools_tokens - users}


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--upstream-config", required=True,
                    help="a config whose model.base_url is the real model server (read by the proxy, never printed)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--api-port", type=int, required=True)
    ap.add_argument("--ws-port", type=int, required=True)
    ap.add_argument("--proxy-port", type=int, required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--idle-cmd")
    ap.add_argument("--results")
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--overhead", action="store_true", help="measure round 0 of the first scenario per component")
    args = ap.parse_args()

    checkout, work = Path(args.checkout).resolve(), Path(args.work).resolve()
    token = (work / "token").read_text(encoding="utf-8").strip()
    capture = work / f"capture-{args.label}.jsonl"
    dlog = work / f"daemon-{args.label}.log"
    home = work / "home"
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home), "LANG": "C.UTF-8",
        "PROMETHEUS_CONFIG_DIR": str(home / ".prometheus"), "XDG_CONFIG_HOME": str(home / ".config"),
        "PYTHONPATH": str(checkout / "src"), "PYTHONSAFEPATH": "1", "PYTHONNOUSERSITE": "1",
    }
    proxy = daemon = None
    results: list[dict] = []
    try:
        proxy = _start([sys.executable, str(HERE / "proxy.py"), str(args.proxy_port), str(capture),
                        str(Path(args.upstream_config).expanduser())], work / f"proxy-{args.label}.log", work,
                       dict(os.environ))
        time.sleep(2)
        daemon = _start([str(checkout / ".venv" / "bin" / "python"), "scripts/daemon.py", "--config",
                         str(work / "prometheus.yaml")], dlog, checkout, env)
        _wait_up(args.api_port, token, daemon)
        for name in args.scenarios.split(","):
            spec = SCENARIOS[name]
            _wait_idle(args.idle_cmd)
            sid = f"beacon:{uuid.uuid4()}"
            if spec.get("bind_workspace"):
                httpx.put(f"http://127.0.0.1:{args.api_port}/api/sessions/{sid}/workspace",
                          headers={"Authorization": f"Bearer {token}"},
                          json={"path": str(work / "workspace")}, timeout=30).raise_for_status()
            t_from = time.time()
            log_off = dlog.stat().st_size
            message = spec["message"].replace("{workspace}", str(work / "workspace"))
            run = asyncio.run(_drive(args.api_port, args.ws_port, token, sid, message, args.timeout))
            if run["timed_out"]:
                # Stop a thrashing turn rather than let it hold the shared slot.
                httpx.post(f"http://127.0.0.1:{args.api_port}/api/chat/interrupt",
                           headers={"Authorization": f"Bearer {token}"}, json={"session_id": sid}, timeout=30)
                time.sleep(5)
            time.sleep(3)
            run["final_text"] = _last_assistant_text(_stored_messages(args.api_port, token, sid))
            caps = _captures(capture, t_from, time.time())
            res = _grade(name, spec, run, caps, _telemetry(home / ".prometheus" / "telemetry.db", sid, t_from),
                         _compactor_lines(dlog, log_off), work / "workspace")
            res["session_id"] = sid
            if args.overhead and not results and caps:
                res["overhead"] = _overhead(args.proxy_port, caps[0])
            results.append(res)
            print(json.dumps({k: res[k] for k in ("scenario", "pass", "checks", "tools_called", "elapsed_s")}))
            with open(work / f"frames-{args.label}-{name}.jsonl", "w", encoding="utf-8") as fh:
                for fr in run["frames"]:
                    fh.write(json.dumps(fr) + "\n")
    finally:
        _stop(daemon)
        _stop(proxy)
    if args.results:
        Path(args.results).write_text(json.dumps({"label": args.label, "results": results}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
