"""Build an isolated scratch daemon setup from an operator config.

Keeps every setting that shapes the prompt and the tool catalog (tools,
context, compaction, bootstrap, anatomy, adapter, model, profiles), so the
scratch daemon sends the model what the operator's daemon sends, and isolates
the rest:

* web on loopback spare ports with a fresh letters-only token;
* no gateways, cron, sentinel, mDNS, pairing, push, tracing, symbiote or
  training capture (tool telemetry stays on, written inside ``out``);
* the model reached only through the logging proxy, with no fallback and no
  other backend;
* workspace, HOME and config dir inside ``out``;
* every string value under a secret-looking key removed;
* MCP servers replaced by the local stub (same tool schemas, no network).

Prompt files (SOUL, AGENTS, ANATOMY, MEMORY, USER, skills/) are copied from
``--source-home`` when given; saved files are replicated as empty files of
the same name, size and mtime (the prompt lists names and sizes only).

Nothing is printed except the paths written. The output config is written
mode 600 and contains the model's address; keep it out of git.

Usage:
  python isolate.py --source-config CFG --out DIR --api-port N --ws-port N
         --proxy-port N --python PY [--source-home DIR] [--mcp-tools JSON]
         [--set dotted.key=yaml_value ...]
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
import shutil
import string
from pathlib import Path
from urllib.parse import urlparse

import yaml

_SECRET_KEY = re.compile(r"(^|_)(token|key|secret|password|apikey|api_key|credentials?)$", re.I)
_PROMPT_FILES = ("SOUL.md", "AGENTS.md", "ANATOMY.md", "MEMORY.md", "USER.md")


def _strip_secrets(node, path=()):
    if isinstance(node, dict):
        for k in list(node):
            v = node[k]
            if isinstance(v, str) and _SECRET_KEY.search(str(k)) and path + (k,) != ("web", "api_token"):
                del node[k]
            else:
                _strip_secrets(v, path + (k,))
    elif isinstance(node, list):
        for item in node:
            _strip_secrets(item, path)


def _set(cfg: dict, dotted: str, value) -> None:
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


# A small, plausible package for the multi-file coding scenario. Each module
# is padded to about 8.8k characters (~2.35k tokens): with the operator's
# ~18.6k overhead, a fresh window of three results crosses the 26,214 line,
# while the largest result stays under the 2,458 at which the projected rule
# binds with a 4096 reply room (same sizing as tests/support/gate_daemon.py).
_MODULES = ("accounts", "audit", "billing", "currency", "invoices", "payments", "reports", "taxes", "ledger_io")


def _module_source(name: str) -> str:
    lines = [f'"""ledger.{name}: part of a small double-entry ledger."""', "", "from __future__ import annotations", ""]
    i = 0
    while sum(len(x) + 1 for x in lines) < 8800:
        i += 1
        fn = "compute_total" if (name == "billing" and i == 1) else f"{name}_step_{i}"
        lines += [
            f"def {fn}(entries: list[dict], *, scale: int = {i}) -> int:",
            "    total = 0",
            "    for entry in entries:",
            "        amount = int(entry.get('amount', 0)) * scale",
            f"        if entry.get('kind') == '{name}':",
            "            total += amount",
            "        elif entry.get('kind') == 'reversal':",
            "            total -= amount",
            "    return total",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-config", required=True)
    ap.add_argument("--source-home")
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-port", type=int, required=True)
    ap.add_argument("--ws-port", type=int, required=True)
    ap.add_argument("--proxy-port", type=int, required=True)
    ap.add_argument("--python", required=True, help="interpreter that runs the stub MCP server")
    ap.add_argument("--mcp-tools", help="JSON list of MCP tool schemas served as the 'context7' stub")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    out = Path(args.out).resolve()
    home = out / "home"
    pdir = home / ".prometheus"
    work = out / "workspace"
    for d in (pdir, home / ".config", work / "src" / "ledger"):
        d.mkdir(parents=True, exist_ok=True)

    with open(args.source_config, encoding="utf-8") as fh:
        cfg = copy.deepcopy(yaml.safe_load(fh) or {})

    token = "".join(random.choice(string.ascii_letters) for _ in range(40))
    upstream_path = urlparse(cfg["model"]["base_url"]).path
    proxy = f"http://127.0.0.1:{args.proxy_port}{upstream_path}"

    cfg["web"] = {
        "enabled": True, "api_port": args.api_port, "ws_port": args.ws_port,
        "tls_api_port": args.api_port + 1, "tls_ws_port": args.ws_port + 1,
        "bind": "127.0.0.1", "trusted_proxies": [], "api_token": token,
        "dashboard_port": args.api_port + 7000,
    }
    gw = cfg.setdefault("gateway", {})
    for k in ("telegram_enabled", "cron_enabled", "slack_enabled", "discord_enabled"):
        gw[k] = False
    for k in ("allowed_chat_ids", "briefing_chat_id", "proxy_url"):
        gw.pop(k, None)
    off = {
        "sentinel.enabled": False, "pairing.requests_enabled": False, "discovery.mdns": False,
        "network.home_network": False, "network.allow_plaintext_lan": False, "push.enabled": False,
        "tracing.enabled": False, "symbiote.enabled": False, "anatomy.scan_on_startup": False,
        "infrastructure.archive_enabled": False,
        "learning.auto_skill_creation": False, "learning.skill_refinement_enabled": False,
        "learning.gepa_enabled": False, "trajectory_export.enabled": False,
        "training.capture_enabled": False, "training.cloud_golden_capture": False,
        "printing_press.auto_update_library": False, "doctor.startup_check": False,
        "heartbeat.maintenance_db": "",
    }
    for k, v in off.items():
        _set(cfg, k, v)
    (cfg.get("anatomy") or {}).pop("ssh_key", None)
    (cfg.get("evals") or {}).pop("judge_base_url", None)
    sec = cfg.setdefault("security", {})
    sec["workspace_root"] = [str(work)]
    sec["bash_write_allow"] = [str(work)]
    sec.pop("grants", None)
    sec.pop("allowed_commands", None)
    cfg["model"]["base_url"] = proxy
    cfg["model"].pop("fallback_url", None)
    cfg["model"].pop("fallback_provider", None)
    cfg["router"] = {"fallback": []}
    cfg["backends"] = {"4090": {"provider": "llama_cpp", "base_url": proxy}}
    for k in ("image_generation", "whisper"):
        cfg.pop(k, None)

    cfg["mcp_servers"] = {}
    if args.mcp_tools:
        stub = Path(__file__).resolve().parent / "stub_mcp.py"
        tools_copy = out / "mcp_tools.json"
        shutil.copyfile(args.mcp_tools, tools_copy)
        cfg["mcp_servers"]["context7"] = {"command": args.python, "args": [str(stub), str(tools_copy)]}

    _strip_secrets(cfg)
    cfg["web"]["api_token"] = token
    for item in args.set:
        key, _, raw = item.partition("=")
        _set(cfg, key, yaml.safe_load(raw))

    if args.source_home:
        src = Path(args.source_home)
        for name in _PROMPT_FILES:
            if (src / name).is_file():
                shutil.copy2(src / name, pdir / name)
        if (src / "skills").is_dir():
            shutil.copytree(src / "skills", pdir / "skills", dirs_exist_ok=True)
        if (src / "files").is_dir():
            (pdir / "files").mkdir(exist_ok=True)
            for f in (src / "files").iterdir():
                if f.is_file():
                    st = f.stat()
                    dst = pdir / "files" / f.name
                    with open(dst, "wb") as fh:
                        fh.truncate(st.st_size)
                    os.utime(dst, (st.st_atime, st.st_mtime))

    for name in _MODULES:
        (work / "src" / "ledger" / f"{name}.py").write_text(_module_source(name), encoding="utf-8")
    (work / "src" / "ledger" / "__init__.py").write_text('"""A small double-entry ledger."""\n', encoding="utf-8")
    wiki = pdir / "wiki"
    (wiki / "entities").mkdir(parents=True, exist_ok=True)
    (wiki / "index.md").write_text(
        "# Wiki index\n\n- [[deploy-checklist]]: the steps before any daemon restart\n", encoding="utf-8")
    (wiki / "entities" / "deploy-checklist.md").write_text(
        "# Deploy checklist\n\n1. Prepare the release on a branch.\n2. Arm the maintenance window.\n"
        "3. Restart, then verify the running version and the model choices.\n", encoding="utf-8")

    cfg_path = out / "prometheus.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    os.chmod(cfg_path, 0o600)
    (out / "token").write_text(token, encoding="utf-8")
    os.chmod(out / "token", 0o600)
    print(cfg_path)


if __name__ == "__main__":
    main()
