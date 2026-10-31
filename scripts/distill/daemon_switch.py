#!/usr/bin/env python3
"""Move the Prometheus daemon's primary model to the Qwen3.8 Max API for a GPU window, and back.

    python3 daemon_switch.py status
    python3 daemon_switch.py capture             # dry run (the default): what would be recorded
    python3 daemon_switch.py capture --execute
    python3 daemon_switch.py to-max --execute
    python3 daemon_switch.py restore --execute
    python3 daemon_switch.py selftest            # can this interpreter run the script at all

Runs ON the daemon's host. ``setups.py`` runs it there as a ``pre_apply`` /
``post_apply`` hook (shipped over SSH on stdin, or from the dead-man's copy),
and it can be run by hand. Every command except ``status`` and ``selftest``
is a dry run unless ``--execute`` is given. It needs only the stdlib and
PyYAML, so the host's system python runs it.

What decides the daemon's primary model, and why this edits two files:

* ``model:`` in the daemon's config file (provider, base_url, model,
  api_key_env). The provider object is built once, at boot.
* ``config_pins.yaml`` in the daemon's config dir. At every boot the daemon
  lays each pinned dotted key over the config IN MEMORY, silently, logging a
  CONFIG DRIFT line. A pin on ``model.provider`` / ``model.base_url`` would
  put the old values back on top of an edited config, so the pins are edited
  in step, and the edit is checked by simulating that boot before anything
  is written.
* ``PUT /api/model`` is not used: it only saves to the config file (applied
  on restart), never touches the pins (so the pin reverts the provider at
  the next boot), refuses a ``base_url``, may call the vendor to check the
  key, and also rewrites ``context.effective_limit``.

``to-max`` (optionally with ``--set section.key=value`` for keys that already
exist, e.g. to pause an idle-time job that would call Max with nobody asking)
sets ``model.provider: qwen``, ``model.model: qwen3.8-max``,
``model.api_key_env: QWEN_API_KEY`` and blanks ``model.base_url`` (and a
foreign ``base_url_env``), with the matching pins. With ``base_url`` blank
the daemon reads the endpoint from ``QWEN_BASE_URL`` in its env file at
boot, so no account-identifying URL is written to a yaml file. This script
never reads or prints the key: it checks only that the variable names are
set (and asks the daemon whether it holds a key).

The Max key is for interactive use only. ``to-max`` refuses (dry run too)
while anything would call the switched primary with nobody asking:
``BACKGROUND_CALLERS`` lists each such caller with the config switch that
turns it off. The memory extractor, and cron jobs that build the primary from
this config, have no such switch in this daemon, so until they do every
``to-max`` refuses.

The daemon is restarted through its own ``POST /api/restart`` (a graceful
shutdown, then a re-exec of the same command line in the SAME process under
the same unit: systemd never sees it stop). A restart is unavoidable because
the provider is built at boot. This script never runs ``systemctl stop`` or
``restart``; the one systemctl call it makes is a ``start`` of a daemon that
is already down. Verification uses status endpoints only (``/health``,
``GET /api/model``, ``GET /api/status``, and the local model server's
``/health``, ``/props``, ``/v1/models``): nothing is ever sent to Max.

Files (0600, in 0700 directories), on the host::

    ~/.local/share/distill-setups/before/<UTC stamp>/
        prometheus.yaml  config_pins.yaml  manifest.json
        switched.json    (written by to-max before it changes anything)
        restored.json    (written when restore has verified)
    ~/.local/share/distill-setups/before/ACTIVE   (the capture a restore uses)

Someone else's edit is never written over. The daemon itself rewrites its
config at any time (a persistent permission grant or its revocation), so:
every write is a compare-and-swap (the file is read again just before the
rename and must still be the bytes this run checked); ``to-max`` looks at the
files again after waiting for an idle daemon; and ``restore`` (and a
rollback) takes only to-max's own lines back out of a file someone else
edited during the window, keeping that edit, after checking that the result
reads back as exactly that.

Refusals (exit 1; nothing more is changed): no capture to restore from while
the config names Max; a file changed since the capture (to-max); a key to-max
changed that someone else changed again during the window, or an edit next to
to-max's lines (restore: merge by hand); a saved change waiting for a restart
that is not to-max's own; ``PROMETHEUS_PROVIDER`` / ``PROMETHEUS_MODEL``
overriding the file; a background caller that would use Max; the local model
server not healthy, or serving a different model, at an executing restore
(the daemon stays on Max rather than pointing at the wrong server; a dry run
only notes it); a to-max that does not verify (rolled back first).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import difflib
import errno
import fcntl
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml

VERSION = 1

DEFAULT_CONFIG = "~/prometheus-deploy/config/prometheus.yaml"
DEFAULT_PINS = "~/.prometheus/config_pins.yaml"
DEFAULT_ENV_FILE = "~/.config/prometheus/env"
DEFAULT_STATE_DIR = "~/.local/share/distill-setups/before"
ACTIVE_FILE = "ACTIVE"
LOCK_NAME = ".daemon-switch.lock"
DEFAULT_API_PORT = 8005
DAEMON_UNIT = "prometheus.service"

MAX_PROVIDER = "qwen"
MAX_MODEL = "qwen3.8-max"
MAX_KEY_ENV = "QWEN_API_KEY"
MAX_URL_ENV = "QWEN_BASE_URL"
TOKEN_ENV = "PROMETHEUS_API_TOKEN"
TOKEN_FILE_ENV = "PROMETHEUS_API_TOKEN_FILE"
PIN_ENVS = ("PROMETHEUS_PROVIDER", "PROMETHEUS_MODEL")
ENV_NAMES = (MAX_KEY_ENV, MAX_URL_ENV, TOKEN_ENV, TOKEN_FILE_ENV) + PIN_ENVS
MODEL_OVERRIDDEN = "model_overridden"

CONFIG_COPY = "prometheus.yaml"
PINS_COPY = "config_pins.yaml"
MANIFEST = "manifest.json"
SWITCHED = "switched.json"
RESTORED = "restored.json"

POLL_S = 2.0
IDLE_POLL_S = 10.0
MIN_UPTIME_S = 5.0

# What calls the PRIMARY model with nobody asking: inside the daemon, or started with its config. During a
# window the primary is the Max Token Plan key, which is for interactive use only, so to-max refuses while
# any of these would run on the switched config (pins applied). Each is (what, gates); it is off when any
# gate (section, key, value when the key is absent) is false. No gates: the daemon cannot turn it off.
BACKGROUND_CALLERS: tuple[tuple[str, tuple[tuple[str, str, Any], ...]], ...] = (
    ("the memory extractor (runs on the primary every 30 min; the daemon has no switch for it)", ()),
    ("cron jobs that build the primary from this config, e.g. the daily briefing (no switch)", ()),
    ("the curator (learning.curator_enabled)", (("learning", "curator_enabled", True),)),
    ("sentinel knowledge synthesis (sentinel.enabled, sentinel.synthesis_enabled)",
     (("sentinel", "enabled", False), ("sentinel", "synthesis_enabled", True))),
    ("GEPA (learning.gepa_enabled)", (("learning", "gepa_enabled", False),)),
)

_sleep = time.sleep
_now = time.monotonic


class Refusal(Exception):
    """A precondition does not hold. Nothing (more) is changed; the command exits 1."""


class Changed(Refusal):
    """A file changed between this run's check and its write. Nothing was written over it."""


def log(msg: str) -> None:
    print(msg, flush=True)


# ── small file helpers ──────────────────────────────────────────────────────


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def file_record(path: Path, data: bytes | None) -> dict[str, Any]:
    if data is None:
        return {"path": str(path), "absent": True}
    st = path.stat()
    return {"path": str(path), "sha256": sha256(data), "size": len(data),
            "mode": stat.S_IMODE(st.st_mode), "mtime_ns": st.st_mtime_ns}


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def write_private(path: Path, data: bytes) -> None:
    """``data`` to ``path`` as 0600 (also when it replaces a looser file)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def replace_file(path: Path, data: bytes, mode: int, mtime_ns: int | None = None, *,
                 expect_sha: str) -> None:
    """Atomically make ``path`` hold ``data`` with ``mode`` (temp file + rename in the same directory).

    A compare-and-swap: the live file is read again just before the rename and must still hash to
    ``expect_sha`` (the bytes the caller checked), else Changed and the file is left alone. The daemon
    rewrites its config at any time; what is left is the gap between that read and the rename.
    """
    real = path.resolve()
    tmp = real.with_name(f".{real.name}.daemon-switch.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            os.write(fd, data)
            os.fsync(fd)
            os.fchmod(fd, mode)
        finally:
            os.close(fd)
        if mtime_ns is not None:
            os.utime(tmp, ns=(mtime_ns, mtime_ns))
        live = read_bytes(real)
        if live is None or sha256(live) != expect_sha:
            raise Changed(f"{path} changed after this run checked it; not writing over it")
        os.replace(tmp, real)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def dump_json(data: Any) -> bytes:
    return (json.dumps(data, indent=1, sort_keys=True, default=str) + "\n").encode()


# ── the env file: names only ────────────────────────────────────────────────


def _env_lines(path: Path) -> Iterator[tuple[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, _, value = line.partition("=")
        yield name.strip(), value.strip()


def env_names(path: Path) -> dict[str, dict[str, bool]]:
    """``{NAME: {"set": bool}}`` for ENV_NAMES: whether the env file gives each one a value.

    Values are looked at only to tell empty from non-empty and are never kept.
    """
    out = {n: {"set": False} for n in ENV_NAMES}
    for name, value in _env_lines(path):
        if name in out:
            out[name]["set"] = value.strip("\"'") != ""
    return out


def env_value(path: Path, name: str) -> str:
    """One variable's value from the env file (used only for the API token)."""
    found = ""
    for n, value in _env_lines(path):
        if n == name:
            found = value
    if len(found) >= 2 and found[0] == found[-1] and found[0] in "\"'":
        found = found[1:-1]
    return found


# ── YAML text edits that keep every other byte ──────────────────────────────

_MISSING = object()


def yaml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    text = str(value)
    if text == "":
        return "''"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:/@+-]*", text) and yaml.safe_load(text) == text:
        return text
    return json.dumps(text)


def _split_value(rest: str) -> tuple[str, str, str]:
    """``(value, gap, comment)`` of what follows ``key:`` and its separator."""
    end = None
    if rest[:1] in ("'", '"'):
        q, i = rest[0], 1
        while i < len(rest):
            if q == '"' and rest[i] == "\\":
                i += 2
                continue
            if rest[i] == q:
                if q == "'" and rest[i + 1:i + 2] == "'":
                    i += 2
                    continue
                end = i + 1
                break
            i += 1
        if end is None:
            raise ValueError("unterminated quoted value")
    m = re.search(r"[ \t]#", rest[end or 0:])
    if m is None:
        value = rest.rstrip(" \t")
        return value, rest[len(value):], ""
    cut = (end or 0) + m.start() + 1
    value = rest[:cut].rstrip(" \t")
    return value, rest[len(value):cut], rest[cut:]


def _content(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith("#")


def _indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _key_rows(lines: list[str], lo: int, hi: int, indent: str, key: str) -> list[int]:
    pat = re.compile(r"^" + re.escape(indent) + re.escape(key) + r":(?:[ \t]|$|\r?\n)")
    return [i for i in range(lo, hi) if _indent(lines[i]) == indent and pat.match(lines[i])]


def _block(lines: list[str], top: int) -> tuple[int, int, str]:
    """``(first, end, child indent)`` of the block mapping under top-level line ``top``."""
    end = top + 1
    while end < len(lines) and not (_content(lines[end]) and not lines[end][:1].isspace()):
        end += 1
    child = None
    for i in range(top + 1, end):
        if _content(lines[i]):
            child = _indent(lines[i])
            break
    if not child:
        raise ValueError(f"{lines[top].split(':')[0]}: is not a block mapping with keys")
    return top + 1, end, child


def set_scalar(text: str, path: tuple[str, ...], value: Any, *,
               insert_before: str | None = None) -> str:
    """``text`` with one scalar at ``path`` (one or two keys) set, every other line byte-identical.

    The key's line keeps its trailing comment and the spacing before it. A
    missing second-level key is inserted just above ``insert_before``'s line
    (or as the block's first key). Anything ambiguous (a duplicate key, a flow
    mapping, a key whose value is a nested block) raises ValueError.
    """
    lines = text.splitlines(keepends=True)
    if len(path) == 1:
        lo, hi, indent, key = 0, len(lines), "", path[0]
    elif len(path) == 2:
        tops = _key_rows(lines, 0, len(lines), "", path[0])
        if len(tops) != 1:
            raise ValueError(f"{path[0]}: found {len(tops)} times at the top level")
        head = lines[tops[0]].split("#", 1)[0].split(":", 1)[1].strip()
        if head:
            raise ValueError(f"{path[0]}: is not a block mapping")
        lo, hi, indent = _block(lines, tops[0])
        key = path[1]
    else:
        raise ValueError("only one or two levels are supported")
    rows = _key_rows(lines, lo, hi, indent, key)
    if len(rows) > 1:
        raise ValueError(f"{'.'.join(path)}: found {len(rows)} times")
    literal = yaml_literal(value)
    if not rows:
        if len(path) == 1:
            raise ValueError(f"{key}: not found")
        at = lo
        if insert_before is not None:
            anchor = _key_rows(lines, lo, hi, indent, insert_before)
            if len(anchor) == 1:
                at = anchor[0]
        nl = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
        lines.insert(at, f"{indent}{key}: {literal}{nl}")
        return "".join(lines)
    i = rows[0]
    line = lines[i]
    nl = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
    body = line[: len(line) - len(nl)]
    m = re.match(r"^(" + re.escape(indent) + re.escape(key) + r":)([ \t]*)(.*)$", body)
    if m is None:
        raise ValueError(f"{'.'.join(path)}: cannot parse its line")
    prefix, sep, rest = m.groups()
    old, gap, comment = _split_value(rest)
    if not old:
        raise ValueError(f"{'.'.join(path)}: has no scalar on its line (a nested block?)")
    if old[:1] in ("{", "[", "|", ">", "&", "*", "!"):
        raise ValueError(f"{'.'.join(path)}: is not a plain scalar")
    lines[i] = f"{prefix}{sep or ' '}{literal}{gap if comment else ''}{comment}{nl}"
    return "".join(lines)


def apply_edits(text: str, edits: list[tuple[tuple[str, ...], Any]],
                inserts_before: dict[str, str] | None = None) -> tuple[str, dict[str, Any]]:
    """The edited text and the document it must parse to; raises ValueError when it does not."""
    doc = yaml.safe_load(text) or {}
    if not isinstance(doc, dict):
        raise ValueError("top level is not a mapping")
    expected = copy.deepcopy(doc)
    out = text
    for path, value in edits:
        out = set_scalar(out, path, value, insert_before=(inserts_before or {}).get(path[-1]))
        node = expected
        for k in path[:-1]:
            node = node.setdefault(k, {})
        node[path[-1]] = value
    if (yaml.safe_load(out) or {}) != expected:
        raise ValueError("the edited text does not read back as the intended change")
    return out, expected


# ── what the daemon will run: a copy of its boot-time pin logic ─────────────


def simulate_boot(config: dict[str, Any], pins: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """``(model section in force, keys a pin corrected)``, as ``daemon.apply_config_pins`` does it."""
    cfg, drifted = boot_config(config, pins)
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return model, drifted


def background_on(config: dict[str, Any]) -> list[str]:
    """The BACKGROUND_CALLERS that would run on ``config`` (the config in force, pins applied)."""
    out = []
    for what, gates in BACKGROUND_CALLERS:
        on = True
        for section, key, absent in gates:
            sec = config.get(section)
            if not (sec.get(key, absent) if isinstance(sec, dict) else absent):
                on = False
        if on:
            out.append(what)
    return out


def boot_config(config: dict[str, Any], pins: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """``(config in force, keys a pin corrected)``, as ``daemon.apply_config_pins`` does it."""
    cfg = copy.deepcopy(config)
    drifted: list[str] = []
    for dotpath, expected in (pins or {}).items():
        if expected is None:
            continue
        parts = str(dotpath).split(".")
        val: Any = cfg
        for p in parts:
            val = val.get(p, _MISSING) if isinstance(val, dict) else _MISSING
            if val is _MISSING:
                break
        if val is _MISSING or str(val) != str(expected):
            obj = cfg
            for p in parts[:-1]:
                obj = obj.setdefault(p, {})
            obj[parts[-1]] = expected
            drifted.append(str(dotpath))
    return cfg, drifted


def parse_set(item: str) -> tuple[tuple[str, str], Any]:
    """``section.key=value`` (a YAML scalar) for ``--set``."""
    key, sep, raw = item.partition("=")
    parts = tuple(key.strip().split("."))
    if not sep or len(parts) != 2 or not all(re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in parts):
        raise Refusal(f"--set {item!r}: use section.key=value")
    if parts[0] == "model":
        raise Refusal(f"--set {item!r}: the model section is the switch's own")
    try:
        value = yaml.safe_load(raw) if raw.strip() else ""
    except yaml.YAMLError:
        raise Refusal(f"--set {item!r}: the value is not a YAML scalar") from None
    if not (value is None or isinstance(value, (bool, int, float, str))):
        raise Refusal(f"--set {item!r}: the value is not a scalar")
    return (parts[0], parts[1]), value


def max_plan(cfg_text: str, pins_text: str | None, sets: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    """The to-max edit of both files, verified, from their before-texts.

    ``sets`` (``section.key=value``, off by default) also changes keys that
    already exist in the config for the window, e.g. to pause a background
    job that would otherwise call the primary (now Max) with nobody asking.
    A pin on such a key is edited in step. Restore puts the bytes back.
    """
    doc = yaml.safe_load(cfg_text) or {}
    model = doc.get("model") if isinstance(doc, dict) else None
    if not isinstance(model, dict):
        raise Refusal("the config has no model: mapping")
    if str(model.get("api_key") or "").strip():
        raise Refusal("model.api_key is set inline in the config; it would outrank QWEN_API_KEY. Not switching")
    want = {"provider": MAX_PROVIDER, "model": MAX_MODEL, "api_key_env": MAX_KEY_ENV}
    edits: list[tuple[tuple[str, ...], Any]] = [
        (("model", k), v) for k, v in want.items() if model.get(k, _MISSING) != v]
    if model.get("base_url") not in (None, ""):
        edits.append((("model", "base_url"), ""))
    if model.get("base_url_env") not in (None, "", MAX_URL_ENV):
        edits.append((("model", "base_url_env"), ""))
    set_paths: set[str] = set()
    for item in sets:
        path, value = parse_set(item)
        section = doc.get(path[0])
        if not isinstance(section, dict) or path[1] not in section or isinstance(section[path[1]], (dict, list)):
            raise Refusal(f"--set {item!r}: {'.'.join(path)} is not an existing scalar in the config")
        # A pin on the key is edited in step even when the config already holds the value.
        set_paths.add(".".join(path))
        if section[path[1]] != value:
            edits.append((path, value))
    try:
        new_cfg, new_doc = apply_edits(cfg_text, edits, {"api_key_env": "provider"})
    except ValueError as e:
        raise Refusal(f"cannot edit the config safely: {e}") from None
    new_model = new_doc["model"]
    new_pins_text, new_pins, pin_edits = None, None, []
    if pins_text is not None:
        pins = yaml.safe_load(pins_text) or {}
        if not isinstance(pins, dict):
            raise Refusal("the pins file is not a mapping")
        for key, val in pins.items():
            if val is None:
                continue
            sub = str(key)[len("model."):] if str(key).startswith("model.") else None
            if sub in ("provider", "model", "api_key_env", "base_url", "base_url_env", "api_key") or (
                    str(key) in set_paths):
                if sub is not None:
                    target = new_model.get(sub, "")
                else:
                    a, b = str(key).split(".", 1)
                    target = new_doc[a][b]
                target = "" if target is None else target
                if str(val) != str(target):
                    pin_edits.append(((str(key),), target))
        try:
            new_pins_text, new_pins = apply_edits(pins_text, pin_edits)
        except ValueError as e:
            raise Refusal(f"cannot edit the pins safely: {e}") from None
    booted, drift = boot_config(new_doc, new_pins)
    in_force = booted.get("model") if isinstance(booted.get("model"), dict) else {}
    # A pin already correcting some other key is the operator's business: it stays as it is.
    # Only the keys this edit touches must come out of the boot exactly as written.
    ours = {".".join(p) for p, _ in edits} | {"model." + k for k in (
        "provider", "model", "api_key_env", "base_url", "base_url_env", "api_key")}
    pre_drift = set(simulate_boot(doc, yaml.safe_load(pins_text) or {} if pins_text is not None else None)[1])
    drift_ours = [d for d in drift if d in ours or d not in pre_drift]
    bad = []
    if in_force.get("provider") != MAX_PROVIDER or in_force.get("model") != MAX_MODEL:
        bad.append(f"provider/model would be {in_force.get('provider')}/{in_force.get('model')}")
    if in_force.get("base_url") or in_force.get("api_key"):
        bad.append("a base_url or inline api_key would still be in force")
    if in_force.get("base_url_env") not in (None, "", MAX_URL_ENV):
        bad.append("a foreign base_url_env would be in force")
    if drift_ours:
        bad.append(f"the pins would still correct {', '.join(drift_ours)} at boot")
    if bad:
        raise Refusal("the edit would not boot on Max: " + "; ".join(bad))
    return {"config": new_cfg, "pins": new_pins_text, "pins_doc": new_pins, "drift": sorted(drift),
            "config_edits": [".".join(p) for p, _ in edits], "pin_edits": [p[0] for p, _ in pin_edits],
            "config_paths": [tuple(p) for p, _ in edits], "pin_paths": [tuple(p) for p, _ in pin_edits],
            "background": background_on(booted)}


_SECRET_KEY = re.compile(r"(?i)(key|token|secret|passw|cred)(?!_env)")


def shown(line: str) -> str:
    """A config line fit for a log: a secret-looking key's value is blanked, comments dropped."""
    body = line.split(" #", 1)[0].strip()
    key, sep, _ = body.partition(":")
    return f"{key}: <redacted>" if sep and _SECRET_KEY.search(key) else body


def safe_pins(pins: Any) -> Any:
    if not isinstance(pins, dict):
        return pins
    return {k: ("<redacted>" if _SECRET_KEY.search(str(k)) else v) for k, v in pins.items()}


def changed_lines(old: str, new: str) -> list[str]:
    """The edit as ``~ L<n>: old -> new`` / ``+ L<n>: new`` lines (secret-looking values blanked)."""
    a, b = old.splitlines(), new.splitlines()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if op == "replace" and i2 - i1 == j2 - j1:
            out += [f"  ~ L{j1 + k + 1}: {shown(a[i1 + k])}  ->  {shown(b[j1 + k])}" for k in range(i2 - i1)]
            continue
        out += [f"  - L{i + 1}: {shown(a[i])}" for i in range(i1, i2)]
        out += [f"  + L{j + 1}: {shown(b[j])}" for j in range(j1, j2)]
    return out


# ── the daemon's API and the local model server (status endpoints only) ─────

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http(method: str, url: str, token: str | None = None, timeout: float = 10.0) -> tuple[int, Any]:
    """``(status, parsed JSON or None)``; status 0 when the server could not be reached."""
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    req.add_header("Accept", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            status, body = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, body = e.code, e.read() or b""
    except (urllib.error.URLError, OSError, ValueError):
        return 0, None
    try:
        return status, json.loads(body.decode("utf-8") or "null")
    except (ValueError, UnicodeDecodeError):
        return status, None


@dataclass
class Ctx:
    config: Path
    pins: Path
    env_file: Path
    state_dir: Path
    api: str | None
    execute: bool = False
    offline: bool = False
    force: bool = False
    idle_wait: float = 300.0
    boot_wait: float = 240.0
    backend_wait: float = 300.0
    notify_cmd: str | None = None
    sets: tuple[str, ...] = ()
    _token: str | None = None

    def api_base(self, doc: dict[str, Any] | None = None) -> str:
        if self.api:
            return self.api.rstrip("/")
        web = (doc or {}).get("web") if isinstance(doc, dict) else None
        port = web.get("api_port") if isinstance(web, dict) else None
        return f"http://127.0.0.1:{int(port or DEFAULT_API_PORT)}"

    def token(self, doc: dict[str, Any] | None) -> str:
        """The daemon's bearer token, as the daemon resolves it. Held in memory only, never printed."""
        if self._token is None:
            web = (doc or {}).get("web") if isinstance(doc, dict) else None
            tok = web.get("api_token") if isinstance(web, dict) else None
            if not (isinstance(tok, str) and tok):
                tok = ""
                named = os.environ.get(TOKEN_FILE_ENV) or env_value(self.env_file, TOKEN_FILE_ENV)
                if named:
                    with contextlib.suppress(OSError):
                        tok = Path(named).expanduser().read_text().strip()
                tok = tok or os.environ.get(TOKEN_ENV, "") or env_value(self.env_file, TOKEN_ENV)
            self._token = tok
        return self._token


def runtime_view(ctx: Ctx, doc: dict[str, Any]) -> dict[str, Any]:
    """What the running daemon says about its model and pins (names, codes and loopback URLs only)."""
    base, tok = ctx.api_base(doc), ctx.token(doc)
    st, health = http("GET", base + "/health")
    if st != 200 or not isinstance(health, dict):
        return {"reachable": False, "health_status": st}
    view: dict[str, Any] = {"reachable": True, "uptime_seconds": health.get("uptime_seconds"),
                            "health_status": health.get("status")}
    st_m, m = http("GET", base + "/api/model", tok)
    if st_m == 200 and isinstance(m, dict):
        warnings = [w.get("code") or w.get("error") for w in m.get("warnings") or [] if isinstance(w, dict)]
        qwen = [p for p in m.get("providers") or [] if isinstance(p, dict) and p.get("provider") == MAX_PROVIDER]
        view["model"] = {"running": m.get("running"), "saved": m.get("saved"), "pending": m.get("pending"),
                         "key": m.get("key"), "warnings": warnings,
                         "qwen_key_present": qwen[0].get("key_present") if qwen else None}
    else:
        view["model_error"] = st_m
    st_s, s = http("GET", base + "/api/status", tok, timeout=30.0)
    if st_s == 200 and isinstance(s, dict):
        cp = s.get("config_pins") if isinstance(s.get("config_pins"), dict) else {}
        view["status"] = {"provider": s.get("provider"), "model": s.get("model"), "state": s.get("state"),
                          "config_pins": {"pins": cp.get("pins"),
                                          "drift_corrected_at_boot": cp.get("drift_corrected_at_boot")}}
    else:
        view["status_error"] = st_s
    return view


def local_base_url(model: dict[str, Any]) -> str | None:
    """Where a local primary is served (the model server restore waits for); None for a cloud one."""
    provider = str(model.get("provider") or "llama_cpp")
    if provider == "llama_cpp":
        return str(model.get("base_url") or "http://localhost:8080").rstrip("/")
    if provider == "ollama":
        return str(model.get("base_url") or "http://localhost:11434").rstrip("/")
    return None


def backend_identity(base: str, provider: str) -> dict[str, Any]:
    """Is the local model server up, and which model does it serve (status endpoints only)."""
    if provider == "ollama":
        st, _ = http("GET", base + "/api/tags", timeout=5.0)
        return {"url": base, "health": st}
    st, _ = http("GET", base + "/health", timeout=5.0)
    out: dict[str, Any] = {"url": base, "health": st}
    if st == 200:
        st_p, props = http("GET", base + "/props", timeout=5.0)
        if st_p == 200 and isinstance(props, dict):
            out["model_path"] = props.get("model_path")
        st_v, models = http("GET", base + "/v1/models", timeout=5.0)
        data = models.get("data") if isinstance(models, dict) else None
        if st_v == 200 and isinstance(data, list) and data and isinstance(data[0], dict):
            out["model_id"] = data[0].get("id")
    return out


def systemctl_user(verb: str, unit: str) -> subprocess.CompletedProcess[str]:
    if verb not in ("is-active", "start"):
        raise ValueError(f"daemon_switch never runs systemctl {verb}")
    try:
        return subprocess.run(["systemctl", "--user", verb, unit], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return subprocess.CompletedProcess([verb], 1, "", str(e))


def wait_idle(ctx: Ctx, doc: dict[str, Any], *, required: bool) -> None:
    """Wait for the daemon to finish a turn before restarting it."""
    deadline = _now() + ctx.idle_wait
    while True:
        st, s = http("GET", ctx.api_base(doc) + "/api/status", ctx.token(doc), timeout=30.0)
        state = s.get("state") if st == 200 and isinstance(s, dict) else None
        if state in (None, "idle"):
            return
        if _now() >= deadline:
            if required:
                raise Refusal(f"the daemon stayed busy ({state}) for {ctx.idle_wait:.0f}s; not interrupting a turn")
            log(f"note: the daemon is still busy ({state}); restarting anyway")
            return
        _sleep(IDLE_POLL_S)


def restart_daemon(ctx: Ctx, doc: dict[str, Any]) -> None:
    """Restart the daemon so it builds its provider again, and wait for the new boot to answer."""
    base, tok = ctx.api_base(doc), ctx.token(doc)
    st, health = http("GET", base + "/health")
    t0 = _now()
    if st != 200 or not isinstance(health, dict):
        state = systemctl_user("is-active", DAEMON_UNIT).stdout.strip() or "unknown"
        if state not in ("inactive", "failed"):
            raise Refusal(f"the daemon's API does not answer while {DAEMON_UNIT} is {state}: "
                          "not restarting it (this tool never stops the daemon)")
        log(f"{DAEMON_UNIT} is {state}: starting it")
        cp = systemctl_user("start", DAEMON_UNIT)
        if cp.returncode != 0:
            raise Refusal(f"systemctl --user start {DAEMON_UNIT} failed: {cp.stderr.strip()[-200:]}")
        u0 = None
    else:
        u0 = float(health.get("uptime_seconds") or 0.0)
        if u0 < MIN_UPTIME_S:
            # The new boot is told apart by its smaller uptime, which needs an old one to compare with.
            _sleep(MIN_UPTIME_S - u0)
            st, health = http("GET", base + "/health")
            if st != 200 or not isinstance(health, dict):
                raise Refusal("the daemon stopped answering just before the restart")
            u0 = float(health.get("uptime_seconds") or 0.0)
        t0 = _now()
        st, body = http("POST", base + "/api/restart", tok)
        if st not in (200, 202):
            code = body.get("error") if isinstance(body, dict) else None
            raise Refusal(f"POST /api/restart answered {st}{' ' + str(code) if code else ''}")
        log("restart requested (POST /api/restart: the daemon re-execs itself)")
    deadline = t0 + ctx.boot_wait
    while True:
        _sleep(POLL_S)
        st, health = http("GET", base + "/health")
        if st == 200 and isinstance(health, dict):
            up = float(health.get("uptime_seconds") or 0.0)
            if u0 is None or up < u0 + (_now() - t0) - 1.0:
                st_m, _ = http("GET", base + "/api/model", tok)
                if st_m == 200:
                    log(f"daemon is back (uptime {up:.0f}s)")
                    return
        if _now() > deadline:
            raise Refusal(f"the daemon did not come back within {ctx.boot_wait:.0f}s")


def runtime_problems(view: dict[str, Any], *, running: dict[str, Any], pins: dict[str, Any] | None,
                     drift: list[str] | None, require_key: bool = False) -> list[str]:
    if not view.get("reachable"):
        return ["the daemon's API does not answer"]
    out = []
    m = view.get("model")
    if not isinstance(m, dict):
        return [f"GET /api/model answered {view.get('model_error')} (token?)"]
    have = m.get("running") or {}
    for k, v in running.items():
        if have.get(k) != v:
            out.append(f"running {k} is {have.get(k)!r}, want {v!r}")
    if m.get("pending"):
        out.append("a saved model change is still waiting for a restart")
    if MODEL_OVERRIDDEN in (m.get("warnings") or []):
        out.append("PROMETHEUS_PROVIDER/PROMETHEUS_MODEL overrides the config")
    if require_key and not (m.get("key") or {}).get("present"):
        out.append(f"the daemon holds no {MAX_KEY_ENV}")
    s = view.get("status")
    if not isinstance(s, dict):
        return out + [f"GET /api/status answered {view.get('status_error')}"]
    cp = s.get("config_pins") or {}
    if pins is not None and (cp.get("pins") or {}) != pins:
        out.append("the pins in force are not the pins on disk")
    got_drift = list(cp.get("drift_corrected_at_boot") or [])
    if drift is not None and sorted(got_drift) != sorted(drift):
        out.append(f"CONFIG DRIFT corrected at boot: {', '.join(got_drift) or 'none'} "
                   f"(want {', '.join(drift) or 'none'})")
    if s.get("provider") not in (None, running.get("provider")):
        out.append(f"/api/status says provider {s.get('provider')!r}")
    return out


# ── captures ────────────────────────────────────────────────────────────────


@dataclass
class Capture:
    dir: Path
    manifest: dict[str, Any]
    config: bytes
    pins: bytes | None
    switched: dict[str, Any] | None

    @property
    def id(self) -> str:
        return self.dir.name


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def read_active(ctx: Ctx) -> Capture | None:
    """The capture a restore uses, verified against its own manifest. Raises Refusal if it is damaged."""
    try:
        name = (ctx.state_dir / ACTIVE_FILE).read_text().strip()
    except FileNotFoundError:
        return None
    d = ctx.state_dir / name
    if not re.fullmatch(r"[0-9TZ_-]+", name) or not d.is_dir():
        raise Refusal(f"{ACTIVE_FILE} names capture {name!r}, which is missing: nothing to restore from")
    manifest = _load_json(d / MANIFEST)
    if not isinstance(manifest, dict):
        raise Refusal(f"capture {name}: {MANIFEST} is missing or unreadable")
    config = read_bytes(d / CONFIG_COPY)
    pins = read_bytes(d / PINS_COPY)
    files = manifest.get("files") or {}
    if config is None or sha256(config) != (files.get("config") or {}).get("sha256"):
        raise Refusal(f"capture {name}: the config copy is missing or does not match its sha256")
    if (files.get("pins") or {}).get("absent"):
        if pins is not None:
            raise Refusal(f"capture {name}: a pins copy exists but the manifest says there was none")
    elif pins is None or sha256(pins) != (files.get("pins") or {}).get("sha256"):
        raise Refusal(f"capture {name}: the pins copy is missing or does not match its sha256")
    switched = _load_json(d / SWITCHED)
    return Capture(d, manifest, config, pins, switched if isinstance(switched, dict) else None)


def file_state(live: bytes | None, before_sha: str | None, after_sha: str | None) -> str:
    """"before", "after" (our own edit), "absent" (and was absent) or "other" (changed by someone else).

    "after" is tested first: a file to-max left as it was (a pins file with no model pin) is both, and
    once a switch is recorded it counts as switched.
    """
    if live is None:
        return "absent" if before_sha is None else "other"
    h = sha256(live)
    if after_sha and h == after_sha:
        return "after"
    if h == before_sha:
        return "before"
    return "other"


def states(ctx: Ctx, cap: Capture) -> dict[str, str]:
    files = cap.manifest["files"]
    sw = cap.switched or {}
    return {
        "config": file_state(read_bytes(ctx.config), files["config"].get("sha256"), sw.get("config_sha256")),
        "pins": file_state(read_bytes(ctx.pins), files["pins"].get("sha256"), sw.get("pins_sha256")),
    }


@contextlib.contextmanager
def locked(ctx: Ctx, wait_s: float = 60.0) -> Iterator[None]:
    """One capture/switch/restore at a time on this host (the dead-man and a person, say)."""
    if not ctx.execute:
        yield
        return
    private_dir(ctx.state_dir)
    fd = os.open(ctx.state_dir / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = _now() + wait_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES) or _now() > deadline:
                    raise Refusal("another daemon_switch run holds the lock") from None
                _sleep(1.0)
        yield
    finally:
        os.close(fd)


def parse_config(data: bytes) -> dict[str, Any]:
    try:
        doc = yaml.safe_load(data.decode("utf-8")) or {}
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        raise Refusal(f"the config does not parse as YAML ({type(e).__name__})") from None
    if not isinstance(doc, dict):
        raise Refusal("the config's top level is not a mapping")
    return doc


def parse_pins(data: bytes | None) -> dict[str, Any] | None:
    if data is None:
        return None
    try:
        pins = yaml.safe_load(data.decode("utf-8")) or {}
    except (yaml.YAMLError, UnicodeDecodeError) as e:
        raise Refusal(f"the pins file does not parse as YAML ({type(e).__name__})") from None
    if not isinstance(pins, dict):
        raise Refusal("the pins file is not a mapping")
    return pins


def is_max(model: dict[str, Any]) -> bool:
    return model.get("provider") == MAX_PROVIDER


def runtime_refusals(view: dict[str, Any], *, ours_pending: bool = False) -> list[str]:
    """Runtime states a capture or switch must not start from.

    ``ours_pending``: the files hold to-max's own recorded edit (a switch cut off before its restart), so
    the change waiting for a restart is that edit, and finishing the switch applies it.
    """
    if not view.get("reachable"):
        return ["the daemon's API does not answer, so its runtime state cannot be recorded"]
    m = view.get("model")
    if not isinstance(m, dict):
        return [f"GET /api/model answered {view.get('model_error')} (no usable API token?)"]
    out = []
    if m.get("pending") and not ours_pending:
        out.append("a saved model change is waiting for a restart; a switch would apply it too")
    if MODEL_OVERRIDDEN in (m.get("warnings") or []):
        out.append("PROMETHEUS_PROVIDER/PROMETHEUS_MODEL in the daemon's environment outrank the config")
    if not isinstance(view.get("status"), dict):
        out.append(f"GET /api/status answered {view.get('status_error')}")
    return out


def take_capture(ctx: Ctx, cfg: bytes, pins: bytes | None) -> dict[str, Any]:
    """The before-state as a manifest (runtime included unless --offline)."""
    doc = parse_config(cfg)
    pins_doc = parse_pins(pins)
    model, drift = simulate_boot(doc, pins_doc)
    manifest: dict[str, Any] = {
        "version": VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "files": {"config": file_record(ctx.config, cfg), "pins": file_record(ctx.pins, pins)},
        "env_file": {"path": str(ctx.env_file), "names": env_names(ctx.env_file)},
        "expected_boot": {"provider": model.get("provider"), "model": model.get("model"),
                          "base_url": local_base_url(model), "drift": drift},
        "pins": pins_doc,
        "runtime": None,
        "backend": None,
    }
    if not ctx.offline:
        view = runtime_view(ctx, doc)
        manifest["runtime"] = view
        base = local_base_url(model)
        if base:
            manifest["backend"] = backend_identity(base, str(model.get("provider") or "llama_cpp"))
    return manifest


def write_capture(ctx: Ctx, cfg: bytes, pins: bytes | None, manifest: dict[str, Any]) -> Path:
    private_dir(ctx.state_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d, n = ctx.state_dir / stamp, 1
    while d.exists():
        n += 1
        d = ctx.state_dir / f"{stamp}-{n}"
    private_dir(d)
    write_private(d / CONFIG_COPY, cfg)
    if pins is not None:
        write_private(d / PINS_COPY, pins)
    write_private(d / MANIFEST, dump_json(manifest))
    write_private(ctx.state_dir / ACTIVE_FILE, (d.name + "\n").encode())
    cap = read_active(ctx)
    if cap is None or cap.dir != d:
        raise Refusal("the capture did not read back")
    return d


def describe_view(view: dict[str, Any] | None) -> str:
    if not view:
        return "not read (--offline)"
    if not view.get("reachable"):
        return f"API unreachable (health {view.get('health_status')})"
    run = (view.get("model") or {}).get("running") or {}
    st = view.get("status") or {}
    return (f"running {run.get('provider')}/{run.get('model') or '-'} ({run.get('kind')}), "
            f"pending {(view.get('model') or {}).get('pending')}, state {st.get('state')}, "
            f"pins {safe_pins((st.get('config_pins') or {}).get('pins'))}, "
            f"drift {((st.get('config_pins') or {}).get('drift_corrected_at_boot'))}")


# ── commands ────────────────────────────────────────────────────────────────


def cmd_capture(ctx: Ctx) -> int:
    with locked(ctx):
        cfg, pins = read_bytes(ctx.config), read_bytes(ctx.pins)
        if cfg is None:
            raise Refusal(f"no config file at {ctx.config}")
        cap = read_active(ctx)
        if cap is not None and cap.switched:
            st = states(ctx, cap)
            if all(v in ("after", "absent") for v in st.values()):
                log(f"already switched under capture {cap.id}; it holds the before-state. Nothing captured")
                return 0
            raise Refusal(f"capture {cap.id} is recorded as switched and the files are {st}; "
                          "run restore first (a new capture now would record the wrong before-state)")
        doc = parse_config(cfg)
        model, _ = simulate_boot(doc, parse_pins(pins))
        if is_max(model) and not ctx.force:
            raise Refusal("the config already boots on qwen: recorded as the before-state, a restore would keep "
                          "Max (pass --force if that is really the state to come back to)")
        manifest = take_capture(ctx, cfg, pins)
        if not ctx.offline:
            bad = runtime_refusals(manifest["runtime"])
            if bad:
                raise Refusal("; ".join(bad))
        names = manifest["env_file"]["names"]
        log(f"config  {ctx.config}  sha256 {manifest['files']['config']['sha256'][:16]}  "
            f"mode {manifest['files']['config']['mode']:o}")
        pr = manifest["files"]["pins"]
        log(f"pins    {ctx.pins}  " + ("absent" if pr.get("absent") else
                                       f"sha256 {pr['sha256'][:16]}  mode {pr['mode']:o}"))
        log(f"env     {ctx.env_file}  set: {', '.join(n for n, v in names.items() if v['set']) or '-'}")
        log(f"boots   {manifest['expected_boot']}")
        log(f"runtime {describe_view(manifest['runtime'])}")
        if manifest["backend"]:
            log(f"backend {manifest['backend']}")
        if not ctx.execute:
            log("[dry run] nothing written; --execute records this as the before-state")
            return 0
        d = write_capture(ctx, cfg, pins, manifest)
        log(f"captured {d} (active)")
        return 0


def _write_switched(cap: Capture, record: dict[str, Any]) -> None:
    write_private(cap.dir / SWITCHED, dump_json(record))
    cap.switched = record


# ── taking to-max's edit back out of a file (restore, rollback) ─────────────


def _get(doc: Any, path: tuple[str, ...]) -> Any:
    for k in path:
        doc = doc.get(k, _MISSING) if isinstance(doc, dict) else _MISSING
        if doc is _MISSING:
            break
    return doc


def _put(doc: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Set ``path`` to ``value`` in ``doc``; ``_MISSING`` removes it."""
    node = doc
    for k in path[:-1]:
        if not isinstance(node.get(k), dict):
            if value is _MISSING:
                return
            node[k] = {}
        node = node[k]
    if value is _MISSING:
        node.pop(path[-1], None)
    else:
        node[path[-1]] = copy.deepcopy(value)


def merge3(before: str, after: str, live: str) -> str:
    """``live`` with the before->after edit taken back out: a three-way merge over ``after``.

    Lines the edit changed go back to ``before``; lines someone else changed since stay as ``live`` has
    them. A change of each that overlaps or touches the other raises Refusal (the same change made on
    both sides, as when the edit never reached the file, counts once).
    """
    a = after.splitlines(keepends=True)

    def hunks(other: str) -> list[tuple[int, int, tuple[str, ...]]]:
        o = other.splitlines(keepends=True)
        return [(i1, i2, tuple(o[j1:j2])) for tag, i1, i2, j1, j2
                in difflib.SequenceMatcher(None, a, o, autojunk=False).get_opcodes() if tag != "equal"]

    ours, theirs = hunks(before), hunks(live)
    for t in theirs:
        if t in ours:
            continue
        for o in ours:
            if not (o[1] < t[0] or t[1] < o[0]):
                raise Refusal(f"an edit made during the window at line {t[0] + 1} of the switched file "
                              f"touches a line to-max changed (line {o[0] + 1})")
    out: list[str] = []
    i = 0
    for i1, i2, repl in sorted(set(ours) | set(theirs)):
        out += a[i:i1]
        out += repl
        i = i2
    out += a[i:]
    return "".join(out)


def merge_back(label: str, before: bytes, after: bytes, live: bytes, touched: list[tuple[str, ...]]) -> bytes:
    """The live file with to-max's edit (``before`` -> ``after``, keys ``touched``) taken out again.

    Refuses when someone else changed a key to-max changed, when the edits are on touching lines, or
    when the result does not parse as exactly the live document with the touched keys at their
    captured values.
    """
    try:
        b_text, a_text, l_text = before.decode("utf-8"), after.decode("utf-8"), live.decode("utf-8")
        b_doc, a_doc, l_doc = (yaml.safe_load(b_text) or {}), (yaml.safe_load(a_text) or {}), (
            yaml.safe_load(l_text) or {})
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        raise Refusal(f"{label}: cannot read it to take to-max's edit out ({type(e).__name__}); "
                      "merge by hand, then restore") from None
    if not all(isinstance(d, dict) for d in (b_doc, a_doc, l_doc)):
        raise Refusal(f"{label}: not a mapping; merge by hand, then restore")
    for path in touched:
        now = _get(l_doc, path)
        if now != _get(a_doc, path) and now != _get(b_doc, path):
            raise Refusal(f"{label}: {'.'.join(path)} was changed again during the window, and to-max had "
                          "changed it; merge by hand, then restore")
    try:
        merged = merge3(b_text, a_text, l_text)
    except Refusal as e:
        raise Refusal(f"{label}: {e}; merge by hand, then restore") from None
    expected = copy.deepcopy(l_doc)
    for path in touched:
        _put(expected, path, _get(b_doc, path))
    try:
        got = yaml.safe_load(merged) or {}
    except yaml.YAMLError:
        got = None
    if got != expected:
        raise Refusal(f"{label}: taking to-max's lines out would not read back as the window's edits with "
                      "to-max's keys at their captured values; merge by hand, then restore")
    return merged.encode("utf-8")


def switched_files(cap: Capture) -> dict[str, tuple[bytes | None, list[tuple[str, ...]]]]:
    """Per file, the bytes to-max wrote and the keys it changed: rebuilt from the capture, checked against
    the hashes in switched.json."""
    sw = cap.switched or {}
    try:
        plan = max_plan(cap.config.decode("utf-8"), cap.pins.decode("utf-8") if cap.pins is not None else None,
                        tuple(sw.get("sets") or ()))
    except (Refusal, UnicodeDecodeError) as e:
        raise Refusal(f"capture {cap.id}: cannot rebuild what to-max wrote ({e}); merge by hand") from None
    cfg = plan["config"].encode("utf-8")
    pins = plan["pins"].encode("utf-8") if plan["pins"] is not None else None
    if sha256(cfg) != sw.get("config_sha256") or (sha256(pins) if pins is not None else None) != sw.get(
            "pins_sha256"):
        raise Refusal(f"capture {cap.id}: the rebuilt switch does not hash as {SWITCHED} records; merge by hand")
    return {"config": (cfg, list(plan["config_paths"])), "pins": (pins, list(plan["pin_paths"]))}


@dataclass
class FileMove:
    """What restore (or a rollback) does to one of the daemon's files."""

    name: str
    path: Path
    state: str
    data: bytes | None  # what the file holds afterwards (None: absent)
    write: bool
    expect_sha: str | None = None  # the live bytes the write replaces (compare-and-swap)
    mode: int | None = None
    mtime_ns: int | None = None
    note: str = ""


def restore_moves(ctx: Ctx, cap: Capture) -> dict[str, FileMove]:
    """Per file: the captured bytes back over to-max's own (mode and mtime too), or, when someone else edited
    the file during the window, that edit kept with to-max's lines taken out. Raises Refusal on a clash."""
    files = cap.manifest["files"]
    st = states(ctx, cap)
    rebuilt: dict[str, tuple[bytes | None, list[tuple[str, ...]]]] | None = None
    out: dict[str, FileMove] = {}
    for name, path, before in (("config", ctx.config, cap.config), ("pins", ctx.pins, cap.pins)):
        live = read_bytes(path)
        state = st[name]
        if state in ("before", "absent"):
            out[name] = FileMove(name, path, state, live, False)
        elif state == "after":
            if live is None or before is None:
                raise Refusal(f"{path}: switched, but there is nothing captured to put back")
            out[name] = FileMove(name, path, state, before, live != before, sha256(live),
                                 int(files[name]["mode"]), int(files[name]["mtime_ns"]))
        elif live is None:
            raise Refusal(f"{path} was deleted since capture {cap.id}; not recreating it. Merge by hand, then restore")
        elif before is None or not cap.switched:
            # Absent at capture (to-max never creates a file), or no switch recorded: nothing in it is to-max's.
            out[name] = FileMove(name, path, state, live, False,
                                 note="changed by someone else; to-max never wrote it, so it is left as it is")
        else:
            if rebuilt is None:
                rebuilt = switched_files(cap)
            after, touched = rebuilt[name]
            if after is None or after == before:
                out[name] = FileMove(name, path, state, live, False,
                                     note="changed by someone else; to-max did not change it")
                continue
            merged = merge_back(str(path), before, after, live, touched)
            out[name] = FileMove(name, path, state, merged, merged != live, sha256(live),
                                 stat.S_IMODE(path.stat().st_mode), None,
                                 note="edited during the window: that edit is kept, to-max's lines are taken out")
    return out


def apply_moves(moves: dict[str, FileMove]) -> list[str]:
    """Write the moves (compare-and-swap each), check they read back; the names written."""
    done = []
    for m in moves.values():
        if not m.write:
            continue
        if m.data is None or m.expect_sha is None or m.mode is None:
            raise Refusal(f"{m.path}: no bytes to write back")
        replace_file(m.path, m.data, m.mode, m.mtime_ns, expect_sha=m.expect_sha)
        if read_bytes(m.path) != m.data:
            raise Refusal(f"{m.path} does not read back as written")
        done.append(m.name)
    return done


# ── to-max ──────────────────────────────────────────────────────────────────


def _refuse_background(plan: dict[str, Any]) -> None:
    if plan["background"]:
        raise Refusal("the Max key is for interactive use only, and on the switched config these would call it "
                      "with nobody asking: " + "; ".join(plan["background"])
                      + ". Not switching until each is off for the window")


def cmd_to_max(ctx: Ctx) -> int:
    with locked(ctx):
        cap = read_active(ctx)
        if cap is None:
            if ctx.execute:
                raise Refusal("no capture: run `capture --execute` first, so a restore has the exact before-state")
            cfg, pins = read_bytes(ctx.config), read_bytes(ctx.pins)
            if cfg is None:
                raise Refusal(f"no config file at {ctx.config}")
            log("[dry run] no capture yet (`capture --execute` must run first); planning from the live files")
            plan = max_plan(cfg.decode("utf-8"), pins.decode("utf-8") if pins is not None else None, ctx.sets)
            _refuse_background(plan)
            _max_preconditions(ctx, parse_config(cfg))
            _print_plan(ctx, cfg, pins, plan)
            return 0
        st = states(ctx, cap)
        doc_before = parse_config(cap.config)
        if "other" in st.values():
            raise Refusal(f"changed since capture {cap.id}: {st}. Someone else edited the daemon's files; "
                          "not writing over it (capture again once it is settled)")
        plan = max_plan(cap.config.decode("utf-8"), cap.pins.decode("utf-8") if cap.pins is not None else None,
                        ctx.sets)
        _refuse_background(plan)
        after_cfg, after_pins = plan["config"].encode(), (plan["pins"].encode() if plan["pins"] is not None else None)
        rec = cap.switched
        if rec is not None:
            if (rec.get("config_sha256"), rec.get("pins_sha256")) != (
                    sha256(after_cfg), sha256(after_pins) if after_pins is not None else None):
                raise Refusal(f"capture {cap.id} was switched with --set {', '.join(rec.get('sets') or []) or 'none'}"
                              f"; this run asks for {', '.join(ctx.sets) or 'none'}. Run restore first")
            if all(v in ("after", "absent") for v in st.values()):
                log(f"already switched under capture {cap.id}"
                    + (f" (with --set {', '.join(rec.get('sets') or [])})" if rec.get("sets") else ""))
                if ctx.offline:
                    return 0
                view = runtime_view(ctx, doc_before)
                bad = runtime_problems(view, running={"provider": MAX_PROVIDER, "model": MAX_MODEL},
                                       pins=parse_pins(read_bytes(ctx.pins)), drift=rec.get("drift"))
                if not bad:
                    if ctx.execute and rec.get("status") != "verified":
                        rec["status"] = "verified"
                        _write_switched(cap, rec)
                    log("runtime: " + describe_view(view))
                    return 0
                log("files are switched but the runtime is not: " + "; ".join(bad) + "; finishing the switch")
        # A switch cut off after writing (before its restart) left its own change pending: that is not a refusal.
        ours_pending = rec is not None and "after" in st.values()
        view = _max_preconditions(ctx, doc_before, ours_pending=ours_pending)
        _print_plan(ctx, cap.config, cap.pins, plan)
        if not ctx.execute:
            log("[dry run] nothing written, no restart; --execute switches")
            return 0
        if ctx.offline:
            raise Refusal("--offline cannot --execute (the switch is verified through the daemon's API)")
        wait_idle(ctx, doc_before, required=True)
        # The daemon rewrites its own config (a permission grant) at any time, a turn in progress included.
        now = states(ctx, cap)
        if now != st:
            raise Refusal(f"the daemon's files changed while this run waited for an idle daemon ({st} -> {now}); "
                          "nothing written. Capture again once they are settled")
        record = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "status": "writing",
                  "sets": list(ctx.sets), "drift": plan["drift"],
                  "config_sha256": sha256(after_cfg),
                  "pins_sha256": sha256(after_pins) if after_pins is not None else None}
        _write_switched(cap, record)
        files = cap.manifest["files"]
        restarted = False
        try:
            for name, path, data, before in (("config", ctx.config, after_cfg, cap.config),
                                             ("pins", ctx.pins, after_pins, cap.pins)):
                # A file to-max leaves as it was is not rewritten (its inode and mtime stay).
                if data is None or data == before or st[name] != "before":
                    continue
                replace_file(path, data, int(files[name]["mode"]), expect_sha=files[name]["sha256"])
            if read_bytes(ctx.config) != after_cfg or (after_pins is not None and read_bytes(ctx.pins) != after_pins):
                raise Refusal("the edited files do not read back as planned")
            record["status"] = "written"
            _write_switched(cap, record)
            restarted = True
            restart_daemon(ctx, doc_before)
            view = runtime_view(ctx, doc_before)
            bad = runtime_problems(view, running={"provider": MAX_PROVIDER, "model": MAX_MODEL,
                                                  "api_key_env": MAX_KEY_ENV},
                                   pins=plan["pins_doc"], drift=plan["drift"], require_key=True)
            if bad:
                raise Refusal("the daemon did not come back on Max: " + "; ".join(bad))
        except (Refusal, OSError) as e:
            log(f"to-max FAILED: {e}. Rolling back to capture {cap.id}")
            _rollback(ctx, cap, doc_before, restarted=restarted)
            raise Refusal(f"to-max did not verify ({e}); rolled back to the captured files") from None
        record["status"] = "verified"
        _write_switched(cap, record)
        log("runtime: " + describe_view(view))
        log(f"switched to {MAX_PROVIDER}/{MAX_MODEL} (capture {cap.id} holds the before-state)")
        return 0


def _max_preconditions(ctx: Ctx, doc: dict[str, Any], *, ours_pending: bool = False) -> dict[str, Any] | None:
    """Refuse a switch the daemon could not boot from, or one that would carry someone else's change."""
    names = env_names(ctx.env_file)
    missing = [n for n in (MAX_KEY_ENV, MAX_URL_ENV) if not names[n]["set"]]
    if missing:
        raise Refusal(f"{', '.join(missing)} not set in {ctx.env_file}: the daemon could not reach Max")
    if any(names[n]["set"] for n in PIN_ENVS):
        raise Refusal("PROMETHEUS_PROVIDER/PROMETHEUS_MODEL are set in the env file and outrank the config")
    if ctx.offline:
        return None
    view = runtime_view(ctx, doc)
    bad = runtime_refusals(view, ours_pending=ours_pending)
    if (view.get("model") or {}).get("qwen_key_present") is False:
        bad.append(f"the daemon reports no {MAX_KEY_ENV} key")
    if bad:
        raise Refusal("; ".join(bad))
    return view


def _rollback(ctx: Ctx, cap: Capture, doc: dict[str, Any], *, restarted: bool) -> None:
    """Undo a to-max that did not verify: its lines out of the files again (an edit someone else made meanwhile
    is kept), the capture unswitched, and the daemon restarted if a restart had been asked for."""
    try:
        done = apply_moves(restore_moves(ctx, cap))
        os.replace(cap.dir / SWITCHED, cap.dir / (SWITCHED + ".rolled-back"))
        cap.switched = None
        log(f"rollback: the captured files are back (written: {', '.join(done) or 'nothing'})")
        if restarted:
            restart_daemon(ctx, doc)
            log("rollback: daemon restarted on the captured files")
        else:
            log("rollback: the daemon was not restarted, so it still runs the captured files")
    except (Refusal, OSError) as e:
        log(f"ROLLBACK INCOMPLETE: {e}. Run restore --execute (capture {cap.id})")


def _print_plan(ctx: Ctx, cfg: bytes, pins: bytes | None, plan: dict[str, Any]) -> None:
    log(f"to-max: {MAX_PROVIDER}/{MAX_MODEL}, key from {MAX_KEY_ENV}, endpoint from {MAX_URL_ENV} at boot")
    log(f"  config {ctx.config}: {', '.join(plan['config_edits']) or 'no change'}")
    for line in changed_lines(cfg.decode("utf-8"), plan["config"]):
        log(line)
    if plan["pins"] is None:
        log(f"  pins {ctx.pins}: none (no pins file)")
    else:
        log(f"  pins {ctx.pins}: {', '.join(plan['pin_edits']) or 'no change'}")
        for line in changed_lines((pins or b"").decode("utf-8"), plan["pins"]):
            log(line)
    log("  then: wait for an idle daemon, POST /api/restart, verify via /api/model + /api/status")


# ── restore ─────────────────────────────────────────────────────────────────


def restore_target(ctx: Ctx, cap: Capture) -> tuple[dict[str, FileMove], dict[str, Any], dict[str, Any] | None,
                                                     dict[str, Any], list[str]]:
    """``(moves, config, pins, model in force, drift)`` for the files as the restore leaves them.

    The daemon boots from those files after the restart, so they, not the runtime recorded at capture
    (which may predate an edit the daemon was never restarted for), are what the restore verifies against.
    """
    moves = restore_moves(ctx, cap)
    cfg = moves["config"].data
    if cfg is None:
        raise Refusal(f"no config file at {ctx.config}")
    doc = parse_config(cfg)
    pins_doc = parse_pins(moves["pins"].data)
    in_force, drift = simulate_boot(doc, pins_doc)
    return moves, doc, pins_doc, in_force, drift


def cmd_restore(ctx: Ctx) -> int:
    with locked(ctx):
        cap = read_active(ctx)
        if cap is None:
            cfg = read_bytes(ctx.config)
            if cfg is None:
                raise Refusal(f"no config file at {ctx.config}")
            model, _ = simulate_boot(parse_config(cfg), parse_pins(read_bytes(ctx.pins)))
            if is_max(model):
                raise Refusal("the daemon's files boot on qwen but there is no active capture to restore from; "
                              "not guessing the before-state")
            log("no active capture and the files are not switched: nothing to restore")
            return 0
        st = states(ctx, cap)
        moves, doc, want_pins, in_force, want_drift = restore_target(ctx, cap)
        if is_max(in_force) and (cap.manifest.get("expected_boot") or {}).get("provider") != MAX_PROVIDER:
            raise Refusal(f"after the restore the files would still boot on qwen, which capture {cap.id} did not; "
                          "not guessing the before-state")
        want_running = {"provider": str(in_force.get("provider") or "llama_cpp"),
                        "model": str(in_force.get("model") or "")}
        need_write = [n for n, m in moves.items() if m.write]
        log(f"restore from capture {cap.id}: files {st}; write back: {', '.join(need_write) or 'nothing'}")
        for m in moves.values():
            if m.note:
                log(f"  {m.name}: {m.note}")
        if ctx.offline:
            log("[dry run --offline] the model server and the runtime were not checked")
            return 0
        view = runtime_view(ctx, doc)
        need_restart = bool(need_write) or bool(runtime_problems(view, running=want_running, pins=want_pins,
                                                                 drift=want_drift))
        log(f"  runtime now: {describe_view(view)}; restart {'needed' if need_restart else 'not needed'}")
        if need_restart:
            if ctx.execute:
                _wait_backend(ctx, cap, in_force)
            else:
                try:
                    _wait_backend(ctx, cap, in_force, wait=False)
                except Refusal as e:
                    log(f"[dry run] note: {e} at the moment; an executing restore waits up to "
                        f"{ctx.backend_wait:.0f}s for it (an apply starts the model server before its post_apply)")
        if not ctx.execute:
            log("[dry run] nothing written; --execute restores")
            return 0
        if need_write:
            wait_idle(ctx, doc, required=False)
            # Again, after the waits: the daemon may have rewritten its config (a grant) meanwhile.
            moves, doc, want_pins, now_force, now_drift = restore_target(ctx, cap)
            if (now_force, now_drift) != (in_force, want_drift):
                raise Refusal("the daemon's files changed while this run waited, and would now boot differently; "
                              "nothing written. Run restore again")
            done = apply_moves(moves)
            log(f"files restored ({', '.join(done) or 'nothing left to write'})")
        if need_restart:
            restart_daemon(ctx, doc)
            view = runtime_view(ctx, doc)
        bad = runtime_problems(view, running=want_running, pins=want_pins, drift=want_drift)
        if bad:
            raise Refusal("files are restored but the daemon does not run as captured: " + "; ".join(bad))
        runtime = cap.manifest.get("runtime") or {}
        st_model = (runtime.get("status") or {}).get("model")
        now_model = (view.get("status") or {}).get("model")
        if st_model and now_model and st_model != now_model:
            log(f"note: /api/status model is {now_model!r} (was {st_model!r} at capture)")
        write_private(cap.dir / RESTORED, dump_json({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                     "runtime": view}))
        (ctx.state_dir / ACTIVE_FILE).unlink()
        log("runtime: " + describe_view(view))
        log(f"restored to capture {cap.id}")
        return 0


def _wait_backend(ctx: Ctx, cap: Capture, model: dict[str, Any], *, wait: bool = True) -> None:
    """The local model server ``model`` names must be healthy, and serve the captured model, before the daemon
    points back at it (otherwise it stays on Max). A dry run looks once instead of waiting."""
    base = local_base_url(model)
    if base is None:
        return
    was = cap.manifest.get("backend") or {}
    deadline = _now() + (ctx.backend_wait if wait else 0.0)
    while True:
        now = backend_identity(base, str(model.get("provider") or "llama_cpp"))
        why = None
        if now.get("health") != 200:
            why = f"{base} is not healthy (health {now.get('health')})"
        else:
            for k in ("model_path", "model_id"):
                if was.get(k) and now.get(k) != was.get(k):
                    why = f"{base} serves {k} {now.get(k)!r}, not the captured {was.get(k)!r}"
        if why is None:
            log(f"model server ok: {base}")
            return
        if _now() >= deadline:
            raise Refusal(why + "; the daemon stays on its current model")
        _sleep(5.0)


def cmd_status(ctx: Ctx) -> int:
    cfg, pins = read_bytes(ctx.config), read_bytes(ctx.pins)
    try:
        cap = read_active(ctx)
    except Refusal as e:
        log(f"active capture: DAMAGED ({e})")
        cap = None
    else:
        if cap is None:
            log("active capture: none")
        else:
            sw = cap.switched or {}
            log(f"active capture: {cap.id} (captured {cap.manifest.get('captured_at')}; "
                f"switch {sw.get('status', 'not done')})")
            log(f"files vs capture: {states(ctx, cap)}")
    if cfg is None:
        log(f"config {ctx.config}: missing")
        return 0
    try:
        doc = parse_config(cfg)
        model, drift = simulate_boot(doc, parse_pins(pins))
    except Refusal as e:
        log(f"config: {e}")
        return 0
    log(f"files boot as: {model.get('provider')}/{model.get('model') or '-'} "
        f"base_url {model.get('base_url') or '(from env)'}; pins correct: {', '.join(drift) or 'nothing'}")
    names = env_names(ctx.env_file)
    log(f"env file names set: {', '.join(n for n, v in names.items() if v['set']) or '-'}")
    if not ctx.offline:
        log("runtime: " + describe_view(runtime_view(ctx, doc)))
    return 0


def cmd_selftest(ctx: Ctx) -> int:
    log(f"selftest ok: python {sys.version.split()[0]}, PyYAML {getattr(yaml, '__version__', '?')}")
    return 0


def notify(ctx: Ctx, command: str, reason: str) -> None:
    if not ctx.notify_cmd:
        log("not notified (no --notify-cmd)")
        return
    msg = f"daemon_switch {command} REFUSED on {socket.gethostname()}\n- {' '.join(reason.split())[:300]}\n"
    try:
        cp = subprocess.run(["sh", "-c", ctx.notify_cmd], input=msg, capture_output=True, text=True, timeout=60)
        log("notify " + ("sent" if cp.returncode == 0 else f"FAILED (exit {cp.returncode})"))
    except (OSError, subprocess.SubprocessError) as e:
        log(f"notify FAILED: {type(e).__name__}")


COMMANDS = {"capture": cmd_capture, "to-max": cmd_to_max, "restore": cmd_restore,
            "status": cmd_status, "selftest": cmd_selftest}


def main(argv: list[str] | None = None) -> int:
    # allow_abbrev=False: an abbreviation such as --exec must not stand in for --execute (setups
    # refuses --execute in a hook and adds it itself only when an apply executes).
    ap = argparse.ArgumentParser(prog="daemon_switch", description=__doc__.split("\n\n")[0], allow_abbrev=False)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--execute", action="store_true", help="really change things (default: dry run)")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="the daemon's config file")
    ap.add_argument("--pins", default=DEFAULT_PINS, help="the daemon's config_pins.yaml")
    ap.add_argument("--env-file", default=DEFAULT_ENV_FILE, help="the daemon's env file (names only)")
    ap.add_argument("--state-dir", default=DEFAULT_STATE_DIR, help="where captures are kept")
    ap.add_argument("--api", default=None, help="daemon API base (default http://127.0.0.1:<web.api_port>)")
    ap.add_argument("--offline", action="store_true", help="dry run without the daemon's API")
    ap.add_argument("--force", action="store_true", help="capture even when the files already boot on qwen")
    ap.add_argument("--idle-wait", type=float, default=300.0, help="seconds to wait for an idle daemon")
    ap.add_argument("--boot-wait", type=float, default=240.0, help="seconds to wait for the restarted daemon")
    ap.add_argument("--backend-wait", type=float, default=300.0,
                    help="seconds restore waits for the local model server")
    ap.add_argument("--notify-cmd", default=None, help="shell command given a refusal's summary on stdin")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="to-max only: also set this existing config key for the window (off by default; "
                         "restore puts the bytes back)")
    args = ap.parse_args(argv)
    ctx = Ctx(config=Path(args.config).expanduser(), pins=Path(args.pins).expanduser(),
              env_file=Path(args.env_file).expanduser(), state_dir=Path(args.state_dir).expanduser(),
              api=args.api, execute=args.execute, offline=args.offline, force=args.force,
              idle_wait=args.idle_wait, boot_wait=args.boot_wait, backend_wait=args.backend_wait,
              notify_cmd=args.notify_cmd, sets=tuple(args.sets))
    if args.offline and args.execute:
        log("REFUSED: --offline is for dry runs only")
        return 1
    try:
        return COMMANDS[args.command](ctx)
    except Refusal as e:
        log(f"REFUSED: {e}")
        if args.execute:
            notify(ctx, args.command, str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
