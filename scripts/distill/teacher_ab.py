#!/usr/bin/env python3
"""Teacher quant A/B: the same real-pattern tasks on two quants of the local teacher, one judge.

The operator's rule (2026-10-30): before mass generation, run the same 30
teacher tasks on the IQ3 quant (arm A) and on a Q4_K_M of the same model (arm
B); if IQ3's pass rate is more than 5 points lower than Q4's, use the bigger
quant with less context, else keep IQ3.

    # 1. Pick the tasks (seeded, stratified by category/mode; ids recorded) and fix
    #    every request setting both arms will use (sampling included):
    uv run python scripts/distill/teacher_ab.py select \\
        --tasks ~/distill/phase2b/tasksets/real-v1.jsonl --out ~/distill/teacher-ab \\
        [--n 30] [--seed teacher-ab-v1] [--pack ~/distill/phase3/request_pack.json] [--repo ~/Prometheus] \\
        [--temperature 0.7 --top-k 20 --top-p 0.8 --min-p 0]
    # 2. One arm per server, WITHOUT judging (one llama-server at a time is fine):
    uv run python scripts/distill/teacher_ab.py arm --out ~/distill/teacher-ab --arm A \\
        --url http://127.0.0.1:18080 [--until 06:20]
    uv run python scripts/distill/teacher_ab.py arm --out ~/distill/teacher-ab --arm B \\
        --url http://127.0.0.1:18080 [--until 06:20]
    # 3. All traces of both arms on ONE judge endpoint (the bigger quant), then the report:
    uv run python scripts/distill/teacher_ab.py judge --out ~/distill/teacher-ab \\
        --judge-url http://127.0.0.1:18080 [--until 06:20]
    uv run python scripts/distill/teacher_ab.py report --out ~/distill/teacher-ab
    # Or all of it against servers that are up at the same time:
    uv run python scripts/distill/teacher_ab.py run --tasks ... --out ... \\
        --url-a http://127.0.0.1:18081 --url-b http://127.0.0.1:18082 --judge-url http://127.0.0.1:18082

``teacher_ab_run.sh`` beside this file drives steps 1-3 on one GPU host with one
llama-server at a time (IQ3 for arm A; then Q4_K_M for arm B and the judge).

Tasks
-----
``--n`` (30) rows of a ``taskset_gen.py`` real-pattern taskset (read with
``teacher_gen.read_taskset_rows``: each row passes taskset_gen's own
``validate_task``; nothing else is accepted), stratified by (category, mode):
proportional shares by largest remainder, at least one per stratum when ``n``
allows, rows of distinct ``task_group`` first, every draw seeded by
``--seed``. Each row becomes a teacher task exactly as ``teacher_gen.py --tasks``
builds it (flags honoured: chat-mode sends no tools, ``inject_errors`` fails the
first tool call, ``force_compaction`` runs the real compactor with a small
window). The built tasks are frozen into ``ab_tasks.jsonl``, so both arms run
byte-identical tasks whatever happens to the repo between them.

Identical settings
------------------
``select`` writes ``ab_settings.json``: the request settings both arms use
(model name, max tokens, tool iterations and timeouts, the compaction window,
concurrency, the system prompt with its date fixed, and EXPLICIT sampling:
temperature, top_k, top_p and min_p on every teacher and summarizer request,
because a server's defaults come from its launch flags and the GGUF's own
metadata, which differ between the two files). ``arm`` takes only the arm, the
URL and a time box; every record carries the hash of those settings, and the
report refuses a decision if the two arms' hashes differ.

The servers are compared too, not only the requests: at its preflight each arm
reads llama-server's ``/props`` and every record carries the model file, the
per-slot context, a hash of the chat template (each server renders with its own
GGUF's template unless one ``--chat-template-file`` is given to both) and the
server's own sampler defaults (repeat/presence penalties, DRY, XTC, sampler
order, ...: a GGUF's metadata can set them). An arm whose server differs from
the other arm's in any of these, or runs on the same model file, is refused
before its first task, and the report refuses a decision on such records.
``models`` checks the two GGUF files before any server starts: two different
files, B (the judge) the bigger one, the same embedded chat template.

Judging
-------
``arm`` runs each task through ``teacher_gen``'s real agent loop and sandbox and
records its deterministic verification (tool success, gym predicates, the
kind's own checks, the served model) and the rendered transcript, but never
asks a judge. ``judge`` then sends every recorded transcript of BOTH arms to
one ``--judge-url`` (teacher_gen's judge prompt, temperature 0,
``--judge-seed``); every judge record names the server's ``/v1/models`` id, and
a resume against another judge is refused. So no arm grades itself in its own
run, and both are graded by the same model. (The plan uses the bigger quant as
the judge: it is also arm B's model, which, if anything, favours B, the
conservative direction for this decision.) A judge that is not arm B's model
(served id and model file of arm B's records) is refused, and judging needs
arm B's records first.

Pass rule (``teacher_gen``'s, unchanged): a task passes when its deterministic
checks pass, it yields at least one row, and the judge passes it (verdict
pass, completed, tools_ok and grounded true, recovery_ok not false, and a
judge that is not a Max model). A provider refusal (context overflow) or a
task timeout is a fail; a server that is down (transport errors) is retried by
the next run, never recorded.

Report
------
``ab_report.json``: per-arm pass rate (with tool success, predicates and judge
pass apart, and by category), the paired table over the tasks both arms
finished (a both pass, b only Q4, c only IQ3, d neither), the paired
difference IQ3 - Q4 with its 95% interval (``score.py``'s Newcombe hybrid score
interval, with its bootstrap and exact McNemar cross-checks) and the decision:
``use Q4_K_M (less context)`` when IQ3's pass rate is MORE than 5 points below
Q4's (computed exactly: 100 * (k_B - k_A) > 5 * n), else ``keep IQ3``. The rule
reads the point estimates, as the operator stated it; the interval is printed
beside it (with 30 tasks it is roughly +-20 points wide). No decision
(``incomplete``) while fewer than ``--min-pairs`` (default: all selected) pairs
are finished and judged; none (``invalid``) when the arms' settings differ,
the arms' servers differ (``/props``: chat template, sampler defaults, per-slot
context) or share one model file, both arms report the same served model, the
judge records name more than one judge or a judge that is not arm B's model,
or nothing was judged. ``ab_summary.txt``: five lines for Telegram.

Resume: every phase appends one JSON line per finished task (a torn last line
is cut) and skips what is done; ``--until`` (HH:MM, next occurrence, or ISO) is
a hard end: no task starts within ``min_task_seconds`` of it and one still
running at it (a task or a judge call) is cancelled and not recorded. An HH:MM
is resolved once per command (``run`` passes the absolute time to every phase;
``time-left --iso`` gives it to the shell driver) and one more than 12 h ahead
is refused: read again after it has passed, it would mean tomorrow.

State stays under ``--out`` (``work/`` holds the Prometheus state roots and the
telemetry database; the sandbox is ``sbx/``). Endpoints are loopback only
(``--allow-lan-endpoints`` at ``select`` also admits private addresses);
hosted APIs and Max models are refused (teacher_gen's endpoint policy).

Exit codes: 0 done (``report``/``run``: a decision was made), 1 not finished
(time box, tasks left) or no decision, 2 usage or input error, 3 a server is
unreachable or the wrong model, or too many errors in a row.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import os
import random
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

log = logging.getLogger("teacher_ab")

_DIR = Path(__file__).resolve().parent
SETTINGS_SCHEMA = "distill-teacher-ab-settings/1"
RECORD_SCHEMA = "distill-teacher-ab-record/1"
REPORT_SCHEMA = "distill-teacher-ab-report/1"
ARMS = ("A", "B")
DEFAULT_LABELS = {"A": "IQ3_XXS", "B": "Q4_K_M"}
DEFAULT_N = 30
DEFAULT_SEED = "teacher-ab-v1"
#: Qwen3's published non-thinking sampling. Explicit on every request of both arms.
DEFAULT_SAMPLING = {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.0}
#: The operator's rule: IQ3 more than this many points below Q4 -> use Q4.
THRESHOLD_PTS = 5
DECISION_Q4 = "use Q4_K_M (less context)"
DECISION_IQ3 = "keep IQ3"
DECISION_INCOMPLETE = "incomplete"
DECISION_INVALID = "invalid"
#: no judge call starts this close to --until (one still running at it is cancelled)
JUDGE_MARGIN_S = 30.0
#: ab_settings.json keys that shape a request or a verdict (hashed into every record)
GEN_KEYS = ("generator", "model_name", "teacher_id", "sampling", "max_tokens", "max_tool_iterations",
            "tool_timeout", "task_timeout", "request_timeout", "compaction_history_tokens", "concurrency",
            "jail", "min_task_seconds", "system_prompt", "judge_max_chars", "allow_lan")
#: llama-server's /props default_generation_settings.params that shape what it samples or
#: how it renders and parses a turn (the server's own defaults: a GGUF's metadata can set them)
PROPS_SAMPLER_KEYS = (
    "temperature", "dynatemp_range", "dynatemp_exponent", "top_k", "top_p", "min_p", "top_n_sigma",
    "xtc_probability", "xtc_threshold", "typical_p", "repeat_last_n", "repeat_penalty", "presence_penalty",
    "frequency_penalty", "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n", "mirostat",
    "mirostat_tau", "mirostat_eta", "adaptive_target", "adaptive_decay", "n_predict", "ignore_eos", "min_keep",
    "samplers", "chat_format", "reasoning_format", "reasoning_in_content")
#: server facts both arms must share (the model file is the one thing that must differ)
SERVER_SAME_KEYS = ("chat_template_sha", "sampler_sha", "n_ctx")


def _load_sibling(name: str, file: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _DIR / file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tg = _load_sibling("distill_teacher_gen", "teacher_gen.py")


class InputError(ValueError):
    """Bad arguments or files: exit 2."""


class ServerError(RuntimeError):
    """A server that is down, refuses, or is the wrong model: exit 3."""


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


class Paths:
    def __init__(self, out: Path) -> None:
        self.out = Path(out).expanduser().resolve()
        self.settings = self.out / "ab_settings.json"
        self.selection = self.out / "ab_selection.json"
        self.tasks = self.out / "ab_tasks.jsonl"
        self.judge = self.out / "judge.jsonl"
        self.report = self.out / "ab_report.json"
        self.summary = self.out / "ab_summary.txt"
        self.work = self.out / "work"
        self.sandbox = self.out / "sbx"

    def arm(self, arm: str) -> Path:
        return self.out / f"arm-{arm}.jsonl"


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_records(path: Path) -> list[dict[str, Any]]:
    """The rows of an append-only file; a torn last line is cut off it."""
    return tg._repair_jsonl(path)


def settings_sha(conf: dict[str, Any]) -> str:
    gen = {k: conf.get(k) for k in GEN_KEYS}
    return hashlib.sha256(json.dumps(gen, sort_keys=True).encode()).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds")


def resolve_until(text: str | None) -> float | None:
    """--until as a timestamp (teacher_gen's rule: HH:MM is the next occurrence and
    refused when that is more than 12 h ahead; or an ISO time). Resolve it ONCE per
    run and pass the ISO form on: an HH:MM re-read after it has passed is tomorrow."""
    try:
        return tg.parse_until(text, max_ahead_hours=tg.UNTIL_HHMM_MAX_AHEAD_H)
    except ValueError as exc:
        raise InputError(f"--until {text!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# What a server is: llama-server's /props, and a GGUF file's own metadata
# ---------------------------------------------------------------------------


def props_record(props: dict[str, Any]) -> dict[str, Any]:
    """The facts of one llama-server (/props) an arm's records carry: the model
    file, the per-slot context, the chat template and the server's own sampling
    defaults (as hashes, with the sampler values beside them)."""
    dgs = props.get("default_generation_settings") or {}
    params = dgs.get("params") if isinstance(dgs.get("params"), dict) else dgs
    sampler = {k: params.get(k) for k in PROPS_SAMPLER_KEYS}
    template = str(props.get("chat_template") or "")
    tool_template = str(props.get("chat_template_tool_use") or "")
    model_path = str(props.get("model_path") or "")
    return {
        "model_path": model_path, "model_file": Path(model_path).name, "model_ftype": props.get("model_ftype"),
        "n_ctx": dgs.get("n_ctx"), "total_slots": props.get("total_slots"),
        "chat_template_sha": _sha16(template + "\x00" + tool_template) if template else "",
        "sampler_sha": _sha16(json.dumps(sampler, sort_keys=True)), "sampler": sampler,
    }


async def server_props(url: str, timeout: float) -> dict[str, Any]:
    """``props_record`` of the server at *url*; ServerError when it has no usable
    /props (every llama-server has one; a server without is not what this compares)."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.get(f"{url}/props")
        body = resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, OSError, ValueError) as exc:
        raise ServerError(f"{url}/props is unreachable ({type(exc).__name__})") from exc
    if not isinstance(body, dict) or not body.get("chat_template") or not body.get("model_path"):
        raise ServerError(f"{url}/props names no chat template or model file: the arms' servers "
                          "cannot be compared (is it llama-server?)")
    return props_record(body)


def server_differences(a: dict[str, Any] | None, b: dict[str, Any] | None) -> list[str]:
    """Why two arms' servers are not like for like: a fact that differs
    (SERVER_SAME_KEYS), or the same model file on both."""
    if not a or not b:
        return ["a server's settings were not recorded"]
    out = [k for k in SERVER_SAME_KEYS if a.get(k) != b.get(k)]
    if a.get("model_path") and a.get("model_path") == b.get("model_path"):
        out.append("the same model file")
    return out


_GGUF_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
               12: "<d"}
_GGUF_STRING, _GGUF_ARRAY = 8, 9


def gguf_metadata(path: Path) -> dict[str, Any]:
    """A GGUF file's key/value metadata (header only; arrays are skipped and given
    as ``<array N>``). ValueError for a file that is not a little-endian GGUF v2/v3."""
    import struct

    out: dict[str, Any] = {}
    with Path(path).open("rb") as fh:
        def read(n: int) -> bytes:
            data = fh.read(n)
            if len(data) != n:
                raise ValueError(f"{path}: truncated GGUF header")
            return data

        def u32() -> int:
            return int(struct.unpack("<I", read(4))[0])

        def u64() -> int:
            return int(struct.unpack("<Q", read(8))[0])

        def string() -> str:
            n = u64()
            if n > 64 << 20:
                raise ValueError(f"{path}: implausible GGUF string length {n}")
            return read(n).decode("utf-8", "replace")

        def value(kind: int) -> Any:
            if kind in _GGUF_FIXED:
                fmt = _GGUF_FIXED[kind]
                return struct.unpack(fmt, read(struct.calcsize(fmt)))[0]
            if kind == _GGUF_STRING:
                return string()
            if kind == _GGUF_ARRAY:
                inner, n = u32(), u64()
                if n > 50_000_000:
                    raise ValueError(f"{path}: implausible GGUF array length {n}")
                if inner in _GGUF_FIXED:
                    fh.seek(n * struct.calcsize(_GGUF_FIXED[inner]), os.SEEK_CUR)
                else:
                    for _ in range(n):
                        value(inner)
                return f"<array {n}>"
            raise ValueError(f"{path}: unknown GGUF value type {kind}")

        if fh.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        version = u32()
        if version not in (2, 3):
            raise ValueError(f"{path}: GGUF version {version} is not supported")
        u64()  # tensor count
        for _ in range(u64()):
            key = string()
            out[key] = value(u32())
    return out


def check_models(a_path: Path, b_path: Path, template_pinned: bool = False) -> tuple[dict[str, Any], list[str]]:
    """Before any server starts: arm A's and arm B's GGUF files are two different
    files, B (the judge, the bigger quant) is the bigger one, and both embed the
    same chat template (each server renders with its own file's template, unless
    *template_pinned*: one --chat-template-file for both). Returns (a summary per
    arm, problems)."""
    problems: list[str] = []
    summary: dict[str, Any] = {}
    stats = {}
    for arm, p in (("A", Path(a_path)), ("B", Path(b_path))):
        try:
            st = p.stat()
            meta = gguf_metadata(p)
        except (OSError, ValueError) as exc:
            problems.append(f"arm {arm} model {p}: {exc}")
            continue
        stats[arm] = st
        template = str(meta.get("tokenizer.chat_template") or "")
        summary[arm] = {"file": p.name, "bytes": st.st_size, "file_type": meta.get("general.file_type"),
                        "chat_template_sha": _sha16(template) if template else "",
                        "sampling": {k: v for k, v in sorted(meta.items()) if k.startswith("general.sampling.")}}
    if len(stats) == 2:
        sa, sb = stats["A"], stats["B"]
        if (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino):
            problems.append("arm A and arm B name the same model file")
        elif sb.st_size <= sa.st_size:
            problems.append(f"arm B's file ({summary['B']['file']}, {sb.st_size} bytes) is not bigger than arm A's "
                            f"({summary['A']['file']}, {sa.st_size} bytes): B is the bigger quant (swapped?)")
        ta, tb = summary["A"]["chat_template_sha"], summary["B"]["chat_template_sha"]
        if template_pinned:
            pass
        elif not ta or not tb:
            problems.append("a model file embeds no chat template")
        elif ta != tb:
            problems.append(f"the two files embed different chat templates ({ta} vs {tb}): the arms would "
                            "render the same messages differently; pass one --chat-template-file to both servers")
    return summary, problems


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def stratum(row: dict[str, Any]) -> str:
    return f"{row['category']}|{row['mode']}"


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional shares of *n* over strata of *sizes* by largest remainder
    (ties by name); every stratum gets one when n allows; never more than a
    stratum holds."""
    keys = sorted(k for k, v in sizes.items() if v > 0)
    total = sum(sizes[k] for k in keys)
    n = min(n, total)
    if n <= 0 or not keys:
        return {k: 0 for k in sizes}
    quota = {k: Fraction(n * sizes[k], total) for k in keys}
    alloc = {k: min(sizes[k], int(quota[k])) for k in keys}
    if n >= len(keys):
        for k in keys:
            alloc[k] = max(1, alloc[k])
    while sum(alloc.values()) > n:  # the floor of one overshot: take from the most over quota
        k = max((k for k in keys if alloc[k] > 1), key=lambda k: (alloc[k] - quota[k], k))
        alloc[k] -= 1
    while sum(alloc.values()) < n:
        k = max((k for k in keys if alloc[k] < sizes[k]), key=lambda k: (quota[k] - alloc[k], -keys.index(k)))
        alloc[k] += 1
    return {k: alloc.get(k, 0) for k in sizes}


def select_stratified(rows: list[dict[str, Any]], n: int, seed: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """*n* rows stratified by (category, mode), seeded. Within a stratum, rows of
    a task group not yet picked come first (two variants of one real prompt
    would test it twice). Returns (rows in run order, per-stratum counts)."""
    by: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        by[stratum(r)].append(r)
    alloc = allocate({k: len(v) for k, v in by.items()}, n)
    used_groups: set[str] = set()
    picks: list[dict[str, Any]] = []
    for key in sorted(by):
        pool = list(by[key])
        random.Random(f"{seed}|stratum|{key}").shuffle(pool)
        fresh = [r for r in pool if r["task_group"] not in used_groups]
        chosen: list[dict[str, Any]] = []
        for r in fresh:
            if len(chosen) == alloc[key]:
                break
            if r["task_group"] not in used_groups:
                chosen.append(r)
                used_groups.add(r["task_group"])
        for r in pool:  # not enough distinct groups: fill from the rest, in the same order
            if len(chosen) == alloc[key]:
                break
            if r not in chosen:
                chosen.append(r)
        picks += chosen
    random.Random(f"{seed}|order").shuffle(picks)
    strata = {k: {"available": len(by[k]), "picked": alloc[k]} for k in sorted(by)}
    return picks, strata


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def build_conf(a: argparse.Namespace) -> dict[str, Any]:
    sampling = dict(DEFAULT_SAMPLING)
    given = tg.sampling_from_args(a)
    sampling.update(given)
    for flag, value in (("--model-name", a.model_name), ("--teacher-id", a.teacher_id),
                        ("--judge-id", a.judge_id), ("--judge-model", a.judge_model)):
        if tg.has_max_token(value):
            raise InputError(f"{flag} {value!r} names a Max model; this script never uses one")
    base = tg.load_pack_prompt(a.pack) or tg.DEFAULT_SYSTEM_PROMPT.format(date=dt.date.today().isoformat())
    if a.concurrency < 1:
        raise InputError("--concurrency must be at least 1")
    return {
        "schema": SETTINGS_SCHEMA, "generator": tg.GENERATOR_VERSION,
        "model_name": a.model_name, "teacher_id": a.teacher_id, "judge_id": a.judge_id or a.teacher_id,
        "judge_model": a.judge_model, "judge_seed": a.judge_seed, "judge_max_chars": a.judge_max_chars,
        "sampling": sampling, "max_tokens": a.max_tokens, "max_tool_iterations": a.max_tool_iterations,
        "tool_timeout": a.tool_timeout, "task_timeout": a.task_timeout, "request_timeout": a.request_timeout,
        "compaction_history_tokens": a.compaction_history_tokens, "concurrency": a.concurrency,
        "jail": a.jail, "min_task_seconds": a.min_task_seconds, "system_prompt": base,
        "allow_lan": bool(a.allow_lan_endpoints),
        "labels": {"A": a.label_a, "B": a.label_b}, "threshold_pts": THRESHOLD_PTS,
        "max_consecutive_errors": a.max_consecutive_errors,
    }


def load_conf(paths: Paths) -> dict[str, Any]:
    if not paths.settings.exists() or not paths.tasks.exists():
        raise InputError(f"{paths.out} has no selection: run `select` first")
    conf = json.loads(paths.settings.read_text(encoding="utf-8"))
    if conf.get("schema") != SETTINGS_SCHEMA:
        raise InputError(f"{paths.settings}: not a {SETTINGS_SCHEMA} file")
    if conf.get("generator") != tg.GENERATOR_VERSION:
        raise InputError(f"{paths.settings} was written by {conf.get('generator')}, this is "
                         f"{tg.GENERATOR_VERSION}: start a new --out")
    return conf


def load_tasks(paths: Paths) -> list[Any]:
    tasks = []
    for line in paths.tasks.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)["task"]
            task = tg.GenTask(**d)
            task.validate()
            if task.origin != tg.ORIGIN_TASKSET:
                raise InputError(f"{paths.tasks}: task {task.id} is not a real-pattern task")
            tasks.append(task)
    return tasks


def gen_settings(conf: dict[str, Any], paths: Paths, url: str) -> Any:
    """teacher_gen Settings for one arm: everything from ab_settings.json, only the URL differs."""
    return tg.Settings(
        out=paths.out, name="teacher-ab", teacher_url=url, model_name=conf["model_name"],
        judge_model=conf["judge_model"], judge_seed=conf["judge_seed"], judge_max_chars=conf["judge_max_chars"],
        allow_lan=conf["allow_lan"], concurrency=conf["concurrency"], min_task_seconds=conf["min_task_seconds"],
        max_tokens=conf["max_tokens"], max_tool_iterations=conf["max_tool_iterations"],
        tool_timeout=conf["tool_timeout"], task_timeout=conf["task_timeout"],
        request_timeout=conf["request_timeout"], compaction_history_tokens=conf["compaction_history_tokens"],
        sandbox_root=paths.sandbox, base_system_prompt=conf["system_prompt"], progress=False,
        jail=conf["jail"], teacher_id=conf["teacher_id"], judge_id=conf["judge_id"],
        task_source=tg.ORIGIN_TASKSET, sampling=dict(conf["sampling"]))


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def cmd_select(a: argparse.Namespace) -> int:
    paths = Paths(a.out)
    conf = build_conf(a)
    taskset = Path(a.tasks).expanduser().resolve()
    rows = tg.read_taskset_rows(str(taskset))
    if a.n < 1:
        raise InputError("--n must be at least 1")
    picks, strata = select_stratified(rows, a.n, a.seed)
    selection = {
        "taskset": taskset.name, "taskset_sha256": file_sha256(taskset), "taskset_rows": len(rows),
        "seed": a.seed, "n": len(picks), "ids": [r["id"] for r in picks],
        "task_groups": len({r["task_group"] for r in picks}), "strata": strata,
        "flags": {f: sum(1 for r in picks if r["flags"][f]) for f in ("suppress_tools", "inject_errors",
                                                                       "force_compaction")},
    }
    if paths.selection.exists():
        old = json.loads(paths.selection.read_text(encoding="utf-8"))
        old_conf = json.loads(paths.settings.read_text(encoding="utf-8")) if paths.settings.exists() else {}
        same_conf = {k: v for k, v in old_conf.items() if k != "system_prompt"} == \
            {k: v for k, v in conf.items() if k != "system_prompt"}
        if old.get("ids") == selection["ids"] and old.get("taskset_sha256") == selection["taskset_sha256"] \
                and same_conf:
            print(json.dumps({"selection": str(paths.selection), "n": len(picks), "status": "already selected"}))
            return 0
        raise InputError(f"{paths.out} already holds another selection or other settings; use a new --out")
    paths.out.mkdir(parents=True, exist_ok=True)
    os.chmod(paths.out, 0o700)
    tg.isolate_state_dirs(paths.work)
    tasks = tg.tasks_from_taskset_rows(picks, Path(a.repo).expanduser())
    kinds = collections.Counter(t.kind for t in tasks)
    selection["kinds"] = dict(sorted(kinds.items()))
    tmp = paths.tasks.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps({"order": i, "row": r, "task": t.to_dict()}, ensure_ascii=False) + "\n"
                           for i, (r, t) in enumerate(zip(picks, tasks))), encoding="utf-8")
    os.replace(tmp, paths.tasks)
    _write_json(paths.settings, conf)
    _write_json(paths.selection, selection)
    print(json.dumps({"selection": str(paths.selection), "n": len(picks), "strata": strata,
                      "kinds": selection["kinds"], "settings_sha": settings_sha(conf)}, indent=1))
    return 0


# ---------------------------------------------------------------------------
# arm
# ---------------------------------------------------------------------------


async def _preflight(url: str, pattern: str, role: str, timeout: float) -> str:
    try:
        served = await tg.preflight(url, timeout)
        return tg.check_model_id(served, pattern, role)
    except tg.EndpointError as exc:
        raise ServerError(str(exc)) from exc


def _until_reached(until: float | None, margin: float) -> bool:
    return until is not None and time.time() > until - margin


async def _one_task(task: Any, settings: Any, conf: dict[str, Any], arm: str, worker: int) -> dict[str, Any] | None:
    """Run one task, verify it deterministically, never judge. None: the server
    failed (transport): not recorded, retried by the next run."""
    out = await tg.run_task(task, settings, worker)
    try:
        if out.transport_error:
            log.warning("arm %s %s: transport error %s", arm, task.id, out.error[:200])
            return None
        base = {"tool_success": False, "predicates_pass": False, "n_rows": 0, "served_models": []}
        if out.rejection:
            det_ok, reasons, record = False, [out.rejection], dict(base)
        else:
            det_ok, reasons, record = tg.deterministic_checks(out, settings)
            base.update(tool_success=record["tool_success"], predicates_pass=record["predicates"]["pass"],
                        served_models=record["served_models"])
            if det_ok:
                base["n_rows"] = len(tg.rows_for(out, settings, dict(record, verified=True)))
                if not base["n_rows"]:
                    det_ok, reasons = False, ["no_rows"]
        return {
            **base, "det_ok": det_ok, "reasons": reasons,
            "transcript": tg.judge_transcript(out, conf["judge_max_chars"]),
            "teacher_calls": len(out.calls), "tokens_in": sum(c.input_tokens for c in out.calls),
            "tokens_out": sum(c.output_tokens for c in out.calls), "duration_s": round(out.duration_s, 1),
            "compactions": out.compactions, "injections_fired": int(out.env.injections_fired),
        }
    finally:
        shutil.rmtree(out.env.root, ignore_errors=True)


async def run_arm(paths: Paths, conf: dict[str, Any], arm: str, url: str, until: float | None) -> dict[str, Any]:
    settings = gen_settings(conf, paths, url)
    try:
        endpoint = settings.teacher_endpoint
    except tg.EndpointError as exc:
        raise InputError(str(exc)) from exc
    served = await _preflight(endpoint, conf["teacher_id"], "teacher", min(30.0, conf["request_timeout"]))
    settings.teacher_served = served
    server = await server_props(endpoint, min(30.0, conf["request_timeout"]))
    path = paths.arm(arm)
    done = {}
    for r in read_records(path):
        done[r["task_id"]] = r
    olds = {r.get("served") for r in done.values()}
    if olds and olds != {served}:
        raise InputError(f"arm {arm} was started on {sorted(map(str, olds))}; this server serves {served!r}")
    sha = settings_sha(conf)
    if any(r.get("settings_sha") != sha for r in done.values()):
        raise InputError(f"{path} holds records made with other settings")
    # the same server as this arm's earlier records, and like for like with the other arm's
    for r in done.values():
        if (r.get("server") or {}).get("model_path") != server["model_path"] \
                or [k for k in SERVER_SAME_KEYS if (r.get("server") or {}).get(k) != server.get(k)]:
            raise InputError(f"arm {arm} was started on another server ({(r.get('server') or {}).get('model_file')}); "
                             f"this one serves {server['model_file']}")
    other = next((r.get("server") for r in read_records(paths.arm("B" if arm == "A" else "A"))), None)
    if other is not None:
        diff = server_differences(server, other)
        if diff:
            raise InputError(f"arm {arm}'s server is not like for like with the other arm's: {', '.join(diff)} "
                             f"({server['model_file']} vs {other.get('model_file')})")
    tasks = [t for t in load_tasks(paths) if t.id not in done]
    queue = collections.deque(tasks)
    state = {"stop": "", "errors": 0, "ran": 0}
    margin = conf["min_task_seconds"]

    async def worker(wid: int) -> None:
        while queue and not state["stop"]:
            if _until_reached(until, margin):
                state["stop"] = "until"
                return
            task = queue.popleft()
            budget = conf["task_timeout"]
            if until is not None:
                budget = min(budget, until - time.time())
            t0 = time.time()
            try:
                rec = await asyncio.wait_for(_one_task(task, settings, conf, arm, wid), budget)
            except asyncio.TimeoutError:
                if until is not None and time.time() >= until - 1:
                    state["stop"] = "until"
                    return
                rec = {"det_ok": False, "reasons": ["task_timeout"], "tool_success": False,
                       "predicates_pass": False, "n_rows": 0, "served_models": [], "transcript": "",
                       "duration_s": round(time.time() - t0, 1)}
            except Exception as exc:  # noqa: BLE001 - a harness fault in one task: retried next run
                log.warning("arm %s %s: %s: %s", arm, task.id, type(exc).__name__, str(exc)[:200])
                rec = None
            if rec is None:
                state["errors"] += 1
                if state["errors"] >= conf["max_consecutive_errors"]:
                    state["stop"] = "consecutive_errors"
                continue
            state["errors"] = 0
            state["ran"] += 1
            row = {"schema": RECORD_SCHEMA, "arm": arm, "label": conf["labels"][arm], "task_id": task.id,
                   "task_group": task.meta.get("task_id"), "category": task.domain, "mode": task.mode,
                   "kind": task.kind, "url": url, "served": served, "settings_sha": sha, "server": server,
                   "sampling": dict(sorted(conf["sampling"].items())),
                   "finished_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), **rec}
            _append(path, row)
            print(f"[arm {arm}] {task.id} {'ok' if rec['det_ok'] else ','.join(rec['reasons'])} "
                  f"({rec.get('duration_s', '-')}s)", file=sys.stderr, flush=True)

    await asyncio.gather(*(worker(i) for i in range(conf["concurrency"])))
    left = len(load_tasks(paths)) - len({r["task_id"] for r in read_records(path)})
    return {"arm": arm, "served": served, "ran": state["ran"], "left": left,
            "stop": state["stop"] or ("done" if not left else "tasks_left")}


def cmd_arm(a: argparse.Namespace) -> int:
    paths = Paths(a.out)
    conf = load_conf(paths)
    tg.isolate_state_dirs(paths.work)
    res = asyncio.run(run_arm(paths, conf, a.arm, a.url, resolve_until(a.until)))
    print(json.dumps(res))
    if res["stop"] == "consecutive_errors":
        return 3
    return 0 if res["left"] == 0 else 1


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------


async def run_judge(paths: Paths, conf: dict[str, Any], judge_url: str, until: float | None) -> dict[str, Any]:
    import httpx

    from prometheus.api.turn_errors import classify_turn_error

    settings = gen_settings(conf, paths, judge_url)
    settings.judge_url = judge_url
    try:
        endpoint = settings.judge_endpoint
    except tg.EndpointError as exc:
        raise InputError(str(exc)) from exc
    served = await _preflight(endpoint, conf["judge_id"], "judge", min(30.0, conf["request_timeout"]))
    server = await server_props(endpoint, min(30.0, conf["request_timeout"]))
    # the plan: every trace of both arms is judged by arm B's model (the bigger quant),
    # so neither arm grades itself in its own run; IQ3 grading both is refused
    b_recs = read_records(paths.arm("B"))
    if not b_recs:
        raise InputError("arm B has no records yet: the judge is arm B's model, so judging runs after arm B")
    b_served = {r.get("served") for r in b_recs}
    b_files = {(r.get("server") or {}).get("model_path") for r in b_recs}
    if b_served != {served} or b_files != {server["model_path"]}:
        raise InputError(f"this judge serves {served!r} from {server['model_file']}; the judge must be arm B's "
                         f"model ({sorted(map(str, b_served))} from "
                         f"{sorted(Path(str(f)).name for f in b_files)})")
    judged = {(r["arm"], r["task_id"]): r for r in read_records(paths.judge)}
    olds = {r.get("judge_served") for r in judged.values()}
    if olds and olds != {served}:
        raise InputError(f"the traces were judged by {sorted(map(str, olds))}; this judge serves {served!r}. "
                         "Every trace is judged by one judge.")
    todo = []
    for arm in ARMS:
        for r in read_records(paths.arm(arm)):
            if (arm, r["task_id"]) not in judged and r.get("transcript"):
                todo.append(r)
    queue = collections.deque(todo)
    state = {"stop": "", "errors": 0, "ran": 0}

    async def worker() -> None:
        while queue and not state["stop"]:
            if _until_reached(until, JUDGE_MARGIN_S):
                state["stop"] = "until"
                return
            r = queue.popleft()
            try:
                if until is None:
                    rec = await tg.judge_text(r["transcript"], settings)
                else:  # a hard end: a call still running at --until is cancelled, not recorded
                    rec = await asyncio.wait_for(tg.judge_text(r["transcript"], settings),
                                                 max(0.0, until - time.time()))
            except asyncio.TimeoutError:
                state["stop"] = "until"
                return
            except httpx.HTTPStatusError as exc:
                kind = str(classify_turn_error(exc).get("kind") or "unknown")
                if kind in tg.RETRY_KINDS:
                    rec = None
                else:  # the judge refuses this transcript (too long, say): it always will
                    rec = {"verdict": "fail", "pass": False, "error": f"judge_{kind}",
                           "reason": f"HTTP {exc.response.status_code}"}
            except (httpx.HTTPError, OSError, ValueError) as exc:
                log.warning("judge %s/%s: %s", r["arm"], r["task_id"], type(exc).__name__)
                rec = None
            if rec is None:
                state["errors"] += 1
                if state["errors"] >= conf["max_consecutive_errors"]:
                    state["stop"] = "consecutive_errors"
                continue
            state["errors"] = 0
            state["ran"] += 1
            rec["kind"] = "ab-judge"
            _append(paths.judge, {"schema": RECORD_SCHEMA, "arm": r["arm"], "task_id": r["task_id"],
                                  "judge_url": judge_url, "judge_served": served,
                                  "judge_model_path": server["model_path"], "judge": rec})

    await asyncio.gather(*(worker() for _ in range(conf["concurrency"])))
    judged = {(r["arm"], r["task_id"]) for r in read_records(paths.judge)}
    left = sum(1 for arm in ARMS for r in read_records(paths.arm(arm))
               if r.get("transcript") and (arm, r["task_id"]) not in judged)
    return {"judge_served": served, "ran": state["ran"], "left": left,
            "stop": state["stop"] or ("done" if not left else "traces_left")}


def cmd_judge(a: argparse.Namespace) -> int:
    paths = Paths(a.out)
    conf = load_conf(paths)
    tg.isolate_state_dirs(paths.work)
    res = asyncio.run(run_judge(paths, conf, a.judge_url, resolve_until(a.until)))
    print(json.dumps(res))
    if res["stop"] == "consecutive_errors":
        return 3
    return 0 if res["left"] == 0 else 1


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def task_passes(rec: dict[str, Any], judge: dict[str, Any] | None) -> bool:
    """teacher_gen's rule: deterministic checks, at least one row, and a judge
    pass from a judge that is not a Max model."""
    return bool(rec.get("det_ok") and rec.get("n_rows", 0) > 0 and judge is not None
                and judge.get("pass") is True and not tg.has_max_token(judge.get("served_model")))


def decide(k_a: int, k_b: int, n: int, threshold_pts: int = THRESHOLD_PTS) -> str:
    """The operator's rule on the point estimates, exactly: IQ3 (A) more than
    *threshold_pts* below Q4 (B) -> Q4."""
    if n <= 0:
        return DECISION_INCOMPLETE
    return DECISION_Q4 if Fraction(100 * (k_b - k_a), n) > threshold_pts else DECISION_IQ3


def _pct(k: int, n: int) -> float | None:
    return round(100.0 * k / n, 1) if n else None


def build_report(paths: Paths, conf: dict[str, Any], min_pairs: int | None = None) -> dict[str, Any]:
    score = _load_sibling("distill_score", "score.py")
    selection = json.loads(paths.selection.read_text(encoding="utf-8"))
    ids = list(selection["ids"])
    recs = {arm: {r["task_id"]: r for r in read_records(paths.arm(arm))} for arm in ARMS}
    judges = {(r["arm"], r["task_id"]): r for r in read_records(paths.judge)}
    problems: list[str] = []
    shas = {r.get("settings_sha") for arm in ARMS for r in recs[arm].values()}
    if len(shas) > 1 or (shas and shas != {settings_sha(conf)}):
        problems.append("the arms' request settings differ")
    served = {arm: sorted({str(r.get("served")) for r in recs[arm].values()}) for arm in ARMS}
    if any(len(v) > 1 for v in served.values()):
        problems.append("an arm was served by more than one model id")
    if served["A"] and served["A"] == served["B"]:
        problems.append(f"both arms report the same served model {served['A'][0]!r}")
    # the servers behind the ids (aliases can differ on one file): the model file, the
    # per-slot context, the chat template and the server's own sampler defaults
    servers = {arm: [json.loads(x) for x in sorted({json.dumps(r.get("server"), sort_keys=True)
                                                    for r in recs[arm].values()})] for arm in ARMS}
    for arm in ARMS:
        if any(s is None for s in servers[arm]):
            problems.append(f"arm {arm}'s server settings (/props) were not recorded")
        elif len(servers[arm]) > 1:
            problems.append(f"arm {arm} ran on more than one server")
    if all(len(servers[arm]) == 1 and servers[arm][0] for arm in ARMS):
        diff = server_differences(servers["A"][0], servers["B"][0])
        if diff:
            problems.append(f"the arms' servers differ: {', '.join(diff)}")
    judge_ids = sorted({str(r.get("judge_served")) for r in judges.values()})
    if len(judge_ids) > 1:
        problems.append("the traces were judged by more than one judge")
    # the plan: all results judged by arm B's model (the bigger quant), never by arm A's
    if judge_ids and served["B"] and judge_ids != served["B"]:
        problems.append(f"the judge ({', '.join(judge_ids)}) is not arm B's model ({', '.join(served['B'])})")
    judge_files = sorted({str(r.get("judge_model_path")) for r in judges.values()})
    b_files = sorted({str(s.get("model_path")) for s in servers["B"] if s})
    if judges and b_files and judge_files != b_files:
        problems.append(f"the judge's model file ({', '.join(Path(f).name for f in judge_files)}) is not "
                        f"arm B's ({', '.join(Path(f).name for f in b_files)})")

    def judge_of(arm: str, tid: str) -> dict[str, Any] | None:
        j = judges.get((arm, tid))
        return j["judge"] if j else None

    def finished(arm: str, tid: str) -> bool:
        r = recs[arm].get(tid)
        return r is not None and (not r.get("transcript") or (arm, tid) in judges)

    pairs = [tid for tid in ids if finished("A", tid) and finished("B", tid)]
    if pairs and not judges:  # every pair finished without a transcript: nothing was graded
        problems.append("no trace was judged")
    outcome = {arm: {tid: task_passes(recs[arm][tid], judge_of(arm, tid)) for tid in pairs} for arm in ARMS}
    a = sum(1 for t in pairs if outcome["A"][t] and outcome["B"][t])
    b = sum(1 for t in pairs if outcome["B"][t] and not outcome["A"][t])
    c = sum(1 for t in pairs if outcome["A"][t] and not outcome["B"][t])
    d = len(pairs) - a - b - c
    # base = Q4 (B), candidate = IQ3 (A): score.py's orientation is candidate minus base
    table = score.PairedCounts(a, b, c, d)
    paired = score.paired_report(table, "teacher-ab") if pairs else None
    n = len(pairs)
    k = {"A": a + c, "B": a + b}
    arms: dict[str, Any] = {}
    for arm in ARMS:
        rows = [recs[arm][t] for t in pairs]
        cats: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"n": 0, "pass": 0})
        for t in pairs:
            cat = recs[arm][t]["category"]
            cats[cat]["n"] += 1
            cats[cat]["pass"] += int(outcome[arm][t])
        reasons = collections.Counter(x for r in rows for x in r.get("reasons", []))
        jp = sum(1 for t in pairs if (judge_of(arm, t) or {}).get("pass") is True)
        arms[arm] = {
            "label": conf["labels"][arm], "served": served[arm][0] if len(served[arm]) == 1 else served[arm],
            "recorded": len(recs[arm]), "pairs": n, "pass": k[arm], "pass_rate_pts": _pct(k[arm], n),
            "tool_success": sum(1 for r in rows if r.get("tool_success")),
            "predicates_pass": sum(1 for r in rows if r.get("predicates_pass")),
            "deterministic_pass": sum(1 for r in rows if r.get("det_ok")),
            "judge_pass": jp,
            "fail_reasons": dict(reasons.most_common()),
            "by_category": {cat: dict(v, pass_rate_pts=_pct(v["pass"], v["n"])) for cat, v in sorted(cats.items())},
            "tokens_out": sum(int(r.get("tokens_out") or 0) for r in rows),
            "duration_s": round(sum(float(r.get("duration_s") or 0) for r in rows), 1),
        }
    need = len(ids) if min_pairs is None else min_pairs
    if problems:
        decision = DECISION_INVALID
    elif n < need or n == 0:
        decision = DECISION_INCOMPLETE
    else:
        decision = decide(k["A"], k["B"], n, conf["threshold_pts"])
    judged_n = sum(1 for arm in ARMS for r in recs[arm].values() if (arm, r["task_id"]) in judges)
    return {
        "schema": REPORT_SCHEMA, "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection": {k2: selection[k2] for k2 in ("taskset", "taskset_sha256", "seed", "n", "ids", "strata",
                                                    "flags", "kinds") if k2 in selection},
        "settings": {"sha": settings_sha(conf), "sampling": conf["sampling"], "max_tokens": conf["max_tokens"],
                     "model_name": conf["model_name"], "concurrency": conf["concurrency"],
                     "compaction_history_tokens": conf["compaction_history_tokens"],
                     "identical_across_arms": not any(p.startswith(("the arms' request settings",
                                                                    "the arms' servers differ")) for p in problems),
                     "servers": {arm: servers[arm][0] if len(servers[arm]) == 1 else servers[arm] for arm in ARMS}},
        "judge": {"served": judge_ids[0] if len(judge_ids) == 1 else judge_ids, "records": len(judges),
                  "judged": judged_n, "temperature": 0, "seed": conf["judge_seed"],
                  "model_file": [Path(f).name for f in judge_files],
                  "is_arm_b_model": bool(judge_ids) and judge_ids == served["B"] and judge_files == b_files},
        "arms": arms,
        "paired": {"orientation": "A minus B (IQ3 minus Q4); a both pass, b only B, c only A, d neither",
                   "table": {"a": a, "b": b, "c": c, "d": d}, **(paired or {})},
        "decision": {
            "verdict": decision,
            "rule": f"use Q4_K_M (less context) if A's pass rate is more than {conf['threshold_pts']} points "
                    "below B's; else keep IQ3 (point estimates, exact)",
            "threshold_pts": conf["threshold_pts"],
            "a_minus_b_pts": round(100.0 * (k["A"] - k["B"]) / n, 2) if n else None,
            "pairs": n, "min_pairs": need, "problems": problems,
        },
    }


def summary_lines(rep: dict[str, Any]) -> list[str]:
    """Five lines for Telegram."""
    A, B = rep["arms"]["A"], rep["arms"]["B"]
    dec = rep["decision"]
    n = dec["pairs"]
    p = rep["paired"]
    head = (f"Teacher A/B: {dec['verdict'].upper()}" if dec["verdict"] in (DECISION_INCOMPLETE, DECISION_INVALID)
            else f"Teacher A/B decision: {dec['verdict']}")
    if dec["verdict"] == DECISION_INCOMPLETE:
        head += f" ({n} of {len(rep['selection']['ids'])} pairs done; rerun to resume)"
    elif dec["verdict"] == DECISION_INVALID:
        head += f" ({'; '.join(dec['problems'])})"

    def rate(arm: dict[str, Any]) -> str:
        return f"{arm['label']} {arm['pass']}/{n}" + (f" ({arm['pass_rate_pts']}%)" if n else "")

    ci = p.get("ci_pts")
    diff = (f"{A['label']} - {B['label']}: {p.get('diff_pts'):+.1f} pts, 95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}] "
            f"(paired Newcombe); McNemar p={p.get('mcnemar_exact_p')}") if ci else "no finished pairs yet"
    t = p["table"]
    return [
        head,
        f"Pass: {rate(A)} vs {rate(B)}",
        diff,
        f"Pairs: both {t['a']}, {B['label']} only {t['b']}, {A['label']} only {t['c']}, neither {t['d']}; "
        f"tool ok {A['tool_success']} vs {B['tool_success']}, judge pass {A['judge_pass']} vs {B['judge_pass']}",
        f"Rule: {A['label']} more than {dec['threshold_pts']} pts below {B['label']} -> {DECISION_Q4}, "
        f"else {DECISION_IQ3}; one judge: {rep['judge']['served']}",
    ]


def cmd_report(a: argparse.Namespace) -> int:
    paths = Paths(a.out)
    conf = load_conf(paths)
    rep = build_report(paths, conf, a.min_pairs)
    _write_json(paths.report, rep)
    lines = summary_lines(rep)
    paths.summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if rep["decision"]["verdict"] in (DECISION_Q4, DECISION_IQ3) else 1


def phase_status(paths: Paths) -> dict[str, Any]:
    """What is done: tasks recorded per arm, traces judged (for the shell driver)."""
    if not paths.selection.exists():
        return {"selected": 0}
    ids = json.loads(paths.selection.read_text(encoding="utf-8"))["ids"]
    recs = {arm: {r["task_id"]: r for r in read_records(paths.arm(arm))} for arm in ARMS}
    judged = {(r["arm"], r["task_id"]) for r in read_records(paths.judge)}
    to_judge = [(arm, t) for arm in ARMS for t, r in recs[arm].items() if r.get("transcript")]
    out: dict[str, Any] = {"selected": len(ids)}
    for arm in ARMS:
        out[f"arm-{arm}"] = {"done": sum(1 for t in ids if t in recs[arm]), "total": len(ids)}
    out["judge"] = {"done": sum(1 for k in to_judge if k in judged), "total": len(to_judge),
                    "arms_complete": all(out[f"arm-{arm}"]["done"] == len(ids) for arm in ARMS)}
    return out


def cmd_time_left(a: argparse.Namespace) -> int:
    """Seconds until --until: exit 0 when at least --min-seconds are left, 1 when
    not, 2 when --until does not parse or (``--max-seconds``) lies further ahead
    than that. Prints ``{"seconds_left", "until_iso"}``, or with ``--iso`` only the
    absolute time: the shell driver resolves an HH:MM ONCE and passes that on (an
    HH:MM read again after it has passed would be tomorrow)."""
    until = resolve_until(a.until)
    if until is None:
        raise InputError("--until is empty")
    left = until - time.time()
    if a.max_seconds is not None and left > a.max_seconds:
        raise InputError(f"--until {a.until!r} is {iso(until)}, {left / 3600:.1f} h away: more than "
                         f"--max-seconds {a.max_seconds:g} (a time already past today?)")
    print(iso(until) if a.iso else json.dumps({"seconds_left": round(left, 1), "until_iso": iso(until)}))
    return 0 if left >= a.min_seconds else 1


def cmd_models(a: argparse.Namespace) -> int:
    """Checks the two GGUF files before any server starts (``check_models``):
    exit 0 when they are fit to compare, 2 otherwise (problems on stderr)."""
    summary, problems = check_models(Path(a.a).expanduser(), Path(a.b).expanduser(), a.template_pinned)
    print(json.dumps({"models": summary, "problems": problems}, indent=1))
    for prob in problems:
        print(f"error: {prob}", file=sys.stderr)
    return 2 if problems else 0


def cmd_status(a: argparse.Namespace) -> int:
    """Prints the phase status; with --need, exit 0 only when that phase is complete."""
    st = phase_status(Paths(a.out))
    print(json.dumps(st))
    if not a.need:
        return 0
    if a.need == "judge":
        j = st.get("judge") or {}
        return 0 if j.get("arms_complete") and j.get("done") == j.get("total") else 1
    arm = st.get(a.need) or {}
    return 0 if arm and arm["done"] == arm["total"] and arm["total"] > 0 else 1


# ---------------------------------------------------------------------------
# run: everything, servers up at once
# ---------------------------------------------------------------------------


def cmd_run(a: argparse.Namespace) -> int:
    # --until is resolved once: every phase gets the same absolute end
    until = resolve_until(a.until)
    until_iso = iso(until) if until is not None else None
    rc = cmd_select(a)
    if rc:
        return rc
    for arm, url in (("A", a.url_a), ("B", a.url_b)):
        rc = cmd_arm(argparse.Namespace(out=a.out, arm=arm, url=url, until=until_iso))
        if rc:
            return rc
    rc = cmd_judge(argparse.Namespace(out=a.out, judge_url=a.judge_url, until=until_iso))
    if rc:
        return rc
    return cmd_report(argparse.Namespace(out=a.out, min_pairs=a.min_pairs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _select_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tasks", required=True, help="a taskset_gen.py real-pattern taskset (JSONL)")
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--seed", default=DEFAULT_SEED)
    p.add_argument("--repo", default=str(_DIR.parent.parent), help="repo docs for force_compaction padding")
    p.add_argument("--pack", help="build.py request pack: its system prompt (date fixed now)")
    p.add_argument("--label-a", default=DEFAULT_LABELS["A"])
    p.add_argument("--label-b", default=DEFAULT_LABELS["B"])
    p.add_argument("--model-name", default=tg.DEFAULT_MODEL_NAME)
    p.add_argument("--teacher-id", default=tg.DEFAULT_TEACHER_ID, help="glob both arms' /v1/models id must match")
    p.add_argument("--judge-id", help="glob the judge's id must match (default --teacher-id)")
    p.add_argument("--judge-model")
    p.add_argument("--judge-seed", type=int, default=tg.DEFAULT_JUDGE_SEED)
    p.add_argument("--judge-max-chars", type=int, default=30000)
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-k", dest="top_k", type=int)
    p.add_argument("--top-p", dest="top_p", type=float)
    p.add_argument("--min-p", dest="min_p", type=float)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-tool-iterations", type=int, default=12)
    p.add_argument("--tool-timeout", type=float, default=60.0)
    p.add_argument("--task-timeout", type=float, default=900.0)
    p.add_argument("--request-timeout", type=float, default=300.0)
    p.add_argument("--compaction-history-tokens", type=int, default=1500)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--min-task-seconds", type=float, default=120.0)
    p.add_argument("--max-consecutive-errors", type=int, default=5)
    p.add_argument("--jail", choices=tg.JAIL_MODES, default="auto")
    p.add_argument("--allow-lan-endpoints", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="pick the tasks and fix the request settings")
    s.add_argument("--out", required=True)
    _select_args(s)
    ar = sub.add_parser("arm", help="run one arm's tasks on one server (no judging)")
    ar.add_argument("--out", required=True)
    ar.add_argument("--arm", required=True, choices=ARMS)
    ar.add_argument("--url", required=True)
    ar.add_argument("--until", help="HH:MM (local, next occurrence; refused when more than 12 h ahead) or ISO "
                                    "time: a hard end")
    j = sub.add_parser("judge", help="judge every recorded trace of both arms on one endpoint")
    j.add_argument("--out", required=True)
    j.add_argument("--judge-url", required=True)
    j.add_argument("--until")
    r = sub.add_parser("report", help="pass rates, the paired interval and the decision")
    r.add_argument("--out", required=True)
    r.add_argument("--min-pairs", type=int, help="pairs needed for a decision (default: every selected task)")
    st = sub.add_parser("status", help="what is done (JSON); --need: exit 0 only when that phase is complete")
    st.add_argument("--out", required=True)
    st.add_argument("--need", choices=("arm-A", "arm-B", "judge"))
    tl = sub.add_parser("time-left", help="exit 0 when at least --min-seconds are left before --until")
    tl.add_argument("--until", required=True)
    tl.add_argument("--min-seconds", type=float, default=0.0)
    tl.add_argument("--max-seconds", type=float, help="refuse (exit 2) an --until further ahead than this")
    tl.add_argument("--iso", action="store_true", help="print only the resolved absolute time")
    mo = sub.add_parser("models", help="check arm A's and arm B's GGUF files before any server starts")
    mo.add_argument("--a", required=True, help="arm A's GGUF (IQ3)")
    mo.add_argument("--b", required=True, help="arm B's GGUF (Q4_K_M, the judge: the bigger file)")
    mo.add_argument("--template-pinned", action="store_true",
                    help="both servers get one --chat-template-file: the embedded templates may differ")
    run = sub.add_parser("run", help="select, both arms, judge and report against servers up at once")
    run.add_argument("--out", required=True)
    _select_args(run)
    run.add_argument("--url-a", required=True)
    run.add_argument("--url-b", required=True)
    run.add_argument("--judge-url", required=True)
    run.add_argument("--until")
    run.add_argument("--min-pairs", type=int)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    a = parse_args(argv)
    try:
        if getattr(a, "out", None):
            out = Path(a.out).expanduser().resolve()
            live = (Path.home() / ".prometheus").resolve()
            if out == live or live in out.parents:
                raise InputError(f"--out must not be inside {live}")
        return {"select": cmd_select, "arm": cmd_arm, "judge": cmd_judge, "report": cmd_report,
                "status": cmd_status, "time-left": cmd_time_left, "models": cmd_models, "run": cmd_run}[a.cmd](a)
    except ServerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (InputError, ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
