"""Every normalization the parity diff applies — and why each one is safe.

A normalization is a place a regression can hide: whatever a rule rewrites,
the diff can no longer see. So the list is closed and explicit. A rule earns a
place here only by being SHOWN to vary between two runs of unchanged code (or
between two machines running the same code), and each one says what it
replaces and what that costs. ``python scripts/parity_harness.py
normalizations`` prints this list; the PR that introduced the harness carries
it verbatim.

Two families:

* REQUEST rules rewrite the JSON body the daemon sends to the model, before it
  is fingerprinted. They are string rewrites inside JSON string values.
* OBSERVABLE rules rewrite the rows and files the daemon persisted. They are
  keyed by (store, table, column) and replace a value with a placeholder;
  ``None`` stays ``None`` so presence/absence is still diffed.

Identifiers are replaced with an ORDINAL placeholder (``<uuid:3>``: the third
distinct value seen), not a constant, so a change in which row references
which — a checkpoint pointing at the wrong turn, a tool result filed under the
wrong call — still shows up.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Rule:
    name: str
    family: str          # "request" | "observable"
    replaces: str        # what the rule rewrites, in words
    why: str             # the evidence that it varies between runs of the same code
    cost: str            # what the diff can no longer see because of it


# ---------------------------------------------------------------------------
# Request rules (string rewrites inside JSON string values)
# ---------------------------------------------------------------------------

# A coding run's clone directory: <parent>/coding/<task_id>-<int(time.time())>.
CLONE_EPOCH = re.compile(r"(/coding/[A-Za-z0-9_.]+-)(1[0-9]{9})\b")


@dataclass(frozen=True)
class TextRule(Rule):
    pattern: re.Pattern = re.compile("")
    replacement: str = ""


REQUEST_RULES: list[TextRule] = [
    TextRule(
        name="env-date",
        family="request",
        replaces="the `- Date: …` line of the system prompt's # Environment section",
        why="the daemon stamps today's date; a trace recorded on one day replays on another",
        cost="a change to how the date line is FORMATTED is invisible (its presence is not)",
        pattern=re.compile(r"(?m)^(- Date: ).*$"),
        replacement=r"\1<date>",
    ),
    TextRule(
        name="env-os",
        family="request",
        replaces="the `- OS: …` line (kernel release string)",
        why="host-dependent by construction: distro.version() / platform.release() "
            "(context/environment.py) — the recording host and a CI runner differ",
        cost="none for the daemon — it only reports the host's value",
        pattern=re.compile(r"(?m)^(- OS: ).*$"),
        replacement=r"\1<os>",
    ),
    TextRule(
        name="env-python",
        family="request",
        replaces="the `- Python: …` line",
        why="the traces were recorded under 3.11.15 and CI replays under 3.12; the "
            "value is platform.python_version()",
        cost="none for the daemon — it only reports the interpreter's value",
        pattern=re.compile(r"(?m)^(- Python: ).*$"),
        replacement=r"\1<python>",
    ),
    TextRule(
        name="coding-clone-epoch",
        family="request",
        replaces="the epoch-seconds suffix of a coding run's clone directory "
                 "(.../coding/<task_id>-<epoch>)",
        why="the clone is named f'{task_id}-{int(time.time())}' (prometheus/__main__.py), "
            "so it changes every run; the model is told the path in its first message",
        cost="a clone placed under a DIFFERENT parent, or a different task id, is still "
             "visible; only the ten digits are hidden",
        pattern=CLONE_EPOCH,
        replacement=r"\1<epoch>",
    ),
]


def _rewrite_strings(obj: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, list):
        return [_rewrite_strings(v, fn) for v in obj]
    if isinstance(obj, dict):
        return {k: _rewrite_strings(v, fn) for k, v in obj.items()}
    return obj


def _apply_text_rules(text: str) -> str:
    for rule in REQUEST_RULES:
        text = rule.pattern.sub(rule.replacement, text)
    return text


def normalize_request(body: Any) -> Any:
    """The request as the diff sees it. Pure; the input is not modified.

    Ordinals restart per request: each request is compared as a unit."""
    ords = _Ordinals()

    def fn(text: str) -> str:
        text = _apply_text_rules(text)
        for rule in REQUEST_ORDINAL_RULES:
            text = ords.sub(rule, text)
        return text
    return _rewrite_strings(body, fn)


def fingerprint(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Observable rules (persisted rows and files)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldRule(Rule):
    """Replace the value of any sqlite column / JSON key with one of ``fields``."""

    fields: frozenset = frozenset()
    placeholder: str = ""


@dataclass(frozen=True)
class PatternRule(Rule):
    """Replace every match inside any string with an ORDINAL placeholder."""

    pattern: re.Pattern = re.compile("")
    label: str = ""


@dataclass(frozen=True)
class PathRule(Rule):
    """A persisted FILE whose content is replaced (existence still compared)."""

    suffixes: tuple = ()
    placeholder: str = ""


FIELD_RULES: list[FieldRule] = [
    FieldRule(
        name="wall-clock-fields",
        family="observable",
        replaces="any column/key named timestamp, created_at, updated_at, started_at, "
                 "ended_at, deleted_at, pinned_at, read_at, last_seen_at, last_mentioned, "
                 "timestamp_iso, mtime, probed_at, ts — including keys inside JSON stored "
                 "in a text column",
        why="written from time.time() / datetime.now() (mtime: the harness creates the "
            "fixture files at run time); differs on every run by construction",
        cost="WHEN a row was written is invisible. Row ORDER is still compared (rows are "
             "dumped in rowid order), and so is whether the field is NULL",
        fields=frozenset({"timestamp", "created_at", "updated_at", "started_at", "ended_at",
                          "deleted_at", "pinned_at", "read_at", "last_seen_at",
                          "last_mentioned", "timestamp_iso", "mtime", "probed_at", "ts"}),
        placeholder="<time>",
    ),
    FieldRule(
        name="durations",
        family="observable",
        replaces="any column/key named duration_ms, latency_ms or wall_seconds",
        why="measured wall time of the operation; varies with host load (wall_seconds: "
            "a coding run's report — 25.7 s recorded against the real model, 0.2 s replayed)",
        cost="how LONG an operation took is invisible here — which is exactly what the "
             "overhead benchmark measures instead",
        fields=frozenset({"duration_ms", "latency_ms", "wall_seconds"}),
        placeholder="<ms>",
    ),
    FieldRule(
        name="node-identity",
        family="observable",
        replaces="any column/key named node_id",
        why="the node keypair is minted fresh at first boot of every isolated root",
        cost="none for a single-node run; a row stamped with the WRONG node would be missed",
        fields=frozenset({"node_id"}),
        placeholder="<node>",
    ),
]

TOOLU = PatternRule(
    name="daemon-minted-tool-use-ids",
    family="request+observable",
    replaces="tool-use ids the DAEMON mints (toolu_<12 or 32 hex>), as <toolu:N>",
    why="minted from uuid4 when a call arrives without a server id — the adapter's "
        "text extraction (enforcer.py, formatter.py) and the providers' fallback; "
        "observed differing between two replays of the compaction and coding traces. "
        "Ids the MODEL SERVER assigned are recorded and replayed verbatim, so they are "
        "not touched",
    cost="none for identity; the ordinal keeps each tool_use paired with its tool_result",
    pattern=re.compile(r"\btoolu_[0-9a-f]{12}(?:[0-9a-f]{20})?\b"),
    label="toolu",
)

REQUEST_ORDINAL_RULES: list[PatternRule] = [TOOLU]

PATTERN_RULES: list[PatternRule] = [
    TOOLU,
    PatternRule(
        name="uuid4",
        family="observable",
        replaces="uuid4 values (32-hex or dashed) anywhere, as <uuid:N> by first appearance",
        why="row ids (lcm_messages.id, subsystem_runs.id, tool_calls.id, ...) are uuid4() — "
            "random by construction",
        cost="none for identity; the ORDINAL keeps which-row-references-which comparable",
        pattern=re.compile(
            r"\b[0-9a-f]{8}-?[0-9a-f]{4}-?4[0-9a-f]{3}-?[89ab][0-9a-f]{3}-?[0-9a-f]{12}\b"),
        label="uuid",
    ),
    PatternRule(
        name="checkpoint-id",
        family="observable",
        replaces="file-checkpoint ids ({epoch_ms}-{hex6}) anywhere, as <ckpt:N>",
        why="minted from the wall clock plus uuid4 (checkpoints/store.py)",
        cost="none for identity; ordinals keep the checkpoint<->file linkage comparable",
        pattern=re.compile(r"\b1[0-9]{12}-[0-9a-f]{6}\b"),
        label="ckpt",
    ),
]

@dataclass(frozen=True)
class KeyValueRule(Rule):
    """In a key/value table, replace the VALUE of the named keys."""

    table: str = ""
    keys: frozenset = frozenset()
    placeholder: str = ""


@dataclass(frozen=True)
class ColumnRule(Rule):
    """Replace one table's named columns (values derived from a volatile input)."""

    table: str = ""
    columns: frozenset = frozenset()
    placeholder: str = ""


COLUMN_RULES: list[ColumnRule] = [
    ColumnRule(
        name="training-pair-hashes",
        family="observable",
        replaces="training.db training_pairs.id and .context_hash",
        why="both are sha256 over input that includes time.time() — the pair context's "
            "`ts` (engine/agent_loop.py::_pair_context) and, for id, the write time "
            "(learning/pair_capture.py)",
        cost="the hash itself; the pair's source, model, tool, rejected/chosen calls and "
             "context (minus ts) are all still compared",
        table="training_pairs",
        columns=frozenset({"id", "context_hash"}),
        placeholder="<hash-of-time>",
    ),
]

KEYVALUE_RULES: list[KeyValueRule] = [
    KeyValueRule(
        name="schema-meta-stamps",
        family="observable",
        replaces="telemetry.db schema_meta values for keys created_at and billing_recorded_since",
        why="both are the wall-clock moment the telemetry DB was created (first boot)",
        cost="when the DB was created; the other schema_meta keys (versions) are compared",
        table="schema_meta",
        keys=frozenset({"created_at", "billing_recorded_since"}),
        placeholder="<time>",
    ),
]

@dataclass(frozen=True)
class ColumnPatternRule(Rule):
    """An ORDINAL pattern applied only to one table (and the paths that embed it)."""

    table: str = ""
    pattern: re.Pattern = re.compile("")
    label: str = ""


COLUMN_PATTERN_RULES: list[ColumnPatternRule] = [
    ColumnPatternRule(
        name="managed-task-id",
        family="observable",
        replaces="background-task ids (<type letter><8 hex>) in tasks.db, as <task:N>",
        why="minted as f'{prefix}{uuid4().hex[:8]}' (tasks/manager.py) — random by "
            "construction; also embedded in the task's output_file path",
        cost="none for identity; ordinals keep id and output_file paired",
        table="tasks",
        pattern=re.compile(r"(?<![0-9a-z])[a-z][0-9a-f]{8}(?![0-9a-f])"),
        label="task",
    ),
]

PATH_RULES: list[PathRule] = [
    PathRule(
        name="boot-stamps",
        family="observable",
        replaces="the CONTENT of .daemon_started and of node/node.key, node/node.pub",
        why="boot wall time, and a keypair minted fresh per isolated root",
        cost="their contents; their existence is still compared",
        suffixes=(".daemon_started", "node/node.key", "node/node.pub"),
        placeholder="<boot-specific>",
    ),
    PathRule(
        name="harness-config-file",
        family="observable",
        replaces="the CONTENT of home/.prometheus/prometheus.yaml",
        why="it is the harness's INPUT (the trace carries it verbatim), rendered with this "
            "run's free ports",
        cost="a daemon WRITE to its own config file (the approval-grant writer can) would "
             "be invisible here; no recorded scenario reaches that path",
        suffixes=("home/.prometheus/prometheus.yaml",),
        placeholder="<harness-input>",
    ),
]


@dataclass(frozen=True)
class ResponseRule(Rule):
    """Replay-only: rewrite a recorded RESPONSE so it names this run's value."""

    pattern: re.Pattern = re.compile("")


RESPONSE_RULES: list[ResponseRule] = [
    ResponseRule(
        name="coding-clone-binding",
        family="response (replay only)",
        replaces="in a recorded model RESPONSE, the clone-directory epoch — rewritten to the "
                 "epoch in the request being answered",
        why="the model was told the clone path and uses it in tool arguments; the replayed "
            "run's clone has a different epoch, so the recorded answer would name a directory "
            "that does not exist and the sandbox refuses it (observed: SANDBOX VIOLATION on "
            "every call of the first coding replay)",
        cost="none that the request diff does not already show: the request carrying the "
             "epoch is itself compared (see coding-clone-epoch)",
        pattern=CLONE_EPOCH,
    ),
]


def _resplit_sse(body: str, fn: Callable[[str], str]) -> str:
    """Apply ``fn`` to each streamed CHANNEL (content, reasoning, each tool
    call's arguments) as one string, then cut it back at the original chunk
    boundaries. The model streams a path as token fragments, so no single
    chunk holds it; ``fn`` must preserve length, which keeps every chunk the
    size it was recorded at — streaming granularity is part of the replay."""
    lines = body.split("\n")
    events: list[tuple[int, dict]] = []
    for i, line in enumerate(lines):
        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
            try:
                events.append((i, json.loads(line[6:])))
            except json.JSONDecodeError:
                pass
    frags: dict[tuple, list[tuple[int, dict, str]]] = {}
    for i, ev in events:
        for choice in ev.get("choices") or []:
            delta = choice.get("delta") or {}
            for key in ("content", "reasoning_content"):
                if isinstance(delta.get(key), str):
                    frags.setdefault((key,), []).append((i, delta, key))
            for tc in delta.get("tool_calls") or []:
                fnc = tc.get("function") or {}
                if isinstance(fnc.get("arguments"), str):
                    frags.setdefault(("tool", tc.get("index", 0)), []).append((i, fnc, "arguments"))
    changed: set[int] = set()
    for parts in frags.values():
        whole = "".join(holder[key] for _, holder, key in parts)
        new = fn(whole)
        if new == whole:
            continue
        if len(new) != len(whole):
            raise ValueError("response binding must preserve length")
        pos = 0
        for i, holder, key in parts:
            n = len(holder[key])
            holder[key] = new[pos:pos + n]
            pos += n
            changed.add(i)
    for i, ev in events:
        if i in changed:
            lines[i] = "data: " + json.dumps(ev, ensure_ascii=False, separators=(",", ":"))
    return "\n".join(lines)


def bind_response(body: str, raw_request: str) -> str:
    """Apply RESPONSE_RULES: bind run-specific values in a recorded answer."""
    for rule in RESPONSE_RULES:
        current = rule.pattern.search(raw_request)
        if current is None:
            continue

        def fn(text: str, r: ResponseRule = rule, c: re.Match = current) -> str:
            return r.pattern.sub(lambda m: m.group(1) + c.group(2), text)
        body = _resplit_sse(body, fn) if "data: " in body else fn(body)
    return body


@dataclass(frozen=True)
class HostPathRule(Rule):
    pass


HOST_PATH_RULE = HostPathRule(
    name="host-paths",
    family="observable",
    replaces="the interpreter path, the venv root and the source checkout root, as "
             "<python>, <venv>, <src>",
    why="a coding run is launched with sys.executable (coding/managed.py), so the command "
        "stored in tasks.db names this host's venv; CI checks out and builds elsewhere",
    cost="a change in WHICH interpreter launches a coding run, if it stayed under the same "
         "venv, would be invisible",
)


def _host_paths() -> list[tuple[str, str]]:
    import sys
    from pathlib import Path as _P
    src = str(_P(__file__).resolve().parents[2])
    pairs = [(sys.executable, "<python>"), (str(_P(sys.executable).resolve()), "<python>"),
             (sys.prefix, "<venv>"), (src, "<src>")]
    return sorted({(a, b) for a, b in pairs if a}, key=lambda ab: -len(ab[0]))


GIT_INTERNALS = PathRule(
    name="git-internals",
    family="observable",
    replaces="everything under a .git/ directory: objects/ is dropped (its FILE NAMES are "
             "hashes), every other file's content is replaced (existence still compared)",
    why="commits made during a coding run (and the fixture commit's index) carry wall-clock "
        "stamps, so every object hash, ref and index differs between runs",
    cost="the committed TREE is not compared here — the working-tree files are, and so is "
         "the run's report (branch, diff_stat)",
    suffixes=(),
    placeholder="<git-internal>",
)


class _Ordinals:
    def __init__(self) -> None:
        self.maps: dict[str, dict[str, int]] = {}

    def sub(self, rule: PatternRule, text: str) -> str:
        table = self.maps.setdefault(rule.label, {})

        def repl(m: re.Match) -> str:
            key = m.group(0).replace("-", "")
            if key not in table:
                table[key] = len(table) + 1
            return f"<{rule.label}:{table[key]}>"
        return rule.pattern.sub(repl, text)


def _norm_text(value: str, ords: _Ordinals) -> str:
    for literal, placeholder in _host_paths():
        value = value.replace(literal, placeholder)
    value = CLONE_EPOCH.sub(r"\1<epoch>", value)
    for rule in PATTERN_RULES:
        value = ords.sub(rule, value)
    return value


def _norm_value(field: str | None, value: Any, ords: _Ordinals) -> Any:
    if field is not None and value is not None:
        for rule in FIELD_RULES:
            if field in rule.fields:
                return rule.placeholder
    if isinstance(value, str):
        stripped = value.lstrip()
        if stripped[:1] in ("{", "["):
            # JSON kept in a text column (summary_json, payload, content_json):
            # the field rules must reach its keys too. Re-serialised the same
            # way on both sides, so formatting cannot differ.
            try:
                inner = json.loads(value)
            except json.JSONDecodeError:
                inner = None
            if isinstance(inner, (dict, list)):
                return {"$json": _norm_value(None, inner, ords)}
        return _norm_text(value, ords)
    if isinstance(value, dict):
        return {k: _norm_value(k, v, ords) for k, v in value.items()}
    if isinstance(value, list):
        return [_norm_value(None, v, ords) for v in value]
    return value


def _norm_store(path: str, dump: dict, ords: _Ordinals) -> dict:
    for rule in PATH_RULES:
        if path.endswith(rule.suffixes):
            return {"replaced": rule.placeholder}
    if "sqlite" in dump:
        out = {}
        for table in sorted(dump["sqlite"]):
            t = dump["sqlite"][table]
            cols = t["columns"]
            raw_rows = t["rows"]
            for crule in COLUMN_PATTERN_RULES:
                if table == crule.table:
                    raw_rows = [[ords.sub(crule, v) if isinstance(v, str) else v for v in row]
                                for row in raw_rows]
            for colrule in COLUMN_RULES:
                if table == colrule.table:
                    raw_rows = [[colrule.placeholder if (c in colrule.columns and v is not None)
                                 else v for c, v in zip(cols, row)] for row in raw_rows]
            rows = [[_norm_value(c, v, ords) for c, v in zip(cols, row)] for row in raw_rows]
            for rule in KEYVALUE_RULES:
                if table == rule.table and cols[:2] == ["key", "value"]:
                    rows = [[r[0], rule.placeholder if r[0] in rule.keys else r[1], *r[2:]]
                            for r in rows]
            out[table] = {"columns": cols, "rows": rows}
        return {"sqlite": out}
    return {k: _norm_value(None, v, ords) for k, v in dump.items()}


def normalize_observables(obs: dict, root: Any = None) -> dict:
    """Apply every observable rule. Deterministic walk order: steps, then
    stores by path, tables by name, rows in rowid order — so identical
    behaviour assigns identical ordinals."""
    ords = _Ordinals()
    obs = json.loads(json.dumps(obs, default=str))
    steps = [_norm_value(None, s, ords) for s in obs.get("steps", [])]
    stores = {}
    for p in sorted(obs.get("stores", {})):
        key = CLONE_EPOCH.sub(r"\1<epoch>", "/" + p)[1:]
        if "/.git/" in "/" + key:
            if "/.git/objects/" in "/" + key:
                continue
            stores[key] = {"replaced": GIT_INTERNALS.placeholder}
            continue
        stores[key] = _norm_store(p, obs["stores"][p], ords)
    return {"steps": steps, "stores": stores}


def all_rules() -> list[Rule]:
    return [*REQUEST_RULES, *FIELD_RULES, *PATTERN_RULES, *COLUMN_PATTERN_RULES, *COLUMN_RULES,
            *KEYVALUE_RULES, *PATH_RULES, GIT_INTERNALS, HOST_PATH_RULE, *RESPONSE_RULES]
