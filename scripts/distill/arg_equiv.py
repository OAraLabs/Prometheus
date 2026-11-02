#!/usr/bin/env python3
"""Argument equivalence for eval-tools: would two calls to the same tool have the same effect?

    PYTHONPATH=src python scripts/distill/arg_equiv.py rescore \\
        --cases <evalsets>/eval-tools.jsonl \\
        --results student-q4=<dir>/eval-tools.jsonl --results base=<dir2>/eval-tools.jsonl ... \\
        [--pair student-q4:base ...] \\
        [--exec-store <sandbox-root>/repo-store/<sha>.git --exec-work <scratch>/work --jail require] \\
        [--json-out report.json] [--markdown-out table.md]
    PYTHONPATH=src python scripts/distill/arg_equiv.py validate --report report.json --labels labels.json

``score.py`` keeps its exact-argument match (``arg_exact``: type-strict JSON
equality, per ``args_match``). This module adds ``arg_equiv`` beside it: the
model's first call names the expected tool, and its arguments would do what
the expected arguments do. Every verdict records the tier that decided it:

exact         the arguments are equal the way ``arg_exact`` compares them.
canonical     equal once both sides are put in a canonical form. Nothing here
              guesses: each rule is an identity of the tool or of the POSIX
              shell, never "close enough".
                * JSON: integral floats are ints (1.0 == 1), keys unordered,
                  ``args_ignore`` dropped (as ``arg_exact`` does).
                * schema: a missing optional argument takes its schema
                  ``default``; an optional argument that is null (with a null
                  or no default) is dropped, as the tool reads it the same.
                * paths (``x-prometheus-path`` arguments and bash ``cwd``):
                  ``~`` and ``~/`` expand to the case's home, a relative path
                  joins the case's working directory, ``.`` parts and doubled
                  slashes go, and a directory path loses its trailing slash.
                  ``..`` is kept (a symlink can make ``a/..`` not the parent).
                * bash ``command``, when it lexes as plain POSIX shell (no
                  ``$``, backquote or backslash outside single quotes, no
                  heredoc, subshell, background job, comment or newline;
                  otherwise only surrounding whitespace is ignored):
                  whitespace between words; quoting, except where it changes
                  meaning (a glob character, brace or tilde unquoted vs
                  quoted stays distinct); an unquoted leading ``~/`` expands
                  to the case's home (exactly what the shell does); a
                  trailing ``;``; a leading ``cd <the case's own cwd> &&|;``
                  (a no-op); ``cd`` targets resolved like paths; and the
                  order of boolean single-letter flags for a few programs
                  whose listed flags take no value and do not override each
                  other (``ls -la`` == ``ls -al`` == ``ls -l -a``; see
                  ``BOOL_FLAGS``). Pipes, ``&&`` vs ``;`` and redirections are
                  kept as written.
              No glob or regex rewriting: two different patterns are never
              called equal here.
execution     side-effect-free calls the gym sandbox can run (bash and
              read_file) are run in the sandbox ``teacher_gen.py`` uses: its
              shell grammar (``SandboxEnv.plan_shell``), its OS jail (bwrap on
              Linux) and the repo fixture cloned from the gym's own repo store
              (committed files of one commit). The case's home maps to the
              sandbox root and its working directory to the fixture. Each
              call runs in two snapshots: A, the fixture as committed (every
              mtime pinned), and B, the same tree perturbed (pad lines
              appended to every text file, pad files added to every
              directory), so two commands that differ only past the end of a
              short file or listing still differ. Each call runs twice in A
              (a run that differs from itself is nondeterministic). The
              verdict is "equivalent" only when both calls exit 0 in every
              run, neither is refused or simulated, nothing is cut short,
              every absolute or ``~`` path either names exists in the
              snapshot, their output is not empty and output and effects on
              the workspace are identical in A and in B. "Not equivalent"
              when both exit 0 cleanly in a snapshot and differ there.
              Anything else (refused, simulated, a missing path, a nonzero
              exit, a timeout, cut output, nondeterminism) is undecidable.
conservative  a tool with side effects (writes, edits, tasks, messages...)
              called with different canonical arguments, or arguments that
              are not a JSON object (the loop refuses those): counted not
              equivalent; never run.
undecidable   a side-effect-free tool (or bash) whose canonical arguments
              differ and that execution could not decide (or no executor
              was given); and an edit_file call that is the other with the
              same text added around both old_str and new_str (the same edit
              exactly when the wider text is in the file, which only the
              file can tell). Counted separately, never guessed: the
              ``arg_equivalence`` rate counts only proven equivalences (a
              lower bound), and ``arg_equivalence_upper`` adds the
              undecidable ones.

The report holds counts, case ids and verdicts only, never call or reply text.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

REPORT_SCHEMA = "distill-arg-equiv/1"

EXACT, CANONICAL, EXECUTION, CONSERVATIVE, UNDECIDABLE = (
    "exact", "canonical", "execution", "conservative", "undecidable")
TIERS = (EXACT, CANONICAL, EXECUTION, CONSERVATIVE, UNDECIDABLE)

#: Tools that change something outside their own reply. Different canonical
#: arguments to one of these count as not equivalent (they are never run).
SIDE_EFFECT_TOOLS = frozenset({
    "write_file", "edit_file", "notebook_edit", "todo_write", "task_create", "task_stop",
    "task_update", "cron_create", "cron_delete", "message", "sessions_send", "sessions_spawn",
    "agent", "wiki_compile", "image_generate", "video_generate", "tts", "download_file",
    "symbiote_graft", "symbiote_harvest", "dashboard", "ask_user", "browser", "skill",
})
#: ``memory`` writes unless both calls only list.
_MEMORY_READ_OPS = frozenset({"list"})
#: Tools the executor can run (side-effect-free in the gym sandbox).
EXECUTABLE_TOOLS = frozenset({"bash", "read_file"})

#: Boolean single-letter flags whose order never matters for that program: none
#: takes a value and none overrides another by position (ls's format and sort
#: flags, where the last one wins, are left out, and so is anything ambiguous).
BOOL_FLAGS: dict[str, frozenset[str]] = {
    "ls": frozenset("aAlhrRdFi"),
    "grep": frozenset("nirvcwEFHhsxI"), "egrep": frozenset("nirvcwHhsxI"),
    "fgrep": frozenset("nirvcwHhsxI"),
    "wc": frozenset("lwcmL"),
    "sort": frozenset("rnufb"),
    "du": frozenset("shac"),
    "cat": frozenset("nbAeEsTv"),
    "rm": frozenset("rfv"),
    "mkdir": frozenset("pv"),
}


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def strict_eq(a: Any, b: Any) -> bool:
    """``score.strict_eq``: JSON equality without bool/int coercion (``true``
    is not ``1``); an int and a float compare by value."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(strict_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(strict_eq(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def norm_json(v: Any) -> Any:
    """Integral floats become ints; containers are rebuilt (keys unordered)."""
    if isinstance(v, bool) or v is None or isinstance(v, (str, int)):
        return v
    if isinstance(v, float):
        return int(v) if v.is_integer() else v
    if isinstance(v, dict):
        return {k: norm_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [norm_json(x) for x in v]
    return v


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str,
                                     separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------
# The case's environment
# ---------------------------------------------------------------------------

_WORKDIR = re.compile(r"^\s*-?\s*Working directory:\s*(\S+)\s*$", re.MULTILINE)
_HOME = re.compile(r"^(/home/[^/]+|/Users/[^/]+|/root)(?:/|$)")


@dataclass(frozen=True)
class CaseEnv:
    """Where the case's calls would have run: its working directory (from the
    system prompt's ``Working directory:`` line) and the home that contains it."""
    cwd: str | None = None
    home: str | None = None


def case_env(messages: list[dict[str, Any]]) -> CaseEnv:
    for m in messages:
        if m.get("role") != "system":
            continue
        content = m.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not isinstance(content, str):
            continue
        hit = _WORKDIR.search(content)
        if hit:
            cwd = _norm_abs(hit.group(1))
            home = _HOME.match(cwd)
            return CaseEnv(cwd=cwd, home=home.group(1) if home else None)
    return CaseEnv()


def _norm_abs(p: str) -> str:
    """An absolute path with ``.`` parts and doubled or trailing slashes gone;
    ``..`` kept."""
    parts = [x for x in p.split("/") if x not in ("", ".")]
    return "/" + "/".join(parts)


def norm_path(value: Any, env: CaseEnv, *, is_dir: bool = False) -> Any:
    """A path argument as the tool resolves it (``~`` = home, relative = under
    the working directory). Left as is when the case does not say where."""
    if not isinstance(value, str) or not value:
        return value
    s = value
    if s == "~" or s.startswith("~/"):
        if env.home is None:
            return value
        s = env.home + s[1:]
    elif s.startswith("~"):
        return value  # ~user: someone else's home
    if not s.startswith("/"):
        if env.cwd is None:
            return value
        s = env.cwd + "/" + s
    out = _norm_abs(s)
    if value.endswith("/") and not is_dir and out != "/":
        out += "/"  # "file/" is not "file": ENOTDIR on a file
    return out


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

#: Characters whose meaning depends on being unquoted (marked with \\x00 in a
#: canonical word so a quoted one never equals an unquoted one).
_LIVE = frozenset("*?[]{}~!")
_OPS = ("&&", "||", ">>", ">&", "<&", "<>", ">|", ";", "|", ">", "<")
_REJECT_OPS = ("<<", "&>", "|&", ";;", "&", "(", ")")
_MARK = "\x00"


class ShellUnsafe(ValueError):
    """The command uses shell features the canonical form does not model."""


def shell_tokens(cmd: str, env: CaseEnv, rules: set[str] | None = None) -> list[tuple[str, str]]:
    """The command as ``(kind, text)`` tokens, kind ``op`` or ``word``. A word's
    text has its quotes removed; a live character outside quotes is prefixed
    with \\x00, and an unquoted leading ``~`` / ``~/`` is the case's home
    (``"tilde"`` is added to *rules*). Raises ShellUnsafe for anything outside
    plain POSIX words and operators."""
    toks: list[tuple[str, str]] = []
    i, n = 0, len(cmd)
    if "\n" in cmd.strip() or "\r" in cmd:
        raise ShellUnsafe("multi-line")
    while i < n:
        c = cmd[i]
        if c in " \t\n":
            i += 1
            continue
        if c in ";&|<>()":
            bad = next((o for o in _REJECT_OPS if cmd.startswith(o, i)), None)
            ok = next((o for o in _OPS if cmd.startswith(o, i)), None)
            if bad and (not ok or len(bad) >= len(ok)):
                raise ShellUnsafe(f"operator {bad}")
            assert ok is not None
            toks.append(("op", ok))
            i += len(ok)
            continue
        if c == "#":
            raise ShellUnsafe("comment")
        buf: list[str] = []
        quoted = False
        start = i
        while i < n and cmd[i] not in " \t\n;&|<>()":
            c = cmd[i]
            if c == "'":
                j = cmd.find("'", i + 1)
                if j < 0:
                    raise ShellUnsafe("unbalanced quote")
                buf.append(cmd[i + 1:j])
                quoted, i = True, j + 1
            elif c == '"':
                j = i + 1
                while j < n and cmd[j] != '"':
                    if cmd[j] in "$`\\":
                        raise ShellUnsafe("expansion in double quotes")
                    j += 1
                if j >= n:
                    raise ShellUnsafe("unbalanced quote")
                buf.append(cmd[i + 1:j])
                quoted, i = True, j + 1
            elif c in "$`\\":
                raise ShellUnsafe("expansion or escape")
            else:
                if c == "~" and i == start and env.home is not None and (
                        i + 1 >= n or cmd[i + 1] in "/ \t;&|<>"):
                    buf.append(env.home)
                    if rules is not None:
                        rules.add("tilde")
                elif c in _LIVE:
                    buf.append(_MARK + c)
                else:
                    buf.append(c)
                i += 1
        word = "".join(buf)
        # an IO number: digits written right against a redirection operator
        if not quoted and word.isdigit() and i < n and cmd[i] in "<>":
            op = next(o for o in _OPS if cmd.startswith(o, i))
            if op in ("&&", "||", ";", "|"):
                raise ShellUnsafe("io number")
            toks.append(("op", word + op))
            i += len(op)
            continue
        toks.append(("word", word))
    return toks


def _commands(toks: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Tokens split at list and pipe operators, each piece followed by its
    operator (as a one-token piece)."""
    out: list[list[tuple[str, str]]] = [[]]
    for t in toks:
        if t[0] == "op" and t[1] in ("&&", "||", ";", "|"):
            out.append([t])
            out.append([])
        else:
            out[-1].append(t)
    return out


def _cluster_flags(words: list[str]) -> tuple[list[str], bool]:
    """``argv`` with a leading run of boolean flags (``BOOL_FLAGS``) merged into
    one sorted cluster; and whether anything moved."""
    allowed = BOOL_FLAGS.get(words[0]) if words else None
    if not allowed:
        return words, False
    letters: list[str] = []
    k = 1
    while k < len(words):
        w = words[k]
        if not (w.startswith("-") and len(w) > 1 and not w.startswith("--")
                and all(ch in allowed for ch in w[1:])):
            break
        letters += list(w[1:])
        k += 1
    if k == 1:
        return words, False
    merged = "-" + "".join(sorted(set(letters)))
    changed = words[1:k] != [merged]
    return [words[0], merged, *words[k:]], changed


def canonical_shell(cmd: str, env: CaseEnv) -> tuple[Any, list[str]]:
    """The canonical form of a shell command and the rules that applied. A
    command outside the modelled subset is its text with surrounding
    whitespace removed (``("raw", text)``)."""
    found: set[str] = set()
    try:
        toks = shell_tokens(cmd, env, found)
    except ShellUnsafe:
        return ("raw", cmd.strip()), []
    rules: list[str] = sorted(found)
    pieces = _commands(toks)
    # trailing separators
    while len(pieces) >= 2 and not pieces[-1] and pieces[-2] == [("op", ";")]:
        pieces = pieces[:-2]
        rules.append("trailing_separator")
    cwd = env.cwd
    out: list[tuple[str, str]] = []
    prev_op = None
    for idx, piece in enumerate(pieces):
        if len(piece) == 1 and piece[0][0] == "op" and piece[0][1] in ("&&", "||", ";", "|"):
            out.append(piece[0])
            prev_op = piece[0][1]
            continue
        words = [t[1] for t in piece if t[0] == "word"]
        only_words = all(t[0] == "word" for t in piece)
        nxt = pieces[idx + 1][0][1] if idx + 1 < len(pieces) and pieces[idx + 1] else None
        in_pipe = prev_op == "|" or nxt == "|"
        if words and words[0] == "cd" and only_words and len(words) <= 2 and not in_pipe:
            target = words[1] if len(words) == 2 else (env.home or "~")
            if _MARK not in target and target != "-":
                resolved = norm_path(target, CaseEnv(cwd=cwd, home=env.home), is_dir=True)
                if resolved != target:
                    rules.append("cd_path")
                if idx == 0 and resolved == env.cwd and nxt in ("&&", ";"):
                    rules.append("cd_to_cwd")
                    # drop this cd and the operator after it
                    pieces[idx + 1] = []
                    continue
                if isinstance(resolved, str) and resolved.startswith("/"):
                    cwd = resolved
                piece = [("word", "cd"), ("word", resolved)]
        elif words:
            argv_end = next((k for k, t in enumerate(piece) if t[0] == "op"), len(piece))
            argv = [t[1] for t in piece[:argv_end]]
            new, changed = _cluster_flags(argv)
            if changed:
                rules.append("flag_order")
            piece = [("word", w) for w in new] + piece[argv_end:]
        out.extend(piece)
        prev_op = None
    return ("sh", tuple(out)), sorted(set(rules))


# ---------------------------------------------------------------------------
# Canonical arguments
# ---------------------------------------------------------------------------


def _props(schema: dict[str, Any]) -> dict[str, Any]:
    p = schema.get("properties") if isinstance(schema, dict) else None
    return p if isinstance(p, dict) else {}


def _path_kind(tool: str, key: str, spec: Any) -> str | None:
    if isinstance(spec, dict) and spec.get("x-prometheus-path"):
        return str(spec["x-prometheus-path"])
    if tool == "bash" and key == "cwd":
        return "dir"
    return None


def canonical_args(tool: str, args: dict[str, Any], schema: dict[str, Any], env: CaseEnv,
                   *, ignore: tuple[str, ...] = (), fill_defaults: bool = True
                   ) -> tuple[dict[str, Any], list[str]]:
    rules: list[str] = []
    props = _props(schema)
    required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
    out = {k: norm_json(v) for k, v in args.items() if k not in ignore}
    if fill_defaults:
        for k, spec in props.items():
            if k not in out and k not in ignore and isinstance(spec, dict) and "default" in spec:
                out[k] = norm_json(spec["default"])
                rules.append("schema_default")
    for k in list(out):
        if out[k] is None and k not in required:
            spec = props.get(k)
            if not isinstance(spec, dict) or spec.get("default", None) is None:
                del out[k]
                rules.append("null_optional")
    for k, spec in props.items():
        kind = _path_kind(tool, k, spec)
        if kind and isinstance(out.get(k), str):
            new = norm_path(out[k], env, is_dir=kind == "dir")
            if new != out[k]:
                rules.append("path")
            out[k] = new
    if tool == "bash" and isinstance(out.get("command"), str):
        out["command"], more = canonical_shell(out["command"], env)
        rules += more
    return out, sorted(set(rules))


def _subset_eq(expected: dict[str, Any], got: dict[str, Any]) -> bool:
    return all(k in got and strict_eq(got[k], v) for k, v in expected.items())


def has_side_effects(tool: str, *calls: dict[str, Any]) -> bool:
    if tool == "memory":
        return not all(isinstance(c, dict) and c.get("operation") in _MEMORY_READ_OPS for c in calls)
    return tool in SIDE_EFFECT_TOOLS


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    equivalent: bool | None
    tier: str
    reason: str = ""
    rules: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"equivalent": self.equivalent, "tier": self.tier}
        if self.reason:
            d["reason"] = self.reason
        if self.rules:
            d["rules"] = list(self.rules)
        return d


def check_call(tool: str, expected: dict[str, Any], got: Any, schema: dict[str, Any], env: CaseEnv,
               *, match: str = "exact", ignore: tuple[str, ...] = (), exact: bool | None = None,
               executor: Any = None) -> Verdict:
    """Is ``got`` (the model's first call's arguments, to ``tool``) equivalent to
    ``expected``? ``exact`` is score.py's ``arg_exact`` when the caller has it."""
    if not isinstance(got, dict):
        return Verdict(False, CONSERVATIVE, "arguments not an object")
    if exact is None:
        e = {k: v for k, v in expected.items() if k not in ignore}
        if match == "subset":
            exact = _subset_eq(e, got)
        else:
            exact = strict_eq({k: v for k, v in got.items() if k not in ignore}, e)
    if exact:
        return Verdict(True, EXACT)
    subset = match == "subset"
    # A subset expectation names only what must hold: its absent keys are not
    # filled in, the call's are (so "absent" there means "the default").
    ce, r1 = canonical_args(tool, expected, schema, env, ignore=ignore, fill_defaults=not subset)
    cg, r2 = canonical_args(tool, got, schema, env, ignore=ignore)
    rules = tuple(sorted(set(r1) | set(r2)))
    if (_subset_eq(ce, cg) if subset else strict_eq(ce, cg)):
        return Verdict(True, CANONICAL, rules=rules)
    if tool == "edit_file" and _edit_context_extension(ce, cg):
        return Verdict(None, UNDECIDABLE, "edit with wider or narrower context: the same edit "
                       "only if the wider text is in the file", rules=rules)
    if has_side_effects(tool, expected, got):
        return Verdict(False, CONSERVATIVE, "side-effect tool, different canonical arguments",
                       rules=rules)
    if executor is None:
        return Verdict(None, UNDECIDABLE, "no executor", rules=rules)
    if tool not in EXECUTABLE_TOOLS:
        return Verdict(None, UNDECIDABLE, "tool not executable here", rules=rules)
    required = set(schema.get("required") or [])
    if subset and not required <= set(expected):
        return Verdict(None, UNDECIDABLE, "expectation is partial", rules=rules)
    full_e, _ = canonical_args(tool, expected, schema, env, ignore=())
    full_g, _ = canonical_args(tool, got, schema, env, ignore=())
    # execution takes the calls as written (the canonical shell form is not a command)
    run_e = {**{k: v for k, v in full_e.items() if k != "command"}, **_exec_fields(expected)}
    run_g = {**{k: v for k, v in full_g.items() if k != "command"}, **_exec_fields(got)}
    v = executor.compare(tool, run_e, run_g, env)
    v.rules = rules
    return v


def _edit_context_extension(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """One edit is the other with the same text added around both its old and
    new string (same file, same replace_all). Both write the same file when the
    wider old string is in the file (the narrower one is then found at the same
    place, if it is unique), and not otherwise: only the file can tell."""
    if a.get("path") != b.get("path") or a.get("replace_all") != b.get("replace_all"):
        return False
    for x, y in ((a, b), (b, a)):
        xo, xn, yo, yn = x.get("old_str"), x.get("new_str"), y.get("old_str"), y.get("new_str")
        if not all(isinstance(s, str) for s in (xo, xn, yo, yn)) or not xo or xo == yo:
            continue
        i = yo.find(xo)
        if i >= 0 and yn == yo[:i] + xn + yo[i + len(xo):]:
            return True
    return False


def _exec_fields(args: dict[str, Any]) -> dict[str, Any]:
    return {"command": args["command"]} if isinstance(args.get("command"), str) else {}


# ---------------------------------------------------------------------------
# Execution in the gym sandbox
# ---------------------------------------------------------------------------

_PAD_LINES = 300
_PAD_FILES = 25
_PINNED_MTIME = 1_767_225_600  # 2026-01-01T00:00:00Z: every snapshot file's mtime
_TRUNCATED = "...[truncated]..."


def _load_teacher_gen() -> Any:
    mod = sys.modules.get("distill_teacher_gen")
    if mod is not None:
        return mod
    path = Path(__file__).resolve().with_name("teacher_gen.py")
    spec = importlib.util.spec_from_file_location("distill_teacher_gen", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["distill_teacher_gen"] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class RunResult:
    status: str                 # ok | nonzero | refused | simulated | timeout | truncated |
                                # missing_path | unmapped | nondeterministic
    output: str = ""            # sha256 of the output (never the output)
    effect: str = ""            # sha256 of what changed in the workspace
    empty: bool = True

    def same(self, other: "RunResult") -> bool:
        return (self.output, self.effect) == (other.output, other.effect)


@dataclass
class GymSandboxExecutor:
    """Runs a side-effect-free call in the gym's sandbox (``teacher_gen.SandboxEnv``)
    over a clone of the gym's repo store. ``work`` is a scratch directory this
    executor owns (it creates and deletes only ``work/A``, ``work/B`` and their
    git directories). The case's working directory, when missing from the
    fixture, is made a link to the fixture (``cwd_is_repo``: the eval cases'
    cwd is a checkout of the same repository)."""

    store: Path
    work: Path
    jail: str = "require"
    cwd_is_repo: bool = True
    timeout: float = 30.0
    _tg: Any = None
    _jail_kind: str = ""
    _pristine: dict[str, set[str]] = field(default_factory=dict)
    _ready: dict[str, bool] = field(default_factory=dict)
    _cache: dict[str, dict[str, RunResult]] = field(default_factory=dict)
    runs: int = 0

    def __post_init__(self) -> None:
        self._tg = _load_teacher_gen()
        self._jail_kind = self._tg.resolve_jail(self.jail)
        self.store = Path(self.store).resolve()
        self.work = Path(self.work).resolve()
        self.work.mkdir(parents=True, exist_ok=True)

    # -- snapshots ------------------------------------------------------------

    def _root(self, snap: str) -> Path:
        return self.work / snap

    def _build(self, snap: str) -> None:
        root = self._root(snap)
        gd = self._tg.repo_git_dir(root)
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(gd, ignore_errors=True)
        root.mkdir(parents=True)
        self._tg.clone_repo_fixture(self.store, root)
        if snap == "B":
            _perturb(root / self._tg.REPO_FIXTURE_DIR)
        _pin_mtimes(root)
        self._pristine[snap] = _manifest(root)
        self._ready[snap] = True

    def _snapshot(self, snap: str) -> Path:
        if not self._ready.get(snap):
            self._build(snap)
        return self._root(snap)

    def _map(self, env: CaseEnv, root: Path) -> tuple[Path, str] | None:
        """(cwd in the sandbox, the case's home as written in commands)."""
        if env.home is None or env.cwd is None:
            return None
        if env.cwd != env.home and not env.cwd.startswith(env.home + "/"):
            return None
        rel = env.cwd[len(env.home):].lstrip("/")
        cwd = root / rel if rel else root
        if rel and not cwd.exists() and self.cwd_is_repo:
            cwd.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(self._tg.REPO_FIXTURE_DIR, cwd)
            _pin_mtimes(root)
            self._pristine[root.name] = _manifest(root)
        return cwd, env.home

    # -- one run --------------------------------------------------------------

    def _run_once(self, tool: str, args: dict[str, Any], env: CaseEnv, snap: str) -> RunResult:
        tg = self._tg
        root = self._snapshot(snap)
        mapped = self._map(env, root)
        if mapped is None:
            return RunResult("unmapped")
        cwd, home = mapped
        sandbox = tg.SandboxEnv(root, tg.GenTask(id="arg-equiv", kind="agentic", domain="coding",
                                                 family="", prompt=""),
                                jail=self._jail_kind, git_dir=tg.repo_git_dir(root))
        self.runs += 1
        try:
            if tool == "bash":
                cmd = _map_home(str(args.get("command") or ""), home, root)
                if _clock_words(tg, cmd):
                    return RunResult("nondeterministic")
                if not _paths_exist(tg, cmd, root, cwd):
                    return RunResult("missing_path")
                plan = sandbox.plan_shell(cmd.strip(), cwd)
                if plan.simulated is not None or any(
                        p.simulated is not None for _o, _t, p, _c in (plan.chain or [])):
                    return RunResult("simulated")
                a = SimpleNamespace(command=cmd, cwd=None,
                                    timeout_seconds=args.get("timeout_seconds") or 30)
                if args.get("cwd"):
                    a.cwd = _map_home(str(args["cwd"]), home, root)
                else:
                    a.cwd = str(cwd)
                res = asyncio.run(sandbox.bash(a, None))
                text = str(res.output)
                meta = getattr(res, "metadata", None) or {}
                if "returncode" not in meta:
                    return RunResult("timeout")
                status = "ok" if meta["returncode"] == 0 else "nonzero"
            elif tool == "read_file":
                path = sandbox.confine(_map_home(str(args.get("path") or ""), home, root), cwd)
                if not path.is_file():
                    return RunResult("missing_path")
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                off, lim = int(args.get("offset") or 0), int(args.get("limit") or 200)
                text, status = "\n".join(lines[off:off + lim]), "ok"
            else:
                return RunResult("refused")
        except tg.SandboxRefusal:
            return RunResult("refused")
        finally:
            effect = self._effects(snap)
        if _TRUNCATED in text:
            return RunResult("truncated")
        return RunResult(status, _digest(text.replace(str(root), "<root>")), effect,
                         empty=not text.strip() or text.strip() == "(no output)")

    def _effects(self, snap: str) -> str:
        """What the run changed in the workspace (a digest; "" for nothing). A
        changed snapshot is rebuilt before its next run."""
        root = self._root(snap)
        shutil.rmtree(root / ".tmp", ignore_errors=True)
        now = _manifest(root)
        before = self._pristine.get(snap, set())
        changed: list[tuple[str, str]] = []
        for p in sorted({_entry_path(e) for e in now - before}):
            path = Path(p)
            if path.is_symlink():
                changed.append((p, "link:" + os.readlink(path)))
            elif path.is_file():
                changed.append((p, hashlib.sha256(path.read_bytes()).hexdigest()))
            else:
                changed.append((p, "dir"))
        gone = sorted({_entry_path(e) for e in before} - {_entry_path(e) for e in now})
        changed += [(p, "-") for p in gone]
        if not changed:
            return ""
        self._ready[snap] = False
        return _digest([(str(Path(p).relative_to(root)) if p.startswith(str(root)) else p, h)
                        for p, h in changed])

    # -- a call, everywhere -----------------------------------------------------

    def run(self, tool: str, args: dict[str, Any], env: CaseEnv) -> dict[str, RunResult]:
        key = _digest([tool, args, env.cwd, env.home])
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        a1 = self._run_once(tool, args, env, "A")
        out = {"A": a1}
        if a1.status == "ok":
            a2 = self._run_once(tool, args, env, "A")
            if a2.status != "ok" or not a1.same(a2):
                out["A"] = RunResult("nondeterministic")
            else:
                out["B"] = self._run_once(tool, args, env, "B")
        self._cache[key] = out
        return out

    def compare(self, tool: str, expected: dict[str, Any], got: dict[str, Any], env: CaseEnv) -> Verdict:
        re_, rg = self.run(tool, expected, env), self.run(tool, got, env)
        for snap in ("A", "B"):
            e, g = re_.get(snap), rg.get(snap)
            if e is None or g is None:
                continue
            if e.status == "ok" and g.status == "ok" and not e.same(g):
                return Verdict(False, EXECUTION, f"results differ in snapshot {snap}")
        ea, ga = re_["A"], rg["A"]
        if ea.status != "ok" or ga.status != "ok":
            return Verdict(None, UNDECIDABLE, f"execution: expected {ea.status}, got {ga.status}")
        eb, gb = re_.get("B"), rg.get("B")
        if eb is None or gb is None or eb.status != "ok" or gb.status != "ok":
            return Verdict(None, UNDECIDABLE,
                           f"execution (perturbed): expected {eb.status if eb else '-'}, "
                           f"got {gb.status if gb else '-'}")
        if ea.empty and not ea.effect:
            return Verdict(None, UNDECIDABLE, "execution: same, but no output and no effect")
        return Verdict(True, EXECUTION, "same output and effects in both snapshots")

    def describe(self) -> dict[str, Any]:
        return {"store": str(self.store), "jail": self._jail_kind, "snapshots": ["A", "B"],
                "pad_lines": _PAD_LINES, "pad_files": _PAD_FILES, "runs": self.runs}

    def cleanup(self) -> None:
        for snap in ("A", "B"):
            root = self._root(snap)
            shutil.rmtree(self._tg.repo_git_dir(root), ignore_errors=True)
            shutil.rmtree(root, ignore_errors=True)


def _entry_path(entry: str) -> str:
    return entry.split("\x00", 1)[0]


def _manifest(root: Path) -> set[str]:
    """Every entry under *root* (except its ``.tmp``) as ``path\\0mtime\\0size``,
    so a write, a touch or a new file shows."""
    out: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath == str(root) and ".tmp" in dirnames:
            dirnames.remove(".tmp")
        for name in [*dirnames, *filenames]:
            p = os.path.join(dirpath, name)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            out.add(f"{p}\x00{st.st_mtime_ns}\x00{st.st_size if not os.path.isdir(p) else 0}")
    return out


def _pin_mtimes(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        for name in [*dirnames, *filenames]:
            os.utime(os.path.join(dirpath, name), (_PINNED_MTIME, _PINNED_MTIME), follow_symlinks=False)
    os.utime(root, (_PINNED_MTIME, _PINNED_MTIME))


def _perturb(tree: Path) -> None:
    """Snapshot B: every text file gets pad lines appended and every directory
    pad files, paths unchanged. Equivalent calls stay equivalent on any
    filesystem state; calls that differ only past a short file's or listing's
    end now differ."""
    for dirpath, dirnames, filenames in os.walk(tree):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for name in filenames:
            p = Path(dirpath) / name
            if name == ".git" or p.is_symlink():
                continue
            try:
                data = p.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:8192]:
                continue
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            pad = "".join(f"zz-pad-line {k:04d}\n" for k in range(_PAD_LINES))
            p.write_bytes(data + (b"" if not data or data.endswith(b"\n") else b"\n") + pad.encode())
        for k in range(_PAD_FILES):
            (Path(dirpath) / f"zz-pad-{k:02d}.txt").write_text(f"zz-pad-file {k:02d}\n")


def _map_home(text: str, home: str, root: Path) -> str:
    """The case's home, written out in a command or path, as the sandbox root."""
    return re.sub(re.escape(home) + r"(?=/|$|[\s'\";|&)<>])", str(root), text)


#: Programs whose output is the clock: never run (two runs a second apart differ).
_CLOCK_PROGRAMS = frozenset({"date"})


def _clock_words(tg: Any, cmd: str) -> bool:
    try:
        toks = tg._lex(cmd)
    except Exception:
        return False
    return any(not t.op and t.text in _CLOCK_PROGRAMS for t in toks)


def _paths_exist(tg: Any, cmd: str, root: Path, cwd: Path) -> bool:
    """Every absolute or ``~`` path the command names under the sandbox root
    exists there (the static part before any wildcard). A path the snapshot
    lacks makes both calls fail the same way, which proves nothing."""
    try:
        toks = tg._lex(cmd)
    except Exception:  # the sandbox's own check refuses it later
        return True
    for t in toks:
        if t.op or not t.text or t.text.startswith("-"):
            continue
        text = t.text
        if "=" in text and not text.startswith(("/", "~")):
            text = text.split("=", 1)[1]
        if text.startswith("~/") or text == "~":
            text = str(root) + text[1:]
        if not text.startswith(str(root)):
            continue
        static = []
        for part in PurePosixPath(text).parts:
            if any(c in part for c in "*?["):
                break
            static.append(part)
        if not Path(*static).exists():
            return False
    return True


# ---------------------------------------------------------------------------
# Rescoring stored eval-tools replies
# ---------------------------------------------------------------------------


def _load_score() -> Any:
    mod = sys.modules.get("distill_score")
    if mod is not None:
        return mod
    spec = importlib.util.spec_from_file_location("distill_score", Path(__file__).resolve().with_name("score.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["distill_score"] = mod
    spec.loader.exec_module(mod)
    return mod


def _kr(S: Any, k: int, n: int) -> dict[str, Any]:
    return S.rate(k, n)


def rescore(cases: list[Any], rows: dict[tuple[str, str], dict[str, Any]], *,
            executor: Any = None, set_name: str = "eval-tools") -> dict[str, Any]:
    """Every scored case's tool outcome and argument verdicts (ids and verdicts
    only), and the rates over them."""
    S = _load_score()
    group = [c for c in cases if c.set == set_name]
    per_case: dict[str, Any] = {}
    errors = missing = 0
    for c in group:
        r = rows.get(c.key)
        if r is None or not S.row_matches(c, r):
            missing += 1
            continue
        if r.get("status") != "ok":
            errors += 1
            continue
        v = S.score_case(c, r, executor=executor)
        entry = {"outcome": v["tool_outcome"], "task_success": bool(v["task_success"])}
        if "arg_exact" in v:
            entry["arg_exact"] = bool(v["arg_exact"])
            entry["arg_equiv"] = v["arg_equiv"]
        per_case[c.id] = entry
    return {"set": set_name, "cases": len(group), "scored": len(per_case), "errors": errors,
            "missing": missing, "per_case": per_case, "metrics": rates(per_case)}


def rates(per_case: dict[str, Any]) -> dict[str, Any]:
    S = _load_score()
    n = len(per_case)
    outs = Counter(e["outcome"] for e in per_case.values())
    with_args = [e for e in per_case.values() if "arg_equiv" in e]
    eq = sum(1 for e in with_args if e["arg_equiv"]["equivalent"] is True)
    und = sum(1 for e in with_args if e["arg_equiv"]["equivalent"] is None)
    tiers = Counter(e["arg_equiv"]["tier"] for e in with_args)
    m = {
        "task_success": S.rate(outs["success"], n),
        "wrong_tool_rate": S.rate(outs["wrong_tool"], n),
        "no_call_rate": S.rate(outs["no_call"], n),
        "extra_call_rate": S.rate(outs["extra_call"], n),
        "other_miss_rate": S.rate(sum(outs[k] for k in S.OTHER_MISS_OUTCOMES), n),
        "arg_exactness": S.rate(sum(1 for e in with_args if e["arg_exact"]), len(with_args)),
        "arg_equivalence": S.rate(eq, len(with_args)),
        "arg_equivalence_upper": S.rate(eq + und, len(with_args)),
    }
    return {**m, "tool_outcomes": dict(outs), "arg_equivalence_tiers": dict(tiers),
            "arg_equivalence_undecidable": und}


def paired(a: dict[str, Any], b: dict[str, Any], key: str) -> dict[str, Any] | None:
    """Candidate ``a`` minus base ``b`` on a per-case boolean: ``wrong_tool``
    (every case both scored) or ``arg_equiv`` (proven equivalent; the cases
    where both named the expected tool). Newcombe 95% interval, in points."""
    S = _load_score()
    pa, pb = a["per_case"], b["per_case"]
    if key == "wrong_tool":
        ids = sorted(set(pa) & set(pb))
        val = lambda e: e["outcome"] == "wrong_tool"  # noqa: E731
    else:
        ids = sorted(i for i in set(pa) & set(pb) if "arg_equiv" in pa[i] and "arg_equiv" in pb[i])
        val = lambda e: e["arg_equiv"]["equivalent"] is True  # noqa: E731
    if not ids:
        return None
    t = [0, 0, 0, 0]  # both, base only, candidate only, neither
    for i in ids:
        x, y = val(pb[i]), val(pa[i])
        t[0 if x and y else 1 if x else 2 if y else 3] += 1
    rep = S.paired_report(S.PairedCounts(*t), f"arg-equiv/{key}", bootstrap=False)
    return {**rep, "base_rate": round(100 * (t[0] + t[1]) / len(ids), 2),
            "candidate_rate": round(100 * (t[0] + t[2]) / len(ids), 2),
            "delta_pts": rep["diff_pts"]}


def _cell(r: dict[str, Any]) -> str:
    if r.get("rate") is None:
        return "n/a"
    return (f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']}) "
            f"[{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}]")


def render(report: dict[str, Any]) -> str:
    lines = ["| Model | Task success | Wrong tool | No call | Extra call | Other misses "
             "| Exact args | Arg equivalence (proven) | Undecidable | Equivalence upper bound |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label, m in report["models"].items():
        x = m["metrics"]
        lines.append(f"| {label} | {_cell(x['task_success'])} | {_cell(x['wrong_tool_rate'])} "
                     f"| {_cell(x['no_call_rate'])} | {_cell(x['extra_call_rate'])} "
                     f"| {_cell(x['other_miss_rate'])} | {_cell(x['arg_exactness'])} "
                     f"| {_cell(x['arg_equivalence'])} | {x['arg_equivalence_undecidable']} "
                     f"| {_cell(x['arg_equivalence_upper'])} |")
    lines += ["", "| Model | Equivalence tiers (cases) | Outcome counts |", "|---|---|---|"]
    for label, m in report["models"].items():
        x = m["metrics"]
        tiers = ", ".join(f"{t} {x['arg_equivalence_tiers'].get(t, 0)}" for t in TIERS)
        outs = ", ".join(f"{k} {v}" for k, v in sorted(x["tool_outcomes"].items()))
        lines.append(f"| {label} | {tiers} | {outs} |")
    if report.get("pairs"):
        lines += ["", "| Candidate vs base | Metric | n | Base | Candidate | Δ (pts) | Paired 95% CI |"
                  " McNemar p |", "|---|---|---|---|---|---|---|---|"]
        for p in report["pairs"]:
            for key, name in (("wrong_tool", "wrong-tool rate"), ("arg_equiv", "arg equivalence")):
                d = p.get(key)
                if not d:
                    continue
                S = _load_score()
                lines.append(f"| {p['candidate']} vs {p['base']} | {name} | {d['n']} | {d['base_rate']:.1f}%"
                             f" | {d['candidate_rate']:.1f}% | {S._diff_text(d)} | {S._ci_text(d)}"
                             f" | {d['mcnemar_exact_p']:.3g} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Validation against hand labels
# ---------------------------------------------------------------------------


def validate(report: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    """Precision and recall of the checker's "equivalent" against hand labels:
    ``{"labels": [{"model": label, "id": case id, "equivalent": true|false}]}``.
    An undecidable verdict counts as "not shown equivalent" (the rate is a
    lower bound), and is also tallied on its own."""
    tp = fp = fn = tn = 0
    und_pos = und_neg = 0
    rows = []
    for lab in labels.get("labels", []):
        m = report["models"].get(lab["model"])
        e = (m or {}).get("per_case", {}).get(lab["id"]) if m else None
        if not e or "arg_equiv" not in e:
            rows.append({**lab, "checker": "missing"})
            continue
        got = e["arg_equiv"]["equivalent"]
        truth = bool(lab["equivalent"])
        if got is None:
            und_pos += truth
            und_neg += not truth
        if got is True and truth:
            tp += 1
        elif got is True:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
        rows.append({**lab, "checker": got, "tier": e["arg_equiv"]["tier"]})
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {"n": tp + fp + fn + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec,
            "undecidable_labelled_equivalent": und_pos, "undecidable_labelled_not": und_neg,
            "rows": rows}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rescore", help="tool outcomes and argument equivalence per model")
    r.add_argument("--cases", action="append", required=True)
    r.add_argument("--results", action="append", required=True, metavar="LABEL=PATH")
    r.add_argument("--pair", action="append", default=[], metavar="CANDIDATE:BASE")
    r.add_argument("--exec-store", help="the gym's repo store (a bare clone); enables execution")
    r.add_argument("--exec-work", help="a scratch directory the executor owns")
    r.add_argument("--jail", default="require", choices=("auto", "require", "off"))
    r.add_argument("--json-out")
    r.add_argument("--markdown-out")
    v = sub.add_parser("validate", help="precision/recall against hand labels")
    v.add_argument("--report", required=True)
    v.add_argument("--labels", required=True)
    v.add_argument("--json-out")
    args = ap.parse_args(argv)
    if args.cmd == "validate":
        res = validate(json.loads(Path(args.report).read_text()), json.loads(Path(args.labels).read_text()))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
        return 0
    S = _load_score()
    executor = None
    if args.exec_store:
        if not args.exec_work:
            print("error: --exec-store needs --exec-work", file=sys.stderr)
            return 2
        executor = GymSandboxExecutor(Path(args.exec_store), Path(args.exec_work), jail=args.jail)
    cases = S.load_cases(args.cases)
    report: dict[str, Any] = {"schema": REPORT_SCHEMA, "scorer_version": S.SCORER_VERSION,
                              "method": "static+execution" if executor else "static",
                              "models": {}, "pairs": []}
    try:
        for spec in args.results:
            label, _, path = spec.partition("=")
            rows, _labels = S.latest_rows(Path(path).expanduser())
            report["models"][label] = rescore(cases, rows, executor=executor)
            print(f"{label}: done", file=sys.stderr)
    finally:
        if executor is not None:
            report["executor"] = executor.describe()
            executor.cleanup()
    for spec in args.pair:
        cand, _, base = spec.partition(":")
        a, b = report["models"][cand], report["models"][base]
        report["pairs"].append({"candidate": cand, "base": base,
                                "wrong_tool": paired(a, b, "wrong_tool"),
                                "arg_equiv": paired(a, b, "arg_equiv")})
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    text = render(report)
    if args.markdown_out:
        Path(args.markdown_out).write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
