"""Extract every config read in ``src/`` and the fallback it uses.

WHY THIS IS AN EXTRACTOR AND NOT A TABLE
-----------------------------------------
The guard that consumes this asserts that the shipped template's value and the
code's live fallback are **equal to each other**. Both sides are read
programmatically, and that is the whole point: a hand-written table of
"expected defaults" would be authored by the same person who wrote the
template, would anchor the same way, and would pass while the two real
artefacts disagreed. Restating a computation in a test reproduces the
computation's bug in the test.

WHAT THIS PASS ACTUALLY PROVES (say it out loud — §2b)
-------------------------------------------------------
It finds dict-key reads whose receiver this pass has already bound to a config
expression, by intra-function dataflow:

    cfg = config.get("memory", {})     ->  binds `cfg` to prefix "memory"
    cfg.get("max_facts", 500)          ->  memory.max_facts, default 500

It does NOT prove a key is a config key: a ``.get()`` on a same-named non-config
dict is indistinguishable. It under-reports too — a read reached through an
attribute (``self.cfg.get(...)``) or through a helper is invisible. So the
guard built on it is a RATCHET over what it can see, never a claim of total
coverage, and the register it maintains is a debt list rather than an
allowlist.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = re.compile(
    r"^(config|cfg|conf|_config|_cfg|full_config|raw_config|"
    r"prometheus_config|settings)$", re.I)

_LITERAL = (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set)


class _Sentinel:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


OPAQUE = _Sentinel("<opaque-default>")
NO_DEFAULT = _Sentinel("<no-default>")
REQUIRED = _Sentinel("<subscript-required>")


@dataclass
class Read:
    """One config read site."""
    key: str
    default: object
    file: str
    line: int

    @property
    def site(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass
class KeyFacts:
    """Everything the pass knows about one dotted config key."""
    key: str
    reads: list[Read] = field(default_factory=list)

    @property
    def literal_defaults(self) -> list[Read]:
        return [r for r in self.reads
                if not isinstance(r.default, _Sentinel)]

    @property
    def distinct_literal_values(self) -> list[object]:
        seen: list[object] = []
        for r in self.literal_defaults:
            if not any(_same(r.default, s) for s in seen):
                seen.append(r.default)
        return seen


def _same(a: object, b: object) -> bool:
    """Value equality that does not conflate True with 1, or False with 0."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def _unwrap(node: ast.AST | None) -> ast.AST | None:
    """`(cfg or {})` / `cfg or {}` -> cfg."""
    while isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and node.values:
        node = node.values[0]
    return node


def _key_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript):
        s = node.slice
        return s.value if isinstance(s, ast.Constant) and isinstance(s.value, str) else None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and node.args):
        a = node.args[0]
        return a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else None
    return None


def _receiver_of(node: ast.AST) -> ast.AST | None:
    if isinstance(node, ast.Subscript):
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.value
    return None


def _literal_value(node: ast.AST, consts: dict) -> object:
    node = _unwrap(node)
    if isinstance(node, _LITERAL):
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError, TypeError):
            return OPAQUE
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    if isinstance(node, ast.UnaryOp):
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError, TypeError):
            return OPAQUE
    return OPAQUE


def _module_consts(tree: ast.Module) -> dict:
    """Module-level literal constants, so `.get(k, DEFAULT_X)` resolves.

    Without this, every key whose default is a named constant lands in the
    "no usable literal" bucket — which would be 30-odd keys that in fact have
    a perfectly good default one name-lookup away.
    """
    out: dict = {}
    for stmt in tree.body:
        target = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            target, value = stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) \
                and stmt.value is not None:
            target, value = stmt.target.id, stmt.value
        if target is None:
            continue
        try:
            out[target] = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            pass
    return out


class _Scope(ast.NodeVisitor):
    def __init__(self, path: str, consts: dict) -> None:
        self.path, self.consts = path, consts
        self.prefixes: dict[str, str] = {}
        self.reads: list[Read] = []

    def seed_params(self, args: ast.arguments) -> None:
        for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
            if CONFIG_NAME.match(a.arg):
                self.prefixes[a.arg] = ""
            elif a.arg.endswith("_config") or a.arg.endswith("_cfg"):
                self.prefixes[a.arg] = a.arg.rsplit("_", 1)[0]

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return
        target = node.targets[0].id
        val = _unwrap(node.value)
        key, recv = _key_of(val), _unwrap(_receiver_of(val)) if val else None
        if key is not None and isinstance(recv, ast.Name):
            base = self.prefixes.get(recv.id)
            if base is not None:
                self.prefixes[target] = f"{base}.{key}" if base else key
                return
        if isinstance(val, ast.Name) and val.id in self.prefixes:
            self.prefixes[target] = self.prefixes[val.id]
        elif CONFIG_NAME.match(target) and not isinstance(val, ast.Name):
            self.prefixes.setdefault(target, "")

    def _record(self, node: ast.AST) -> None:
        key = _key_of(node)
        if key is None:
            return
        recv = _unwrap(_receiver_of(node))
        if not isinstance(recv, ast.Name):
            return
        base = self.prefixes.get(recv.id)
        if base is None:
            return
        if isinstance(node, ast.Subscript):
            default: object = REQUIRED
        elif len(node.args) >= 2:
            default = _literal_value(node.args[1], self.consts)
        else:
            default = NO_DEFAULT
        self.reads.append(
            Read(f"{base}.{key}" if base else key, default, self.path, node.lineno))

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self._record(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._record(node)
        self.generic_visit(node)


def extract_config_reads(src_root: Path, repo_root: Path) -> dict[str, KeyFacts]:
    """Every config read the pass can resolve, grouped by dotted key."""
    facts: dict[str, KeyFacts] = {}
    for py in sorted(src_root.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel, consts = str(py.relative_to(repo_root)), _module_consts(tree)

        module = _Scope(rel, consts)
        for stmt in tree.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                module.visit(stmt)
        inherited = dict(module.prefixes)
        collected = list(module.reads)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sc = _Scope(rel, consts)
                sc.prefixes.update(inherited)
                sc.seed_params(node.args)
                for s in node.body:
                    sc.visit(s)
                collected.extend(sc.reads)

        for r in collected:
            facts.setdefault(r.key, KeyFacts(r.key)).reads.append(r)
    return facts


# --- template helpers -------------------------------------------------------

def flatten(node: object, prefix: tuple[str, ...] = ()) -> dict[str, object]:
    """Every dotted path in a nested mapping -> its value."""
    out: dict[str, object] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            path = prefix + (str(k),)
            out[".".join(path)] = v
            out.update(flatten(v, path))
    return out


def open_maps(node: object, prefix: tuple[str, ...] = ()) -> set[str]:
    """Paths the template declares as user-keyed OPEN MAPS by giving them ``{}``."""
    opens: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            path = prefix + (str(k),)
            if isinstance(v, dict) and not v:
                opens.add(".".join(path))
            else:
                opens |= open_maps(v, path)
    return opens


EMPTYISH = (None, "", [], {})


def equivalent(template_value: object, code_default: object) -> bool:
    """Is the template's value the same statement as the code's fallback?

    Exact equality, plus one deliberate equivalence class: an absent code
    default (``cfg.get(k)`` -> None) and a template value of null / "" / [] /
    {} are the SAME STATEMENT in two spellings — "nothing configured". Without
    this, 44 keys whose template placeholder is an empty string would read as
    mismatches against a None fallback and the guard would be noise.

    True/1 and False/0 are NOT conflated: a bool default and an int template
    value are a real disagreement.
    """
    if isinstance(code_default, _Sentinel):
        return any(_same(template_value, e) for e in EMPTYISH)
    return _same(template_value, code_default)
