"""THE INVARIANT: a config read may not swallow a failure into a substituted
default. If it substitutes, it records.

WHAT THIS EXISTS TO PREVENT
---------------------------
Eight subsystems read prometheus.yaml like this::

    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
        section = data.get("<section>", {})
    except (OSError, Exception):
        section = {}

That handler converts "I could not read your configuration" into "you did not
configure anything." Those are different facts, and the code made them
indistinguishable — which is how a path that had never resolved to a real file
on any checkout survived two years and ~6000 tests. A wrong path that raises is
found on the first boot; a wrong path behind a bare except is found by
accident.

WHERE THE LINE IS DRAWN, AND WHY
--------------------------------
The violation is a BROAD catch that substitutes: bare ``except:``,
``except Exception``, or the redundant ``except (OSError, Exception)``
(``Exception`` already subsumes ``OSError`` — the belt-and-braces was the
smell). A NARROW catch passes, because naming the exceptions you mean IS the
remediation being asked for, and because absence is legitimately the answer for
some files: a missing lock file means the daemon is not running, a missing
sticker cache means an empty cache. ``config/ephemeral.py`` had already reasoned
this out correctly — ``except FileNotFoundError`` returns ``{}`` while the
error path logs — and a guard that flagged it would be turned off within a
week, which is worse than no guard.

The sites deliberately left with narrow silent substitutions are enumerated in
docs/audits/SILENT-FAILURE-AUDIT.md. The previous remediation of this same
shape stopped at four of eight modules and left no record of where it stopped,
because the document IT cited was never committed. That is the mistake this
file and that document exist not to repeat.

BOTH DIRECTIONS
---------------
``test_detector_catches_the_known_bad_shapes`` replays the forms that actually
shipped. A detector that silently matched nothing would produce the same green
as a clean tree — and the equivalent replay check in
test_context_limit_invariant.py has already caught a real hole in its own
detector once, which then immediately surfaced a site hand-reading had missed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import prometheus

SRC_ROOT = Path(prometheus.__file__).resolve().parent

# A try body that both READS A FILE and PARSES it is a config read. Requiring
# the file read is what keeps skill/wiki frontmatter parsing — safe_load over a
# string already in memory — out of scope; those are not configuration.
FILE_READ_MARKERS = ("open(", ".read_text(", ".read_bytes(")
PARSE_MARKERS = ("safe_load", "yaml.load", "json.load")

# Anything that makes the failure observable.
RECORD_NAMES = (
    "log", "logger", "logging", "warn", "error", "exception", "critical",
    "record_silent_failure", "_record", "print",
)


def _module(body: list[ast.stmt]) -> ast.Module:
    return ast.Module(body=body, type_ignores=[])


def _is_config_read(node: ast.Try) -> bool:
    src = ast.unparse(_module(node.body))
    return (any(m in src for m in FILE_READ_MARKERS)
            and any(m in src for m in PARSE_MARKERS))


def _is_broad(handler: ast.ExceptHandler) -> tuple[bool, str]:
    """Bare, Exception, or the redundant (OSError, Exception)."""
    if handler.type is None:
        return True, "bare `except:`"
    src = ast.unparse(handler.type)
    if src == "Exception" or src == "BaseException":
        return True, f"broad `except {src}`"
    names = {n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)}
    if "Exception" in names or "BaseException" in names:
        if names & {"OSError", "IOError"}:
            return True, f"redundant `except {src}` — Exception subsumes OSError"
        return True, f"broad `except {src}`"
    return False, ""


def _records(handler: ast.ExceptHandler) -> bool:
    """Logs, raises, writes the ledger, or otherwise surfaces the exception."""
    mod = _module(handler.body)
    for node in ast.walk(mod):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            fn = node.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else "")
            if any(r in name.lower() for r in RECORD_NAMES):
                return True
            if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                    and fn.value.id.lower() in ("log", "logger", "logging")):
                return True
    # A handler that names the caught exception is surfacing it somehow.
    if handler.name and handler.name in ast.unparse(mod):
        return True
    return False


def _substitutes(handler: ast.ExceptHandler) -> bool:
    """Assigns or returns a value indistinguishable from success-with-nothing."""
    for node in ast.walk(_module(handler.body)):
        value = (node.value if isinstance(node, (ast.Assign, ast.Return, ast.AnnAssign))
                 else None)
        if value is None:
            continue
        if isinstance(value, ast.Dict) and not value.keys:
            return True
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)) and not value.elts:
            return True
        if isinstance(value, (ast.Constant, ast.Name)):
            return True
    return False


def _unguarded_none_get(node: ast.Try) -> list[tuple[int, str]]:
    """``data = yaml.safe_load(fh)`` then ``data.get(...)`` with no guard.

    THE FOURTH STATE. An empty (or all-comment) file parses to None, so
    ``.get`` raises AttributeError — which the surrounding bare handler then
    swallowed, making "your config file is empty" indistinguishable from "you
    have no config file". They call for different fixes.

    Two things make it safe, and both are accepted:

    * a guard — ``or {}`` on the parse, or an ``isinstance(data, dict)`` check
      in the same body. Then None never reaches ``.get``.
    * an enclosing handler that RECORDS. Then the AttributeError is surfaced
      rather than swallowed, which is the whole invariant; ``.get`` on a
      possibly-None value is untidy but not silent.
    """
    if any(_records(h) for h in node.handlers):
        return []

    body_src = ast.unparse(_module(node.body))
    unguarded: dict[str, int] = {}
    for stmt in ast.walk(_module(node.body)):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        src = ast.unparse(stmt.value)
        if "safe_load" not in src and "json.load" not in src:
            continue
        # `or {}` on the parse itself makes it safe.
        if isinstance(stmt.value, ast.BoolOp) and isinstance(stmt.value.op, ast.Or):
            continue
        # ...so does an explicit isinstance check on the same name.
        if f"isinstance({target.id}, dict)" in body_src:
            continue
        unguarded[target.id] = stmt.lineno

    hits: list[tuple[int, str]] = []
    if not unguarded:
        return hits
    for call in ast.walk(_module(node.body)):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "get"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in unguarded):
            hits.append((call.lineno,
                         f"{call.func.value.id}.get(...) on a possibly-None "
                         f"parse result"))
    # The chained form: yaml.safe_load(...).get(...)
    for call in ast.walk(_module(node.body)):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "get"
                and isinstance(call.func.value, ast.Call)
                and "safe_load" in ast.unparse(call.func.value)):
            hits.append((call.lineno, "safe_load(...).get(...) — unguarded"))
    return hits


def scan(source: str) -> list[tuple[int, str]]:
    """Every violation in *source*, as (line, why)."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Try) or not _is_config_read(node):
            continue
        for handler in node.handlers:
            broad, why = _is_broad(handler)
            if broad and _substitutes(handler) and not _records(handler):
                hits.append((handler.lineno, f"{why} substitutes silently"))
            elif broad and not _records(handler):
                hits.append((handler.lineno, f"{why} swallows silently"))
        hits.extend(_unguarded_none_get(node))
    return sorted(set(hits))


def _python_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py")
                  if "__pycache__" not in p.parts)


# ── the scan is real ──────────────────────────────────────────────────────

def test_the_package_is_actually_walked() -> None:
    files = _python_files()
    assert len(files) > 100, f"expected the whole package, found {len(files)}"
    assert any(p.name == "checker.py" for p in files)
    assert any(p.name == "budget.py" for p in files)


@pytest.mark.parametrize("path", _python_files(),
                         ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_silent_config_read(path: Path) -> None:
    hits = scan(path.read_text(encoding="utf-8"))
    assert not hits, (
        f"{path.relative_to(SRC_ROOT.parent.parent)} swallows a config read: "
        + "; ".join(f"line {ln}: {why}" for ln, why in hits)
        + ". Catch what you mean (OSError / yaml.YAMLError), and name the path "
        "attempted and the value substituted — see prometheus.config.load, "
        "which does this once for everything that reads prometheus.yaml. "
        "'I could not read your configuration' and 'you did not configure "
        "anything' are different facts."
    )


# ── both directions: the detector must catch what actually shipped ────────

def test_detector_catches_the_known_bad_shapes() -> None:
    """Replay. A detector matching nothing looks exactly like a clean tree."""

    # 1. The shape in all four unremediated modules.
    redundant = '''
try:
    with open(config_path) as fh:
        data = yaml.safe_load(fh)
    section = data.get("security", {})
except (OSError, Exception):
    section = {}
'''
    hits = scan(redundant)
    assert hits, "missed the (OSError, Exception) shape"
    assert any("redundant" in why for _, why in hits)
    assert any("possibly-None" in why for _, why in hits), (
        "missed the unguarded .get on a safe_load result in the same body"
    )

    # 2. A bare except around a yaml load.
    bare = '''
try:
    with open(p) as fh:
        cfg = yaml.safe_load(fh) or {}
except:
    cfg = {}
'''
    assert any("bare" in why for _, why in scan(bare)), "missed bare except"

    # 3. The fourth state on its own — chained, no enclosing assignment.
    chained = '''
try:
    section = yaml.safe_load(open(p).read()).get("context", {})
except Exception:
    section = {}
'''
    assert any("unguarded" in why or "possibly-None" in why
               for _, why in scan(chained)), "missed safe_load(...).get(...)"


def test_detector_accepts_the_remediated_shape() -> None:
    """The fix must pass, or the guard just bans config reads."""
    good = '''
try:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
except (OSError, yaml.YAMLError) as exc:
    log.error("config %s: UNREADABLE — %s (%s); using %s",
              subsystem, path, exc, substituting)
    record_silent_failure(subsystem, "load", exc, {"path": str(path)})
    raw = {}
'''
    assert scan(good) == [], f"the remediated shape was flagged: {scan(good)}"


def test_detector_accepts_a_narrow_silent_absence() -> None:
    """A narrow catch is the remediation; absence is the answer for some files.

    config/ephemeral.py, gateway/status.py, sticker_cache.py and the golden
    trace watermark all rely on this. Flagging them would get the guard
    disabled — see the module docstring.
    """
    narrow = '''
try:
    with open(p) as fh:
        return json.load(fh)
except FileNotFoundError:
    return {}
'''
    assert scan(narrow) == []


def test_detector_ignores_frontmatter_parsing() -> None:
    """safe_load over an in-memory string is not a config read."""
    frontmatter = '''
try:
    data = yaml.safe_load(block)
except yaml.YAMLError:
    data = None
'''
    assert scan(frontmatter) == []
