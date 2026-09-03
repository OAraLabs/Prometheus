"""THE INVARIANT: no bare integer literal is used as a context limit or token
budget anywhere in ``src/prometheus/web/``, ``src/prometheus/gateway/`` or
``src/prometheus/context/``.

WHY THIS FILE EXISTS
--------------------
``create_app()`` had no resolved context budget in scope, so the one route
that needed one typed the old default. ``/api/lcm/{session_id}`` reported
``"limit": 24000`` four times over, on a daemon whose config said 72000 and
whose llama.cpp server reported 32768 — three numbers, none of them each
other.

The mislabelled denominator was the visible half. The damaging half was
``lcm_engine.assemble(session_id, token_budget=24000)``: the endpoint ran a
real assembly against a fabricated window, so ``total_tokens``,
``fresh_count``, ``summary_count`` and ``compression_ratio`` were all computed
against a context that does not exist. Beacon was not showing a stale view of
the real state — it was showing a parallel one. 9888/32768 (30%) rendered
as 41%.

This is the same failure class ``daemon.py`` documents having already fixed on
the agent-loop path ("a config effective_limit that outlived a model swap
silently won... n_ctx=32768 came to be budgeted at 72000"). Review did not
catch it there either; a passing build is what caught it. Hence a test, not a
convention.

WHY THREE PACKAGES
------------------
Fixing only ``web/`` made the surfaces disagree with each other, which is
worse than the uniform wrongness before it: Beacon reported the detected
32768 while every chat gateway's /context reported a different number
entirely, and nothing on either surface said which was authoritative. So the
guard covers every package that resolves or displays a context window:

  web/       the REST + WS surfaces (/api/lcm, /api/status, web slash)
  gateway/   the chat surfaces (/context on Telegram, Slack, Discord)
  context/   the resolver and the ENFORCEMENT path (ContextCompactor), where
             a fabricated number does not merely mislead — it cuts prompts

``context/budget.py`` is the one place a fallback integer is legitimate, and
it is named (``LEGACY_FALLBACK_LIMIT``) rather than inline, which is exactly
the distinction this guard draws: a named constant is a decision, a bare
literal in a call is an accident.

WHAT IS AND IS NOT FLAGGED
--------------------------
Flagged: an int literal bound to a name that MEANS a context window
(``token_budget``, ``effective_limit``, ``n_ctx``, …) as a keyword argument, a
dict entry, or an assignment. Also flagged: a ``"limit"`` dict entry sitting
next to context-shaped siblings (``total_tokens``, ``compression_ratio``, …),
which is the exact shape of the bug.

Not flagged: pagination and feed limits (``recent(limit=100)``,
``_SEARCH_MAX_LIMIT = 50``). Those are genuinely constants of this layer and
have nothing to do with a model's context window. A guard that failed on them
would be turned off within a week, which is worse than no guard.

CAPABILITY FLAGS — THE SAME DEFECT IN A BOOLEAN
-----------------------------------------------
The fourth instance in a week was not an integer. ``GET /api/models`` carried
``"vision": False`` on the ``local`` row as a literal ("Phase 1") while the
daemon had probed the server, found an mmproj, and logged "Vision: enabled
(multimodal)". Same shape: a surface asserting a constant where the system
held a detected value; a client (Beacon) then believed the local model could
not see and withheld the picture. So a capability the system can DETECT may
never be a literal in a catalog row either: a ``"vision"`` (or
``"supports_vision"``) entry whose value is a bool constant, inside a dict that
is shaped like a catalog row (a ``"provider"``, ``"key"`` or ``"model"``
sibling), is flagged. ``bool(preset.get("vision", False))`` is not — that is a
read of a DECLARED flag with absence meaning False, which is the documented
posture for cloud presets whose capability cannot be probed.

BOTH DIRECTIONS
---------------
A guard that flags nothing because its AST walk silently matches nothing looks
identical to a clean tree. ``test_detector_catches_the_original_bug`` replays
the four literals that shipped and asserts every one is caught, so a green
result from the real scan means the detector is looking, not blind.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import prometheus.context as context_pkg
import prometheus.gateway as gateway_pkg
import prometheus.web as web_pkg

GUARDED_ROOTS = {
    "web": Path(web_pkg.__file__).resolve().parent,
    "gateway": Path(gateway_pkg.__file__).resolve().parent,
    "context": Path(context_pkg.__file__).resolve().parent,
}
REPO_ROOT = GUARDED_ROOTS["web"].parents[2]

# Names that mean "a model's context window" or "how much of it may be spent".
# A bare integer bound to any of these is a fabricated budget by definition:
# the real value is resolved at runtime from the server or the config.
CONTEXT_NAMES = frozenset({
    "token_budget",
    "effective_limit",
    "context_limit",
    "detected_limit",
    "detected_context_size",
    "reserved_output",
    "max_context_tokens",
    "n_ctx",
    "cloud_default_limit",
})

# Siblings that identify a dict as a context-window payload, so a plain
# ``"limit"`` key inside it is a context limit rather than a page size.
CONTEXT_SIBLINGS = frozenset({
    "total_tokens",
    "compression_ratio",
    "fresh_count",
    "summary_count",
    "limit_source",
})

# Capabilities the system DETECTS (llama.cpp ``/props`` modalities, a provider
# class attribute). A bool literal bound to one of these inside a catalog row is
# an assertion standing where a probe result belongs.
CAPABILITY_NAMES = frozenset({"vision", "supports_vision"})

# Siblings that identify a dict as a catalog / model row, so a ``"vision"``
# literal inside it is a capability claim about a model rather than, say, a
# feature toggle in an unrelated payload.
CATALOG_SIBLINGS = frozenset({"provider", "key", "model"})


def _is_bool_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, bool)


def _is_int_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(
        node.value, bool
    )


class _ContextLiteralVisitor(ast.NodeVisitor):
    """Collect (line, name, value) for every bare int used as a budget."""

    def __init__(self) -> None:
        self.hits: list[tuple[int, str, int]] = []

    def visit_Call(self, node: ast.Call) -> None:
        for kw in node.keywords:
            if kw.arg in CONTEXT_NAMES and _is_int_literal(kw.value):
                self.hits.append((kw.value.lineno, f"{kw.arg}=", kw.value.value))

        # The dict-get default: ``ctx.get("effective_limit", 24000)``. This is
        # the exact shape that shipped in ContextCompactor.from_config and in
        # media_services, and it is the most deceptive of the family — the
        # literal reads as a harmless fallback while being the number actually
        # enforced whenever the key is absent. It is invisible to the kwarg and
        # assignment rules: the literal is a positional argument, and the
        # enclosing assignment's value is a Call (``int(...)``), not a constant.
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value in CONTEXT_NAMES
            and _is_int_literal(node.args[1])
        ):
            self.hits.append((
                node.args[1].lineno,
                f'.get("{node.args[0].value}", …)',
                node.args[1].value,
            ))
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        keys = {
            k.value for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        contextual = bool(keys & CONTEXT_SIBLINGS)
        catalog_row = bool(keys & CATALOG_SIBLINGS)
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            name = key.value
            flagged = name in CONTEXT_NAMES or (name == "limit" and contextual)
            if flagged and _is_int_literal(value):
                self.hits.append((value.lineno, f'"{name}":', value.value))
            # A detectable capability asserted as a constant in a model row.
            if name in CAPABILITY_NAMES and catalog_row and _is_bool_literal(value):
                self.hits.append((value.lineno, f'"{name}":', value.value))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_int_literal(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.lower() in CONTEXT_NAMES:
                    self.hits.append((node.lineno, f"{target.id} =", node.value.value))
                elif (
                    isinstance(target, ast.Attribute)
                    and target.attr.lower() in CONTEXT_NAMES
                ):
                    self.hits.append((node.lineno, f".{target.attr} =", node.value.value))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            node.value is not None
            and _is_int_literal(node.value)
            and isinstance(node.target, ast.Name)
            and node.target.id.lower() in CONTEXT_NAMES
        ):
            self.hits.append((node.lineno, f"{node.target.id} =", node.value.value))
        self.generic_visit(node)

    def _visit_def(self, node: ast.AST) -> None:
        """Parameter defaults — ``def f(effective_limit: int = 24000)``.

        The same accident wearing a signature: every caller that omits the
        argument silently inherits a window nothing agreed to, and the call
        sites look clean.
        """
        args = node.args
        positional = args.posonlyargs + args.args
        padded = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        for arg, default in list(zip(positional, padded)) + list(
            zip(args.kwonlyargs, args.kw_defaults)
        ):
            if default is not None and _is_int_literal(default):
                if arg.arg.lower() in CONTEXT_NAMES:
                    self.hits.append(
                        (default.lineno, f"{arg.arg}=", default.value)
                    )
        self.generic_visit(node)

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def


def _scan(source: str) -> list[tuple[int, str, int]]:
    visitor = _ContextLiteralVisitor()
    visitor.visit(ast.parse(source))
    return visitor.hits


def _python_files() -> list[Path]:
    return sorted(
        p
        for root in GUARDED_ROOTS.values()
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
    )


@pytest.mark.parametrize("package", sorted(GUARDED_ROOTS))
def test_every_guarded_package_is_actually_scanned(package: str) -> None:
    """The walk finds files in EACH root. A guard over an empty set is not a
    guard, and a typo'd root would silently stop covering a whole package."""
    root = GUARDED_ROOTS[package]
    files = [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
    assert len(files) >= 3, f"expected the {package} package, found {files}"


def test_the_three_known_surfaces_are_covered() -> None:
    """Name the files this guard exists for, so a move can't silently drop
    one out of scope."""
    names = {p.name for p in _python_files()}
    assert {"server.py", "commands.py", "compactor.py", "budget.py"} <= names


@pytest.mark.parametrize(
    "path", _python_files(), ids=lambda p: f"{p.parent.name}/{p.name}"
)
def test_no_hardcoded_context_budget(path: Path) -> None:
    hits = _scan(path.read_text(encoding="utf-8"))
    assert not hits, (
        f"{path.relative_to(REPO_ROOT)} hardcodes a context budget or a "
        f"detectable capability: "
        + ", ".join(f"line {ln}: {name} {val}" for ln, name, val in hits)
        + ". Resolve it through prometheus.context.budget.resolve_effective_limit "
        "instead — a literal here is a window that exists nowhere in the system. "
        "On a display path it renders a confident wrong denominator; on an "
        "enforcement path (compactor, truncation) it cuts real prompts. If a "
        "fallback genuinely belongs here, give it a NAME "
        "(LEGACY_FALLBACK_LIMIT) so it reads as a decision, not an accident."
    )


def test_detector_catches_the_original_bug() -> None:
    """Replay the four literals that shipped; every one must be caught.

    Without this, a detector that silently matched nothing would produce the
    same green as a clean tree.
    """
    shipped = '''
def get_lcm_state(session_id, lcm_engine):
    if not lcm_engine:
        return {
            "session_id": session_id,
            "total_tokens": 0,
            "limit": 24000,
            "compression_ratio": 0,
            "fresh_count": 0,
            "summary_count": 0,
        }
    result = lcm_engine.assemble(session_id, token_budget=24000)
    return {
        "session_id": session_id,
        "total_tokens": result.total_tokens,
        "limit": 24000,
        "compression_ratio": result.compression_ratio,
        "fresh_count": 0,
        "summary_count": 0,
    }
'''
    hits = _scan(shipped)
    assert len(hits) == 3, f"expected all three literals, got {hits}"
    assert {v for _, _, v in hits} == {24000}


def test_detector_ignores_pagination_limits() -> None:
    """Page sizes are not context windows. Flagging them would kill the guard."""
    benign = '''
_SEARCH_MAX_LIMIT = 50
recent = signal_bus.recent(limit=100)
page = {"items": [], "limit": 20, "offset": 0}
'''
    assert _scan(benign) == []


def test_detector_catches_the_gateway_shapes() -> None:
    """The literals this PR removed from the gateway and the compactor.

    Both were invisible in review: one sat on an except-branch that only fires
    when config loading fails, the other was a dict-get default two lines
    above a min() that made it look harmless.
    """
    gateway_except = """
try:
    budget = TokenBudget.from_config(model=model_name)
    effective_limit = budget.effective_limit
except Exception:
    effective_limit = 24000
    reserved_output = 2000
"""
    assert {v for _, _, v in _scan(gateway_except)} == {24000, 2000}

    compactor_default = """
effective_limit = int(ctx.get("effective_limit", 24000))
"""
    # The dict-get default is a positional arg, not a kwarg — caught because
    # the ASSIGNMENT target is a context name.
    assert {v for _, _, v in _scan(compactor_default)} == {24000}

    signature_default = """
def truncate(text, *, effective_limit: int = 24000, reserved_output=2000):
    return text
"""
    assert {v for _, _, v in _scan(signature_default)} == {24000, 2000}


def test_detector_catches_a_context_limit_assignment() -> None:
    """The other shape the same mistake takes: a module-level constant."""
    variant = '''
effective_limit = 72000
DEFAULTS = {"token_budget": 32768}
'''
    hits = _scan(variant)
    assert {v for _, _, v in hits} == {72000, 32768}


def test_detector_catches_the_shipped_vision_literal() -> None:
    """Replay of #387: the ``local`` catalog row as it shipped. The system had
    detected vision (llama.cpp ``/props`` modalities → ``supports_vision``); the
    row said False because someone typed False. One hit, and it is that one."""
    shipped = '''
catalog = [{
    "key": "local", "label": "Local", "provider": primary_provider,
    "model": primary_model, "is_default": True, "available": True,
    "auth": None,
    "vision": False,
}]
'''
    hits = _scan(shipped)
    assert [(name, val) for _, name, val in hits] == [('"vision":', False)]


def test_detector_ignores_a_declared_capability_read() -> None:
    """The cloud rows READ a declared flag with absence-is-False. That is a
    config read, not an assertion, and it must stay legal — the cloud APIs
    publish no capability to detect."""
    declared = '''
row = {
    "key": key, "provider": preset.get("provider", "unknown"), "model": model,
    "vision": bool(preset.get("vision", False)),
}
'''
    assert _scan(declared) == []


def test_detector_ignores_a_vision_flag_outside_a_model_row() -> None:
    """``"vision"`` as a feature toggle in an unrelated payload is not a claim
    about a model. Only a row shaped like the catalog (provider/key/model
    sibling) is in scope, so the rule cannot fire on a settings dict."""
    toggle = '''
features = {"vision": False, "tts": True, "search": True}
'''
    assert _scan(toggle) == []
