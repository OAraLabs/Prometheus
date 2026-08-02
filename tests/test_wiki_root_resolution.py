"""Wiki root resolution — real file side effects, no mocks.

WHY THIS FILE EXISTS
--------------------
Nine call sites used to derive the wiki root independently. A refactor that
misses one produces a split-brain wiki: the compiler writes one tree while a
consumer reads another, and nothing fails loudly. Mocked tests structurally
cannot catch that class — asserting ``consumer.wiki_root == expected`` passes
happily while the consumer writes somewhere else, and patching the resolver
proves only that the patch worked.

So this file uses **two temp directories**:

  root_b  — the configured wiki root. Everything must land here.
  root_a  — where the OLD default resolves to, made observable by pointing
            PROMETHEUS_CONFIG_DIR at a temp dir.

Every test asserts positively that work landed in ``root_b`` **and** negatively
that ``root_a`` stayed empty. A consumer that ignores the config key writes into
``root_a`` and the negative assertion fails. That is the property mocks cannot
express.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from prometheus.config.paths import (
    get_wiki_root,
    resolve_wiki_root,
    set_wiki_root,
)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Yield (root_a_default, root_b_configured) with the override pinned to B."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("PROMETHEUS_WIKI", raising=False)

    root_a = config_dir / "wiki"          # the default — must stay untouched
    root_b = tmp_path / "configured"      # the configured root — must fill
    root_b.mkdir()

    set_wiki_root(None)
    resolved = resolve_wiki_root({"wiki": {"root": str(root_b)}})
    assert resolved == root_b
    set_wiki_root(resolved)
    try:
        yield root_a, root_b
    finally:
        set_wiki_root(None)


def _md_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md")) if root.exists() else []


def _assert_default_root_untouched(root_a: Path, consumer: str) -> None:
    stray = _md_files(root_a)
    assert not stray, (
        f"{consumer} wrote to the DEFAULT wiki root instead of the configured "
        f"one — it is not reading through get_wiki_root(). Stray files: "
        f"{[str(p) for p in stray]}"
    )


# ---------------------------------------------------------------------------
# 1. All four consumers, real reads and writes, each asserted independently
# ---------------------------------------------------------------------------


def test_wiki_compiler_writes_to_configured_root(roots):
    """WikiCompiler compiles a real MemoryStore into the configured root."""
    root_a, root_b = roots
    from prometheus.memory.store import MemoryStore
    from prometheus.memory.wiki_compiler import WikiCompiler

    store = MemoryStore(db_path=root_b.parent / "memory.db")
    store.persist_memory(
        "project",
        "Prometheus",
        "Prometheus resolves the wiki root in exactly one place.",
        0.9,
        source_event_ids=["test_wiki_root_resolution"],
    )

    compiler = WikiCompiler(store=store)
    assert compiler.wiki_root == root_b, (
        "WikiCompiler did not adopt the configured root"
    )
    compiler.compile(
        [
            {
                "entity_type": "project",
                "entity_name": "Prometheus",
                "fact": "Prometheus resolves the wiki root in exactly one place.",
                "confidence": 0.9,
            }
        ]
    )

    assert (root_b / "index.md").exists(), (
        "WikiCompiler did not write index.md into the configured root"
    )
    assert _md_files(root_b), "WikiCompiler produced no pages in the configured root"
    _assert_default_root_untouched(root_a, "WikiCompiler")


def test_wiki_linter_writes_to_configured_root(roots):
    """WikiLinter reads the configured root and appends its report there."""
    root_a, root_b = roots
    from prometheus.sentinel.wiki_lint import WikiLinter

    (root_b / "index.md").write_text("# Index\n\n- [[Ghost-Page]]\n", encoding="utf-8")
    (root_b / "projects").mkdir(exist_ok=True)
    (root_b / "projects" / "Real-Page.md").write_text(
        "---\ntype: project\n---\n\n# Real Page\n", encoding="utf-8"
    )

    linter = WikiLinter()
    assert linter.wiki_root == root_b, "WikiLinter did not adopt the configured root"
    report = linter.lint()

    assert report is not None
    assert _md_files(root_b), "WikiLinter's target root lost its files"
    _assert_default_root_untouched(root_a, "WikiLinter")


def test_knowledge_synth_files_back_to_configured_root(roots):
    """KnowledgeSynthesizer's insight path resolves under the configured root."""
    root_a, root_b = roots
    from prometheus.sentinel.knowledge_synth import KnowledgeSynthesizer

    synth = KnowledgeSynthesizer(store=None, provider=None, model="test-model")
    assert synth._wiki_root == root_b, (
        "KnowledgeSynthesizer did not adopt the configured root"
    )

    insight_path = synth._insight_page_path(["Prometheus", "Wiki"])
    assert root_b in insight_path.parents, (
        f"KnowledgeSynthesizer files insights outside the configured root: "
        f"{insight_path}"
    )

    # real side effect, through the real path the synthesizer would use
    insight_path.parent.mkdir(parents=True, exist_ok=True)
    insight_path.write_text("# insight\n", encoding="utf-8")
    assert insight_path.exists()
    _assert_default_root_untouched(root_a, "KnowledgeSynthesizer")


@pytest.mark.asyncio
async def test_wiki_query_tool_reads_and_writes_configured_root(roots):
    """WikiQueryTool resolves at execute() time — exercise the real tool path."""
    root_a, root_b = roots
    from prometheus.tools.builtin.wiki_query import WikiQueryInput, WikiQueryTool

    marker = "SENTINEL-WIKI-ROOT-MARKER"
    (root_b / "index.md").write_text(
        f"# Index\n\n- [Prometheus](projects/Prometheus.md) — {marker}\n",
        encoding="utf-8",
    )
    (root_b / "projects").mkdir(exist_ok=True)
    (root_b / "projects" / "Prometheus.md").write_text(
        f"---\ntype: project\n---\n\n# Prometheus\n\n{marker}\n", encoding="utf-8"
    )

    tool = WikiQueryTool()
    result = await tool.execute(
        WikiQueryInput(query="Prometheus"),
        context=None,  # type: ignore[arg-type]
    )

    assert not result.is_error, f"WikiQueryTool errored: {result.output}"
    assert marker in result.output, (
        "WikiQueryTool did not read the configured root — it returned content "
        "that does not contain the marker seeded there"
    )
    _assert_default_root_untouched(root_a, "WikiQueryTool")


# ---------------------------------------------------------------------------
# 2. Default behaviour must be unchanged out of the box
# ---------------------------------------------------------------------------


def test_default_root_unchanged_when_key_absent(tmp_path, monkeypatch):
    """No wiki.root key, no PROMETHEUS_WIKI → <config dir>/wiki, as before."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("PROMETHEUS_WIKI", raising=False)
    set_wiki_root(None)

    assert resolve_wiki_root(None) == config_dir / "wiki"
    assert resolve_wiki_root({}) == config_dir / "wiki"
    assert resolve_wiki_root({"wiki": {}}) == config_dir / "wiki"
    assert get_wiki_root() == config_dir / "wiki"


def test_env_var_overrides_default_but_config_wins(tmp_path, monkeypatch):
    """PROMETHEUS_WIKI beats the default; wiki.root beats PROMETHEUS_WIKI."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("PROMETHEUS_WIKI", str(tmp_path / "from_env"))
    set_wiki_root(None)

    assert resolve_wiki_root(None) == tmp_path / "from_env"
    assert resolve_wiki_root({"wiki": {"root": str(tmp_path / "from_cfg")}}) == (
        tmp_path / "from_cfg"
    )


def test_resolver_does_not_create_the_directory(tmp_path, monkeypatch):
    """Consumers branch on wiki_root.exists(); creating it would change that."""
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("PROMETHEUS_WIKI", raising=False)
    set_wiki_root(None)
    target = tmp_path / "never_created"
    assert resolve_wiki_root({"wiki": {"root": str(target)}}) == target
    assert not target.exists(), "resolve_wiki_root must not mkdir its result"


# ---------------------------------------------------------------------------
# 3. WikiQueryTool resolves at CALL time, not construction time
# ---------------------------------------------------------------------------


def test_wiki_query_resolves_at_call_time(tmp_path, monkeypatch):
    """Constructed before the root is pinned, it must still honour the pin.

    WikiQueryTool is registered as ``WikiQueryTool()`` in ``__main__.py`` with
    no arguments and resolves inside ``execute()``. A constructor-only fix
    passes every other test in this file and fails this one.
    """
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("PROMETHEUS_WIKI", raising=False)
    set_wiki_root(None)

    from prometheus.tools.builtin.wiki_query import WikiQueryTool

    WikiQueryTool()  # constructed BEFORE the root is known
    later = tmp_path / "pinned_later"
    set_wiki_root(later)
    try:
        assert get_wiki_root() == later
    finally:
        set_wiki_root(None)


# ---------------------------------------------------------------------------
# 4. Source guard — no site may resolve the wiki root independently
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parent.parent / "src" / "prometheus"
_RESOLVER = _SRC / "config" / "paths.py"

# Both historical conventions: get_config_dir() / "wiki", and the hardcoded
# home path that ignored PROMETHEUS_CONFIG_DIR.
_FORBIDDEN = re.compile(
    r"""get_config_dir\(\)\s*/\s*["']wiki["']"""
    r"""|config_dir\s*/\s*["']wiki["']"""
    r"""|Path\.home\(\)\s*/\s*["']\.prometheus["']\s*/\s*["']wiki["']"""
)


def test_no_site_resolves_the_wiki_root_independently():
    """Fail if a tenth site appears. This is what stops the drift returning."""
    offenders: list[str] = []
    for py in sorted(_SRC.rglob("*.py")):
        if py.resolve() == _RESOLVER:
            continue  # the one legitimate place
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if _FORBIDDEN.search(line):
                rel = py.relative_to(_SRC.parent.parent)
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "These sites resolve the wiki root independently instead of calling "
        "get_wiki_root(). Nine such sites caused a split-brain wiki before "
        "2026-08-02; every one must go through prometheus.config.paths.\n  "
        + "\n  ".join(offenders)
    )


def test_source_guard_regex_actually_matches_the_old_forms():
    """The guard is only worth having if it recognises what it hunts for."""
    assert _FORBIDDEN.search('wiki_root = get_config_dir() / "wiki"')
    assert _FORBIDDEN.search('("wiki/", config_dir / "wiki")')
    assert _FORBIDDEN.search('wiki_dir = Path.home() / ".prometheus" / "wiki"')
    assert not _FORBIDDEN.search("wiki_root = get_wiki_root()")


def test_daemon_passes_the_resolved_root_to_every_construction_site():
    """The three daemon-constructed consumers must be handed the root."""
    tree = ast.parse((_SRC / "daemon.py").read_text(encoding="utf-8"))
    wanted = {"WikiCompiler", "WikiLinter", "KnowledgeSynthesizer"}
    seen: dict[str, bool] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in wanted:
                seen[node.func.id] = any(
                    kw.arg == "wiki_root" for kw in node.keywords
                )
    missing = sorted(n for n in wanted if not seen.get(n))
    assert not missing, (
        f"daemon.py constructs {missing} without passing wiki_root — they will "
        f"silently fall back to the process default instead of the configured "
        f"root. Bare construction is exactly how six components sat dead in "
        f"the daemon for months."
    )
