"""Brain-vault tools — real files, real symlinks, both directions.

The vault is READ-ONLY to Prometheus and that has to be a property, not a
policy: ``is_read_only()`` returning True is a claim the tool makes about
itself, and a claim cannot fail. So the write guard here parses the module's
AST and fails the build if a write call appears at all — the same shape as the
config-drift and tool-name guards, and the only version that survives someone
adding a "small" cache write later.

Both directions throughout (§2c). Every confinement test has an admission
twin: refusing everything would satisfy a suite of refusals perfectly, and a
vault tool that reads nothing looks exactly like a vault tool that is
correctly locked down.
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
from pathlib import Path

import pytest

from prometheus.config.paths import (
    get_vault_root,
    resolve_vault_root,
    set_vault_root,
)
from prometheus.tools.builtin.vault import VaultReadTool, VaultSearchTool

MODULE = (
    Path(__file__).resolve().parents[1]
    / "src" / "prometheus" / "tools" / "builtin" / "vault.py"
)

# The operator's ACTUAL vault, taken from the environment rather than named
# here. Two reasons: this repo is public and a real vault path does not belong
# in a tracked file (the pre-commit hook enforces that), and a hardcoded name
# would make the real-symlink test below skip forever the moment anyone's vault
# lived somewhere else. Set PROMETHEUS_VAULT to run it.
_REAL_VAULT_ENV = os.environ.get("PROMETHEUS_VAULT")
REAL_VAULT = Path(_REAL_VAULT_ENV).expanduser() if _REAL_VAULT_ENV else None
REAL_SYMLINK = (REAL_VAULT / ".venv" / "bin" / "python") if REAL_VAULT else None


# ---------------------------------------------------------------------------
# Fixture vault — mirrors the real tree's shape, including its escape surface
# ---------------------------------------------------------------------------

@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A miniature brain vault, pinned as the process root."""
    root = tmp_path / "brain-vault"
    (root / "wiki" / "sources" / "concepts").mkdir(parents=True)
    (root / "wiki" / "sources" / "projects").mkdir(parents=True)
    (root / "raw" / "claude-chats").mkdir(parents=True)
    (root / "notes").mkdir()

    (root / "CLAUDE.md").write_text("# Vault Router\n", encoding="utf-8")
    (root / "wiki" / "index.md").write_text(
        "# Brain Wiki Index\n\n"
        "- [Standing Principles](sources/concepts/Standing-Principles.md)"
        " — the compressed cost of every repeated failure\n"
        "- [Prometheus](sources/projects/Prometheus.md)"
        " — the sovereign agent\n",
        encoding="utf-8",
    )
    (root / "wiki" / "sources" / "concepts" / "Standing-Principles.md").write_text(
        "---\ntype: concept\n---\n\n# Standing Principles\n\n"
        "Say what each check actually proves.\n"
        "An import check is not a scope check.\n"
        "A green suite proves the code runs, not that anything calls it.\n",
        encoding="utf-8",
    )
    (root / "wiki" / "sources" / "projects" / "Prometheus.md").write_text(
        "---\ntype: project\n---\n\n# Prometheus\n\n"
        "The daemon runs a llama.cpp backend.\n",
        encoding="utf-8",
    )
    (root / "raw" / "claude-chats" / "2026-04-24-prometheus3.md").write_text(
        "raw conversation export, unsummarised\n", encoding="utf-8",
    )
    (root / "notes" / "scratch.md").write_text("human scratch\n", encoding="utf-8")

    # The escape surface, reproduced with the SAME shape the real vault has:
    # .venv/bin/python is an absolute symlink to an interpreter outside the
    # tree. tests below exercise the real one too where it exists.
    outside = tmp_path / "outside-the-vault.txt"
    outside.write_text("SECRET OUTSIDE THE VAULT\n", encoding="utf-8")
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(outside)
    (root / ".venv" / "lib64").symlink_to("lib")  # internal, like the real one
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    # A symlink escaping from a CONTENT directory. The .venv one above is
    # refused by two independent controls (containment AND the non-content
    # exclusion), so a test using it cannot tell which one fired — mutation V2
    # disabled the symlink resolve and that test stayed green. This one sits
    # in wiki/, where only containment can save it.
    (root / "wiki" / "sneaky.md").symlink_to(outside)

    monkeypatch.delenv("PROMETHEUS_VAULT", raising=False)
    set_vault_root(root)
    try:
        yield root, outside
    finally:
        set_vault_root(None)


def _search(query: str) -> str:
    r = asyncio.run(VaultSearchTool().execute(
        VaultSearchTool.input_model(query=query), None,
    ))
    return r.output


def _search_result(query: str):
    return asyncio.run(VaultSearchTool().execute(
        VaultSearchTool.input_model(query=query), None,
    ))


def _read_result(path: str):
    return asyncio.run(VaultReadTool().execute(
        VaultReadTool.input_model(path=path), None,
    ))


# ---------------------------------------------------------------------------
# 1. Read-only, structurally
# ---------------------------------------------------------------------------

_WRITE_FUNCS = {
    "write_text", "write_bytes", "mkdir", "touch", "unlink", "rmdir",
    "rename", "replace", "symlink_to", "hardlink_to", "chmod", "remove",
    "rmtree", "copy", "copy2", "copytree", "move", "makedirs",
}


def test_the_module_contains_no_write_call_at_all():
    """Not "unused" — ABSENT. A read-only flag is a claim the tool makes about
    itself and claims do not fail; this parses the source and fails the build
    if a write appears, including one added in good faith later."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    found = sorted({
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _WRITE_FUNCS
    })
    assert not found, (
        f"vault.py calls {found} — the brain vault is read-only to Prometheus. "
        f"Its CLAUDE.md §1 gives raw/ to nobody, wiki/memory/ to the Prometheus "
        f"compiler and notes/ to Will; 'read anywhere, write nowhere' is the "
        f"standing rule for every agent touching it."
    )


def test_the_module_never_opens_a_file_for_writing():
    """The other half: ``open(p, 'w')`` is not an attribute call."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open":
            modes = [
                a.value for a in node.args[1:2]
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            ] + [
                kw.value.value for kw in node.keywords
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant)
            ]
            for m in modes:
                assert not set(m) & set("wax+"), f"vault.py opens a file with mode {m!r}"


def test_the_write_guard_would_actually_catch_a_write():
    """A guard never observed failing is a claim, not a check."""
    sample = ast.parse("from pathlib import Path\nPath('x').write_text('y')\n")
    found = {
        n.func.attr for n in ast.walk(sample)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in _WRITE_FUNCS
    }
    assert found == {"write_text"}


@pytest.mark.parametrize("tool", [VaultSearchTool(), VaultReadTool()])
def test_tools_declare_read_only(tool):
    assert tool.is_read_only(None) is True


# ---------------------------------------------------------------------------
# 2. Confinement — refusal AND admission
# ---------------------------------------------------------------------------

def test_traversal_out_of_the_vault_is_refused(vault):
    root, outside = vault
    r = _read_result("../outside-the-vault.txt")
    assert r.is_error
    assert "escapes the brain vault root" in r.output
    assert "SECRET" not in r.output


def test_an_absolute_path_outside_the_vault_is_refused(vault):
    root, outside = vault
    r = _read_result(str(outside))
    assert r.is_error
    assert "SECRET" not in r.output


def test_a_symlink_pointing_out_of_the_vault_is_refused(vault):
    """The escape surface reproduced from the real tree: .venv/bin/python is
    an absolute symlink to an interpreter outside the vault."""
    root, outside = vault
    assert (root / ".venv" / "bin" / "python").is_symlink()  # precondition
    r = _read_result(".venv/bin/python")
    assert r.is_error
    assert "SECRET" not in r.output


def test_a_symlink_escaping_from_a_CONTENT_directory_is_refused(vault):
    """The test that actually pins the symlink guard.

    ``.venv/bin/python`` is refused by TWO controls — containment and the
    non-content exclusion — so disabling the symlink resolve leaves that test
    green (observed: mutation V2 survived it). This link lives in ``wiki/``,
    where the exclusion does not apply, so only resolve-then-confine can
    refuse it. §3b: if deleting the control a test is named after still leaves
    something else producing the same refusal, the test does not test it."""
    root, outside = vault
    link = root / "wiki" / "sneaky.md"
    assert link.is_symlink() and link.resolve() == outside.resolve()

    r = _read_result("wiki/sneaky.md")
    assert r.is_error, "a symlink out of wiki/ was followed"
    assert "escapes the brain vault root" in r.output
    assert "SECRET" not in r.output


def test_a_symlink_escape_is_not_reachable_through_search_either(vault):
    """Search reads page bodies too, so the same escape must be closed there.

    This found a real bug: confinement had been applied to vault_read only,
    while vault_search's content sweep used ``rglob`` and read whatever it
    yielded — including a symlink out of the tree, whose contents it then
    printed. Two entry points, one boundary; guarding the obvious one is half
    a control."""
    out = _search("SECRET")
    assert "SECRET OUTSIDE THE VAULT" not in out


def test_an_index_entry_cannot_traverse_out_of_the_vault(vault):
    """index.md is a file like any other, so its links are untrusted input:
    ``[label](../../../secret)`` is a traversal wearing a friendly name."""
    root, outside = vault
    (root / "wiki" / "index.md").write_text(
        "# Index\n\n- [Innocent Page](../../outside-the-vault.txt)"
        " — SECRET bait\n",
        encoding="utf-8",
    )
    out = _search("SECRET bait")
    assert "SECRET OUTSIDE THE VAULT" not in out


def test_non_content_directories_are_refused(vault):
    root, _ = vault
    r = _read_result(".git/config")
    assert r.is_error
    assert "non-content" in r.output


@pytest.mark.skipif(
    REAL_SYMLINK is None or not REAL_SYMLINK.is_symlink(),
    reason=(
        "PROMETHEUS_VAULT is unset or its .venv symlink is absent. The "
        "fixture tests above reproduce the exact shape (an absolute symlink "
        "out of the tree, from both an excluded and a content directory); "
        "this one additionally exercises the OPERATOR'S REAL vault. Export "
        "PROMETHEUS_VAULT=<your vault> to run it."
    ),
)
def test_the_real_vaults_own_symlink_is_refused():
    """Against the genuine article, not a stand-in.

    ``~/brain-vault/.venv/bin/python`` really does point at
    ``~/.local/share/uv/python/.../bin/python3.11``. If confinement is wrong,
    this reads an interpreter binary from outside the vault."""
    set_vault_root(REAL_VAULT)
    try:
        r = _read_result(".venv/bin/python")
        assert r.is_error
        assert "escapes the brain vault root" in r.output or "non-content" in r.output
    finally:
        set_vault_root(None)


def test_reads_inside_the_vault_are_ADMITTED(vault):
    """The admission half. Every test above asserts a refusal, and a tool that
    refused everything would pass all of them."""
    root, _ = vault
    r = _read_result("wiki/sources/concepts/Standing-Principles.md")
    assert not r.is_error
    assert "Say what each check actually proves." in r.output


@pytest.mark.parametrize("path,needle", [
    ("wiki/index.md", "Wiki Index"),
    ("raw/claude-chats/2026-04-24-prometheus3.md", "unsummarised"),
    ("notes/scratch.md", "human scratch"),
    ("CLAUDE.md", "Vault Router"),
])
def test_every_readable_zone_is_actually_readable(vault, path, needle):
    """CLAUDE.md §1/§6: raw/ and notes/ are READABLE, just never writable.
    A confinement bug that locked them out would be invisible to the refusal
    tests and would quietly halve the vault."""
    r = _read_result(path)
    assert not r.is_error, r.output
    assert needle in r.output


# ---------------------------------------------------------------------------
# 3. Search — content, ranking, and the VISIBLE MISS
# ---------------------------------------------------------------------------

def test_search_returns_real_content_from_real_files(vault):
    """Side-effect test: this cannot pass if the tool is registered but
    unwired, or if it returns a canned string."""
    out = _search("standing principles")
    assert "Standing-Principles.md" in out
    assert "Say what each check actually proves." in out


def test_search_finds_a_term_that_appears_only_in_a_page_body(vault):
    """The index carries a one-line summary per page, so a body-only term is
    invisible to index scoring. Without the content sweep the tool would
    answer 'nothing found' for a fact the vault plainly contains."""
    out = _search("llama.cpp")
    assert "Prometheus.md" in out, out


def test_a_miss_names_its_own_scope_and_the_raw_tree(vault):
    """THE VISIBLE MISS. An empty result that says nothing invites the reader
    to conclude the vault has nothing on the subject — when raw/ was never
    searched at all. This is the LCM summary_store shape: 'no results' from
    the day the engine landed, and nothing ever said why."""
    r = _search_result("zzzznotinthevault")
    assert not r.is_error
    out = r.output
    assert "No matches" in out
    assert "wiki/" in out
    assert "raw/" in out
    assert "sources:" in out, "the escalation route must be named"
    assert "1 unsummarised source file" in out or "unsummarised source files" in out


def test_the_raw_file_count_in_a_miss_is_counted_not_hardcoded(vault):
    """A hardcoded figure becomes a lie the first time an ingest runs."""
    root, _ = vault
    before = _search_result("zzzznotinthevault").output
    assert " 1 unsummarised" in before, before

    (root / "raw" / "claude-chats" / "another.md").write_text("x", encoding="utf-8")
    after = _search_result("zzzznotinthevault").output
    assert " 2 unsummarised" in after, after


def test_the_journal_does_not_outrank_the_page_that_answers_the_question(vault):
    """Observed live before it was fixed: a query matched Standing-Principles
    at score 2 and wiki/log.md at 61, because the log is a chronological
    journal of every finding ever recorded and wins any raw term count.

    Index-first ordering hid it whenever the index matched — which is how it
    would have shipped. This asserts the content sweep's own ordering, on a
    term that is NOT in the index, so index-first cannot mask the result."""
    root, _ = vault
    (root / "wiki" / "log.md").write_text(
        "- [FINDING] scopecheck scopecheck scopecheck scopecheck scopecheck\n",
        encoding="utf-8",
    )
    (root / "wiki" / "sources" / "concepts" / "Scope.md").write_text(
        "---\ntype: concept\n---\n\n# Scope\n\nA scopecheck is the real page.\n",
        encoding="utf-8",
    )
    out = _search("scopecheck")
    assert "Scope.md" in out and "log.md" in out, out
    assert out.index("Scope.md") < out.index("log.md"), (
        "the append-only journal outranked the entity page the fact was "
        "compiled into:\n" + out
    )


def test_search_is_scoped_to_wiki_and_does_not_return_raw_pages(vault):
    """Scope is a decision, so it gets a test: raw/ is reachable by vault_read
    and by a page's sources: frontmatter, not by search."""
    out = _search("unsummarised")
    assert "2026-04-24-prometheus3.md" not in out


# ---------------------------------------------------------------------------
# 4. Loud failure on an absent / unusable root
# ---------------------------------------------------------------------------

def test_an_absent_vault_root_fails_loudly_naming_the_path(tmp_path, monkeypatch):
    monkeypatch.delenv("PROMETHEUS_VAULT", raising=False)
    missing = tmp_path / "no-vault-here"
    set_vault_root(missing)
    try:
        for r in (_search_result("anything"), _read_result("wiki/index.md")):
            assert r.is_error
            assert str(missing) in r.output
            assert "vault.root" in r.output
            assert "Prometheus wiki" in r.output, (
                "the error must disambiguate the two roots — that confusion is "
                "the whole reason they are separately configured"
            )
    finally:
        set_vault_root(None)


def test_a_root_that_is_a_file_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv("PROMETHEUS_VAULT", raising=False)
    f = tmp_path / "not-a-dir"
    f.write_text("x", encoding="utf-8")
    set_vault_root(f)
    try:
        r = _search_result("anything")
        assert r.is_error and "not a directory" in r.output
    finally:
        set_vault_root(None)


def test_a_root_without_a_wiki_tree_says_so_rather_than_returning_nothing(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PROMETHEUS_VAULT", raising=False)
    root = tmp_path / "bare"
    root.mkdir()
    set_vault_root(root)
    try:
        r = _search_result("anything")
        assert r.is_error
        assert "no wiki/ tree" in r.output
    finally:
        set_vault_root(None)


# ---------------------------------------------------------------------------
# 5. Root resolution — the #131 idiom, and no second way to name it
# ---------------------------------------------------------------------------

def test_resolution_order_config_then_env_then_default(tmp_path, monkeypatch):
    monkeypatch.delenv("PROMETHEUS_VAULT", raising=False)
    set_vault_root(None)
    assert resolve_vault_root() == Path.home() / "brain-vault"

    monkeypatch.setenv("PROMETHEUS_VAULT", str(tmp_path / "from-env"))
    assert resolve_vault_root() == tmp_path / "from-env"

    assert resolve_vault_root({"vault": {"root": str(tmp_path / "from-cfg")}}) == (
        tmp_path / "from-cfg"
    ), "config must outrank env"


def test_resolve_does_not_create_the_directory(tmp_path, monkeypatch):
    """An absent vault is a reportable state. Creating it would turn a loud
    failure into a silent no-results."""
    monkeypatch.setenv("PROMETHEUS_VAULT", str(tmp_path / "nope"))
    resolved = resolve_vault_root()
    assert not resolved.exists()


def test_the_vault_root_is_not_the_prometheus_wiki_root(tmp_path, monkeypatch):
    """They must not be conflated — different corpora, different owners,
    different write rules."""
    from prometheus.config.paths import get_wiki_root

    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("PROMETHEUS_VAULT", raising=False)
    set_vault_root(None)
    assert get_vault_root() != get_wiki_root()


_SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"


# Path CONSTRUCTION naming the vault, as opposed to prose that mentions it.
# The first version of this guard matched the bare string and flagged the
# tool's own docstring and description — which is exactly the over-firing that
# trains people to weaken a guard rather than fix a bug. It hunts a `/` or a
# Path(...) immediately before the name, which is what derivation looks like.
_FORBIDDEN_VAULT_PATH = re.compile(
    r"""(?:Path\s*\(|home\s*\(\s*\)\s*|/\s*)["']~?/?brain-vault["']"""
)


def test_no_second_hardcoded_vault_path_in_src():
    """The #131 source guard, applied to the second root. One function names
    this location; anything else is a split-brain waiting to happen."""
    offenders = []
    for path in _SRC.rglob("*.py"):
        if path.name == "paths.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if _FORBIDDEN_VAULT_PATH.search(line):
                offenders.append(f"{path.relative_to(_SRC)}:{i}: {line.strip()}")
    assert not offenders, (
        "these sites DERIVE the brain vault path instead of calling "
        "get_vault_root():\n  " + "\n  ".join(offenders)
    )


def test_the_source_guard_recognises_what_it_hunts():
    """Only worth having if it matches the forms it exists to catch — and
    only tolerable if it leaves prose alone."""
    assert _FORBIDDEN_VAULT_PATH.search('vault = Path.home() / "brain-vault"')
    assert _FORBIDDEN_VAULT_PATH.search("root = Path('~/brain-vault')")
    assert _FORBIDDEN_VAULT_PATH.search('p = base / "brain-vault"')
    # Prose and config keys must NOT trip it.
    assert not _FORBIDDEN_VAULT_PATH.search("# the brain-vault repo is read-only")
    assert not _FORBIDDEN_VAULT_PATH.search('"Search the BRAIN VAULT "')
    assert not _FORBIDDEN_VAULT_PATH.search("root = get_vault_root()")


# ---------------------------------------------------------------------------
# 6. Registration — through the REAL entry point
# ---------------------------------------------------------------------------

def test_the_tools_are_in_the_registry_the_daemon_actually_builds():
    """Not a constructed registry — ``create_tool_registry`` is what the daemon
    and CLI both use. MemoryTool existed for six weeks without being
    registered, and MEMORY.md sat at 0 bytes the whole time; a test against a
    hand-built registry would have passed throughout."""
    from prometheus.__main__ import create_tool_registry

    registry = create_tool_registry({})
    names = {t.name for t in registry.list_tools()}
    assert "vault_search" in names, sorted(names)
    assert "vault_read" in names, sorted(names)


def test_the_tools_are_ADVERTISED_not_merely_registered():
    """The assertion the test above should have been all along.

    ``test_the_tools_are_in_the_registry_the_daemon_actually_builds`` passed on
    the day this feature shipped, and the feature did not work: deferred loading
    advertised 8 of 52 tools and neither vault tool was among them, so the model
    was never offered them and said — correctly — that it had no brain vault.

    Membership in the registry is necessary and not sufficient. This asserts the
    tool is either in the shipped default's advertised set or deliberately
    classified as deferred with a tested discovery path.
    """
    from tests.support.advertisement import advertised_names, registered_names
    from tests.test_tool_advertisement import DEFERRED_BY_DESIGN

    for name in ("vault_search", "vault_read"):
        assert name in registered_names()
        assert name in advertised_names() or name in DEFERRED_BY_DESIGN, (
            f"{name} is registered but invisible to the model"
        )


def test_the_registered_instances_are_the_real_classes():
    """Registered-but-wrong is its own failure mode."""
    from prometheus.__main__ import create_tool_registry

    registry = create_tool_registry({})
    assert isinstance(registry.get("vault_search"), VaultSearchTool)
    assert isinstance(registry.get("vault_read"), VaultReadTool)


def test_the_advertised_examples_match_the_real_schemas():
    """§1c: a tool's example is an interface claim. Assert its keys are a
    subset of the schema, or it teaches the model a parameter that does not
    exist."""
    for tool in (VaultSearchTool(), VaultReadTool()):
        props = set(tool.input_model.model_json_schema()["properties"])
        assert set(tool.example_call) <= props, tool.name


def test_the_descriptions_say_brain_vault_not_bare_vault():
    """symbiote.backup.vault_root already exists and means something else.
    One word for two things in one config file is the collision class that
    has already cost a session."""
    for tool in (VaultSearchTool(), VaultReadTool()):
        assert "BRAIN VAULT" in tool.description or "brain vault" in tool.description
        assert "  " not in tool.description, (
            f"{tool.name} description has a double space — left behind when the\n"
            f"pre-commit hook forced the private repo name out of it"
        )
