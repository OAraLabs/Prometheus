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


def _read_result(path: str, **kwargs):
    return asyncio.run(VaultReadTool().execute(
        VaultReadTool.input_model(path=path, **kwargs), None,
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


# ---------------------------------------------------------------------------
# 7. Paging — offset windows, outline, honest notices
# ---------------------------------------------------------------------------
#
# The defect this section pins: the vault's biggest pages were truncated at
# 48k with a TRAILING notice, the daemon's default per-result truncator
# (tool_result_max=4000 tokens -> first 16,000 chars kept) then beheaded the
# payload, and the model saw neither the tail of the page nor any notice
# saying how to get it. Both directions throughout, per §2c: the new offset
# REACHES what the cap hid, AND the un-paged read still says what remains and
# how to continue — from the HEAD of the result, where truncation layers
# cannot drop it.

from prometheus.tools.builtin.vault import _window  # noqa: E402

_SENTINEL_TAIL = "TAIL-SENTINEL-PAST-THE-OLD-CAP"
_SENTINEL_EOF = "FINAL-LINE-SENTINEL"
_LONG_LINE_CHARS = 15_000  # longer than the default window: forces a hard cut


def _build_paged_page() -> str:
    """~70k chars engineered to meet every boundary shape at once.

    Unique numbered 108-char lines mean a 12,000-char boundary lands mid-line
    essentially always (the snap must engage), and one 15k-char line with no
    newline is longer than the whole window (the hard-cut branch must engage,
    mid-line by construction). Sentinels sit past the old 48k cap and on the
    final line, and one heading sits deep enough to be a real jump target.
    """
    parts = []
    for i in range(220):
        parts.append(f"line-{i:04d} " + "x" * 97)
    parts.append("## Deep-Section marker")
    parts.append("N" * _LONG_LINE_CHARS)
    for i in range(220, 400):
        parts.append(f"line-{i:04d} " + "y" * 97)
    parts.append(_SENTINEL_TAIL)
    for i in range(400, 500):
        parts.append(f"line-{i:04d} " + "z" * 97)
    parts.append(_SENTINEL_EOF)
    return "\n".join(parts) + "\n"


@pytest.fixture
def paged(tmp_path, monkeypatch):
    """A vault whose one big page exercises every windowing branch."""
    root = tmp_path / "brain-vault"
    (root / "wiki").mkdir(parents=True)
    text = _build_paged_page()
    (root / "wiki" / "Big-Page.md").write_text(text, encoding="utf-8")
    (root / "wiki" / "small.md").write_text(
        "# Small\n\nfits in one window\n", encoding="utf-8",
    )
    monkeypatch.delenv("PROMETHEUS_VAULT", raising=False)
    set_vault_root(root)
    try:
        yield root, text
    finally:
        set_vault_root(None)


_CONTINUE_OFFSET = re.compile(r'"offset": (\d+)\}')


def test_windows_tile_and_reassemble_the_exact_file(paged):
    """THE COMPLETENESS CLAIM: concatenating every window IS the file.

    Newline-snapping is where tiling would break, so the fixture forces both
    boundary shapes: unique 108-char lines put every default boundary
    mid-line (each end must snap back to a newline), and the 15k no-newline
    line forces the hard cut (the next window then STARTS mid-line). The
    walk asserts both branches actually fired — a fixture whose boundaries
    happened to land on newlines would prove nothing about the snap."""
    _, text = paged
    snapped = hard_cut = False
    chunks: list[str] = []
    offset = 0
    while True:
        start, end = _window(text, offset, 12_000)
        assert end > start, "a window made no progress"
        chunks.append(text[start:end])
        if end == len(text):
            break
        if text[end - 1] == "\n":
            snapped = True
        else:
            hard_cut = True
        offset = end  # tiling: next window starts exactly where this ended
    assert "".join(chunks) == text, "windows dropped or doubled characters"
    assert len(chunks) >= 5, "fixture did not span multiple windows"
    assert snapped, "no boundary exercised the newline snap"
    assert hard_cut, "no boundary exercised the mid-line hard cut"


def test_a_sentinel_past_the_old_48k_cap_is_reachable_by_offset(paged):
    """The admission half: what the single-shot cap hid, offset reaches."""
    _, text = paged
    pos = text.index(_SENTINEL_TAIL)
    assert pos > 48_000, "fixture regressed — sentinel no longer past the old cap"
    r = _read_result("wiki/Big-Page.md", offset=pos - 50)
    assert not r.is_error, r.output
    assert _SENTINEL_TAIL in r.output

    tail = _read_result("wiki/Big-Page.md", offset=len(text) - 500)
    assert _SENTINEL_EOF in tail.output, "the final line is still unreachable"
    assert f"of {len(text)}]" in tail.output


def test_the_unpaged_read_still_truncates_and_says_how_to_continue(paged):
    """The refusal half: no offset -> the tail stays hidden AND the result
    says so honestly — true total, real next offset, positioned at the HEAD
    where the daemon's head-keeping truncators cannot drop it (the old
    trailing notice died there every single time)."""
    _, text = paged
    r = _read_result("wiki/Big-Page.md")
    assert not r.is_error
    assert _SENTINEL_TAIL not in r.output

    m = _CONTINUE_OFFSET.search(r.output)
    assert m, f"no continue offset in output head: {r.output[:300]!r}"
    _, expected_end = _window(text, 0, 12_000)
    assert int(m.group(1)) == expected_end, "the notice names a fake next window"
    assert "[partial view — " in r.output
    assert f"of {len(text)}]" in r.output, "the header must state the TRUE size"
    assert r.output.index('"offset"') < 2_000, "continue notice is not head-positioned"


def test_the_continue_offset_chains_reads_without_gap_or_overlap(paged):
    """Interface-level tiling: the seam line appears in exactly one window."""
    _, text = paged
    first = _read_result("wiki/Big-Page.md")
    nxt = int(_CONTINUE_OFFSET.search(first.output).group(1))
    seam = text[nxt:nxt + 30]  # unique numbered lines make this unambiguous
    assert seam not in first.output, "window 1 leaked past its stated end"
    second = _read_result("wiki/Big-Page.md", offset=nxt)
    assert seam in second.output, "window 2 does not start at the stated offset"


def test_a_page_that_fits_carries_no_paging_furniture(paged):
    """The negative direction: paging must be invisible when nothing is cut."""
    r = _read_result("wiki/small.md")
    assert not r.is_error
    assert "fits in one window" in r.output
    assert re.search(r"\(\d+ chars\)", r.output), "complete reads state their size"
    assert "[partial view" not in r.output
    assert "## Outline" not in r.output
    assert '"offset"' not in r.output


def test_offset_errors_are_loud_and_name_the_valid_range(paged):
    """A bad address gets an error naming the real bounds, not empty success."""
    _, text = paged
    r = _read_result("wiki/Big-Page.md", offset=-5)
    assert r.is_error
    assert "offset" in r.output.lower()

    past = _read_result("wiki/Big-Page.md", offset=len(text) + 7)
    assert past.is_error
    assert str(len(text)) in past.output, "the error must name the true size"
    assert "past the end" in past.output


def test_max_chars_is_clamped_not_trusted(paged):
    """Floor 1000, ceiling 48000 — a wild value degrades to a sane window."""
    _, text = paged
    tiny = _read_result("wiki/Big-Page.md", max_chars=10)
    n1 = int(_CONTINUE_OFFSET.search(tiny.output).group(1))
    assert 500 <= n1 <= 1_000, f"floor not applied: first window ended at {n1}"

    huge = _read_result("wiki/Big-Page.md", max_chars=10_000_000)
    n2 = int(_CONTINUE_OFFSET.search(huge.output).group(1))
    assert 40_000 < n2 <= 48_000, f"ceiling not applied: first window ended at {n2}"


def test_the_outline_lists_real_headings_with_exact_offsets(paged):
    """An outline entry is a jump target: its offset must be the real one."""
    _, text = paged
    off = text.index("## Deep-Section marker")
    r = _read_result("wiki/Big-Page.md")
    assert "## Outline — 1 heading(s)" in r.output
    assert f"@{off} ## Deep-Section marker" in r.output


def test_a_crowded_outline_elides_the_middle_but_keeps_the_tail(tmp_path, monkeypatch):
    """Tail headings are the point — they are what single-shot reads never
    showed — so elision drops the MIDDLE, states what it dropped, and keeps
    both ends."""
    root = tmp_path / "brain-vault"
    (root / "wiki").mkdir(parents=True)
    parts = []
    for i in range(30):
        parts.append(f"## H-{i:02d}")
        parts.append("filler " * 120)
    text = "\n".join(parts) + "\n"
    (root / "wiki" / "crowded.md").write_text(text, encoding="utf-8")
    monkeypatch.delenv("PROMETHEUS_VAULT", raising=False)
    set_vault_root(root)
    try:
        tail_off = text.index("## H-29")
        assert tail_off > 12_000, "fixture regressed — tail heading inside window 1"
        r = _read_result("wiki/crowded.md")
        assert "## Outline — 30 heading(s)" in r.output
        assert "@0 ## H-00" in r.output
        assert f"@{tail_off} ## H-29" in r.output, "tail heading lost to elision"
        assert "+6 more headings between @" in r.output, "elision went silent"
    finally:
        set_vault_root(None)


def test_frontmatter_comment_lines_stay_out_of_the_outline(tmp_path, monkeypatch):
    """Standing-Principles keeps a changelog of '#   + …' YAML comments in its
    frontmatter. To a naive heading regex those are level-1 headings; an
    outline advertising jump targets into YAML teaches the model garbage."""
    root = tmp_path / "brain-vault"
    (root / "wiki").mkdir(parents=True)
    text = (
        "---\ntype: concept\n"
        "#   + changelog-entry-that-looks-like-a-heading\n"
        "---\n\n# Real Title\n\n"
        + "prose " * 2_600
        + "\n## Real-Section\n\nmore prose\n"
    )
    (root / "wiki" / "fm.md").write_text(text, encoding="utf-8")
    monkeypatch.delenv("PROMETHEUS_VAULT", raising=False)
    set_vault_root(root)
    try:
        r = _read_result("wiki/fm.md")
        assert "## Outline" in r.output, "over-window page must carry an outline"
        assert not re.search(r"@\d+ #   \+", r.output), (
            "a YAML comment line became an outline jump target"
        )
        assert re.search(r"@\d+ ## Real-Section", r.output)
    finally:
        set_vault_root(None)


def test_the_default_window_survives_the_default_daemon_truncator(paged):
    """THE LAYER-COUPLING PIN. The daemon's per-result truncator
    (tool_result_max, default 4000 tokens at 4 chars/token) keeps heads and
    appends its own trailer. A default vault_read window must pass through it
    UNTOUCHED, or the continue notice dies in the middle layer and paging is
    dead on arrival — which is precisely how the original 48k single-shot
    behaved in the live daemon. Asserted as an identity, worst case included:
    a max-length outline over long headings plus a long vault-relative path."""
    from prometheus.context.truncation import ToolResultTruncator

    truncator = ToolResultTruncator(4000)
    r = _read_result("wiki/Big-Page.md")
    assert truncator.truncate("vault_read", r.output) == r.output, (
        "a default window no longer fits tool_result_max=4000 — shrink the "
        "window or the outline"
    )

    root, _ = paged
    crowded_dir = root / "wiki" / "sources" / "concepts"
    crowded_dir.mkdir(parents=True)
    name = "A-Deliberately-Long-Concept-Page-Name-Padding-The-Worst-Case-Header.md"
    parts = []
    for i in range(30):
        parts.append(f"## Heading-{i:02d} " + "verbose-heading-text-" * 5)
        parts.append("body " * 400)
    (crowded_dir / name).write_text("\n".join(parts) + "\n", encoding="utf-8")
    worst = _read_result(f"wiki/sources/concepts/{name}")
    assert "## Outline" in worst.output  # precondition: the expensive path fired
    assert truncator.truncate("vault_read", worst.output) == worst.output, (
        "worst-case window + outline exceeds tool_result_max=4000"
    )


def test_paging_does_not_open_the_confinement_door(vault):
    """The refusal twin: an offset read is still a read — same guard, same
    refusal. A symlink out of wiki/ must not become reachable because the
    request carried a parameter."""
    r = _read_result("wiki/sneaky.md", offset=10)
    assert r.is_error
    assert "escapes the brain vault root" in r.output
    assert "SECRET" not in r.output


def test_a_directory_read_ignores_offset_harmlessly(vault):
    """Decided, not accidental: listings are small; offset is meaningless
    there and must not error a legitimate browse."""
    r = _read_result("wiki", offset=7)
    assert not r.is_error
    assert "is a directory" in r.output


def test_search_context_lines_carry_jump_offsets(vault):
    """The search→read handoff: search reads FULL bodies and can cite a line
    a windowed read never reaches, so every context line names its offset."""
    root, _ = vault
    text = (root / "wiki" / "sources" / "projects" / "Prometheus.md").read_text(
        encoding="utf-8",
    )
    off = text.index("The daemon runs a llama.cpp backend.")
    out = _search("llama.cpp")
    assert f"@{off} The daemon runs a llama.cpp backend." in out, out


@pytest.mark.skipif(
    REAL_VAULT is None or not (REAL_VAULT / "wiki").is_dir(),
    reason=(
        "PROMETHEUS_VAULT is unset or has no wiki/ tree. The fixture tests "
        "above cover every windowing branch; this one additionally proves the "
        "OPERATOR'S REAL over-cap pages are now fully reachable."
    ),
)
def test_the_real_standing_principles_tail_is_reachable():
    """Against the genuine article: the page whose hidden tail motivated the
    feature. Its outline must offer a jump target past the old 48k cap, and
    reading at that offset must return content the un-paged read hid."""
    set_vault_root(REAL_VAULT)
    try:
        sp = "wiki/sources/concepts/Standing-Principles.md"
        first = _read_result(sp)
        if first.is_error:
            pytest.skip(f"real vault lacks {sp}")
        deep = [
            int(m.group(1))
            for m in re.finditer(r"@(\d+) ", first.output)
            if int(m.group(1)) > 48_000
        ]
        if not deep:
            pytest.skip("no outline entry past 48k — page shape changed")
        r = _read_result(sp, offset=deep[-1])
        assert not r.is_error
        assert len(r.output) > 200
        assert first.output[-4_000:] != r.output[-4_000:], (
            "the deep read returned the same content as the un-paged read"
        )
    finally:
        set_vault_root(None)
