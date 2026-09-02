"""Item 7-S — context-file discovery: the small fixes.

* ``GEMINI.md`` and ``.github/copilot-instructions.md`` are conventions too.
* One aggregate cap bounds the whole "# Project Instructions" section — the
  upward walk reaches ``/``, so N levels × 12 K was unbounded. What does not
  fit is OMITTED AND NAMED, never dropped silently.
* Two documents share the name AGENTS.md: the subagent registry at
  ``~/.prometheus/AGENTS.md`` (its own loader, gated by
  ``bootstrap.load_agents``) and a repo's own AGENTS.md (a project convention
  file). Discovery now excludes the registry FILE by path — always — and no
  longer excludes the NAME, so ``load_agents: false`` stops throwing away
  every repo's AGENTS.md.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from prometheus.context.prometheusmd import (
    CONVENTION_FILES,
    DEFAULT_MAX_TOTAL_CHARS,
    discover_project_files,
    load_project_files_prompt,
)
from prometheus.context.prompt_assembler import project_files_section


# --------------------------------------------------------------------------- #
# new conventions
# --------------------------------------------------------------------------- #


def test_gemini_and_copilot_are_conventions():
    assert "GEMINI.md" in CONVENTION_FILES
    assert ".github/copilot-instructions.md" in CONVENTION_FILES
    # Priority: our own file first, Claude/agents before Gemini, editor rules after.
    order = [CONVENTION_FILES.index(n) for n in ("PROMETHEUS.md", "CLAUDE.md", "AGENTS.md", "GEMINI.md", ".cursorrules")]
    assert order == sorted(order)


def test_gemini_md_is_discovered(tmp_path):
    (tmp_path / "GEMINI.md").write_text("# gemini rules\n")
    files = discover_project_files(str(tmp_path))
    assert [p.name for p, _ in files] == ["GEMINI.md"]


def test_copilot_instructions_are_discovered_from_dot_github(tmp_path):
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "copilot-instructions.md").write_text("# copilot\n")
    files = discover_project_files(str(tmp_path))
    assert [str(p.relative_to(tmp_path)) for p, _ in files] == [".github/copilot-instructions.md"]
    prompt = load_project_files_prompt(str(tmp_path))
    assert "# copilot" in prompt


def test_one_file_per_level_claude_beats_gemini(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("claude\n")
    (tmp_path / "GEMINI.md").write_text("gemini\n")
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "copilot-instructions.md").write_text("copilot\n")
    files = discover_project_files(str(tmp_path))
    assert [p.name for p, _ in files] == ["CLAUDE.md"]


def test_a_directory_named_like_a_convention_is_not_a_file(tmp_path):
    (tmp_path / "GEMINI.md").mkdir()
    assert discover_project_files(str(tmp_path)) == []


# --------------------------------------------------------------------------- #
# aggregate cap
# --------------------------------------------------------------------------- #


def _tree(tmp_path: Path, sizes: list[int]) -> Path:
    """Nested dirs d0/d1/…, each with a PROMETHEUS.md of the given size. Returns the deepest."""
    cur = tmp_path
    for i, n in enumerate(sizes):
        cur = cur / f"d{i}"
        cur.mkdir()
        (cur / "PROMETHEUS.md").write_text(f"L{i}:" + ("x" * (n - 4)) + "\n")  # exactly n chars
    return cur


def test_default_cap_is_bounded_and_read_from_config():
    assert DEFAULT_MAX_TOTAL_CHARS == 48000
    # The assembler's literal fallback must equal the module default and the template.
    import re
    src = (Path(__file__).resolve().parents[1] / "src/prometheus/context/prompt_assembler.py").read_text()
    m = re.search(r'project_files_max_total_chars",\s*(\d+)\)', src)
    assert m and int(m.group(1)) == DEFAULT_MAX_TOTAL_CHARS


def test_nearest_files_survive_and_omissions_are_named(tmp_path):
    deepest = _tree(tmp_path, [3000, 3000, 3000, 3000])
    prompt = load_project_files_prompt(str(deepest), max_total_chars=7000)
    # Deepest two fit (6000), the third would need 3000 with 1000 remaining →
    # partial (≥ _MIN_PARTIAL_CHARS) so it is truncated; the fourth is omitted.
    assert "L3:" in prompt and "L2:" in prompt and "L1:" in prompt
    assert "L0:" not in prompt
    assert "...[truncated: aggregate project-file cap]..." in prompt
    assert "1 project instruction file(s) omitted — aggregate cap of 7000 chars reached:" in prompt
    assert str(tmp_path / "d0" / "PROMETHEUS.md") in prompt


def test_tiny_remainder_omits_rather_than_shipping_a_stub(tmp_path):
    deepest = _tree(tmp_path, [3000, 3000])
    prompt = load_project_files_prompt(str(deepest), max_total_chars=3500)
    assert "L1:" in prompt
    assert "L0:" not in prompt and "truncated: aggregate" not in prompt
    assert "1 project instruction file(s) omitted" in prompt


def test_cap_none_is_unbounded_and_under_cap_says_nothing(tmp_path):
    deepest = _tree(tmp_path, [3000, 3000, 3000])
    unbounded = load_project_files_prompt(str(deepest), max_total_chars=None)
    roomy = load_project_files_prompt(str(deepest), max_total_chars=DEFAULT_MAX_TOTAL_CHARS)
    assert unbounded == roomy
    assert "omitted" not in roomy and "aggregate" not in roomy


def test_per_file_cap_still_applies_inside_the_aggregate(tmp_path):
    (tmp_path / "PROMETHEUS.md").write_text("y" * 5000 + "\n")
    prompt = load_project_files_prompt(str(tmp_path), max_chars_per_file=1000, max_total_chars=10000)
    assert "...[truncated]..." in prompt and "aggregate" not in prompt


def test_section_reads_the_aggregate_key_from_config(tmp_path):
    deepest = _tree(tmp_path, [3000, 3000, 3000])
    with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=tmp_path / "cfg"):
        section = project_files_section({"context": {"project_files_max_total_chars": 4000}}, deepest)
    assert "L2:" in section and "L0:" not in section
    assert "aggregate cap of 4000 chars" in section


# --------------------------------------------------------------------------- #
# AGENTS.md — the registry file vs a repo's own
# --------------------------------------------------------------------------- #


def test_exclude_paths_skips_that_file_only(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("# repo agents\n")
    (tmp_path / "AGENTS.md").write_text("# registry\n")  # an ancestor sharing the name
    files = discover_project_files(str(repo), exclude_paths=(tmp_path / "AGENTS.md",))
    assert [p for p, _ in files] == [repo / "AGENTS.md"]


def test_exclude_paths_compares_resolved_paths(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# registry\n")
    link_dir = tmp_path / "link"
    link_dir.symlink_to(tmp_path, target_is_directory=True)
    # Named through the symlink, excluded by identity.
    assert discover_project_files(str(tmp_path), exclude_paths=(link_dir / "AGENTS.md",)) == []


def test_repo_agents_md_survives_load_agents_false(tmp_path):
    """The conflation fix: turning the subagent registry off must not drop a repo's AGENTS.md."""
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "AGENTS.md").write_text("REGISTRY_MARKER\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("REPO_AGENTS_MARKER\n")
    with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=cfg):
        section = project_files_section({"bootstrap": {"load_agents": False}}, repo)
    assert "REPO_AGENTS_MARKER" in section
    assert "REGISTRY_MARKER" not in section


def test_registry_file_is_never_rediscovered_even_with_load_agents_true(tmp_path):
    """cwd under the config dir: the registry is loaded by its own bootstrap section, not twice."""
    cfg = tmp_path
    (cfg / "AGENTS.md").write_text("REGISTRY_MARKER\n")
    work = cfg / "workspace"
    work.mkdir()
    with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=cfg):
        section = project_files_section({"bootstrap": {"load_agents": True}}, work)
    assert section is None


def test_registry_gate_still_holds_in_the_full_prompt(tmp_path):
    """build_runtime_system_prompt: load_agents false keeps the REGISTRY out while the repo file stays in."""
    from prometheus.context.prompt_assembler import build_runtime_system_prompt

    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "AGENTS.md").write_text("REGISTRY_MARKER\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("REPO_AGENTS_MARKER\n")
    with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=cfg):
        prompt = build_runtime_system_prompt(cwd=str(repo), config={"bootstrap": {"load_agents": False}})
    assert "REPO_AGENTS_MARKER" in prompt
    assert "REGISTRY_MARKER" not in prompt
