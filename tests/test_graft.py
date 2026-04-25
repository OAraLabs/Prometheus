"""GraftEngine — provenance, allowed-roots guard, source rewriting."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.symbiote.code_scanner import DangerousCodeScanner
from prometheus.symbiote.graft import GraftEngine, GraftedFile, GraftReport
from prometheus.symbiote.harvest import (
    AdaptationStep,
    ExtractedModule,
    HarvestReport,
)
from prometheus.symbiote.license_gate import LicenseCheck, LicenseVerdict


def _harvest(
    *,
    repo_full_name: str = "alice/mit-tool",
    modules: list[ExtractedModule] | None = None,
    plan: list[AdaptationStep] | None = None,
) -> HarvestReport:
    return HarvestReport(
        repo_full_name=repo_full_name,
        repo_url="https://example.com/" + repo_full_name + ".git",
        license=LicenseCheck(
            spdx_id="MIT",
            verdict=LicenseVerdict.ALLOW,
            source="github_api",
        ),
        problem_statement="x",
        modules_extracted=modules or [],
        total_lines_extracted=sum((m.line_count for m in (modules or [])), 0),
        external_dependencies=[],
        adaptation_plan=plan or [],
        security_scan_summary="all clean",
        sandbox_path="",
        harvest_dir="",
        timestamp="2026-04-25T00:00:00Z",
    )


class TestProvenanceHeader:
    def test_header_format(self, tmp_path):
        engine = GraftEngine(project_root=tmp_path, run_tests=False)
        header = engine.generate_provenance_header(
            repo_full_name="alice/mit-tool",
            repo_url="https://example.com/alice/mit-tool",
            original_path="src/x.py",
            license_spdx="MIT",
            modifications="adapted imports",
            timestamp="2026-04-25T00:00:00Z",
        )
        assert "Source: alice/mit-tool" in header
        assert "Original: src/x.py" in header
        assert "License: MIT" in header
        assert "Modified: adapted imports" in header
        assert "Harvested: 2026-04-25T00:00:00Z via SYMBIOTE" in header
        # Each header line begins with '#'.
        for line in header.strip().splitlines():
            assert line.startswith("#")


class TestAllowedRoots:
    def test_resolves_under_src_prometheus(self, tmp_path):
        engine = GraftEngine(project_root=tmp_path, run_tests=False)
        target = engine._resolve_target("src/prometheus/grafted/x.py")
        assert target is not None
        assert str(target).endswith("src/prometheus/grafted/x.py")

    def test_resolves_under_tests(self, tmp_path):
        engine = GraftEngine(project_root=tmp_path, run_tests=False)
        target = engine._resolve_target("tests/test_grafted.py")
        assert target is not None

    def test_rejects_config_path(self, tmp_path):
        engine = GraftEngine(project_root=tmp_path, run_tests=False)
        target = engine._resolve_target("config/prometheus.yaml")
        assert target is None

    def test_rejects_traversal(self, tmp_path):
        engine = GraftEngine(project_root=tmp_path, run_tests=False)
        target = engine._resolve_target("src/prometheus/../../etc/passwd")
        assert target is None


class TestSourceRewriting:
    def test_rewrites_donor_imports(self, tmp_path):
        engine = GraftEngine(project_root=tmp_path, run_tests=False)
        harvest = _harvest(repo_full_name="owner/cool-pkg")
        adapted = engine._adapt_content(
            "import cool_pkg\nfrom cool_pkg import x\n",
            harvest,
        )
        # The donor root is inferred as 'cool_pkg' (lowercased, dashes→underscores).
        assert "import prometheus.cool_pkg" in adapted
        assert "from prometheus.cool_pkg import x" in adapted

    def test_normalizes_logger(self, tmp_path):
        engine = GraftEngine(project_root=tmp_path, run_tests=False)
        harvest = _harvest()
        adapted = engine._adapt_content(
            'log = logging.getLogger("custom.name")\n',
            harvest,
        )
        assert "logging.getLogger(__name__)" in adapted

    def test_infer_donor_root(self):
        assert GraftEngine._infer_donor_root("alice/cool-tool") == "cool_tool"
        assert GraftEngine._infer_donor_root("nope") is None
        assert GraftEngine._infer_donor_root("") is None


class TestGraftFlow:
    def test_aborts_when_target_outside_allowed_roots(self, tmp_path):
        engine = GraftEngine(project_root=tmp_path, run_tests=False)
        harvest = _harvest(
            modules=[ExtractedModule(original_path="src/x.py", content="x = 1\n")],
            plan=[AdaptationStep(
                action="create",
                target_path="config/dangerous.yaml",
                description="x",
                source_module="src/x.py",
            )],
        )
        report = asyncio.run(engine.graft(harvest))
        assert report.aborted is True
        assert "outside allowed roots" in report.abort_reason

    def test_aborts_on_dangerous_adapted(self, tmp_path):
        engine = GraftEngine(project_root=tmp_path, run_tests=False)
        harvest = _harvest(
            modules=[ExtractedModule(
                original_path="src/x.py",
                content='exec("hi")\n',
            )],
            plan=[AdaptationStep(
                action="create",
                target_path="src/prometheus/grafted/x.py",
                description="x",
                source_module="src/x.py",
            )],
        )
        report = asyncio.run(engine.graft(harvest))
        assert report.aborted is True
        assert "DangerousCodeScanner" in report.abort_reason

    def test_writes_adapted_file_with_provenance(self, tmp_path):
        # Build a minimal project tree under tmp_path so PROMETHEUS.md exists.
        (tmp_path / "src" / "prometheus" / "grafted").mkdir(parents=True)
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_wiring.py").write_text("# wiring tests\n")
        (tmp_path / "PROMETHEUS.md").write_text("# Prometheus\n")

        engine = GraftEngine(project_root=tmp_path, run_tests=False)
        harvest = _harvest(
            modules=[ExtractedModule(
                original_path="src/x.py",
                content="def add(a, b):\n    return a + b\n",
                line_count=2,
            )],
            plan=[AdaptationStep(
                action="create",
                target_path="src/prometheus/grafted/x.py",
                description="adapted",
                source_module="src/x.py",
            )],
        )
        report = asyncio.run(engine.graft(harvest))
        assert report.aborted is False
        assert len(report.files_created) == 1
        # The file got a provenance header.
        target = tmp_path / "src" / "prometheus" / "grafted" / "x.py"
        assert target.exists()
        text = target.read_text()
        assert "Source: alice/mit-tool" in text
        assert "License: MIT" in text
        # PROMETHEUS.md got an entry.
        prom_md = (tmp_path / "PROMETHEUS.md").read_text()
        assert "Grafted via SYMBIOTE" in prom_md
        assert "alice/mit-tool" in prom_md
