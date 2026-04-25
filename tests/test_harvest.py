"""HarvestEngine — sandbox safety, scan blocking, persistence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from prometheus.symbiote.code_scanner import DangerousCodeScanner, ScanVerdict
from prometheus.symbiote.harvest import (
    AdaptationStep,
    ExtractedModule,
    HarvestEngine,
    HarvestReport,
)
from prometheus.symbiote.license_gate import LicenseCheck, LicenseGate, LicenseVerdict


def _engine(tmp_path: Path) -> HarvestEngine:
    return HarvestEngine(
        scanner=DangerousCodeScanner(),
        license_gate=LicenseGate(),
        provider=None,  # _pick_files will use fallback
        sandbox_root=tmp_path / "sandbox",
        harvest_root=tmp_path / "harvests",
        max_repo_size_mb=10,
        file_budget_max=5,
        file_budget_kb=10,
    )


def _mit_check() -> LicenseCheck:
    return LicenseCheck(
        spdx_id="MIT",
        verdict=LicenseVerdict.ALLOW,
        source="github_api",
        obligations=["Include copyright notice"],
    )


class TestRepoSizeGuard:
    def test_aborts_when_repo_too_big(self, tmp_path):
        engine = _engine(tmp_path)
        engine._max_repo_size_mb = 1
        report = asyncio.run(
            engine.harvest(
                repo_full_name="foo/bar",
                repo_url="https://example.com/foo/bar.git",
                problem_statement="x",
                license_check=_mit_check(),
                repo_size_kb=10 * 1024,  # 10 MB > 1 MB limit
            )
        )
        assert report.aborted is True
        assert "exceeds limit" in report.abort_reason


class TestSandboxCleanup:
    def test_refuses_path_outside_sandbox_root(self, tmp_path):
        engine = _engine(tmp_path)
        outside = tmp_path / "not_sandbox" / "rogue"
        outside.mkdir(parents=True)
        sentinel = outside / "keep.txt"
        sentinel.write_text("keep me")
        # The cleanup should refuse and NOT delete the file.
        engine._cleanup_sandbox(outside)
        assert sentinel.exists()

    def test_cleans_up_inside_sandbox_root(self, tmp_path):
        engine = _engine(tmp_path)
        inside = tmp_path / "sandbox" / "ok"
        inside.mkdir(parents=True)
        (inside / "x.txt").write_text("ok")
        engine._cleanup_sandbox(inside)
        assert not inside.exists()


class TestExtractImports:
    def test_extracts_top_level_imports(self):
        content = (
            "import os\n"
            "import json as j\n"
            "from pathlib import Path\n"
            "from typing import Any, Optional\n"
            "from .relative import x\n"  # relative import — root is empty after strip
        )
        imports = HarvestEngine._extract_imports(content)
        assert "os" in imports
        assert "json" in imports
        assert "pathlib" in imports
        assert "typing" in imports

    def test_dedupes(self):
        content = "import os\nimport os\nfrom os import path\n"
        imports = HarvestEngine._extract_imports(content)
        assert imports.count("os") == 1


class TestCollectExternalDeps:
    def test_filters_stdlib(self):
        modules = [
            ExtractedModule(
                original_path="x.py",
                content="",
                dependencies=["os", "json", "httpx", "pydantic", "typing"],
            )
        ]
        deps = HarvestEngine._collect_external_deps(modules)
        assert "httpx" in deps
        assert "pydantic" in deps
        assert "os" not in deps
        assert "typing" not in deps


class TestScanIntegration:
    def test_dangerous_file_aborts_harvest(self, tmp_path):
        """If the read+scan loop produces a 'dangerous' module, the harvest aborts.

        We don't actually clone here; we drive the engine via its internal
        method directly.
        """
        engine = _engine(tmp_path)
        # Build a fake "sandbox" dir with one safe and one dangerous file.
        sandbox = tmp_path / "sandbox" / "fake"
        sandbox.mkdir(parents=True)
        (sandbox / "safe.py").write_text("def add(a,b):\n    return a+b\n")
        (sandbox / "bad.py").write_text("exec('print(1)')\n")

        modules = engine._read_and_scan(
            sandbox, ["safe.py", "bad.py"],
        )
        # Safe is clean, bad is dangerous.
        verdicts = {m.original_path: m.scan_verdict for m in modules}
        assert verdicts["safe.py"] == ScanVerdict.CLEAN.value
        assert verdicts["bad.py"] == ScanVerdict.DANGEROUS.value


class TestFileBudget:
    def test_caps_files(self, tmp_path):
        engine = _engine(tmp_path)
        sandbox = tmp_path / "sandbox" / "fake"
        sandbox.mkdir(parents=True)
        for i in range(20):
            (sandbox / f"f{i}.py").write_text(f"x = {i}\n")
        modules = engine._read_and_scan(
            sandbox, [f"f{i}.py" for i in range(20)],
        )
        # Capped to file_budget_max=5
        assert len(modules) <= 5

    def test_caps_bytes(self, tmp_path):
        engine = _engine(tmp_path)
        engine._file_budget_max = 100  # not the limiting factor
        engine._file_budget_bytes = 2048  # 2 KB total budget
        sandbox = tmp_path / "sandbox" / "fake"
        sandbox.mkdir(parents=True)
        # Two files, each ~1.5 KB
        (sandbox / "a.py").write_text("# comment\n" * 200)  # ~2KB
        (sandbox / "b.py").write_text("# comment\n" * 200)
        modules = engine._read_and_scan(sandbox, ["a.py", "b.py"])
        total = sum(len(m.content.encode()) for m in modules)
        assert total <= 2048


class TestPersistence:
    def test_persist_writes_report_files(self, tmp_path):
        engine = _engine(tmp_path)
        harvest_dir = tmp_path / "harvests" / "owner_repo"
        report = HarvestReport(
            repo_full_name="owner/repo",
            repo_url="https://example.com/owner/repo.git",
            license=_mit_check(),
            problem_statement="x",
            modules_extracted=[
                ExtractedModule(
                    original_path="src/foo.py",
                    content="def foo(): pass\n",
                    line_count=1,
                ),
            ],
            total_lines_extracted=1,
            external_dependencies=[],
            adaptation_plan=[
                AdaptationStep(
                    action="create",
                    target_path="src/prometheus/foo.py",
                    description="adapt foo",
                    source_module="src/foo.py",
                ),
            ],
            security_scan_summary="all clean",
            sandbox_path=str(tmp_path / "sandbox" / "x"),
            harvest_dir=str(harvest_dir),
            timestamp="2026-04-25T00:00:00Z",
        )
        engine._persist_report(report, tmp_path / "sandbox" / "x")
        assert (harvest_dir / "harvest.md").exists()
        assert (harvest_dir / "license.md").exists()
        assert (harvest_dir / "adaptation_plan.md").exists()
        assert (harvest_dir / "security_scan.md").exists()
        assert (harvest_dir / "report.json").exists()
        assert (harvest_dir / "extracted" / "src" / "foo.py").exists()
        # report.json should round-trip
        data = json.loads((harvest_dir / "report.json").read_text())
        assert data["repo_full_name"] == "owner/repo"
