"""GraftEngine — Phase 3 of SYMBIOTE: integrate harvested code into Prometheus.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT

Pipeline:
  1. For each AdaptationStep, build a Prometheus-shaped file:
       - Add provenance header (matches existing donor-file convention)
       - Rewrite imports of the donor package to ``prometheus.<target>``
       - Adjust logging to use the Prometheus logger pattern
  2. DangerousCodeScanner re-runs on every adapted file before write.
  3. Write files only under ``src/prometheus/`` or ``tests/`` (safety guard).
  4. Generate integration tests appended to ``tests/test_wiring.py``.
  5. Run the full test suite. Pass/fail captured in the GraftReport.
  6. Append a new section to ``PROMETHEUS.md``.

The grafting itself is structural — the LLM-assisted adaptation can be
layered on top by callers (HarvestEngine produces a plan, GraftEngine
executes it). For Session A the engine performs deterministic transforms;
LLM-driven re-writing can be added in a follow-up sprint without changing
the public interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from prometheus.symbiote.code_scanner import (
    DangerousCodeScanner,
    ScanVerdict,
)
from prometheus.symbiote.harvest import (
    AdaptationStep,
    ExtractedModule,
    HarvestReport,
)

log = logging.getLogger(__name__)


_PROMETHEUS_MD_HEADER = "## Grafted via SYMBIOTE"

_PROVENANCE_TEMPLATE = (
    "# Source: {repo_full_name} ({repo_url})\n"
    "# Original: {original_path}\n"
    "# License: {license_spdx}\n"
    "# Modified: {modifications}\n"
    "# Harvested: {timestamp} via SYMBIOTE\n\n"
)


@dataclass
class GraftedFile:
    path: str
    original_source: str
    lines_added: int
    provenance_header: str
    scan_verdict: str = ScanVerdict.CLEAN.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraftReport:
    repo_full_name: str
    files_created: list[GraftedFile] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    tests_added: list[str] = field(default_factory=list)
    tests_passed: bool = False
    tests_output: str = ""
    prometheus_md_updated: bool = False
    timestamp: str = ""
    aborted: bool = False
    abort_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_full_name": self.repo_full_name,
            "files_created": [f.to_dict() for f in self.files_created],
            "files_modified": list(self.files_modified),
            "tests_added": list(self.tests_added),
            "tests_passed": self.tests_passed,
            "tests_output": self.tests_output[-2000:],  # tail only
            "prometheus_md_updated": self.prometheus_md_updated,
            "timestamp": self.timestamp,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }

    def to_telegram_summary(self) -> str:
        if self.aborted:
            return f"Graft aborted ({self.repo_full_name}): {self.abort_reason}"
        status = "passed" if self.tests_passed else "FAILED"
        lines = [
            f"Graft complete: {self.repo_full_name}",
            f"Created: {len(self.files_created)} file(s)",
            f"Modified: {len(self.files_modified)} file(s)",
            f"Tests added: {len(self.tests_added)}",
            f"Tests {status}",
            f"PROMETHEUS.md updated: {self.prometheus_md_updated}",
        ]
        return "\n".join(lines)


class GraftEngine:
    """Run Phase 3 of SYMBIOTE."""

    # Roots inside which graft is allowed to write. Anything else triggers an abort.
    _ALLOWED_ROOTS = ("src/prometheus/", "tests/")

    def __init__(
        self,
        scanner: DangerousCodeScanner | None = None,
        project_root: Path | None = None,
        *,
        run_tests: bool = True,
    ) -> None:
        self._scanner = scanner or DangerousCodeScanner()
        self._project_root = (project_root or Path(__file__).resolve().parents[3]).resolve()
        self._run_tests = bool(run_tests)

    async def graft(self, harvest_report: HarvestReport) -> GraftReport:
        """Execute the graft pipeline using the harvest report's plan."""
        timestamp = _now_iso()
        report = GraftReport(
            repo_full_name=harvest_report.repo_full_name,
            timestamp=timestamp,
        )

        if harvest_report.aborted:
            report.aborted = True
            report.abort_reason = (
                f"Source harvest was aborted: {harvest_report.abort_reason}"
            )
            return report
        if not harvest_report.adaptation_plan:
            report.aborted = True
            report.abort_reason = "Harvest report has no adaptation plan"
            return report

        modules_by_path: dict[str, ExtractedModule] = {
            m.original_path: m for m in harvest_report.modules_extracted
        }

        # Apply each step.
        for step in harvest_report.adaptation_plan:
            target = self._resolve_target(step.target_path)
            if target is None:
                report.aborted = True
                report.abort_reason = (
                    f"Target {step.target_path!r} is outside allowed roots "
                    + repr(self._ALLOWED_ROOTS)
                )
                return report
            module = modules_by_path.get(step.source_module)
            if module is None:
                log.debug(
                    "GraftEngine: step references missing source module %s",
                    step.source_module,
                )
                continue
            grafted = self._graft_one(step, module, target, harvest_report, timestamp)
            if grafted is None:
                report.aborted = True
                report.abort_reason = f"DangerousCodeScanner rejected adapted file {step.target_path}"
                return report
            if grafted.path in report.files_modified or any(
                f.path == grafted.path for f in report.files_created
            ):
                continue
            if step.action.lower() == "modify" or step.action.lower() == "extend":
                report.files_modified.append(grafted.path)
            else:
                report.files_created.append(grafted)

        # Generate integration tests.
        if report.files_created:
            test_funcs = self._append_wiring_tests(report.files_created, harvest_report)
            report.tests_added = test_funcs

        # Run pytest.
        if self._run_tests:
            passed, output = await self._run_pytest()
            report.tests_passed = passed
            report.tests_output = output
        else:
            report.tests_passed = True
            report.tests_output = "(skipped)"

        # Update PROMETHEUS.md.
        try:
            self._update_prometheus_md(report, harvest_report)
            report.prometheus_md_updated = True
        except OSError:
            log.exception("GraftEngine: failed to update PROMETHEUS.md")

        return report

    # ------------------------------------------------------------------
    # Per-file graft
    # ------------------------------------------------------------------

    def _resolve_target(self, target_path: str) -> Path | None:
        """Validate the target and return its absolute Path, or None if disallowed.

        Resolves the path BEFORE checking the allowed-roots prefix, so a
        ``..`` traversal that lands outside the allowed roots is rejected
        even when the literal input string starts with an allowed prefix.
        """
        if not target_path:
            return None
        rel = target_path.lstrip("/")
        candidate = (self._project_root / rel).resolve()
        try:
            relative = candidate.relative_to(self._project_root)
        except ValueError:
            return None
        relative_str = relative.as_posix()
        if not any(
            relative_str == root.rstrip("/")
            or relative_str.startswith(root)
            for root in self._ALLOWED_ROOTS
        ):
            return None
        return candidate

    def _graft_one(
        self,
        step: AdaptationStep,
        module: ExtractedModule,
        target: Path,
        harvest_report: HarvestReport,
        timestamp: str,
    ) -> GraftedFile | None:
        """Adapt and write one file. Returns None if the scanner rejects it."""
        provenance = self.generate_provenance_header(
            repo_full_name=harvest_report.repo_full_name,
            repo_url=harvest_report.repo_url,
            original_path=module.original_path,
            license_spdx=harvest_report.license.spdx_id or "UNKNOWN",
            modifications=step.description or "imports adapted to prometheus.*",
            timestamp=timestamp,
        )
        adapted = self._adapt_content(module.content, harvest_report)

        # Re-scan the adapted file before write.
        scan = self._scanner.scan_content(adapted, file_path=str(target))
        if scan.is_dangerous:
            return None

        full = provenance + adapted
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(full, encoding="utf-8")
        rel = target.relative_to(self._project_root).as_posix()
        return GraftedFile(
            path=rel,
            original_source=f"{harvest_report.repo_full_name}:{module.original_path}",
            lines_added=full.count("\n"),
            provenance_header=provenance.strip(),
            scan_verdict=scan.verdict.value,
        )

    def generate_provenance_header(
        self,
        *,
        repo_full_name: str,
        repo_url: str,
        original_path: str,
        license_spdx: str,
        modifications: str,
        timestamp: str | None = None,
    ) -> str:
        """Public so tests can verify the format directly."""
        ts = timestamp or _now_iso()
        return _PROVENANCE_TEMPLATE.format(
            repo_full_name=repo_full_name,
            repo_url=repo_url,
            original_path=original_path,
            license_spdx=license_spdx,
            modifications=modifications.strip().replace("\n", " "),
            timestamp=ts,
        )

    def _adapt_content(self, content: str, harvest_report: HarvestReport) -> str:
        """Apply deterministic source rewrites.

        - Replace top-level imports of the donor package name with a
          ``prometheus.<package>``-shaped placeholder.
        - Normalise logger creation to use ``logging.getLogger(__name__)``.
        """
        donor_root = self._infer_donor_root(harvest_report.repo_full_name)
        if donor_root:
            # Naïve rename: ``import donor`` / ``from donor.X`` → ``prometheus.<donor>``.
            target_pkg = f"prometheus.{donor_root}"
            content = re.sub(
                rf"\bimport\s+{re.escape(donor_root)}\b",
                f"import {target_pkg}",
                content,
            )
            content = re.sub(
                rf"\bfrom\s+{re.escape(donor_root)}(\.|\s+import)",
                lambda m: f"from {target_pkg}{m.group(1)}",
                content,
            )
        # Normalise logger calls.
        content = re.sub(
            r"logging\.getLogger\([^\)]*\)",
            "logging.getLogger(__name__)",
            content,
        )
        return content

    @staticmethod
    def _infer_donor_root(repo_full_name: str) -> str | None:
        """``owner/some-repo`` → ``some_repo``."""
        if not repo_full_name or "/" not in repo_full_name:
            return None
        repo = repo_full_name.split("/", 1)[1]
        normalised = re.sub(r"[^A-Za-z0-9]", "_", repo).strip("_")
        return normalised.lower() or None

    # ------------------------------------------------------------------
    # Tests + docs
    # ------------------------------------------------------------------

    def _append_wiring_tests(
        self,
        grafted: list[GraftedFile],
        harvest_report: HarvestReport,
    ) -> list[str]:
        """Append a smoke-test class to tests/test_wiring.py for each graft batch."""
        wiring_path = self._project_root / "tests" / "test_wiring.py"
        if not wiring_path.exists():
            return []
        ts_slug = re.sub(r"[^A-Za-z0-9]", "_", harvest_report.repo_full_name)
        class_name = f"TestSymbioteGraft_{ts_slug}_{int(_now_epoch())}"
        lines: list[str] = []
        lines.append("\n\n")
        lines.append(f"class {class_name}:\n")
        lines.append(f'    """Smoke tests for graft from {harvest_report.repo_full_name}."""\n\n')
        for f in grafted:
            module_path = f.path
            if not module_path.startswith("src/prometheus/"):
                continue
            module = module_path[len("src/"):].replace("/", ".").rsplit(".", 1)[0]
            test_name = "test_import_" + re.sub(
                r"[^A-Za-z0-9]", "_", module
            )
            lines.append(f"    def {test_name}(self):\n")
            lines.append(f'        """{module} imports cleanly after graft."""\n')
            lines.append("        import importlib\n")
            lines.append(f"        importlib.import_module({module!r})\n\n")
        new_text = "".join(lines)
        try:
            with wiring_path.open("a", encoding="utf-8") as fh:
                fh.write(new_text)
        except OSError:
            log.exception("GraftEngine: failed to append tests")
            return []
        return [class_name]

    def _update_prometheus_md(
        self,
        report: GraftReport,
        harvest_report: HarvestReport,
    ) -> None:
        prom_md = self._project_root / "PROMETHEUS.md"
        if not prom_md.exists():
            return
        existing = prom_md.read_text(encoding="utf-8")
        section = self._render_md_section(report, harvest_report)
        if _PROMETHEUS_MD_HEADER not in existing:
            new_text = existing.rstrip() + "\n\n" + _PROMETHEUS_MD_HEADER + "\n\n" + section
        else:
            # Append a new entry under the existing header.
            new_text = existing.rstrip() + "\n\n" + section
        prom_md.write_text(new_text, encoding="utf-8")

    @staticmethod
    def _render_md_section(
        report: GraftReport,
        harvest_report: HarvestReport,
    ) -> str:
        files = "\n".join(f"  - `{f.path}`" for f in report.files_created)
        return (
            f"### {harvest_report.repo_full_name} "
            f"({harvest_report.license.spdx_id or 'UNKNOWN'}) — {report.timestamp}\n"
            f"- Source: {harvest_report.repo_url}\n"
            f"- Files added:\n{files or '  - (none)'}\n"
            f"- Tests: {'passed' if report.tests_passed else 'FAILED'}\n"
        )

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------

    async def _run_pytest(self) -> tuple[bool, str]:
        """Run the project test suite. Returns (passed, output)."""
        cmd = ["python3", "-m", "pytest", "tests/", "-q", "--tb=short", "-x"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self._project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, "pytest timed out after 300s"
            output = stdout.decode("utf-8", errors="replace")
            return proc.returncode == 0, output
        except FileNotFoundError as exc:
            return False, f"pytest not available: {exc}"


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _now_epoch() -> float:
    return datetime.utcnow().timestamp()
