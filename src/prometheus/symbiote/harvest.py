"""HarvestEngine — Phase 2 of SYMBIOTE: clone, scan, extract, plan.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT

Pipeline:
  1. Validate repo size against ``max_repo_size_mb``.
  2. ``git clone --depth 1`` to ``~/.prometheus/symbiote/sandbox/{name}/``
     under a 60s timeout.
  3. Scan directory tree, identify candidate source files (skipping VCS,
     caches, virtualenvs, build artifacts).
  4. LLM picks up to 15 most-relevant files for the problem statement.
  5. Read selected files (capped to 50KB total content).
  6. Run ``DangerousCodeScanner`` on every file. ANY dangerous → abort.
  7. LLM produces an adaptation plan (where each module goes in Prometheus).
  8. Persist HarvestReport to ``~/.prometheus/symbiote/harvests/...``.
  9. Delete the sandbox; keep the extracted/ copy alongside the report.

Sandbox safety: cleanup verifies the path is rooted under SANDBOX_ROOT
before any removal — defense against path-traversal bugs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prometheus.config.paths import get_config_dir
from prometheus.symbiote.code_scanner import (
    DangerousCodeScanner,
    ScanResult,
    ScanVerdict,
)
from prometheus.symbiote.license_gate import LicenseCheck, LicenseGate

if TYPE_CHECKING:
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)


def _symbiote_root() -> Path:
    return get_config_dir() / "symbiote"


def _sandbox_root() -> Path:
    return _symbiote_root() / "sandbox"


def _harvest_root() -> Path:
    return _symbiote_root() / "harvests"


_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".github", ".gitlab", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", "node_modules", "dist", "build", "target",
    ".idea", ".vscode", "site-packages",
})

_INTERESTING_SUFFIXES: frozenset[str] = frozenset({
    ".py", ".pyi", ".md", ".rst", ".txt",
    ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json",
})

_SOURCE_SUFFIXES: frozenset[str] = frozenset({".py", ".pyi"})

_MAX_REPO_SIZE_MB_DEFAULT = 100
_FILE_BUDGET_MAX_DEFAULT = 15
_FILE_BUDGET_KB_DEFAULT = 50
_CLONE_TIMEOUT_DEFAULT = 60


_FILE_PICK_PROMPT = """\
You select source files most likely to contain the solution to a capability need.

Capability need:
{problem_statement}

Repository directory tree:
{tree}

Pick at most 15 file paths from the tree. Prefer core implementation files
(.py) over tests, docs, and configs. Output ONLY a single JSON object on
one line:
{{"files": ["path/to/file.py", "path/to/other.py"]}}
"""


_ADAPT_PROMPT = """\
You plan how to adapt extracted modules into Prometheus.

Capability need:
{problem_statement}

Prometheus conventions:
- Tools subclass BaseTool in src/prometheus/tools/base.py
- Engines live under src/prometheus/<package>/
- All grafted files start with a provenance header
- Tests under tests/ — integration tests in test_wiring.py

Extracted modules:
{modules}

Produce an adaptation plan as JSON on one line:
{{"steps": [{{"action":"create"|"modify"|"extend",
              "target_path":"src/prometheus/...",
              "description":"...",
              "source_module":"<original_path>"}}, ...]}}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExtractedModule:
    original_path: str
    content: str
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    line_count: int = 0
    scan_verdict: str = ScanVerdict.CLEAN.value
    scan_findings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptationStep:
    action: str
    target_path: str
    description: str
    source_module: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HarvestReport:
    repo_full_name: str
    repo_url: str
    license: LicenseCheck
    problem_statement: str
    modules_extracted: list[ExtractedModule]
    total_lines_extracted: int
    external_dependencies: list[str]
    adaptation_plan: list[AdaptationStep]
    security_scan_summary: str
    sandbox_path: str
    harvest_dir: str
    timestamp: str
    aborted: bool = False
    abort_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_full_name": self.repo_full_name,
            "repo_url": self.repo_url,
            "license": {
                "spdx_id": self.license.spdx_id,
                "verdict": self.license.verdict.value,
                "source": self.license.source,
                "obligations": list(self.license.obligations),
            },
            "problem_statement": self.problem_statement,
            "modules_extracted": [m.to_dict() for m in self.modules_extracted],
            "total_lines_extracted": self.total_lines_extracted,
            "external_dependencies": list(self.external_dependencies),
            "adaptation_plan": [s.to_dict() for s in self.adaptation_plan],
            "security_scan_summary": self.security_scan_summary,
            "sandbox_path": self.sandbox_path,
            "harvest_dir": self.harvest_dir,
            "timestamp": self.timestamp,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }

    def to_telegram_summary(self) -> str:
        if self.aborted:
            return f"Harvest aborted ({self.repo_full_name}): {self.abort_reason}"
        lines = [
            f"Harvest complete: {self.repo_full_name}",
            f"License: {self.license.spdx_id or '?'} ({self.license.verdict.value})",
            f"Modules: {len(self.modules_extracted)} ({self.total_lines_extracted} lines)",
            f"Scan: {self.security_scan_summary}",
            f"Plan: {len(self.adaptation_plan)} step(s)",
            f"Saved to: {self.harvest_dir}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HarvestEngine
# ---------------------------------------------------------------------------


class HarvestEngine:
    """Run Phase 2 of SYMBIOTE."""

    def __init__(
        self,
        scanner: DangerousCodeScanner | None = None,
        license_gate: LicenseGate | None = None,
        provider: ModelProvider | None = None,
        *,
        model: str = "default",
        max_repo_size_mb: int = _MAX_REPO_SIZE_MB_DEFAULT,
        file_budget_max: int = _FILE_BUDGET_MAX_DEFAULT,
        file_budget_kb: int = _FILE_BUDGET_KB_DEFAULT,
        clone_timeout: int = _CLONE_TIMEOUT_DEFAULT,
        sandbox_root: Path | None = None,
        harvest_root: Path | None = None,
    ) -> None:
        self._scanner = scanner or DangerousCodeScanner()
        self._gate = license_gate or LicenseGate()
        self._provider = provider
        self._model = model
        self._max_repo_size_mb = int(max_repo_size_mb)
        self._file_budget_max = int(file_budget_max)
        self._file_budget_bytes = int(file_budget_kb) * 1024
        self._clone_timeout = int(clone_timeout)
        self._sandbox_root = sandbox_root or _sandbox_root()
        self._harvest_root = harvest_root or _harvest_root()

    async def harvest(
        self,
        repo_full_name: str,
        repo_url: str,
        problem_statement: str,
        license_check: LicenseCheck,
        *,
        repo_size_kb: int | None = None,
    ) -> HarvestReport:
        """Execute the harvest pipeline. Always returns a HarvestReport."""
        timestamp = _now_iso()
        repo_slug = repo_full_name.replace("/", "_")
        sandbox_path = self._sandbox_root / f"{repo_slug}_{int(time.time())}"
        harvest_dir = self._harvest_root / f"{repo_slug}_{int(time.time())}"

        report = HarvestReport(
            repo_full_name=repo_full_name,
            repo_url=repo_url,
            license=license_check,
            problem_statement=problem_statement,
            modules_extracted=[],
            total_lines_extracted=0,
            external_dependencies=[],
            adaptation_plan=[],
            security_scan_summary="",
            sandbox_path=str(sandbox_path),
            harvest_dir=str(harvest_dir),
            timestamp=timestamp,
        )

        # Repo size guard.
        if repo_size_kb is not None:
            mb = repo_size_kb / 1024.0
            if mb > self._max_repo_size_mb:
                report.aborted = True
                report.abort_reason = (
                    f"Repo size {mb:.1f}MB exceeds limit {self._max_repo_size_mb}MB"
                )
                return report

        # Clone.
        try:
            await self._clone(repo_url, sandbox_path)
        except Exception as exc:
            report.aborted = True
            report.abort_reason = f"Clone failed: {exc}"
            self._cleanup_sandbox(sandbox_path)
            return report

        try:
            tree = self._build_dir_tree(sandbox_path)
            picked = await self._pick_files(problem_statement, tree)
            modules = self._read_and_scan(sandbox_path, picked)
            if any(m.scan_verdict == ScanVerdict.DANGEROUS.value for m in modules):
                bad = [m.original_path for m in modules
                       if m.scan_verdict == ScanVerdict.DANGEROUS.value]
                report.aborted = True
                report.abort_reason = (
                    "DangerousCodeScanner blocked harvest. Files: " + ", ".join(bad[:5])
                )
                report.modules_extracted = modules
                report.security_scan_summary = self._scan_summary(modules)
                self._persist_report(report, sandbox_path)
                self._cleanup_sandbox(sandbox_path)
                return report

            adaptation_plan = await self._plan_adaptation(problem_statement, modules)

            report.modules_extracted = modules
            report.total_lines_extracted = sum(m.line_count for m in modules)
            report.external_dependencies = self._collect_external_deps(modules)
            report.adaptation_plan = adaptation_plan
            report.security_scan_summary = self._scan_summary(modules)

            self._persist_report(report, sandbox_path)
        finally:
            self._cleanup_sandbox(sandbox_path)

        return report

    # ------------------------------------------------------------------
    # Clone
    # ------------------------------------------------------------------

    async def _clone(self, repo_url: str, sandbox_path: Path) -> None:
        """Run ``git clone --depth 1`` with a hard timeout."""
        sandbox_path.parent.mkdir(parents=True, exist_ok=True)
        if sandbox_path.exists():
            shutil.rmtree(sandbox_path)
        cmd = [
            "git", "clone", "--depth", "1", "--filter=blob:limit=1m",
            "--quiet", repo_url, str(sandbox_path),
        ]
        log.info("HarvestEngine: cloning %s", repo_url)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._clone_timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"git clone timed out after {self._clone_timeout}s")
        if proc.returncode != 0:
            raise RuntimeError(
                f"git clone exit {proc.returncode}: "
                + (stderr.decode("utf-8", errors="replace")[:300] if stderr else "")
            )

    # ------------------------------------------------------------------
    # Directory scan
    # ------------------------------------------------------------------

    def _build_dir_tree(self, repo_path: Path, *, max_lines: int = 200) -> str:
        """Build a compact tree string for the LLM file picker."""
        lines: list[str] = []
        for path in sorted(repo_path.rglob("*")):
            rel = path.relative_to(repo_path)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            if path.is_dir():
                continue
            if path.suffix.lower() not in _INTERESTING_SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            lines.append(f"{rel} ({size}B)")
            if len(lines) >= max_lines:
                lines.append("... (truncated)")
                break
        return "\n".join(lines)

    async def _pick_files(self, problem_statement: str, tree: str) -> list[str]:
        """Use the LLM to pick relevant files. Falls back to all .py files."""
        if self._provider is None:
            return self._fallback_pick(tree)
        prompt = _FILE_PICK_PROMPT.format(
            problem_statement=problem_statement,
            tree=tree,
        )
        try:
            text = await self._call_provider(prompt, max_tokens=512)
        except Exception:
            log.debug("HarvestEngine: file pick failed", exc_info=True)
            return self._fallback_pick(tree)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return self._fallback_pick(tree)
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return self._fallback_pick(tree)
        files = obj.get("files") if isinstance(obj, dict) else None
        if not isinstance(files, list) or not files:
            return self._fallback_pick(tree)
        cleaned: list[str] = []
        for entry in files:
            if isinstance(entry, str) and entry.strip():
                cleaned.append(entry.strip())
            if len(cleaned) >= self._file_budget_max:
                break
        return cleaned or self._fallback_pick(tree)

    def _fallback_pick(self, tree: str) -> list[str]:
        """Pick the first N .py files mentioned in the tree."""
        out: list[str] = []
        for line in tree.splitlines():
            parts = line.split(" (")
            if not parts:
                continue
            path = parts[0].strip()
            if path.endswith(tuple(_SOURCE_SUFFIXES)):
                out.append(path)
            if len(out) >= self._file_budget_max:
                break
        return out

    # ------------------------------------------------------------------
    # Read + scan
    # ------------------------------------------------------------------

    def _read_and_scan(
        self,
        repo_path: Path,
        picked: list[str],
    ) -> list[ExtractedModule]:
        modules: list[ExtractedModule] = []
        bytes_used = 0
        for rel in picked[: self._file_budget_max]:
            path = (repo_path / rel).resolve()
            try:
                path.relative_to(repo_path.resolve())
            except ValueError:
                log.warning("HarvestEngine: rejected path traversal: %s", rel)
                continue
            if not path.exists() or not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            content_bytes = len(content.encode("utf-8", errors="ignore"))
            if bytes_used + content_bytes > self._file_budget_bytes:
                # Trim if needed.
                remaining = max(0, self._file_budget_bytes - bytes_used)
                if remaining < 256:
                    break
                content = content[: remaining]
                content_bytes = len(content.encode("utf-8", errors="ignore"))
            bytes_used += content_bytes

            scan: ScanResult = self._scanner.scan_content(content, file_path=rel)
            module = ExtractedModule(
                original_path=rel,
                content=content,
                description="",
                dependencies=self._extract_imports(content) if rel.endswith(tuple(_SOURCE_SUFFIXES)) else [],
                line_count=content.count("\n") + (0 if content.endswith("\n") else 1),
                scan_verdict=scan.verdict.value,
                scan_findings=[
                    {
                        "severity": f.severity,
                        "rule": f.rule,
                        "line": f.line,
                        "detail": f.detail,
                    }
                    for f in scan.findings
                ],
            )
            modules.append(module)
        return modules

    @staticmethod
    def _extract_imports(content: str) -> list[str]:
        """Parse top-level import names without importing the module."""
        imports: list[str] = []
        for line in content.splitlines()[:200]:
            s = line.strip()
            if s.startswith("import "):
                rest = s[len("import "):].split(" as ")[0]
                root = rest.split(".")[0].split(",")[0].strip()
                if root:
                    imports.append(root)
            elif s.startswith("from "):
                rest = s[len("from "):].split(" import ")[0]
                root = rest.split(".")[0].strip()
                if root:
                    imports.append(root)
        # Dedupe preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for name in imports:
            if name not in seen and not name.startswith("_"):
                seen.add(name)
                out.append(name)
        return out

    @staticmethod
    def _collect_external_deps(modules: list[ExtractedModule]) -> list[str]:
        """Filter imports to ones that look like external packages."""
        # Stdlib heuristic: ignore imports from a small known-stdlib set.
        stdlib = {
            "abc", "argparse", "asyncio", "ast", "base64", "collections",
            "concurrent", "contextlib", "copy", "dataclasses", "datetime",
            "enum", "functools", "glob", "hashlib", "io", "itertools",
            "json", "logging", "math", "os", "pathlib", "queue", "random",
            "re", "shutil", "signal", "socket", "sqlite3", "ssl", "string",
            "subprocess", "sys", "tempfile", "threading", "time", "traceback",
            "types", "typing", "uuid", "warnings", "weakref", "xml",
        }
        out: list[str] = []
        seen: set[str] = set()
        for m in modules:
            for imp in m.dependencies:
                if imp in stdlib:
                    continue
                if imp not in seen:
                    seen.add(imp)
                    out.append(imp)
        return out

    @staticmethod
    def _scan_summary(modules: list[ExtractedModule]) -> str:
        sus = sum(1 for m in modules if m.scan_verdict == ScanVerdict.SUSPICIOUS.value)
        bad = sum(1 for m in modules if m.scan_verdict == ScanVerdict.DANGEROUS.value)
        if bad:
            return f"{bad} dangerous, {sus} suspicious"
        if sus:
            return f"all clean except {sus} suspicious"
        return "all clean"

    # ------------------------------------------------------------------
    # Adaptation plan
    # ------------------------------------------------------------------

    async def _plan_adaptation(
        self,
        problem_statement: str,
        modules: list[ExtractedModule],
    ) -> list[AdaptationStep]:
        if self._provider is None or not modules:
            return []
        modules_str = "\n".join(
            f"- {m.original_path} ({m.line_count} lines, deps={m.dependencies})"
            for m in modules
        )
        prompt = _ADAPT_PROMPT.format(
            problem_statement=problem_statement,
            modules=modules_str,
        )
        try:
            text = await self._call_provider(prompt, max_tokens=1024)
        except Exception:
            log.debug("HarvestEngine: adaptation plan failed", exc_info=True)
            return []
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return []
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []
        steps_raw = obj.get("steps") if isinstance(obj, dict) else None
        if not isinstance(steps_raw, list):
            return []
        steps: list[AdaptationStep] = []
        for entry in steps_raw:
            if not isinstance(entry, dict):
                continue
            steps.append(AdaptationStep(
                action=str(entry.get("action", "create"))[:32],
                target_path=str(entry.get("target_path", ""))[:200],
                description=str(entry.get("description", ""))[:500],
                source_module=str(entry.get("source_module", ""))[:200],
            ))
        return steps

    # ------------------------------------------------------------------
    # Persistence + cleanup
    # ------------------------------------------------------------------

    def _persist_report(self, report: HarvestReport, sandbox_path: Path) -> None:
        harvest_dir = Path(report.harvest_dir)
        harvest_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir = harvest_dir / "extracted"
        extracted_dir.mkdir(parents=True, exist_ok=True)

        # Write extracted file copies.
        for module in report.modules_extracted:
            target = extracted_dir / module.original_path
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.write_text(module.content, encoding="utf-8")
            except OSError:
                log.warning("HarvestEngine: failed to write %s", target)

        # harvest.md — human-readable summary
        try:
            (harvest_dir / "harvest.md").write_text(self._render_harvest_md(report), encoding="utf-8")
            (harvest_dir / "license.md").write_text(self._render_license_md(report.license), encoding="utf-8")
            (harvest_dir / "adaptation_plan.md").write_text(
                self._render_plan_md(report), encoding="utf-8",
            )
            (harvest_dir / "security_scan.md").write_text(
                self._render_scan_md(report), encoding="utf-8",
            )
            (harvest_dir / "report.json").write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            log.exception("HarvestEngine: failed to write harvest report files")

    @staticmethod
    def _render_harvest_md(report: HarvestReport) -> str:
        return (
            f"# Harvest: {report.repo_full_name}\n\n"
            f"- URL: {report.repo_url}\n"
            f"- Timestamp: {report.timestamp}\n"
            f"- Problem: {report.problem_statement}\n"
            f"- License: {report.license.spdx_id} ({report.license.verdict.value})\n"
            f"- Modules: {len(report.modules_extracted)} "
            f"({report.total_lines_extracted} lines)\n"
            f"- Scan: {report.security_scan_summary}\n"
            f"- External deps: {', '.join(report.external_dependencies) or 'none'}\n"
            f"- Aborted: {report.aborted} {report.abort_reason}\n"
        )

    @staticmethod
    def _render_license_md(license_check: LicenseCheck) -> str:
        body = [
            f"# License: {license_check.spdx_id or 'UNKNOWN'}",
            f"- Verdict: {license_check.verdict.value}",
            f"- Source: {license_check.source}",
            "## Obligations",
        ]
        if license_check.obligations:
            for ob in license_check.obligations:
                body.append(f"- {ob}")
        else:
            body.append("- (none)")
        if license_check.raw_text:
            body.append("\n## Raw text (first 8KB)\n")
            body.append("```")
            body.append(license_check.raw_text)
            body.append("```")
        return "\n".join(body) + "\n"

    @staticmethod
    def _render_plan_md(report: HarvestReport) -> str:
        lines = [f"# Adaptation plan ({len(report.adaptation_plan)} step(s))\n"]
        for step in report.adaptation_plan:
            lines.append(
                f"- **{step.action}** `{step.target_path}` from `{step.source_module}`\n"
                f"  {step.description}\n"
            )
        return "\n".join(lines) if report.adaptation_plan else "# Adaptation plan\n\n_No steps generated._\n"

    @staticmethod
    def _render_scan_md(report: HarvestReport) -> str:
        lines = ["# Security scan", f"\nSummary: {report.security_scan_summary}\n"]
        for m in report.modules_extracted:
            if not m.scan_findings:
                continue
            lines.append(f"\n## {m.original_path} — {m.scan_verdict}\n")
            for f in m.scan_findings:
                lines.append(
                    f"- L{f.get('line', 0)} **{f.get('severity', '')}**"
                    f" {f.get('rule', '')}: {f.get('detail', '')}"
                )
        return "\n".join(lines) + "\n"

    def _cleanup_sandbox(self, sandbox_path: Path) -> None:
        """Remove the cloned repo. Refuses to delete anything outside SANDBOX_ROOT."""
        try:
            resolved = sandbox_path.resolve()
            root = self._sandbox_root.resolve()
            resolved.relative_to(root)
        except (ValueError, OSError):
            log.warning(
                "HarvestEngine: refusing to clean up path outside sandbox root: %s",
                sandbox_path,
            )
            return
        if resolved.exists():
            try:
                shutil.rmtree(resolved)
            except OSError:
                log.exception("HarvestEngine: cleanup failed for %s", resolved)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def _call_provider(self, prompt: str, *, max_tokens: int = 512) -> str:
        from prometheus.engine.messages import ConversationMessage
        from prometheus.providers.base import (
            ApiMessageRequest,
            ApiTextDeltaEvent,
        )

        request = ApiMessageRequest(
            model=self._model,
            messages=[ConversationMessage.from_user_text(prompt)],
            max_tokens=max_tokens,
        )
        text_parts: list[str] = []
        async for event in self._provider.stream_message(request):  # type: ignore[union-attr]
            if isinstance(event, ApiTextDeltaEvent):
                text_parts.append(event.text)
        return "".join(text_parts)


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
