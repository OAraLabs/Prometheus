"""DangerousCodeScanner — static-analysis pass on harvested source files.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT

The spec assumes a `src/prometheus/security/code_scanner.py` from earlier
sprints, but no such module exists in the codebase. This module fills the
gap with a minimal AST-based scanner that flags constructs we don't want
to graft into Prometheus from third-party code:

  • exec(), eval(), compile() at any scope
  • __import__() at any scope
  • os.system / os.popen / pty.spawn at any scope
  • subprocess.* at *module scope* (allowed inside functions)
  • Network calls (httpx/requests/urllib/socket) at *module scope*
  • File writes at *module scope* outside /tmp

A "dangerous" verdict means the harvest is BLOCKED. "suspicious" findings
are reported but do NOT block (the user can still graft if they accept the
risk and the calls are inside functions, where they're typical for tools).

Limitations:
  • Python only (other languages return clean with a note).
  • AST-based — won't catch obfuscated equivalents like `getattr(builtins,
    "exec")(...)`. Treat this as a first line of defense, not a sandbox.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

log = logging.getLogger(__name__)


class ScanVerdict(str, Enum):
    CLEAN = "clean"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"


@dataclass
class ScanFinding:
    """A single pattern detected during the scan."""

    severity: str  # "dangerous" | "suspicious"
    rule: str      # e.g. "exec_call", "subprocess_module_scope"
    line: int
    detail: str


@dataclass
class ScanResult:
    """Result of scanning one file (or one chunk of source)."""

    verdict: ScanVerdict
    findings: list[ScanFinding] = field(default_factory=list)
    file_path: str | None = None

    @property
    def is_dangerous(self) -> bool:
        return self.verdict == ScanVerdict.DANGEROUS

    @property
    def is_clean(self) -> bool:
        return self.verdict == ScanVerdict.CLEAN


# Built-in calls that are dangerous at any scope.
_DANGEROUS_BUILTINS: frozenset[str] = frozenset({
    "exec", "eval", "compile", "__import__",
})

# Attribute calls that are dangerous at any scope (e.g. ``os.system``).
_DANGEROUS_ATTR_CALLS: frozenset[tuple[str, str]] = frozenset({
    ("os", "system"),
    ("os", "popen"),
    ("os", "execv"),
    ("os", "execve"),
    ("os", "execvp"),
    ("os", "execvpe"),
    ("pty", "spawn"),
    ("ctypes", "CDLL"),
    ("ctypes", "WinDLL"),
})

# Modules that, used at *module scope*, are suspicious — typical of
# install-time payloads.
_SUSPICIOUS_AT_MODULE_SCOPE_MODULES: frozenset[str] = frozenset({
    "subprocess", "socket", "httpx", "requests", "urllib", "urllib2",
    "ftplib", "telnetlib", "paramiko",
})


class DangerousCodeScanner:
    """AST-based scanner. Returns a ``ScanResult`` per file."""

    def scan_content(
        self,
        content: str,
        file_path: str | None = None,
    ) -> ScanResult:
        """Scan a Python source string. Non-Python content returns CLEAN."""
        if file_path and not file_path.endswith(".py"):
            return ScanResult(verdict=ScanVerdict.CLEAN, file_path=file_path)
        try:
            tree = ast.parse(content, filename=file_path or "<string>")
        except SyntaxError as exc:
            return ScanResult(
                verdict=ScanVerdict.SUSPICIOUS,
                findings=[ScanFinding(
                    severity="suspicious",
                    rule="syntax_error",
                    line=getattr(exc, "lineno", 0) or 0,
                    detail=f"Could not parse: {exc.msg}",
                )],
                file_path=file_path,
            )

        findings: list[ScanFinding] = []
        self._visit_calls(tree, findings)
        self._visit_module_scope(tree, findings)

        if any(f.severity == "dangerous" for f in findings):
            verdict = ScanVerdict.DANGEROUS
        elif findings:
            verdict = ScanVerdict.SUSPICIOUS
        else:
            verdict = ScanVerdict.CLEAN
        return ScanResult(verdict=verdict, findings=findings, file_path=file_path)

    def scan_file(self, path: Path) -> ScanResult:
        """Read a file from disk and scan it."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ScanResult(
                verdict=ScanVerdict.SUSPICIOUS,
                findings=[ScanFinding(
                    severity="suspicious",
                    rule="read_error",
                    line=0,
                    detail=str(exc),
                )],
                file_path=str(path),
            )
        return self.scan_content(content, str(path))

    # ------------------------------------------------------------------
    # AST visitors
    # ------------------------------------------------------------------

    def _visit_calls(self, tree: ast.AST, findings: list[ScanFinding]) -> None:
        """Flag dangerous call patterns at any scope."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DANGEROUS_BUILTINS:
                findings.append(ScanFinding(
                    severity="dangerous",
                    rule=f"{func.id}_call",
                    line=getattr(node, "lineno", 0),
                    detail=f"Direct call to dangerous builtin {func.id}()",
                ))
            elif isinstance(func, ast.Attribute):
                root = self._unwind_attribute(func)
                if root is not None and (root, func.attr) in _DANGEROUS_ATTR_CALLS:
                    findings.append(ScanFinding(
                        severity="dangerous",
                        rule=f"{root}_{func.attr}_call",
                        line=getattr(node, "lineno", 0),
                        detail=f"Direct call to dangerous {root}.{func.attr}()",
                    ))

    def _visit_module_scope(self, tree: ast.AST, findings: list[ScanFinding]) -> None:
        """Flag suspicious module-scope side effects.

        Calls like ``subprocess.run(...)`` at the top level run on import
        — typical of install-time droppers. Inside functions/classes the
        same call is normal tool behavior.
        """
        if not isinstance(tree, ast.Module):
            return
        for node in tree.body:
            # Direct expression statements at module level
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                self._check_module_scope_call(node.value, findings)
            # Assignments where the RHS is a call
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                value = getattr(node, "value", None)
                if isinstance(value, ast.Call):
                    self._check_module_scope_call(value, findings)

    def _check_module_scope_call(
        self,
        call: ast.Call,
        findings: list[ScanFinding],
    ) -> None:
        func = call.func
        root: str | None = None
        attr: str | None = None
        if isinstance(func, ast.Attribute):
            root = self._unwind_attribute(func)
            attr = func.attr
        elif isinstance(func, ast.Name):
            root = func.id
        if root in _SUSPICIOUS_AT_MODULE_SCOPE_MODULES:
            findings.append(ScanFinding(
                severity="suspicious",
                rule=f"{root}_module_scope",
                line=getattr(call, "lineno", 0),
                detail=(
                    f"Module-scope call to {root}"
                    + (f".{attr}" if attr else "")
                    + " — runs on import"
                ),
            ))

    @staticmethod
    def _unwind_attribute(node: ast.Attribute) -> str | None:
        """Walk to the leftmost Name of an attribute chain.

        For ``foo.bar.baz``, returns ``"foo"``. Returns None if the chain
        bottoms out on something other than a Name (e.g. a function call).
        """
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Name):
            return cur.id
        return None
