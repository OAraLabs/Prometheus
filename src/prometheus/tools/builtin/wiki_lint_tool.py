"""WikiLintTool — agent can trigger wiki lint on demand.

Source: Novel code for Prometheus Sprint 9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

if TYPE_CHECKING:
    from prometheus.sentinel.wiki_lint import WikiLinter

# Module-level singleton (set by daemon.py at startup)
_wiki_linter: WikiLinter | None = None


def set_wiki_linter(linter: WikiLinter) -> None:
    """Register the WikiLinter so the tool can access it."""
    global _wiki_linter  # noqa: PLW0603
    _wiki_linter = linter


class WikiLintInput(BaseModel):
    """Arguments for wiki_lint."""

    severity: str | None = Field(
        default=None,
        description="Filter results by severity: 'error', 'warning', or 'info'. Omit for all.",
    )
    auto_fix: bool = Field(
        default=False,
        description="Automatically fix safe issues (broken links, missing crossrefs).",
    )


class WikiLintTool(BaseTool):
    """Run health checks on the Prometheus wiki."""

    name = "wiki_lint"
    description = (
        "Scan the wiki for health issues: orphan pages, broken links, "
        "stale pages, duplicate entities, missing cross-references, "
        "and category imbalance. Optionally auto-fix safe issues."
    )
    input_model = WikiLintInput

    def is_read_only(self, arguments: WikiLintInput) -> bool:
        return not arguments.auto_fix

    async def execute(
        self, arguments: WikiLintInput, context: ToolExecutionContext
    ) -> ToolResult:
        if _wiki_linter is None:
            return ToolResult(
                output="WikiLinter not initialised. Is the daemon running with sentinel enabled?",
                is_error=True,
            )

        result = _wiki_linter.lint()

        # Filter by severity if requested
        issues = result.issues
        if arguments.severity:
            issues = [i for i in issues if i.severity == arguments.severity]

        # Auto-fix if requested
        fixed = 0
        if arguments.auto_fix and result.has_issues:
            fixed = _wiki_linter.auto_fix(result)

        output = _wiki_linter.summary(issues)
        if fixed:
            output += f"\n\nAuto-fixed {fixed} issue(s)."

        return ToolResult(output=output)
