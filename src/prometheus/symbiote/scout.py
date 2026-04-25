"""ScoutEngine — Phase 1 of SYMBIOTE: research GitHub for solutions.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT

Pipeline:
  1. LLM generates 2-3 GitHub search queries from a problem statement.
  2. Execute searches via GitHubSearchTool, deduplicate by full_name.
  3. License gate: drop BLOCK verdicts, flag WARN.
  4. Fetch READMEs (already done by GitHubSearchTool.search()).
  5. LLM scores each remaining candidate's relevance with GBNF-enforced
     JSON output {"score": float, "rationale": str}.
  6. Classify each candidate: recommended | viable | risky | blocked.
  7. Return a ScoutReport.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from prometheus.symbiote.github_search import GitHubSearchTool
from prometheus.symbiote.license_gate import (
    LicenseCheck,
    LicenseGate,
    LicenseVerdict,
)

if TYPE_CHECKING:
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)


_QUERY_GEN_PROMPT = """\
You translate capability needs into 2-3 GitHub search queries.

Capability need:
{problem_statement}

Rules:
- Each line is one GitHub search query string.
- Use specific technical terms, not natural language sentences.
- 2-6 words per query is ideal.
- Output ONLY the queries, one per line. No commentary, no numbering.

Example for "I need YAML validation with schemas":
yaml schema validation python
strictyaml pydantic config
yaml type checking
"""


_SCORE_PROMPT = """\
You evaluate the relevance of a GitHub repository to a capability need.

Problem: {problem_statement}

Repository: {full_name}
Description: {description}
Language: {language}
Stars: {stars}
Topics: {topics}

README excerpt:
{readme}

Score relevance from 0.0 to 1.0:
- 1.0 = Direct, well-maintained solution
- 0.7 = Strong match, minor gaps
- 0.5 = Partial match or tangential
- 0.3 = Weak match, mostly irrelevant
- 0.0 = Not relevant

Respond with EXACTLY one JSON object on a single line:
{{"score": <float>, "rationale": "<one sentence>"}}
"""


_SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}


@dataclass
class ScoutCandidate:
    """One repository surfaced during a scout."""

    full_name: str
    url: str
    description: str
    stars: int
    language: str | None
    license_check: LicenseCheck
    last_pushed: str | None
    topics: list[str]
    size_kb: int
    readme_excerpt: str
    relevance_score: float
    relevance_rationale: str
    recommendation: str  # "recommended" | "viable" | "risky" | "blocked"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict converts the enum to a string already via str(Enum) — but
        # LicenseCheck.verdict needs explicit handling.
        d["license_check"]["verdict"] = self.license_check.verdict.value
        return d


@dataclass
class ScoutReport:
    """Full scout output for one problem statement."""

    problem_statement: str
    candidates: list[ScoutCandidate]
    search_queries_used: list[str]
    timestamp: str
    total_found: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_statement": self.problem_statement,
            "candidates": [c.to_dict() for c in self.candidates],
            "search_queries_used": list(self.search_queries_used),
            "timestamp": self.timestamp,
            "total_found": self.total_found,
            "notes": self.notes,
        }

    def to_telegram_summary(self) -> str:
        if not self.candidates:
            base = f"Scout: no viable candidates for {self.problem_statement!r}."
            if self.notes:
                base += f" ({self.notes})"
            return base
        lines = [
            f"Scout: {len(self.candidates)} candidate(s) for {self.problem_statement!r}",
            f"Queries: {', '.join(self.search_queries_used)}",
        ]
        for c in self.candidates[:5]:
            lic = c.license_check.spdx_id or "?"
            lines.append(
                f"  • {c.full_name} ({lic}, ★{c.stars}) "
                f"[{c.recommendation}] score={c.relevance_score:.2f}"
            )
        return "\n".join(lines)


class ScoutEngine:
    """Run Scout pipeline. The LLM does the heavy lifting (queries + scoring)."""

    def __init__(
        self,
        github_search: GitHubSearchTool,
        license_gate: LicenseGate,
        provider: ModelProvider | None = None,
        model: str = "default",
        *,
        max_candidates: int = 8,
    ) -> None:
        self._github = github_search
        self._gate = license_gate
        self._provider = provider
        self._model = model
        self._max_candidates = max(1, int(max_candidates))

    async def scout(
        self,
        problem_statement: str,
        *,
        language: str = "Python",
        min_stars: int = 10,
    ) -> ScoutReport:
        """Run the full pipeline. Always returns a ScoutReport."""
        if not problem_statement.strip():
            return ScoutReport(
                problem_statement=problem_statement,
                candidates=[],
                search_queries_used=[],
                timestamp=_now_iso(),
                total_found=0,
                notes="empty problem statement",
            )

        queries = await self._generate_search_queries(problem_statement)
        if not queries:
            queries = [self._fallback_query(problem_statement, language)]

        seen: dict[str, dict[str, Any]] = {}
        for q in queries:
            try:
                results = await self._github.search(
                    query=q,
                    language=language,
                    min_stars=min_stars,
                    max_results=5,
                    license_filter=None,  # license gate runs in-process
                )
            except Exception:
                log.exception("ScoutEngine: search failed for query %r", q)
                results = []
            for item in results:
                full_name = item.get("full_name")
                if full_name and full_name not in seen:
                    seen[full_name] = item
        total_found = len(seen)
        if not seen:
            return ScoutReport(
                problem_statement=problem_statement,
                candidates=[],
                search_queries_used=queries,
                timestamp=_now_iso(),
                total_found=0,
                notes="no GitHub results",
            )

        # License gate first — drop BLOCKED, keep ALLOW + WARN + UNKNOWN
        # so users can see why a repo is excluded if score requested.
        kept: list[tuple[dict[str, Any], LicenseCheck]] = []
        blocked_count = 0
        for item in list(seen.values())[: self._max_candidates]:
            lic = self._gate.check({"license": item.get("license_obj")})
            if lic.verdict == LicenseVerdict.BLOCK:
                blocked_count += 1
                continue
            kept.append((item, lic))

        # Score each remaining candidate via LLM.
        scored: list[ScoutCandidate] = []
        for item, lic in kept:
            score, rationale = await self._score_relevance(problem_statement, item)
            candidate = self._build_candidate(item, lic, score, rationale)
            scored.append(candidate)

        scored.sort(key=lambda c: c.relevance_score, reverse=True)
        notes = f"{blocked_count} blocked by license" if blocked_count else ""
        return ScoutReport(
            problem_statement=problem_statement,
            candidates=scored,
            search_queries_used=queries,
            timestamp=_now_iso(),
            total_found=total_found,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # LLM steps
    # ------------------------------------------------------------------

    async def _generate_search_queries(self, problem: str) -> list[str]:
        if self._provider is None:
            return [self._fallback_query(problem, "Python")]
        prompt = _QUERY_GEN_PROMPT.format(problem_statement=problem)
        try:
            text = await self._call_provider(prompt, max_tokens=256)
        except Exception:
            log.debug("ScoutEngine: query generation failed", exc_info=True)
            return [self._fallback_query(problem, "Python")]
        queries: list[str] = []
        for line in text.splitlines():
            line = line.strip().strip('"').strip("'")
            if not line:
                continue
            # Strip leading numbering / bullets
            line = re.sub(r"^[\d]+[\).]\s*", "", line)
            line = re.sub(r"^[-*]\s+", "", line)
            if line:
                queries.append(line)
            if len(queries) >= 3:
                break
        return queries or [self._fallback_query(problem, "Python")]

    async def _score_relevance(
        self,
        problem: str,
        candidate: dict[str, Any],
    ) -> tuple[float, str]:
        if self._provider is None:
            return 0.5, "no provider; default score"
        prompt = _SCORE_PROMPT.format(
            problem_statement=problem,
            full_name=candidate.get("full_name", "?"),
            description=(candidate.get("description") or "")[:300],
            language=candidate.get("language") or "?",
            stars=candidate.get("stars", 0),
            topics=", ".join(candidate.get("topics") or [])[:200],
            readme=(candidate.get("readme_excerpt") or "")[:1500],
        )
        try:
            text = await self._call_provider(prompt, max_tokens=256)
        except Exception:
            log.debug("ScoutEngine: score call failed", exc_info=True)
            return 0.0, "score call failed"
        return self._parse_score(text)

    async def _call_provider(self, prompt: str, *, max_tokens: int = 256) -> str:
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_candidate(
        self,
        item: dict[str, Any],
        license_check: LicenseCheck,
        score: float,
        rationale: str,
    ) -> ScoutCandidate:
        candidate = ScoutCandidate(
            full_name=item.get("full_name", ""),
            url=item.get("url", ""),
            description=item.get("description", "") or "",
            stars=int(item.get("stars", 0)),
            language=item.get("language"),
            license_check=license_check,
            last_pushed=item.get("last_pushed"),
            topics=list(item.get("topics") or []),
            size_kb=int(item.get("size_kb", 0)),
            readme_excerpt=(item.get("readme_excerpt") or "")[:500],
            relevance_score=max(0.0, min(1.0, float(score))),
            relevance_rationale=str(rationale)[:500],
            recommendation="blocked",  # filled in below
        )
        candidate.recommendation = self._classify(candidate)
        return candidate

    @staticmethod
    def _classify(candidate: ScoutCandidate) -> str:
        v = candidate.license_check.verdict
        if v == LicenseVerdict.BLOCK or v == LicenseVerdict.UNKNOWN:
            return "blocked"
        score = candidate.relevance_score
        stars = candidate.stars
        if v == LicenseVerdict.WARN:
            return "risky"
        # ALLOW
        if score >= 0.7 and stars >= 50:
            return "recommended"
        if score >= 0.4:
            return "viable"
        return "risky"

    @staticmethod
    def _parse_score(text: str) -> tuple[float, str]:
        """Parse a {"score": float, "rationale": str} JSON object from LLM output."""
        if not text or not text.strip():
            return 0.0, "empty response"
        # Try direct parse first.
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # Strip markdown fences / extract first JSON object
            stripped = re.sub(r"```(?:json)?\s*\n?", "", text).strip()
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
                if not m:
                    return 0.0, "could not parse"
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError:
                    return 0.0, "could not parse"
        if not isinstance(obj, dict):
            return 0.0, "non-dict response"
        score = obj.get("score", 0.0)
        try:
            score_f = max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            score_f = 0.0
        rationale = str(obj.get("rationale", ""))[:500]
        return score_f, rationale

    @staticmethod
    def _fallback_query(problem: str, language: str | None) -> str:
        """Build a simple GitHub query from the problem statement."""
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", problem.lower())
        # Drop common stopwords; keep up to 4 distinctive words.
        stop = {"i", "we", "need", "want", "for", "the", "a", "and", "or",
                "to", "with", "of", "is", "it", "in", "on", "from"}
        keep = [w for w in words if w not in stop][:4]
        parts = [" ".join(keep) or problem.strip()]
        if language:
            parts.append(f"language:{language}")
        return " ".join(parts)


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"
