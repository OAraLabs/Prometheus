"""ScoutEngine — score parsing, recommendation classification, license gating."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.symbiote.github_search import GitHubSearchTool, GitHubClient
from prometheus.symbiote.license_gate import (
    LicenseCheck,
    LicenseGate,
    LicenseVerdict,
)
from prometheus.symbiote.scout import (
    ScoutCandidate,
    ScoutEngine,
    ScoutReport,
)


def _make_search_tool():
    """A GitHubSearchTool whose .search() returns a fixed list."""
    tool = GitHubSearchTool(client=GitHubClient(token=None))
    tool.search = AsyncMock(return_value=[])  # type: ignore[method-assign]
    return tool


class TestParseScore:
    def test_parses_clean_json(self):
        score, reason = ScoutEngine._parse_score(
            '{"score": 0.85, "rationale": "good fit"}'
        )
        assert score == 0.85
        assert reason == "good fit"

    def test_clamps_above_one(self):
        score, _ = ScoutEngine._parse_score('{"score": 1.5, "rationale": "x"}')
        assert score == 1.0

    def test_clamps_below_zero(self):
        score, _ = ScoutEngine._parse_score('{"score": -0.5, "rationale": "x"}')
        assert score == 0.0

    def test_strips_markdown_fence(self):
        score, _ = ScoutEngine._parse_score(
            '```json\n{"score": 0.7, "rationale": "x"}\n```'
        )
        assert score == 0.7

    def test_extracts_from_preamble(self):
        score, _ = ScoutEngine._parse_score(
            'Sure, here is the verdict: {"score": 0.6, "rationale": "x"}'
        )
        assert score == 0.6

    def test_empty_returns_zero(self):
        score, reason = ScoutEngine._parse_score("")
        assert score == 0.0


class TestClassify:
    def _mk_candidate(
        self,
        verdict: LicenseVerdict,
        score: float,
        stars: int = 100,
    ) -> ScoutCandidate:
        return ScoutCandidate(
            full_name="foo/bar",
            url="",
            description="",
            stars=stars,
            language="Python",
            license_check=LicenseCheck(
                spdx_id="MIT",
                verdict=verdict,
                source="github_api",
            ),
            last_pushed=None,
            topics=[],
            size_kb=0,
            readme_excerpt="",
            relevance_score=score,
            relevance_rationale="",
            recommendation="",
        )

    def test_blocked_when_license_blocks(self):
        c = self._mk_candidate(LicenseVerdict.BLOCK, 0.9, 1000)
        assert ScoutEngine._classify(c) == "blocked"

    def test_blocked_when_unknown(self):
        c = self._mk_candidate(LicenseVerdict.UNKNOWN, 0.9, 1000)
        assert ScoutEngine._classify(c) == "blocked"

    def test_risky_when_warn(self):
        c = self._mk_candidate(LicenseVerdict.WARN, 0.9, 1000)
        assert ScoutEngine._classify(c) == "risky"

    def test_recommended_when_high_score_high_stars(self):
        c = self._mk_candidate(LicenseVerdict.ALLOW, 0.8, 200)
        assert ScoutEngine._classify(c) == "recommended"

    def test_viable_when_mid_score(self):
        c = self._mk_candidate(LicenseVerdict.ALLOW, 0.5, 200)
        assert ScoutEngine._classify(c) == "viable"

    def test_risky_when_low_score(self):
        c = self._mk_candidate(LicenseVerdict.ALLOW, 0.2, 200)
        assert ScoutEngine._classify(c) == "risky"


class TestFallbackQuery:
    def test_strips_stopwords_and_appends_language(self):
        q = ScoutEngine._fallback_query("I need YAML schema validation", "Python")
        assert "yaml" in q.lower()
        assert "language:Python" in q


class TestScoutPipeline:
    def test_empty_problem_returns_empty(self):
        gate = LicenseGate()
        engine = ScoutEngine(_make_search_tool(), gate, provider=None)
        report = asyncio.run(engine.scout(""))
        assert isinstance(report, ScoutReport)
        assert report.candidates == []
        assert report.notes == "empty problem statement"

    def test_pipeline_blocks_gpl_candidates(self):
        gate = LicenseGate()
        tool = _make_search_tool()

        async def fake_search(**kwargs):
            return [
                {
                    "full_name": "alice/mit-tool",
                    "url": "https://x",
                    "description": "MIT licensed",
                    "stars": 100,
                    "language": "Python",
                    "license": "MIT",
                    "license_obj": {"spdx_id": "MIT"},
                    "last_pushed": "2026-01-01T00:00:00Z",
                    "topics": [],
                    "size_kb": 50,
                    "readme_excerpt": "MIT readme",
                },
                {
                    "full_name": "bob/gpl-tool",
                    "url": "https://x",
                    "description": "GPL licensed",
                    "stars": 100,
                    "language": "Python",
                    "license": "GPL-3.0-only",
                    "license_obj": {"spdx_id": "GPL-3.0-only"},
                    "last_pushed": "2026-01-01T00:00:00Z",
                    "topics": [],
                    "size_kb": 50,
                    "readme_excerpt": "GPL readme",
                },
            ]

        tool.search = fake_search  # type: ignore[method-assign]
        engine = ScoutEngine(tool, gate, provider=None)
        report = asyncio.run(engine.scout("yaml validation"))
        # MIT survives, GPL is blocked.
        names = {c.full_name for c in report.candidates}
        assert "alice/mit-tool" in names
        assert "bob/gpl-tool" not in names
        assert "1 blocked by license" in report.notes

    def test_pipeline_with_no_results(self):
        gate = LicenseGate()
        tool = _make_search_tool()

        async def fake_search(**kwargs):
            return []

        tool.search = fake_search  # type: ignore[method-assign]
        engine = ScoutEngine(tool, gate, provider=None)
        report = asyncio.run(engine.scout("any problem"))
        assert report.candidates == []
        assert report.notes == "no GitHub results"
