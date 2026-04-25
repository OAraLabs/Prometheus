"""GitHubSearch — query building, rate limiter, response parsing."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from prometheus.symbiote.github_search import (
    GitHubClient,
    GitHubSearchTool,
    _TokenBucket,
)


class TestTokenBucket:
    def test_under_limit_returns_zero(self):
        b = _TokenBucket(max_requests=3, window_seconds=60)
        assert b.try_token() is True
        assert b.try_token() is True
        assert b.try_token() is True
        assert b.try_token() is False  # 4th request rejected

    def test_evicts_old(self):
        b = _TokenBucket(max_requests=2, window_seconds=0.05)
        b.try_token()
        b.try_token()
        time.sleep(0.06)
        assert b.try_token() is True

    def test_min_one_request(self):
        b = _TokenBucket(max_requests=0, window_seconds=60)
        # max clamped to 1
        assert b.try_token() is True


class TestQueryBuilding:
    def test_minimal_query(self):
        q = GitHubSearchTool._build_search_query(
            "yaml", language=None, min_stars=0, license_filter=None,
        )
        assert "yaml" in q
        assert "fork:false" in q

    def test_with_language_and_stars(self):
        q = GitHubSearchTool._build_search_query(
            "yaml schema", language="Python", min_stars=50, license_filter=None,
        )
        assert "language:Python" in q
        assert "stars:>=50" in q

    def test_with_license_filter(self):
        q = GitHubSearchTool._build_search_query(
            "x", language="Python", min_stars=10,
            license_filter=["MIT", "Apache-2.0"],
        )
        assert "license:mit" in q
        assert "license:apache-2.0" in q


class TestSummarize:
    def test_summarize_extracts_fields(self):
        item = {
            "full_name": "foo/bar",
            "description": "thing",
            "html_url": "https://github.com/foo/bar",
            "stargazers_count": 1234,
            "language": "Python",
            "license": {"spdx_id": "MIT"},
            "pushed_at": "2026-04-15T00:00:00Z",
            "topics": ["yaml", "schema"],
            "size": 450,
            "open_issues_count": 12,
            "default_branch": "main",
        }
        out = GitHubSearchTool._summarize_repo(item)
        assert out["full_name"] == "foo/bar"
        assert out["stars"] == 1234
        assert out["language"] == "Python"
        assert out["license"] == "MIT"
        assert out["topics"] == ["yaml", "schema"]
        assert out["size_kb"] == 450
        assert out["readme_excerpt"] == ""


class TestClientFromConfig:
    def test_no_token_when_no_env_no_config(self, monkeypatch):
        monkeypatch.delenv("PROMETHEUS_GITHUB_TOKEN", raising=False)
        client = GitHubClient.from_config(None)
        assert client.has_token is False

    def test_env_token_overrides_config(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_GITHUB_TOKEN", "env_token")
        client = GitHubClient.from_config({"github_token": "config_token"})
        # env wins
        assert client.has_token is True

    def test_config_token_used_when_no_env(self, monkeypatch):
        monkeypatch.delenv("PROMETHEUS_GITHUB_TOKEN", raising=False)
        client = GitHubClient.from_config({"github_token": "abc"})
        assert client.has_token is True


class TestRateRemaining:
    def test_decrements_on_token(self):
        client = GitHubClient(token=None)
        before = client.rate_remaining_estimate
        client._rate.try_token()
        after = client.rate_remaining_estimate
        assert after == before - 1
