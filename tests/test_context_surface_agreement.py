"""THE INVARIANT: every surface that reports a context window reports the SAME
window for the same model and the same detected value.

The individual numbers are implementation. This equality is the actual
contract, and it is the thing that broke: PR #358 fixed /api/lcm to resolve
the real window (32768, detected) but did not reach the gateway command layer,
which kept calling ``TokenBudget.from_config(model=model_name)`` with neither
``local_model=`` nor ``detected_limit=``. The resolver cannot reach its
"detected" branch without both, so /context fell through to the configured
global while Beacon showed the detected one — and nothing on either surface
said which was authoritative.

Disagreeing surfaces are worse than the uniform wrongness that preceded them:
before, one number was wrong everywhere and a single fix corrected all of it.
After, an operator comparing Beacon to /context has to know which code path
each went through to know which to believe.

So this file asserts agreement DIRECTLY, by driving both real
implementations — the FastAPI route and the gateway formatter — from the same
inputs and comparing what each one reports. A test that re-derived the
expected number from the resolver would only prove the resolver.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from prometheus.gateway.commands import cmd_context
from prometheus.web.server import create_app

LOCAL_MODEL = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
CLOUD_MODEL = "claude-sonnet-4-5"
SERVER_N_CTX = 32768
CONFIG_LIMIT = 72000

CONFIG = {
    "model": {"model": LOCAL_MODEL, "provider": "llama_cpp"},
    "context": {
        "effective_limit": CONFIG_LIMIT,
        "cloud_default_limit": 1_000_000,
        "model_overrides": {CLOUD_MODEL: {"effective_limit": 200_000}},
    },
}


@dataclass
class _FakeResult:
    total_tokens: int = 9888
    compression_ratio: float = 0.42
    fresh_messages: list = field(default_factory=list)
    summaries: list = field(default_factory=list)


class _Engine:
    def assemble(self, session_id: str, token_budget: int) -> _FakeResult:
        return _FakeResult()


def _beacon_limit(model: str, detected: int | None) -> int | None:
    """What /api/lcm reports, through the real route."""
    config = {**CONFIG, "model": {**CONFIG["model"], "model": model}}
    client = TestClient(create_app(
        config, lcm_engine=_Engine(),
        detected_context_size=detected, local_model=LOCAL_MODEL,
    ))
    return client.get("/api/lcm/desktop:1").json()["limit"]


def _context_limit(model: str, detected: int | None) -> int | None:
    """What /context reports, through the real formatter.

    Parsed out of the rendered reply on purpose: the number a user reads is
    the thing under test, not an intermediate the formatter happens to hold.
    """
    text = cmd_context(
        "system prompt", model,
        local_model=LOCAL_MODEL, detected_limit=detected, config=CONFIG,
    )
    match = re.search(r"Window size:\s+([\d,]+) tokens", text)
    if match is None:
        assert "Window size:     unknown" in text, text
        return None
    return int(match.group(1).replace(",", ""))


@pytest.mark.parametrize(
    "model, detected, expected",
    [
        (LOCAL_MODEL, SERVER_N_CTX, SERVER_N_CTX),   # the live case
        (LOCAL_MODEL, None, CONFIG_LIMIT),           # backend unreachable
        (CLOUD_MODEL, SERVER_N_CTX, 200_000),        # per-model override
        ("some-other-cloud-model", SERVER_N_CTX, 1_000_000),  # cloud default
    ],
    ids=["detected", "backend-down", "override", "cloud-default"],
)
def test_beacon_and_context_command_agree(model, detected, expected):
    beacon = _beacon_limit(model, detected)
    gateway = _context_limit(model, detected)

    assert beacon == gateway, (
        f"/api/lcm says {beacon} and /context says {gateway} for model "
        f"{model!r} with detected={detected}. Two surfaces reporting "
        "different windows is the bug — an operator cannot tell which is "
        "authoritative."
    )
    assert beacon == expected


def test_the_exact_divergence_this_fixes():
    """The live numbers, stated as the regression they were.

    config effective_limit 72000, server n_ctx 32768: Beacon resolved 32768
    while /context reported something else entirely. Both must now say 32768.
    """
    assert _beacon_limit(LOCAL_MODEL, SERVER_N_CTX) == SERVER_N_CTX
    assert _context_limit(LOCAL_MODEL, SERVER_N_CTX) == SERVER_N_CTX


def test_both_surfaces_report_unknown_together():
    """Agreement has to hold on the unresolved state too, or the surfaces
    diverge again exactly where a number is least trustworthy."""
    client = TestClient(create_app({}, lcm_engine=_Engine()))
    assert client.get("/api/lcm/s").json()["limit"] is None

    text = cmd_context("system prompt", LOCAL_MODEL, config={})
    assert "Window size:     unknown" in text
    assert "24,000" not in text and "24000" not in text


def test_context_reply_names_its_source():
    """A number without provenance is not comparable to another number.

    /api/status carries `source`; /context has to say the same thing, or
    comparing the two surfaces means comparing two bare integers.
    """
    detected = cmd_context("p", LOCAL_MODEL, local_model=LOCAL_MODEL,
                           detected_limit=SERVER_N_CTX, config=CONFIG)
    assert "Source: detected" in detected

    fallback = cmd_context("p", LOCAL_MODEL, local_model=LOCAL_MODEL,
                           detected_limit=None, config=CONFIG)
    assert "Source: config" in fallback

    unknown = cmd_context("p", LOCAL_MODEL, config={})
    assert "Source: unknown" in unknown


def test_status_and_context_agree_on_the_source_label():
    """Not just the number — the provenance must match too."""
    client = TestClient(create_app(
        CONFIG, lcm_engine=_Engine(),
        detected_context_size=SERVER_N_CTX, local_model=LOCAL_MODEL,
    ))
    status_source = client.get("/api/status").json()["context"]["source"]
    text = cmd_context("p", LOCAL_MODEL, local_model=LOCAL_MODEL,
                       detected_limit=SERVER_N_CTX, config=CONFIG)
    assert status_source == "detected"
    assert f"Source: {status_source}" in text


def test_gateway_does_not_read_the_config_from_disk_when_given_one(monkeypatch):
    """One detection, one resolution.

    The disk fallback resolves DEFAULTS_PATH, which points one directory ABOVE
    the repo root — it does not exist on this checkout or on the deploy, so
    from_config() swallows the OSError and silently resolves against an empty
    config. Passing the daemon's loaded dict is what makes /context reflect
    the file the daemon is actually running.
    """
    from prometheus.context.budget import TokenBudget

    reads: list[str] = []

    def _explode(cls, *args, **kwargs):
        reads.append("from_config")
        raise AssertionError("must not touch the disk")

    monkeypatch.setattr(TokenBudget, "from_config", classmethod(_explode))

    text = cmd_context("p", LOCAL_MODEL, local_model=LOCAL_MODEL,
                       detected_limit=SERVER_N_CTX, config=CONFIG)

    assert reads == [], (
        "cmd_context read the config from disk despite being handed the "
        "daemon's loaded dict"
    )
    # And it still produced the right answer from the dict alone.
    assert f"{SERVER_N_CTX:,}" in text


def test_disk_fallback_still_works_for_callers_without_a_config():
    """Not every caller has the loaded dict; that path must not crash."""
    text = cmd_context("p", LOCAL_MODEL)
    assert "Context Window" in text
    assert "Source:" in text
