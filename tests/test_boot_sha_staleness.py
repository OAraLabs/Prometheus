"""Boot-SHA staleness detector ("merged-but-dark").

boot_sha = the repo HEAD the process loaded (frozen at startup); tree_head =
the live repo HEAD now. When they differ, new code is on disk the running
process isn't executing — the exact signal the deferred #57/#64/#65 restarts
lacked. Surfaced on /api/status (full SHAs, bearer-gated), /health (bare bool,
unauthenticated), and nudged once-per-drift by the heartbeat.

Side-effect tests: assert the rendered endpoint payloads and the observed
``_notify`` calls — not internal call counts for their own sake. The endpoint
tests need fastapi and skip without it (like the rest of the web suite); the
heartbeat tests have no such dependency and always run.
"""

from __future__ import annotations

import asyncio

import pytest

import prometheus.gateway.heartbeat as hb_mod
from prometheus.gateway.heartbeat import Heartbeat


# ── Surface: /api/status (full SHAs) + /health (bare bool) ──────────────


def _client(monkeypatch, *, boot_sha: str, tree_sha: str):
    pytest.importorskip("fastapi")  # web suite is skip-if-absent
    from fastapi.testclient import TestClient

    import prometheus.web.server as server_mod
    from prometheus.web.server import create_app

    # No token in the test env → /api/* is reachable without a header.
    monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)
    monkeypatch.setattr(server_mod, "git_head_sha", lambda *a, **k: tree_sha)
    return TestClient(create_app({}, boot_sha=boot_sha))


def test_status_reports_boot_sha_and_not_stale_when_in_sync(monkeypatch):
    client = _client(monkeypatch, boot_sha="aaaa1111", tree_sha="aaaa1111")
    body = client.get("/api/status").json()
    assert body["running_sha"] == "aaaa1111"  # the frozen boot value is held
    assert body["tree_head"] == "aaaa1111"
    assert body["stale"] is False


def test_status_stale_true_when_tree_advanced(monkeypatch):
    client = _client(monkeypatch, boot_sha="aaaa1111", tree_sha="bbbb2222")
    body = client.get("/api/status").json()
    assert body["running_sha"] == "aaaa1111"
    assert body["tree_head"] == "bbbb2222"
    assert body["stale"] is True


def test_health_carries_bare_stale_bool_and_leaks_no_sha(monkeypatch):
    client = _client(monkeypatch, boot_sha="aaaa1111", tree_sha="bbbb2222")
    body = client.get("/health").json()
    assert body["stale"] is True
    assert "running_sha" not in body and "tree_head" not in body


def test_off_git_boot_sha_unknown_never_stale(monkeypatch):
    # boot captured off a git checkout → can't determine → never flag stale.
    client = _client(monkeypatch, boot_sha="unknown", tree_sha="bbbb2222")
    assert client.get("/api/status").json()["stale"] is False
    assert client.get("/health").json()["stale"] is False


# ── Heartbeat: de-duped once-per-drift notify (no fastapi dependency) ────


def _recording_heartbeat(boot_sha):
    hb = Heartbeat(boot_sha=boot_sha)
    sent: list[str] = []

    async def _rec(text, *, chat_id=None):
        sent.append(text)

    hb._notify = _rec  # type: ignore[assignment]
    return hb, sent


def _set_tree(monkeypatch, sha):
    monkeypatch.setattr(hb_mod, "git_head_sha", lambda *a, **k: sha)


def test_notify_fires_exactly_once_for_a_standing_drift(monkeypatch):
    hb, sent = _recording_heartbeat("aaaa1111")
    _set_tree(monkeypatch, "bbbb2222")
    for _ in range(5):  # five ticks, same drift
        asyncio.run(hb._check_staleness())
    assert len(sent) == 1
    assert "aaaa1111" in sent[0] and "bbbb2222" in sent[0]


def test_notify_fires_again_on_a_new_tree_head(monkeypatch):
    hb, sent = _recording_heartbeat("aaaa1111")
    _set_tree(monkeypatch, "bbbb2222")
    asyncio.run(hb._check_staleness())
    asyncio.run(hb._check_staleness())  # still bbbb — no repeat
    _set_tree(monkeypatch, "cccc3333")  # new commit on disk
    asyncio.run(hb._check_staleness())
    assert len(sent) == 2


def test_zero_notify_after_simulated_restart_boot_equals_tree(monkeypatch):
    # Post-restart the new process's boot_sha == current tree → silent.
    hb, sent = _recording_heartbeat("cccc3333")
    _set_tree(monkeypatch, "cccc3333")
    for _ in range(5):
        asyncio.run(hb._check_staleness())
    assert sent == []


def test_resync_resets_dedup_so_a_later_drift_renotifies(monkeypatch):
    hb, sent = _recording_heartbeat("aaaa1111")
    _set_tree(monkeypatch, "bbbb2222")
    asyncio.run(hb._check_staleness())          # drift → notify (1)
    _set_tree(monkeypatch, "aaaa1111")          # tree pulled back to boot
    asyncio.run(hb._check_staleness())          # in sync → silent + reset
    _set_tree(monkeypatch, "bbbb2222")          # drifts again
    asyncio.run(hb._check_staleness())          # → notify (2)
    assert len(sent) == 2


def test_unknown_boot_sha_never_notifies(monkeypatch):
    hb, sent = _recording_heartbeat("unknown")
    _set_tree(monkeypatch, "bbbb2222")
    asyncio.run(hb._check_staleness())
    assert sent == []


def test_unknown_tree_head_never_notifies(monkeypatch):
    # git unavailable at check time → can't determine → no false nudge.
    hb, sent = _recording_heartbeat("aaaa1111")
    _set_tree(monkeypatch, "unknown")
    asyncio.run(hb._check_staleness())
    assert sent == []
