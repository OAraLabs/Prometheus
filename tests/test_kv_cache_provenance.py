"""KV cache type is RECORDED, never inferred — and an unrecorded run stays
distinguishable from a confirmed-f16 one.

Prometheus does not launch llama-server, so it cannot set the K/V cache
quantisation; it can only attribute a run to whatever the server was running.
That attribution is the entire point: a gym A/B comparing q8_0 against f16 KV
is unfalsifiable once the runs are over unless each row says which it was.

The temptation this file exists to block is defaulting the unknown case to
f16, which is llama.cpp's own default and therefore the plausible guess. A
plausible guess silently converts "we never looked" into "we checked and it
was f16", and every comparison built on top inherits that. So the assertions
below are mostly about what must NOT appear.

MEASURED, NOT ASSUMED: checked 2026-08-31 against the live endpoint (llama.cpp
build b1-9d57ce456), /props publishes no cache type at any level and /slots
and /metrics are disabled (HTTP 501). ``unreported`` is therefore the correct
answer on that server today, and test_probe_reads_the_field_when_present pins
the reader for the build that eventually does publish it.
"""

from __future__ import annotations

import sqlite3

import httpx
import pytest

from prometheus.gym.report import _kv_cache
from prometheus.gym.store import GymStore
from prometheus.providers.llama_cpp import LlamaCppProvider

# The shape the live server returns today: n_ctx and nothing about the cache.
LIVE_PROPS_TODAY = {
    "default_generation_settings": {"n_ctx": 32768, "params": {"temperature": 0.7}},
    "build_info": "b1-9d57ce456",
    "model_path": "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
}


def _provider_with_props(monkeypatch, payload, *, status: int = 200):
    class _Resp:
        status_code = status

        def raise_for_status(self):
            if status != 200:
                raise httpx.HTTPStatusError("boom", request=None, response=None)

        def json(self):
            return payload

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url):
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return LlamaCppProvider(base_url="http://stub:8080")


async def test_absent_field_is_unreported_never_f16(monkeypatch):
    """The live case. The forbidden outcome is a confident 'f16'."""
    provider = _provider_with_props(monkeypatch, LIVE_PROPS_TODAY)
    result = await provider.detect_kv_cache_types()

    assert result["source"] == "unreported"
    assert result["k"] is None and result["v"] is None
    assert "b1-9d57ce456" in result["detail"]
    # The whole point, stated as an assertion:
    assert "f16" not in repr(result).lower()


@pytest.mark.parametrize(
    "payload, where",
    [
        ({**LIVE_PROPS_TODAY, "cache_type_k": "q8_0", "cache_type_v": "q8_0"},
         "props"),
        ({**LIVE_PROPS_TODAY,
          "default_generation_settings": {"n_ctx": 32768,
                                          "cache_type_k": "q8_0",
                                          "cache_type_v": "q8_0"}},
         "default_generation_settings"),
        ({**LIVE_PROPS_TODAY,
          "default_generation_settings": {
              "n_ctx": 32768,
              "params": {"type_k": "q8_0", "type_v": "q8_0"}}},
         "params"),
    ],
    ids=["top-level", "generation-settings", "params-type_k"],
)
async def test_probe_reads_the_field_when_present(monkeypatch, payload, where):
    """For the build that eventually publishes it, at whichever level."""
    provider = _provider_with_props(monkeypatch, payload)
    result = await provider.detect_kv_cache_types()

    assert (result["k"], result["v"]) == ("q8_0", "q8_0")
    assert result["source"] == "props"
    assert where in result["detail"]
    assert provider.server_kv_cache == result


async def test_unreachable_server_is_not_unreported(monkeypatch):
    """Three distinct states, not two: could not ask ≠ asked and got nothing."""
    def _boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "AsyncClient", _boom)
    result = await LlamaCppProvider(base_url="http://dead:8080").detect_kv_cache_types()
    assert result["source"] == "unreachable"
    assert result["k"] is None and result["v"] is None


async def test_no_config_key_claims_to_control_the_cache_type():
    """Prometheus does not launch llama-server; a key here would be a lie.

    Guards against someone 'helpfully' adding model.cache_type_k later: it
    would look like a control, set nothing, and make every run recorded under
    it wrong in a way nothing would catch.
    """
    from pathlib import Path

    import prometheus

    repo = Path(prometheus.__file__).resolve().parents[2]
    # The TRACKED template every install copies from. The live
    # config/prometheus.yaml is gitignored and does not travel into worktrees,
    # so asserting on it would make this test skip exactly where it matters.
    template = repo / "config" / "prometheus.yaml.default"
    assert template.exists(), f"shipped config template missing at {template}"

    targets = [template]
    live = repo / "config" / "prometheus.yaml"
    if live.exists():
        targets.append(live)

    for path in targets:
        text = path.read_text(encoding="utf-8")
        for forbidden in ("cache_type_k:", "cache_type_v:", "kv_cache_type:",
                          "kv_cache_k:", "kv_cache_v:"):
            assert forbidden not in text, (
                f"{forbidden} in {path.name} claims to govern the KV cache. "
                "Prometheus does not launch llama-server — it records the "
                "cache type, it cannot set it, and a key that looks like a "
                "control but sets nothing is worse than no key."
            )


# ---------------------------------------------------------------------------
# Gym persistence — the attribution has to survive the run
# ---------------------------------------------------------------------------

def _row(**over):
    row = dict(
        series="s2", experiment="e1", task_id="t1", run_idx=0,
        model="qwen", category="tools", success=1, emission_pass=1,
        execution_pass=1, fail_reasons=None, tools_called="[]", latency_ms=1.0,
        retries=0, repairs=0, dropped_malformed=0, feedback_retries=0,
        breaker_tripped=0, error=None, manifest_sha="a", taskset_sha="b",
        kv_cache_k=None, kv_cache_v=None, kv_cache_source="unreported",
    )
    row.update(over)
    return row


def test_gym_rows_carry_the_cache_attribution(tmp_path):
    store = GymStore(tmp_path / "gym.db")
    store.record_run(**_row(kv_cache_k="q8_0", kv_cache_v="q8_0",
                            kv_cache_source="props"))
    store.record_run(**_row(run_idx=1))  # unreported arm
    rows = store.runs("s2", "e1")
    assert [r["kv_cache_source"] for r in rows] == ["props", "unreported"]
    assert rows[0]["kv_cache_k"] == "q8_0" and rows[1]["kv_cache_k"] is None
    store.close()


def test_migration_adds_columns_to_a_pre_existing_db(tmp_path):
    """An old gym.db must open, not crash — and its rows read 'not recorded'."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE gym_runs (series TEXT NOT NULL, experiment TEXT NOT NULL, "
        "task_id TEXT NOT NULL, run_idx INTEGER NOT NULL, timestamp REAL NOT NULL, "
        "model TEXT NOT NULL, category TEXT NOT NULL, success INTEGER NOT NULL, "
        "fail_reasons TEXT, tools_called TEXT, latency_ms REAL NOT NULL DEFAULT 0, "
        "retries INTEGER NOT NULL DEFAULT 0, repairs INTEGER NOT NULL DEFAULT 0, "
        "dropped_malformed INTEGER NOT NULL DEFAULT 0, "
        "feedback_retries INTEGER NOT NULL DEFAULT 0, "
        "breaker_tripped INTEGER NOT NULL DEFAULT 0, error TEXT, "
        "manifest_sha TEXT NOT NULL, taskset_sha TEXT NOT NULL, "
        "PRIMARY KEY (series, experiment, task_id, run_idx))"
    )
    conn.execute(
        "INSERT INTO gym_runs VALUES ('s','e','t',0,1.0,'m','c',1,NULL,'[]',"
        "1.0,0,0,0,0,0,NULL,'a','b')"
    )
    conn.commit()
    conn.close()

    store = GymStore(db)
    have = {r[1] for r in store._conn.execute("PRAGMA table_info(gym_runs)")}
    assert {"kv_cache_k", "kv_cache_v", "kv_cache_source"} <= have
    assert store.runs("s", "e")[0]["kv_cache_source"] is None  # never recorded
    store.close()


# ---------------------------------------------------------------------------
# The report says so out loud
# ---------------------------------------------------------------------------

def _rows(dicts):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE r (kv_cache_k TEXT, kv_cache_v TEXT, kv_cache_source TEXT)")
    conn.executemany("INSERT INTO r VALUES (?,?,?)", dicts)
    return conn.execute("SELECT * FROM r").fetchall()


def test_report_prints_confirmed_cache_types():
    line = _kv_cache(_rows([("q8_0", "q8_0", "props")] * 3))
    assert "q8_0" in line and "reported by the server" in line


def test_report_prints_the_absence_rather_than_omitting_it():
    """An unknown cache type must be visible in the artifact, not silent."""
    line = _kv_cache(_rows([(None, None, "unreported")] * 3))
    assert "unknown" in line and "NOT inferred" in line
    assert "f16" not in line


def test_report_flags_an_arm_that_mixed_cache_types():
    """Mixed within one arm invalidates it as an arm — say so."""
    line = _kv_cache(_rows([("q8_0", "q8_0", "props"), (None, None, "unreported")]))
    assert "MIXED" in line
