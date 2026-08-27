"""GET /api/usage — token usage and cost per model (branch feat/api-usage).

THE ASSERTION THAT MATTERS: a zero cost has four different meanings and the endpoint must never
collapse them. `qwen3.8-max` carried 89M tokens — 57% of everything this box ever spent — through
an Alibaba Token Plan (a flat subscription), and rendered as $0.00 for months because "no price
row" and "no per-token price exists" were the same value. Here they are different fields, and
`cost_usd is None` is asserted directly rather than implied.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.telemetry.cost import billing_for  # noqa: E402
from prometheus.telemetry.tracker import ToolCallTelemetry, set_telemetry_handle  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

TOKEN_PLAN = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
METERED = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _seed(tmp_path):
    """A tracker holding one run per billing situation, with real-shaped token counts."""
    tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    now = time.time()
    rows = [
        # (model, input, output, age_days)
        ("qwen3.8-max", 89_042_118, 1_197_108, 1),          # subscription (Token Plan host)
        ("grok-4.5", 6_017_769, 65_930, 2),                  # metered, priced
        ("/home/x/models/Qwen3.8-27B.gguf", 32_351_937, 180_694, 3),  # local
        ("gemma4-26b", 8_936_391, 407_338, 40),              # unknown + outside a 30d window
    ]
    for model, i, o, age in rows:
        tel.record_run(
            "llm", "completion", "success",
            input_tokens=i, output_tokens=o, model=model, session_id="s1",
        )
        # backdate the row so the window filter has something to exclude
        tel._conn.execute(
            "UPDATE subsystem_runs SET timestamp = ? WHERE model = ?", (now - age * 86400, model)
        )
    tel._conn.commit()
    return tel


def _client(tmp_path):
    tel = _seed(tmp_path)
    set_telemetry_handle(tel)
    app = create_app({"providers": {"qwen": {"model": "qwen3.8-max", "base_url": TOKEN_PLAN}}})
    return TestClient(app), tel


def test_subscription_tokens_are_never_a_dollar_figure(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN_BASE_URL", TOKEN_PLAN)
    client, _ = _client(tmp_path)
    body = client.get("/api/usage").json()
    qwen = next(m for m in body["models"] if m["model"] == "qwen3.8-max")
    assert qwen["billing"] == "subscription"
    # The whole point: None, not 0.0. A client cannot render "$0.00" from this.
    assert qwen["cost_usd"] is None
    assert qwen["input_tokens"] == 89_042_118  # the tokens are still REAL and still reported
    assert "plan" in qwen["billing_reason"]


def test_metered_model_carries_a_real_cost(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/api/usage").json()
    grok = next(m for m in body["models"] if m["model"] == "grok-4.5")
    assert grok["billing"] == "metered"
    # the endpoint rounds to 4dp deliberately (a cent is $0.01; 4dp keeps sub-cent models honest)
    assert grok["cost_usd"] == round(6_017_769 * 3.0 / 1e6 + 65_930 * 15.0 / 1e6, 4)


def test_local_and_unknown_are_distinguished(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/api/usage").json()
    by = {m["model"]: m for m in body["models"]}
    assert by["/home/x/models/Qwen3.8-27B.gguf"]["billing"] == "local"
    assert by["/home/x/models/Qwen3.8-27B.gguf"]["cost_usd"] is None
    # An unrecognised name is NOT quietly called local or metered — it is an open question.
    assert by["gemma4-26b"]["billing"] == "unknown"
    assert by["gemma4-26b"]["cost_usd"] is None
    assert "PRICING" in by["gemma4-26b"]["billing_reason"]


def test_totals_say_what_share_of_tokens_the_dollars_cover(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/api/usage").json()
    totals = body["totals"]
    # dollars come only from the metered model, so coverage must be a small share of the tokens
    assert totals["cost_usd"] > 0
    assert totals["cost_covers_input_tokens"] == 6_017_769
    assert 0 < totals["cost_covers_share"] < 0.06
    assert totals["input_tokens"] == 89_042_118 + 6_017_769 + 32_351_937 + 8_936_391
    # every billing mode present is accounted for, so a client can show the split
    assert set(body["coverage"]) == {"subscription", "metered", "local", "unknown"}


def test_window_filter_excludes_older_rows(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/api/usage?days=30").json()
    assert body["window_days"] == 30
    assert "gemma4-26b" not in {m["model"] for m in body["models"]}  # 40 days old
    assert "qwen3.8-max" in {m["model"] for m in body["models"]}
    # tracking_since is the FIRST tokened row ever, not the window start — a client must be able
    # to say how far back the data itself goes.
    assert body["tracking_since"] is not None


def test_tracking_since_is_reported_so_a_window_is_honest(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/api/usage").json()
    assert body["tracking_since"].endswith("Z")


def test_no_telemetry_handle_degrades_instead_of_500(tmp_path):
    set_telemetry_handle(None)
    app = create_app({})
    body = TestClient(app).get("/api/usage").json()
    assert body["models"] == [] and body["tracking_since"] is None


def test_billing_for_refuses_to_guess_a_metered_host(tmp_path):
    """The pay-as-you-go case: same model, non-plan host, no price row → unknown, not $0."""
    assert billing_for("qwen3.8-max", TOKEN_PLAN)[0] == "subscription"
    assert billing_for("qwen3.8-max", METERED)[0] == "unknown"
    assert billing_for("grok-4.5", "https://api.x.ai/v1")[0] == "metered"
    assert billing_for("/models/x.gguf", TOKEN_PLAN)[0] == "local"  # structural beats nominal
    assert billing_for("", None)[0] == "unknown"
