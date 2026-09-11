"""#284 — billing is a property of WHEN the tokens were spent, not of the model.

`/api/usage` classified every row by resolving the provider's CURRENT `base_url`.
That reads as a property of the model and is not one: it is a property of the
configuration file, re-derived on every request. `qwen3.8-max` has carried
280,501,476 input tokens under an Alibaba Token Plan. The plan lapses on
2026-09-14. Under read-time classification, the morning after, those same 280M
already-spent tokens stop being `subscription` — not because anything about them
changed, but because a hostname in the config did.

So the test that matters here is not "does the column exist". It is: CHANGE THE
CONFIG OUT FROM UNDER A ROW AND SEE WHETHER ITS PAST HOLDS. Every test below
does that, or checks that a row which was never stamped is labelled as inferred
rather than dressed up as a fact.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.telemetry.cost import billing_modes_from_config  # noqa: E402
from prometheus.telemetry.tracker import ToolCallTelemetry, set_telemetry_handle  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

TOKEN_PLAN = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
LAPSED = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"  # same box, plan expired

MODEL = "qwen3.8-max"
# The real number, as measured on the daemon on 2026-09-11 while the plan was live.
SPENT_UNDER_THE_PLAN = 280_501_476


def _tracker(tmp_path, resolver=None):
    tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    if resolver is not None:
        tel.billing_resolver = resolver
    return tel


def _client(tel, base_url):
    """A live app whose config says `base_url` RIGHT NOW, whatever the rows say."""
    set_telemetry_handle(tel)
    app = create_app({"providers": {"qwen": {"model": MODEL, "base_url": base_url}}})
    return TestClient(app)


def _stored_mode(tel, model=MODEL):
    row = tel._conn.execute(
        "SELECT billing_mode FROM subsystem_runs WHERE model = ? ORDER BY id DESC LIMIT 1",
        (model,),
    ).fetchone()
    return row[0] if row else None


def _model_row(body, model=MODEL):
    return next(m for m in body["models"] if m["model"] == model)


# ── the write half ──────────────────────────────────────────────────────────


def test_the_row_carries_the_mode_it_was_billed_under(tmp_path):
    """The stamp lands on the row, in the database, not in a derived view."""
    tel = _tracker(tmp_path, resolver={MODEL: "subscription"}.get)
    tel.record_run(
        "llm", "completion", "success",
        input_tokens=SPENT_UNDER_THE_PLAN, output_tokens=1_197_108, model=MODEL,
    )
    assert _stored_mode(tel) == "subscription", (
        "the mode was not written to the row — everything downstream is then a "
        "re-derivation of today's config, which is the defect"
    )


def test_a_resolver_that_raises_does_not_cost_the_row_its_telemetry(tmp_path):
    """Billing is a nice-to-have on a telemetry row. Losing the row is not."""
    def explode(_model):
        raise RuntimeError("provider registry is mid-reload")

    tel = _tracker(tmp_path, resolver=explode)
    tel.record_run("llm", "completion", "success", input_tokens=10, output_tokens=1, model=MODEL)
    count = tel._conn.execute("SELECT COUNT(*) FROM subsystem_runs").fetchone()[0]
    assert count == 1, "a throwing resolver swallowed the whole row"
    assert _stored_mode(tel) is None, "a failed resolution must read as 'not recorded', not a guess"


def test_no_resolver_stores_null_rather_than_guessing(tmp_path):
    """A bare tracker has no idea how anything is billed and must say so."""
    tel = _tracker(tmp_path)
    tel.record_run("llm", "completion", "success", input_tokens=10, output_tokens=1, model=MODEL)
    assert _stored_mode(tel) is None


# ── the harm: a lapsed plan re-billing the past ─────────────────────────────


def test_a_lapsed_plan_does_not_reclassify_tokens_already_spent(tmp_path):
    """THE finding. Same rows, same model — only the config moves.

    Before: the route resolves the model's CURRENT base_url, so the day the
    Token Plan lapses, 280M tokens spent under it are re-reported as metered and
    a dollar figure is invented for traffic that was never billed per token.
    """
    tel = _tracker(tmp_path, resolver={MODEL: "subscription"}.get)
    tel.record_run(
        "llm", "completion", "success",
        input_tokens=SPENT_UNDER_THE_PLAN, output_tokens=1_197_108, model=MODEL,
    )

    # The plan lapses. Nothing about the rows changes; the config does.
    body = _client(tel, LAPSED).get("/api/usage").json()
    row = _model_row(body)

    assert row["billing"] == "subscription", (
        "tokens spent under a flat plan were re-billed when the plan lapsed — "
        f"the route now calls them {row['billing']!r}"
    )
    assert row["billing_source"] == "recorded"
    assert row["cost_usd"] is None, "a dollar figure was invented for flat-plan traffic"
    assert row["input_tokens"] == SPENT_UNDER_THE_PLAN, "the tokens are still real"


def test_the_dated_observation_outlives_the_config_that_produced_it(tmp_path):
    """Rows written before the column existed still get the past they had.

    They cannot be stamped — nobody knows, per-row, and inventing one would put a
    rotting inference somewhere permanent. What CAN be written down is what the
    config said, with the date it said it. That is evidence, and it is labelled
    `observed`, not `recorded`.
    """
    tel = _tracker(tmp_path)  # no resolver: these rows land with billing_mode NULL
    tel.record_run(
        "llm", "completion", "success",
        input_tokens=SPENT_UNDER_THE_PLAN, output_tokens=1_197_108, model=MODEL,
    )
    tel.observe_billing_modes({MODEL: "subscription"})  # while the plan is still live

    seen = tel.observed_billing_modes()
    assert seen[MODEL]["mode"] == "subscription"
    assert seen[MODEL]["first_observed"] <= time.time()

    body = _client(tel, LAPSED).get("/api/usage").json()  # the plan has now lapsed
    row = _model_row(body)
    assert row["billing"] == "subscription"
    assert row["billing_source"] == "observed", (
        "an unstamped row was reported as though the mode had been recorded on it"
    )
    assert "observed as of" in row["billing_reason"]


def test_the_observation_records_a_mode_and_never_a_hostname(tmp_path):
    """Conventions: nothing that persists carries a real infrastructure identifier.

    The natural implementation keeps what it matched on — `billing_for` returns
    (mode, reason) and the reason reads "flat plan (token-plan.…aliyuncs.com)".
    Keeping the tuple wholesale "for debuggability" writes a live host into a
    database that is backed up, copied between machines, and read by /api/usage.

    This drives `billing_modes_from_config` rather than hand-passing a dict,
    because that helper is the thing the daemon actually calls — a test that
    constructs its own clean input cannot fail when the helper stops producing
    clean output.
    """
    tel = _tracker(tmp_path)
    modes = billing_modes_from_config(
        {"providers": {"qwen": {"model": MODEL, "base_url": TOKEN_PLAN}}}
    )
    # Vacuity guard only — deliberately loose. If this were `== "subscription"`
    # it would catch a leaked reason before the leak assertion ever ran, and the
    # assertion that names the actual rule would never be the one with teeth.
    assert "subscription" in str(modes.get(MODEL)), (
        f"the helper did not resolve the plan at all, so the leak check below "
        f"would pass vacuously: {modes}"
    )
    tel.observe_billing_modes(modes)
    stored = tel._conn.execute(
        "SELECT value FROM schema_meta WHERE key = ?", (tel.BILLING_OBSERVED_KEY,)
    ).fetchone()[0]
    for leak in ("token-plan", "aliyuncs", "http", "dashscope", ".com"):
        assert leak not in stored, f"the persisted observation leaked {leak!r}: {stored}"
    assert "subscription" in stored


def test_an_unstamped_row_with_no_observation_says_it_is_inferring(tmp_path):
    """Rule 8's other half: the fallback must not look like the answer.

    With nothing recorded and nothing observed, the route can only ask today's
    config — which is the pre-#284 behaviour, and is fine, so long as it does not
    present the result as a fact about the row.
    """
    tel = _tracker(tmp_path)
    tel.record_run("llm", "completion", "success", input_tokens=5_000, output_tokens=10, model=MODEL)
    body = _client(tel, TOKEN_PLAN).get("/api/usage").json()
    row = _model_row(body)
    assert row["billing"] == "subscription"  # same answer as before...
    assert row["billing_source"] == "inferred"  # ...but no longer claiming to be evidence
    assert "current config, not the row's" in row["billing_reason"]


# ── a history that splits ───────────────────────────────────────────────────


def test_a_model_billed_two_ways_is_not_collapsed_into_one(tmp_path):
    """The case that only exists because the mode is recorded.

    Before the plan lapses the tokens are flat-rate; after, the same model name
    is metered. One `billing` field per model cannot hold both, and whichever it
    picks silently relabels the other half.
    """
    tel = _tracker(tmp_path, resolver={MODEL: "subscription"}.get)
    tel.record_run(
        "llm", "completion", "success",
        input_tokens=SPENT_UNDER_THE_PLAN, output_tokens=1_000_000, model=MODEL,
    )
    tel.billing_resolver = {MODEL: "metered"}.get  # the plan lapses; the daemon rewires
    tel.record_run(
        "llm", "completion", "success",
        input_tokens=4_000_000, output_tokens=50_000, model=MODEL,
    )

    body = _client(tel, LAPSED).get("/api/usage").json()
    row = _model_row(body)

    modes = {b["billing"]: b for b in row["billing_breakdown"]}
    assert set(modes) == {"subscription", "metered"}, (
        f"a split history collapsed to {sorted(modes)}"
    )
    assert modes["subscription"]["input_tokens"] == SPENT_UNDER_THE_PLAN
    assert modes["metered"]["input_tokens"] == 4_000_000
    assert row["billing_modes"] == ["metered", "subscription"], (
        "the headline claims a single billing mode for a history that has two"
    )
    # `billing_source` is about PROVENANCE, not which mode won: both halves were
    # stamped on their rows, so both are `recorded` and there is nothing mixed
    # about how we know. That distinction is the reason they are separate fields.
    assert row["billing_source"] == "recorded"
    assert modes["subscription"]["cost_usd"] is None, "flat-plan half was priced"

    # Coverage must count the metered half only — a headline claiming to cover
    # 284M tokens when it priced 4M is the same defect one level up.
    assert body["totals"]["cost_covers_input_tokens"] == 4_000_000
    assert body["coverage"]["subscription"]["input_tokens"] == SPENT_UNDER_THE_PLAN


PRICED = "grok-4.5"  # (3.0, 15.0) per Mtok — a model the price table actually knows


def test_only_the_metered_half_of_a_split_history_is_priced(tmp_path):
    """The gap a mutation found: the split test above used an UNPRICED model.

    `qwen3.8-max` has no PRICING row, so `cost_usd` was None on every segment no
    matter which tokens the route reached for — and pricing the whole model
    instead of the metered segment survived, silently, with the suite green.
    Same split, on a model the price table knows, so the arithmetic is forced to
    be visible.
    """
    tel = _tracker(tmp_path, resolver={PRICED: "subscription"}.get)
    tel.record_run(
        "llm", "completion", "success",
        input_tokens=100_000_000, output_tokens=10_000_000, model=PRICED,
    )
    tel.billing_resolver = {PRICED: "metered"}.get  # the plan lapses
    tel.record_run(
        "llm", "completion", "success",
        input_tokens=1_000_000, output_tokens=100_000, model=PRICED,
    )

    body = _client(tel, LAPSED).get("/api/usage").json()
    row = _model_row(body, PRICED)
    metered = next(b for b in row["billing_breakdown"] if b["billing"] == "metered")
    subscription = next(b for b in row["billing_breakdown"] if b["billing"] == "subscription")

    # 1M in @ $3/Mtok + 0.1M out @ $15/Mtok = $3.00 + $1.50 = $4.50
    assert metered["cost_usd"] == 4.5, (
        f"the metered half was priced as {metered['cost_usd']} — if this is ~$453 "
        f"the route priced the whole model, including 100M flat-plan tokens"
    )
    assert subscription["cost_usd"] is None, "the flat-plan half was given a dollar figure"
    assert row["cost_usd"] == 4.5, "the model total must be the sum of its priced segments"
    assert body["totals"]["cost_usd"] == 4.5
    assert body["totals"]["cost_covers_input_tokens"] == 1_000_000


def test_an_existing_database_gains_the_column_and_keeps_its_rows_null(tmp_path):
    """The additive migration, driven against a DB that predates the column.

    Every row on the daemon today was written without it. They must survive the
    upgrade, keep their tokens, and read as "not recorded" — not as a default
    that a later reader mistakes for a measurement. This is the same failure the
    `latency_ms` rebuild existed to undo, one column over.
    """
    import sqlite3

    db = tmp_path / "telemetry.db"
    pre = ToolCallTelemetry(db_path=db)
    pre.record_run("llm", "completion", "success", input_tokens=7_777, output_tokens=11, model=MODEL)
    pre.close() if hasattr(pre, "close") else pre._conn.close()

    # Drop the column the way an older build would have left the table: no
    # billing_mode at all. SQLite can drop a column in place since 3.35.
    raw = sqlite3.connect(db)
    raw.execute("ALTER TABLE subsystem_runs DROP COLUMN billing_mode")
    cols = {r[1] for r in raw.execute("PRAGMA table_info(subsystem_runs)")}
    assert "billing_mode" not in cols, "the fixture failed to produce a pre-#284 database"
    raw.commit()
    raw.close()

    post = ToolCallTelemetry(db_path=db)  # __init__ runs _migrate_schema
    cols = {r[1] for r in post._conn.execute("PRAGMA table_info(subsystem_runs)")}
    assert "billing_mode" in cols, "the additive migration did not add the column"
    row = post._conn.execute(
        "SELECT input_tokens, billing_mode FROM subsystem_runs WHERE model = ?", (MODEL,)
    ).fetchone()
    assert row[0] == 7_777, "the migration lost the row's tokens"
    assert row[1] is None, "a migrated row was given a billing mode nobody recorded"
