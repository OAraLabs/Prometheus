"""Billing mode is a property of WHEN a call happened (#284).

THE DEFECT
----------
`/api/usage` classified a model's billing by resolving the provider's CURRENT
`base_url` at read time. That makes billing look like a property of the MODEL.
It is a property of WHEN the call was made, and the difference is not academic:
the live configuration WAS the historical record, so repointing one environment
variable reclassified 280M already-spent tokens from `subscription` to
`unknown` — retroactively, for months of rows nobody touched.

THE TEST THAT IS THE DEFECT
----------------------------
`test_a_recorded_row_does_not_reclassify_when_the_config_moves`. Everything else
here is scaffolding around it. It seeds a row stamped at write time, then builds
the app pointed at a DIFFERENT host, and asserts the answer did not move.

Hosts below are `.invalid` (RFC 2606) on purpose — they exercise the same
`SUBSCRIPTION_HOST_MARKERS` prefixes without adding real infrastructure
identifiers to the repository.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.telemetry.cost import billing_host_of, billing_stamp  # noqa: E402
from prometheus.telemetry.tracker import (  # noqa: E402
    ToolCallTelemetry,
    set_telemetry_handle,
)
from prometheus.web.server import create_app  # noqa: E402

PLAN_HOST = "token-plan.ap-southeast-1.example.invalid"
PLAN_URL = f"https://{PLAN_HOST}/compatible-mode/v1"
METERED_URL = "https://dashscope-intl.example.invalid/compatible-mode/v1"
MODEL = "qwen3.8-max"


def _app_pointed_at(url: str):
    return TestClient(
        create_app({"providers": {"qwen": {"model": MODEL, "base_url": url}}})
    )


def _model_row(client, model=MODEL):
    body = client.get("/api/usage").json()
    return next(m for m in body["models"] if m["model"] == model), body


# ── the write-time stamp ─────────────────────────────────────────────────────


def test_record_run_persists_the_stamp(tmp_path):
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    tel.record_run(
        "llm", "completion", "success",
        input_tokens=10, output_tokens=1, model=MODEL,
        billing_mode="subscription",
    )
    row = tel._conn.execute(
        "SELECT billing_mode FROM subsystem_runs WHERE model = ?", (MODEL,),
    ).fetchone()
    assert row == ("subscription",)


def test_an_unstamped_row_is_null_not_a_default(tmp_path):
    """NULL must stay NULL: "nobody recorded this" is not a billing mode."""
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    tel.record_run("llm", "completion", "success", input_tokens=10, model=MODEL)
    row = tel._conn.execute(
        "SELECT billing_mode FROM subsystem_runs WHERE model = ?", (MODEL,),
    ).fetchone()
    assert row == (None,)


def test_the_host_is_classified_and_then_dropped(tmp_path):
    """Rule: nothing that persists carries a real infrastructure identifier.

    The host is genuinely useful — keeping it would let history be re-derived
    if `SUBSCRIPTION_HOST_MARKERS` ever gains an entry. It goes anyway. What
    `billing_stamp` returns is the verdict alone, and this asserts the return
    type rather than the contents of one example, because a tuple that still
    carries the host would pass any value-based check written around it.
    """
    class _P:
        _base_url = "https://metered.example.invalid/v1?key=SUPERSECRET"

    # The helper still parses a host — transiently, to classify against it.
    host = billing_host_of(_P())
    assert host == "metered.example.invalid"
    assert "SUPERSECRET" not in (host or ""), "the query string survived parsing"
    assert "?" not in (host or "")

    stamped = billing_stamp(MODEL, _P())
    assert isinstance(stamped, str), (
        f"billing_stamp returned {stamped!r} — a tuple here means the host is "
        f"being handed to the writer, which is how it reaches the database"
    )
    assert "example.invalid" not in stamped and "SUPERSECRET" not in stamped


def test_a_provider_with_no_url_stamps_nothing_rather_than_guessing(tmp_path):
    class _P:
        pass

    assert billing_host_of(_P()) is None
    # No host to classify against -> falls back to name-only classification,
    # which for an unpriced model is an honest "unknown" rather than a guess.
    assert billing_stamp(MODEL, _P()) == "unknown"


# ── the read path ────────────────────────────────────────────────────────────


def test_a_recorded_row_does_not_reclassify_when_the_config_moves(tmp_path, monkeypatch):
    """THE DEFECT. A stamped row is history and history does not move."""
    monkeypatch.setenv("QWEN_BASE_URL", METERED_URL)
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    tel.record_run(
        "llm", "completion", "success",
        input_tokens=89_042_118, output_tokens=1_197_108, model=MODEL,
        billing_mode="subscription",
    )
    set_telemetry_handle(tel)

    # The box has been repointed at the METERED host — the 9/14 event.
    row, _ = _model_row(_app_pointed_at(METERED_URL))

    assert row["billing"] == "subscription", (
        "the stamped row followed the CURRENT base_url. This is exactly the "
        "defect: 280M tokens spent under a flat plan reclassified because an "
        "env var moved months later."
    )
    assert row["billing_source"] == "recorded"
    assert row["cost_usd"] is None, "subscription tokens are never a dollar figure"
    assert row["input_tokens"] == 89_042_118, "the tokens are still real"


def test_an_unstamped_row_is_labelled_inferred_not_presented_as_history(
    tmp_path, monkeypatch
):
    """Read-time classification survives as a FALLBACK, and must say so."""
    monkeypatch.setenv("QWEN_BASE_URL", PLAN_URL)
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    tel.record_run("llm", "completion", "success", input_tokens=100, model=MODEL)
    set_telemetry_handle(tel)

    row, _ = _model_row(_app_pointed_at(PLAN_URL))
    assert row["billing"] == "subscription"          # same answer as before...
    assert row["billing_source"] == "inferred"        # ...but no longer silent
    assert "inferred" in row["billing_reason"]


def test_a_model_spanning_the_cutover_stays_one_row_and_reports_mixed(
    tmp_path, monkeypatch
):
    """One row per model, ALWAYS — the split goes in `billing_breakdown`.

    Splitting `models` would change a shape every client renders, on the day a
    plan lapses. The truth is still fully available, additively.
    """
    monkeypatch.setenv("QWEN_BASE_URL", METERED_URL)
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    tel.record_run(
        "llm", "completion", "success", input_tokens=1_000, output_tokens=10,
        model=MODEL, billing_mode="subscription",
    )
    tel.record_run(
        "llm", "completion", "success", input_tokens=250, output_tokens=5,
        model=MODEL, billing_mode="metered",
    )
    set_telemetry_handle(tel)

    body = _app_pointed_at(METERED_URL).get("/api/usage").json()
    rows = [m for m in body["models"] if m["model"] == MODEL]
    assert len(rows) == 1, f"the model was split into {len(rows)} API rows"

    row = rows[0]
    assert row["billing"] == "mixed"
    assert row["input_tokens"] == 1_250, "the model total must still be the total"

    modes = {b["billing"]: b for b in row["billing_breakdown"]}
    assert set(modes) == {"subscription", "metered"}
    assert modes["subscription"]["input_tokens"] == 1_000
    assert modes["metered"]["input_tokens"] == 250
    assert sum(b["input_tokens"] for b in row["billing_breakdown"]) == row["input_tokens"]


def test_billing_source_reports_the_weakest_provenance_present(tmp_path, monkeypatch):
    """Sticky-weakest, the rule #450's latency source uses.

    An aggregate has already blended provenances by the time anyone reads it,
    so reporting the strongest present overstates what is known.
    """
    monkeypatch.setenv("QWEN_BASE_URL", PLAN_URL)
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    tel.record_run(
        "llm", "completion", "success", input_tokens=10, model=MODEL,
        billing_mode="subscription",
    )
    tel.record_run("llm", "completion", "success", input_tokens=10, model=MODEL)
    set_telemetry_handle(tel)

    row, _ = _model_row(_app_pointed_at(PLAN_URL))
    assert row["billing_source"] == "inferred", (
        "one unstamped row must drag the model's source down — a partly-"
        "recorded aggregate is not a recorded one"
    )


def test_a_row_older_than_the_boundary_reads_as_backfilled(tmp_path, monkeypatch):
    """`recorded` and `backfilled` are different claims and must not collapse.

    One says what was true when the call was made. The other says what a human
    concluded afterwards from a dated configuration. Only the boundary can tell
    them apart once both are just strings in a column.
    """
    monkeypatch.setenv("QWEN_BASE_URL", PLAN_URL)
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    tel.record_run(
        "llm", "completion", "success", input_tokens=10, model=MODEL,
        billing_mode="subscription",
    )
    # Backdate it to before the writer took over — i.e. the backfill's territory.
    boundary = tel.billing_boundary()
    assert boundary is not None, "v3 must stamp a boundary at init"
    tel._conn.execute(
        "UPDATE subsystem_runs SET timestamp = ? WHERE model = ?",
        (boundary - 86_400, MODEL),
    )
    tel._conn.commit()
    set_telemetry_handle(tel)

    row, _ = _model_row(_app_pointed_at(PLAN_URL))
    assert row["billing"] == "subscription"
    assert row["billing_source"] == "backfilled"


def test_the_boundary_does_not_move_when_the_database_is_reopened(tmp_path):
    """If it moved, already-stamped rows would drift into looking backfilled."""
    db = tmp_path / "t.db"
    first = ToolCallTelemetry(db_path=db)
    b1 = first.billing_boundary()
    first._conn.close()
    time.sleep(0.01)
    second = ToolCallTelemetry(db_path=db)
    assert second.billing_boundary() == b1


# ── the conventions, and the gaps a mutation found ───────────────────────────


def test_nothing_in_the_schema_or_the_response_stores_a_host(tmp_path, monkeypatch):
    """The rule is about what PERSISTS, so check the persisted things directly.

    A column named `billing_host` is the obvious way to keep the evidence, and
    the first implementation of this change had one. Asserting on the schema
    and on the served body — rather than on any one value — is what stops it
    coming back under a different name.
    """
    monkeypatch.setenv("QWEN_BASE_URL", PLAN_URL)
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    tel.record_run(
        "llm", "completion", "success", input_tokens=10, model=MODEL,
        billing_mode="subscription",
    )
    cols = {r[1] for r in tel._conn.execute("PRAGMA table_info(subsystem_runs)")}
    assert not [c for c in cols if "host" in c or "url" in c], (
        f"a column is storing an address: {sorted(cols)}"
    )
    set_telemetry_handle(tel)

    body = _app_pointed_at(PLAN_URL).get("/api/usage").text
    for leak in ("token-plan", "example.invalid", "http://", "https://"):
        assert leak not in body, f"/api/usage served {leak!r} back to the client"


def test_an_existing_database_gains_the_column_and_keeps_its_rows_null(tmp_path):
    """The additive migration, driven against a DB that predates the column.

    Every row on the daemon today was written without it. They must survive the
    upgrade, keep their tokens, and read as "not recorded" — not as a default a
    later reader mistakes for a measurement.
    """
    import sqlite3

    db = tmp_path / "t.db"
    pre = ToolCallTelemetry(db_path=db)
    pre.record_run("llm", "completion", "success", input_tokens=7_777, model=MODEL)
    pre._conn.close()

    raw = sqlite3.connect(db)
    raw.execute("ALTER TABLE subsystem_runs DROP COLUMN billing_mode")
    cols = {r[1] for r in raw.execute("PRAGMA table_info(subsystem_runs)")}
    assert "billing_mode" not in cols, "the fixture failed to produce a pre-v3 database"
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


PRICED = "grok-4.5"  # (3.0, 15.0) per Mtok — a model the price table actually knows


def test_only_the_metered_half_of_a_split_history_is_priced(tmp_path, monkeypatch):
    """A gap a mutation found: the spanning test above uses an UNPRICED model.

    `qwen3.8-max` has no PRICING row, so `cost_usd` is None on every segment no
    matter which tokens the route reaches for — and pricing the WHOLE model
    instead of the metered segment survives, silently, with the suite green.
    Same split, on a model the price table knows, so the arithmetic is visible.
    """
    monkeypatch.setenv("QWEN_BASE_URL", METERED_URL)
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    tel.record_run(
        "llm", "completion", "success", input_tokens=100_000_000,
        output_tokens=10_000_000, model=PRICED, billing_mode="subscription",
    )
    tel.record_run(
        "llm", "completion", "success", input_tokens=1_000_000,
        output_tokens=100_000, model=PRICED, billing_mode="metered",
    )
    set_telemetry_handle(tel)

    body = _app_pointed_at(METERED_URL).get("/api/usage").json()
    row = next(m for m in body["models"] if m["model"] == PRICED)
    seg = {b["billing"]: b for b in row["billing_breakdown"]}

    # 1M in @ $3/Mtok + 0.1M out @ $15/Mtok = $3.00 + $1.50 = $4.50
    assert seg["metered"]["cost_usd"] == 4.5, (
        f"the metered half priced at {seg['metered']['cost_usd']} — if that is "
        f"~$453 the route priced the whole model, 100M flat-plan tokens included"
    )
    assert seg["subscription"]["cost_usd"] is None, "the flat-plan half was priced"
    assert row["cost_usd"] == 4.5, "the model total must be the sum of its priced segments"
    assert body["totals"]["cost_usd"] == 4.5
    assert body["totals"]["cost_covers_input_tokens"] == 1_000_000, (
        "coverage claimed the flat-plan tokens were covered by a dollar figure "
        "that never touched them"
    )


# ── the seam that actually produces the stamp ────────────────────────────────
#
# Everything above hands `record_run` a `billing_mode` directly. That tests the
# column and the reader, and NOTHING about the thing that fills it in on the
# daemon. A mutation proved it: replacing the envelope's stamp with `None` left
# all thirteen of those tests green. These drive `LLMCallEnvelope` with a fake
# provider and read the row back out of SQLite.


class _StreamingProvider:
    """Minimal provider with a base_url, which is the only bit that matters."""

    def __init__(self, url: str, fail: bool = False) -> None:
        self._base_url = url
        self.fail = fail

    async def stream_message(self, request):
        if self.fail:
            raise RuntimeError("provider exploded")
        from prometheus.providers.base import ApiTextDeltaEvent

        yield ApiTextDeltaEvent(text="ok")


def _drive_call(tel, url, *, fail=False, model=MODEL):
    """The `call()` path — the one six subsystems use."""
    import asyncio

    from prometheus.learning.llm_envelope import LLMCallEnvelope

    env = LLMCallEnvelope(subsystem="curator", telemetry=tel, on_failure="return_none")
    asyncio.run(env.call(
        provider=_StreamingProvider(url, fail=fail),
        model=model, prompt="p", operation="_call_model",
    ))


def _modes(tel):
    return [
        r[0] for r in tel._conn.execute(
            "SELECT billing_mode FROM subsystem_runs WHERE subsystem = 'curator'"
        )
    ]


def test_the_envelope_stamps_the_row_without_being_told_the_mode(tmp_path):
    """Nobody passes `billing_mode` on the daemon — the envelope resolves it.

    From the LIVE provider, which is the only object that knows which host the
    tokens were actually spent against. Config cannot answer this: it answers
    "where would this model go if it were called now".
    """
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    _drive_call(tel, PLAN_URL)
    assert _modes(tel) == ["subscription"], (
        "the row reached the database unstamped — every reader then falls back "
        "to read-time classification, which is the defect"
    )


def test_a_failed_call_is_stamped_too(tmp_path):
    """A failed row is the one an operator most needs to attribute.

    `call()` routes success and failure through two different abbreviated
    writers, and fixing only one is how the `model` column ended up 100% empty
    for six subsystems (see `_record_failure`'s comment).
    """
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    _drive_call(tel, PLAN_URL, fail=True)
    assert _modes(tel) == ["subscription"]


def test_the_stamp_follows_the_provider_not_the_model_name(tmp_path):
    """Same model, different host, different answer — which is the point.

    If the mode were a property of the model this would be `subscription` both
    times, and the column would be recording the same mistake it replaced.
    """
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    _drive_call(tel, PLAN_URL)
    _drive_call(tel, METERED_URL)
    assert _modes(tel) == ["subscription", "unknown"], (
        "the stamp did not move when the provider did"
    )


def test_a_provider_that_cannot_be_classified_costs_the_label_not_the_row(tmp_path):
    """Never raises: telemetry is not worth losing to label it."""
    class _Hostile:
        @property
        def _base_url(self):
            raise RuntimeError("provider is mid-reload")

        async def stream_message(self, request):
            from prometheus.providers.base import ApiTextDeltaEvent

            yield ApiTextDeltaEvent(text="ok")

    import asyncio

    from prometheus.learning.llm_envelope import LLMCallEnvelope

    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    env = LLMCallEnvelope(subsystem="curator", telemetry=tel, on_failure="return_none")
    asyncio.run(env.call(provider=_Hostile(), model=MODEL, prompt="p",
                         operation="_call_model"))
    assert _modes(tel) == [None], "a hostile provider cost the row, not just the label"
