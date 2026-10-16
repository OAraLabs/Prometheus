"""Context windows, loud and consistent (audit 2026-10-16, fix 5a).

* One matcher for every per-model override (context/model_keys.py): the
  window, the output cap and the compaction threshold pick the same entry,
  by exact id or by GGUF file name.
* A cloud model with a vendor-documented window is budgeted at it; any other
  falls to ``cloud_default_limit`` with a WARNING once.
* The startup report (context/window_report.py) WARNS on a window that is a
  guess and on an override no served model matches.
* The reply room is the request's ``max_tokens`` everywhere
  (providers/output_caps.reply_room); ``context.reserved_output`` is not read.

The far side, through the real daemon: tests/test_local_gate_context_window.py.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from prometheus.context import budget as budget_mod
from prometheus.context.budget import KNOWN_CLOUD_WINDOWS, TokenBudget, resolve_effective_limit
from prometheus.context.compactor import ContextCompactor, compaction_threshold_from_config
from prometheus.context.model_keys import gguf_name, override_key
from prometheus.context.window_report import known_models, report_at_startup, unmatched_overrides
from prometheus.providers.output_caps import DEFAULT_REPLY_TOKENS, LOCAL_OUTPUT_CAP, OutputCaps, reply_room

SERVED = "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"
MOVED = "/srv/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf"
RENAMED = "/models/Qwen3.8-27B-UD-Q4_K_XL-v2.gguf"


# ---------------------------------------------------------------- matcher


@pytest.mark.parametrize("key", [SERVED, MOVED, "Qwen3.8-27B-UD-Q4_K_XL.gguf", "Qwen3.8-27B-UD-Q4_K_XL"])
def test_an_override_matches_its_gguf_by_file_name(key):
    assert override_key(SERVED, [key, "qwen3.8-max"]) == key


def test_the_exact_id_wins_over_a_file_name_match():
    assert override_key(SERVED, [MOVED, SERVED]) == SERVED


def test_a_renamed_gguf_is_another_model():
    assert override_key(RENAMED, [SERVED]) is None


def test_two_keys_naming_the_same_file_apply_neither(caplog):
    with caplog.at_level(logging.WARNING):
        assert override_key("/x/Same.gguf", ["/a/Same.gguf", "/b/Same.gguf"]) is None
    assert any("none is applied" in r.getMessage() for r in caplog.records)


def test_names_that_are_not_gguf_files_match_exactly_only():
    # A dotted cloud id must not be cut at its "suffix", and a slashed one at its last segment.
    assert gguf_name("qwen3.8-max") is None
    assert override_key("qwen3.8-max", ["qwen3"]) is None
    assert override_key("openrouter/vendor/model-a", ["model-a"]) is None


def test_window_cap_and_threshold_pick_the_same_entry():
    entry = {"effective_limit": 20000, "max_output_tokens": 3000, "compaction_threshold_tokens": 15000}
    config = {"context": {"effective_limit": 24000, "model_overrides": {MOVED: entry}}}
    limit, source = resolve_effective_limit(config["context"], model=SERVED, local_model=SERVED, detected_limit=32768)
    assert (limit, source) == (20000, "model_override")
    cap, cap_source = OutputCaps.from_config(config).resolve(SERVED, default=4096)
    assert (cap, cap_source) == (3000, "model_override")
    assert compaction_threshold_from_config(20000, config, SERVED)[0] == 15000
    compactor = ContextCompactor(
        provider=SimpleNamespace(), effective_limit=24000, model=SERVED, model_overrides={MOVED: 20000},
        threshold_overrides={MOVED: entry},
    )
    assert compactor.limit_for(SERVED) == 20000
    assert compactor.threshold_for(SERVED)[0] == 15000


# ---------------------------------------------------------------- cloud windows


@pytest.fixture
def fresh_warnings(monkeypatch):
    monkeypatch.setattr(budget_mod, "_warned_cloud_default", set())


@pytest.mark.parametrize("model,window", sorted(KNOWN_CLOUD_WINDOWS.items()))
def test_a_known_cloud_model_is_budgeted_at_its_documented_window(model, window, fresh_warnings, caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_effective_limit({}, model=model, local_model=SERVED) == (window, "known")
    assert not caplog.records


def test_a_known_window_never_exceeds_a_lowered_cloud_default():
    ctx = {"cloud_default_limit": 200_000}
    assert resolve_effective_limit(ctx, model="grok-4.5", local_model=SERVED) == (200_000, "known")


def test_an_override_still_beats_a_known_window():
    ctx = {"model_overrides": {"kimi-k2.6": {"effective_limit": 128_000}}}
    assert resolve_effective_limit(ctx, model="kimi-k2.6", local_model=SERVED) == (128_000, "model_override")


def test_the_cloud_default_warns_once_per_model(fresh_warnings, caplog):
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert resolve_effective_limit({}, model="qwen3.8-max", local_model=SERVED)[1] == "cloud_default"
    lines = [r.getMessage() for r in caplog.records if "cloud_default_limit" in r.getMessage()]
    assert len(lines) == 1 and "qwen3.8-max" in lines[0]


# ---------------------------------------------------------------- startup report


class _Registry:
    def __init__(self, *rows):
        self._rows = rows

    def specs(self):
        return tuple(SimpleNamespace(name=n, model=m, models=(), is_primary=p, context_limit=None)
                     for n, m, p, _ in self._rows)

    def status(self, name):
        for n, m, _, n_ctx in self._rows:
            if n == name:
                return SimpleNamespace(model=m, n_ctx=n_ctx, loaded_models=(), available_models=())
        return None

    def detected_windows(self):
        return {n: SimpleNamespace(model=m, n_ctx=c) for n, m, _, c in self._rows if c}


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_an_override_no_served_model_matches_is_reported(caplog):
    config = {"context": {"model_overrides": {SERVED: {}, "/models/Gone.gguf": {}, "claude-sonnet-5": {}}}}
    reg = _Registry(("local", SERVED, True, 32768))
    with caplog.at_level(logging.WARNING):
        report = report_at_startup(config, local_model=SERVED, detected_limit=32768, registry=reg)
    assert report["unmatched_overrides"] == ["/models/Gone.gguf"]
    assert [m for m in _warnings(caplog) if "/models/Gone.gguf" in m]
    assert not [m for m in _warnings(caplog) if "claude-sonnet-5" in m], "a preset model is reachable"


def test_a_window_the_server_did_not_report_warns_with_its_number(caplog):
    with caplog.at_level(logging.WARNING):
        report_at_startup({"context": {"effective_limit": 24000}}, local_model=SERVED, detected_limit=None)
    assert [m for m in _warnings(caplog) if "24000" in m and "context window" in m]


def test_a_detected_window_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING):
        report_at_startup({"context": {"effective_limit": 24000}}, local_model=SERVED, detected_limit=32768)
    assert not _warnings(caplog)


def test_a_cloud_primary_has_no_window_line(caplog):
    with caplog.at_level(logging.WARNING):
        report_at_startup({"context": {}}, local_model="qwen3.8-max", detected_limit=None, local_server=False)
    assert not _warnings(caplog)


def test_an_unsized_backend_warns(caplog):
    reg = _Registry(("local", SERVED, True, 32768), ("mini", "qwen2.5:7b-instruct", False, None))
    with caplog.at_level(logging.WARNING):
        report_at_startup({"context": {"effective_limit": 24000}}, local_model=SERVED, detected_limit=32768,
                          registry=reg)
    assert [m for m in _warnings(caplog) if "'mini'" in m and "24000" in m]


def test_reserved_output_in_config_is_reported_as_unread(caplog):
    with caplog.at_level(logging.WARNING):
        report_at_startup({"context": {"reserved_output": 2000}}, local_model=SERVED, detected_limit=32768)
    assert [m for m in _warnings(caplog) if "reserved_output" in m and "no longer read" in m]


def test_known_models_include_the_presets_and_backends():
    reg = _Registry(("local", SERVED, True, 32768), ("mini", "qwen2.5:7b-instruct", False, 32768))
    names = known_models({"slash_commands": {"kimi": {"model": "kimi-k2.6", "provider": "kimi"}}},
                         local_model=SERVED, registry=reg)
    assert {SERVED, "qwen2.5:7b-instruct", "kimi-k2.6"} <= names
    assert unmatched_overrides([MOVED, "kimi-k2.6", "gpt-4o-old"], names) == ["gpt-4o-old"]


# ---------------------------------------------------------------- reply room


def test_the_reply_room_is_the_output_cap():
    config = {"context": {"reserved_output": 2000, "model_overrides": {SERVED: {"max_output_tokens": 3000}}}}
    assert reply_room(config, SERVED, local=True) == (3000, "model_override")
    assert reply_room({}, SERVED, local=True)[0] == LOCAL_OUTPUT_CAP
    assert reply_room({}, "some-cloud-model", local=False)[0] == DEFAULT_REPLY_TOKENS


def test_a_route_max_tokens_reaches_the_reply_room_with_its_provider():
    route = SimpleNamespace(provider_name="openai_compat", route_max_tokens=12000)
    assert reply_room({}, "some-cloud-model", local=False, provider=route) == (12000, "route")


def test_token_budget_reserves_the_reply_room_not_reserved_output():
    config = {"context": {"effective_limit": 32768, "reserved_output": 2000,
                          "model_overrides": {SERVED: {"max_output_tokens": 3000}}}}
    b = TokenBudget.from_loaded_config(config, model=SERVED, local_model=SERVED, detected_limit=32768)
    assert b.reserved_output == 3000
    assert TokenBudget.from_loaded_config(config, model=SERVED, reply_room=1234).reserved_output == 1234


def test_context_reports_the_reply_room(tmp_path):
    from prometheus.gateway.commands import cmd_context

    config = {"context": {"effective_limit": 32768, "reserved_output": 2000,
                          "model_overrides": {SERVED: {"max_output_tokens": 3000}}}}
    text = cmd_context("You are Prometheus.", SERVED, local_model=SERVED, detected_limit=32768, config=config)
    assert "Reply room:      3,000 tokens" in text and "2,000" not in text
    text = cmd_context("You are Prometheus.", SERVED, local_model=SERVED, detected_limit=32768, config=config,
                       reply_room=2500)
    assert "Reply room:      2,500 tokens" in text
