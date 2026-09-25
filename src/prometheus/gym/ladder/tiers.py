"""Adapter tiers for the ladder: a forced-tier factory and observation-only counters.

A tier sweep runs the SAME model on the SAME server three times — adapter tier
off, light and full — so the difference shows what the adapter layer adds.

Forcing a tier touches no daemon code. ``forced_adapter_factory`` calls the
daemon's own ``create_adapter`` and replaces only its tier decision
(``_get_adapter_tier``) for the duration of that one call, so every other
choice — formatter, strictness, retry budget, adaptive settings — is exactly
what the daemon builds for that tier. Copying ``create_adapter``'s branches
instead would drift the first time the daemon changed one.

``instrument_adapter`` counts what the adapter did without changing any return
value: calls recovered from the model's text, text calls tier off could not
recover, turns carrying ``<tool_call>`` / ``<function=`` markup, and each
retry-or-abort decision. The instance keeps its class's behaviour — the
subclass only observes — and a circuit-breaker tier bump (the loop
``copy.copy``s the adapter) carries the same counter dict, so a run's counts
cover the bumped adapter too.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, cast

from prometheus.adapter import ModelAdapter
from prometheus.adapter.retry import RetryAction

ADAPTER_TIERS = ("off", "light", "full")

# Every counter a row carries (``adapter_counts``); ``tiers_seen`` is a list.
COUNTERS = ("calls_from_text", "text_calls_missed", "xml_markup_turns",
            "adapter_retries", "adapter_aborts", "adapter_escalations")


def forced_adapter_factory(
    tier: str, model_cfg: dict[str, Any], adapter_cfg: dict[str, Any] | None,
) -> Callable[[], ModelAdapter]:
    """A fresh-per-run adapter factory that builds the daemon's adapter for ``tier``."""
    if tier not in ADAPTER_TIERS:
        raise ValueError(f"unknown adapter tier {tier!r}; expected one of {ADAPTER_TIERS}")
    import prometheus.__main__ as daemon

    def factory() -> ModelAdapter:
        real = daemon._get_adapter_tier
        daemon._get_adapter_tier = lambda provider, model: tier  # type: ignore[assignment]
        try:
            return daemon.create_adapter(model_cfg, adapter_cfg)
        finally:
            daemon._get_adapter_tier = real  # type: ignore[assignment]

    return factory


class _CountingAdapter(ModelAdapter):
    """ModelAdapter that counts; every method returns exactly what the parent returns."""

    _ladder_counts: dict[str, Any]

    def _saw(self) -> dict[str, Any]:
        counts = self._ladder_counts
        counts["tiers_seen"].add(self.tier)
        return counts

    def extract_tool_calls(self, text: str, tool_registry: Any = None):  # type: ignore[no-untyped-def]
        out = super().extract_tool_calls(text, tool_registry)
        counts = self._saw()
        if text and ("<tool_call" in text or "<function=" in text):
            counts["xml_markup_turns"] += 1
        if out:
            counts["calls_from_text"] += len(out)
        elif self.tier == self.TIER_OFF and text:
            # What light/full would have recovered from this same text — tier
            # off returns [] without looking. Counting only: never returned.
            try:
                counts["text_calls_missed"] += len(self.enforcer.extract_tool_calls(text, tool_registry))
            except Exception:  # noqa: BLE001 — an observer must never fail the run
                pass
        return out

    def validate_and_repair(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        self._saw()
        return super().validate_and_repair(*args, **kwargs)

    def handle_retry(self, tool_name: str, error: str, tool_registry: Any):  # type: ignore[no-untyped-def]
        action, message = super().handle_retry(tool_name, error, tool_registry)
        key = {RetryAction.RETRY: "adapter_retries", RetryAction.ABORT: "adapter_aborts"}.get(
            action, "adapter_escalations")
        self._saw()[key] += 1
        return action, message


def instrument_adapter(adapter: Any) -> dict[str, Any] | None:
    """Start counting on ``adapter`` in place; the counter dict, or None when
    the adapter is not a plain ModelAdapter (nothing is changed then)."""
    if type(adapter) is not ModelAdapter:
        return None
    adapter.__class__ = _CountingAdapter
    counted = cast(_CountingAdapter, adapter)
    counts: dict[str, Any] = {name: 0 for name in COUNTERS}
    counts["tiers_seen"] = {counted.tier}
    counted._ladder_counts = counts
    return counts


def counts_for_row(counts: dict[str, Any] | None) -> dict[str, Any] | None:
    if counts is None:
        return None
    return {**{k: counts[k] for k in COUNTERS}, "tiers_seen": sorted(counts["tiers_seen"])}


class ProviderRetryCounter(logging.Handler):
    """Counts provider-level HTTP retries (``providers.retry.stream_with_retry``
    logs one WARNING per retry). Infrastructure noise, reported apart from the
    adapter's retries — it is not something the adapter layer adds."""

    LOGGER = "prometheus.providers.retry"

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "retrying in" in record.getMessage():
            self.count += 1

    def __enter__(self) -> "ProviderRetryCounter":
        logger = logging.getLogger(self.LOGGER)
        self._previous_level = logger.level
        if logger.getEffectiveLevel() > logging.WARNING:
            logger.setLevel(logging.WARNING)
        logger.addHandler(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        logger = logging.getLogger(self.LOGGER)
        logger.removeHandler(self)
        logger.setLevel(self._previous_level)
