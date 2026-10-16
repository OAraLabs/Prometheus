"""Say at startup where each context window came from, and warn when it is a guess.

Audit 2026-10-16 found three silent cases:

* a window that fell back to ``context.effective_limit`` because the server
  reported none was logged at INFO, when it is logged at all;
* an override that no served model matches (a renamed or moved GGUF, a model
  the operator stopped running) was ignored with no message;
* a cloud model with no override was budgeted at ``cloud_default_limit``
  (1,000,000) with nothing to say so. That one warns in
  :func:`prometheus.context.budget.resolve_effective_limit`, once per model.

:func:`report_at_startup` logs a WARNING for the first two and keeps the
result for ``/api/status`` (:func:`last_report`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

from prometheus.context.budget import FALLBACK_SOURCES, resolve_effective_limit
from prometheus.context.model_keys import override_key

log = logging.getLogger(__name__)

_last: dict[str, Any] = {"unmatched_overrides": []}


def last_report() -> dict[str, Any]:
    """What the last :func:`report_at_startup` found (``unmatched_overrides``)."""
    return dict(_last)


def _strings(*values: Any) -> list[str]:
    out = []
    for v in values:
        if isinstance(v, str) and v:
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out.extend(x for x in v if isinstance(x, str) and x)
    return out


def known_models(config: Mapping[str, Any], *, local_model: str | None, registry: Any = None) -> set[str]:
    """Every model id this daemon serves or can route a session to.

    The served local model, every backend's served and listed models, every
    slash-command preset's models, the router's fallback entries and the
    escalation teacher. An override for any other name can never apply.
    """
    names = set(_strings(local_model))
    if registry is not None:
        for spec in registry.specs():
            st = registry.status(spec.name)
            names.update(_strings(spec.model, list(spec.models),
                                  getattr(st, "model", None),
                                  list(getattr(st, "loaded_models", ()) or ()),
                                  list(getattr(st, "available_models", ()) or ())))
    try:
        from prometheus.router.model_router import OVERRIDE_PRESETS, resolve_model_choices

        for key in set(OVERRIDE_PRESETS) | set((config.get("slash_commands") or {})):
            names.update(_strings(list(resolve_model_choices(key, dict(config)))))
    except Exception:  # noqa: BLE001 — a preset that cannot resolve names nothing
        log.debug("window report: preset models unavailable", exc_info=True)
    for entry in ((config.get("router") or {}).get("fallback") or []):
        if isinstance(entry, Mapping):
            names.update(_strings(entry.get("model")))
    names.update(_strings((config.get("escalation") or {}).get("teacher_model")))
    return names


def unmatched_overrides(overrides: Iterable[str], models: Iterable[str]) -> list[str]:
    """The override keys no model in *models* matches (context/model_keys.py rules)."""
    keys = [str(k) for k in overrides]
    matched = {override_key(m, keys) for m in models}
    return sorted(k for k in keys if k not in matched)


def report_at_startup(
    config: Mapping[str, Any],
    *,
    local_model: str | None,
    detected_limit: int | None,
    registry: Any = None,
    local_server: bool = True,
) -> dict[str, Any]:
    """Log each window's source, WARN on a fallback or a dead override, and keep the result.

    *local_server* is False when the primary is a cloud provider, which has no
    window to detect; its window line is skipped.
    """
    ctx = dict(config.get("context") or {})

    limit, source = resolve_effective_limit(
        ctx, model=local_model, local_model=local_model, detected_limit=detected_limit,
    )
    if not local_server:
        pass
    elif source in FALLBACK_SOURCES:
        log.warning(
            "context window for the local model %r: the server reported no n_ctx, so it is budgeted at %s "
            "tokens from context.effective_limit (source %s). That is a guess: right only while it matches "
            "the server. Pin it with context.model_overrides.<model>.effective_limit.",
            local_model, limit if limit is not None else "an unknown number of", source,
        )
    else:
        log.info("context window for the local model %r: %s tokens (%s)", local_model, limit, source)

    if registry is not None:
        windows = registry.detected_windows()
        for spec in registry.specs():
            if spec.is_primary:
                continue
            st = registry.status(spec.name)
            model = getattr(st, "model", None) or spec.model
            b_limit, b_source = resolve_effective_limit(
                ctx, model=model, backend=spec.name, detected=windows, backend_hint=spec.context_limit,
            )
            trained = (getattr(st, "extra", None) or {}).get("trained_ctx")
            if b_limit and isinstance(trained, int) and 0 < trained < b_limit:
                log.warning(
                    "context window for backend %r (%s): budgeted at %d tokens but the model is trained for "
                    "%d, the most Ollama serves. Set backends.%s.context_limit to %d.",
                    spec.name, model, b_limit, trained, spec.name, trained,
                )
            if b_source in FALLBACK_SOURCES:
                log.warning(
                    "context window for backend %r (%s): nothing measured and no context_limit, so it is "
                    "budgeted at %s tokens from context.effective_limit (source %s).",
                    spec.name, model, b_limit if b_limit is not None else "an unknown number of", b_source,
                )

    if "reserved_output" in ctx:
        log.warning(
            "context.reserved_output (%s) is no longer read: the reply room is each model's output cap, "
            "the max_tokens its requests ask for (context.model_overrides.<model>.max_output_tokens). "
            "Remove the key.", ctx["reserved_output"],
        )

    overrides = list((ctx.get("model_overrides") or {}).keys())
    models = known_models(config, local_model=local_model, registry=registry)
    dead = unmatched_overrides(overrides, models)
    for key in dead:
        log.warning(
            "context.model_overrides.%s matches no model this daemon serves or routes to, so it is never "
            "applied. A renamed GGUF is another model: key the entry by the id the server reports now.", key,
        )
    _last["unmatched_overrides"] = dead
    return last_report()
