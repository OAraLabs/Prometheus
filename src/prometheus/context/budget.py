"""TokenBudget — context window tracking for Sprint 4.

Tracks estimated token usage by category (system_prompt, messages, tool_results)
and signals when the context is approaching its limit.

Usage:
    budget = TokenBudget.from_config(model="qwen3.5-32b")
    budget.add("system", system_prompt)
    budget.add("messages", message_text)
    if budget.is_approaching_limit():
        # trigger ContextCompressor
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any

from prometheus.context.token_estimation import estimate_tokens

# Last-resort figure for callers that must have *a* number. It is deliberately
# NOT reachable from resolve_effective_limit(): a caller that cannot resolve a
# real window is told so ("unknown") and decides for itself whether a made-up
# denominator is acceptable. /api/lcm decided it is not — see web/server.py.
LEGACY_FALLBACK_LIMIT = 24000
DEFAULT_RESERVED_OUTPUT = 2000

# Source labels returned alongside a resolved limit. A number without one is
# not interpretable: 32768 from the server and 32768 from a config file that
# happens to agree are different facts, and only the first survives a model
# swap.
LIMIT_SOURCES = ("model_override", "detected", "cloud_default", "config", "unknown")


def _model_overrides(ctx: dict[str, Any]) -> dict[str, int]:
    """Flatten ``context.model_overrides.<model>.effective_limit``."""
    out: dict[str, int] = {}
    for name, entry in (ctx.get("model_overrides") or {}).items():
        if isinstance(entry, dict) and "effective_limit" in entry:
            try:
                out[str(name)] = int(entry["effective_limit"])
            except (TypeError, ValueError):
                continue
    return out


def resolve_effective_limit(
    ctx: dict[str, Any],
    *,
    model: str | None = None,
    local_model: str | None = None,
    detected_limit: int | None = None,
    backend: str | None = None,
    detected: Mapping[str, Any] | None = None,
    backend_hint: int | None = None,
) -> tuple[int | None, str]:
    """Resolve the context window IN FORCE, and say where it came from.

    This is the one implementation of the precedence rules. Everything that
    reports or enforces a context budget resolves through here, so a reported
    number and an enforced number cannot drift — the drift is the bug this
    exists to prevent, and it has shipped twice: once as a config
    ``effective_limit`` that outlived a model swap (daemon.py), and once as a
    literal ``24000`` typed into a web route that had no budget in scope
    (/api/lcm, which then ASSEMBLED against the fabricated number).

    Precedence, most specific first — the same order
    :meth:`ContextCompactor.limit_for` enforces:

      1. an explicit per-model override — an operator said so
      2. the window the local server REPORTED, when *model* is the local one
      3. ``cloud_default_limit`` for a session routed to a cloud provider
      4. the configured global, which is a HINT: it is the right answer only
         while the backend is unreachable, and the wrong answer the moment
         the served model changes underneath it

    Returns ``(None, "unknown")`` when none of the four yields a number.
    That is a real state, not a defect: nothing has been detected and nothing
    configured, and substituting a plausible integer there is what produced a
    confidently-wrong 41% utilisation reading on a window that did not exist.

    MULTI-BACKEND (2026-09). ``local_model`` / ``detected_limit`` describe ONE
    box — the boot primary. With several boxes the window is per backend, so
    the resolver also takes *backend* (the registry name serving this turn,
    ``"local"`` for the primary, ``None`` for a cloud override) and *detected*
    (the registry's ``detected_windows()``: name → a ``DetectedWindow`` with
    ``.model`` and ``.n_ctx``). A detected window applies only while the model
    it was reported for is the model being resolved — a box that restarted
    onto another GGUF must not be budgeted at the old size. *backend_hint* is
    the operator's ``backends.<name>.context_limit``, for a box the probe could
    not size. The source names the backend: ``detected:4090``,
    ``backend_config:mini`` (``detected`` stays the bare word for the primary,
    so existing readers of that string keep working). The single-box kwargs
    remain as the shorthand every existing caller already uses.

    A local backend with nothing detected and no hint falls to the configured
    global — NEVER to ``cloud_default_limit``: that number describes a cloud
    API, and a 27B behind llama-server is not one.
    """
    overrides = _model_overrides(ctx)

    if model and model in overrides:
        return overrides[model], "model_override"

    if backend:
        window = (detected or {}).get(backend)
        if window is not None and getattr(window, "n_ctx", None):
            reported_for = getattr(window, "model", None)
            if not model or not reported_for or reported_for == model:
                source = "detected" if backend == "local" else f"detected:{backend}"
                return int(window.n_ctx), source
        if backend_hint:
            try:
                hint = int(backend_hint)
            except (TypeError, ValueError):
                hint = 0
            if hint > 0:
                return hint, f"backend_config:{backend}"

    if model and local_model and model == local_model and detected_limit:
        return int(detected_limit), "detected"

    if backend:
        # A LOCAL backend with nothing detected: skip the cloud default (below)
        # and take the configured global as the hint it is.
        configured = ctx.get("effective_limit")
        try:
            value = int(configured)
        except (TypeError, ValueError):
            return None, "unknown"
        return (value, "config") if value > 0 else (None, "unknown")

    if model and local_model and model != local_model:
        from prometheus.context.compactor import DEFAULT_CLOUD_LIMIT

        configured_cloud = ctx.get("cloud_default_limit", DEFAULT_CLOUD_LIMIT)
        try:
            return int(configured_cloud), "cloud_default"
        except (TypeError, ValueError):
            return None, "unknown"

    configured = ctx.get("effective_limit")
    try:
        value = int(configured)
    except (TypeError, ValueError):
        return None, "unknown"
    if value <= 0:
        return None, "unknown"
    return value, "config"


@dataclass
class TokenBudget:
    """Tracks estimated token usage across context categories.

    Args:
        effective_limit:  Total token budget for this session (model context window).
        reserved_output:  Tokens reserved for model output (subtracted from headroom).
        model_overrides:  Per-model effective_limit overrides (from prometheus.yaml).
    """

    effective_limit: int
    reserved_output: int = DEFAULT_RESERVED_OUTPUT
    model_overrides: dict[str, int] = field(default_factory=dict)
    # WHERE effective_limit came from — one of LIMIT_SOURCES. Carried on the
    # budget rather than recomputed by callers, so a surface that displays the
    # number can display its provenance without a second resolution that might
    # disagree with the first.
    #
    # "unknown" means nothing resolved and ``effective_limit`` is
    # LEGACY_FALLBACK_LIMIT — a placeholder, NOT a measurement. A caller that
    # can say "unknown" (a /context reply, an API field) must check this and
    # say so rather than printing the placeholder as a figure. Defaults to
    # "unknown" so a hand-constructed TokenBudget never claims a provenance it
    # does not have.
    limit_source: str = "unknown"

    # Internal usage tracking by category
    _usage: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._usage = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model: str | None = None,
        config_path: str | None = None,
        *,
        local_model: str | None = None,
        detected_limit: int | None = None,
    ) -> TokenBudget:
        """Build a TokenBudget from prometheus.yaml context section.

        Args:
            model: Active model name — the one serving this session.
            config_path: Path to prometheus.yaml; omit and it resolves via
                ``config.defaults.resolve_config_path()`` (the same search
                order the CLI and daemon use).
            local_model: The model the local inference server has loaded.
                Needed to tell "this session is on the local model" from "this
                session was routed to a cloud provider", which get different
                budgets. Omit and resolution falls back to exact-match only.
            detected_limit: Context size the local server reported. Used only
                when *model* is the local one.

        This mirrors :meth:`ContextCompactor.limit_for` deliberately — a
        reported budget that disagrees with the enforced one is worse than no
        report, because it is consulted precisely when something looks wrong.
        """
        from prometheus.config.load import load_config_file

        explicit = config_path is not None
        if config_path is None:
            from prometheus.config.defaults import resolve_config_path
            config_path = str(resolve_config_path())

        load = load_config_file(
            config_path,
            subsystem="token_budget",
            substituting=f"an unresolved window (limit_source='unknown', "
                         f"reserved_output={DEFAULT_RESERVED_OUTPUT})",
            explicit=explicit,
        )
        ctx = load.section("context")

        return cls.from_loaded_config(
            {"context": ctx},
            model=model,
            local_model=local_model,
            detected_limit=detected_limit,
        )

    @classmethod
    def from_loaded_config(
        cls,
        config: dict[str, Any] | None,
        *,
        model: str | None = None,
        local_model: str | None = None,
        detected_limit: int | None = None,
    ) -> TokenBudget:
        """Same resolution as :meth:`from_config`, from an ALREADY-LOADED dict.

        The daemon has the parsed prometheus.yaml in hand; re-reading it from
        disk would make the reported budget a second, independently-loaded
        opinion of the enforced one. Both routes run through
        :func:`resolve_effective_limit`, which is the point.

        A TokenBudget must carry a number, so an unresolvable window falls
        back to ``LEGACY_FALLBACK_LIMIT`` here — and ``limit_source`` is then
        ``"unknown"``, which is how a caller tells a placeholder from a
        measurement. Callers that would rather report "unknown" than a
        fabricated denominator check that field (or call
        :func:`resolve_effective_limit` directly).
        """
        ctx = (config or {}).get("context") or {}
        limit, _source = resolve_effective_limit(
            ctx,
            model=model,
            local_model=local_model,
            detected_limit=detected_limit,
        )
        return cls(
            effective_limit=(
                limit if limit is not None else LEGACY_FALLBACK_LIMIT
            ),
            reserved_output=ctx.get("reserved_output", DEFAULT_RESERVED_OUTPUT),
            model_overrides=_model_overrides(ctx),
            limit_source=_source,
        )

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add(self, category: str, text: str) -> None:
        """Add estimated tokens for *text* under *category*.

        Common categories: "system", "messages", "tool_results".
        Categories are cumulative — call add() each time new content arrives.
        """
        self._usage[category] = self._usage.get(category, 0) + estimate_tokens(text)

    def reset(self) -> None:
        """Clear all tracked usage."""
        self._usage.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def used(self) -> int:
        """Total estimated tokens used across all categories."""
        return sum(self._usage.values())

    def usage_by_category(self) -> dict[str, int]:
        """Return a copy of the per-category usage dict."""
        return dict(self._usage)

    def headroom(self) -> int:
        """Tokens available before hitting the limit (after reserving output space)."""
        available = self.effective_limit - self.reserved_output
        return max(0, available - self.used)

    def is_approaching_limit(self, threshold: float = 0.75) -> bool:
        """Return True when usage has consumed *threshold* of the available budget.

        Args:
            threshold: Fraction of (effective_limit - reserved_output) at which
                       to trigger compression. Default 0.75 (75%).
        """
        available = self.effective_limit - self.reserved_output
        if available <= 0:
            return True
        return self.used >= available * threshold
