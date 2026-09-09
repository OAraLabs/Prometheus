"""What the live config puts IN FORCE, and where each value came from.

WHY THIS EXISTS
---------------
``scripts/deploy_guard.sh`` refuses to start a deploy clone that is not
``origin/main``, so merged-but-not-running CODE announces itself. The CONFIG
had no analogue. On 2026-09-07 the live config pinned the cloud tool-call
ceiling well BELOW the shipped default it had just been raised to, and nothing
said so: ``/health`` was green, the tracked tree was clean, the deploy guard
passed, and every cloud turn ran at a fraction of the intended ceiling. It was
found by reading the file.

(The numbers are deliberately not restated here.
``test_no_hardcoded_cap_defaults_outside_shipped_defaults`` forbids it, and it
is right to: a ceiling written down twice is a ceiling that drifts, which is
the failure that produced this module.)

``tests/test_config_drift.py`` catches the PRESENCE half of this class — a live
key absent from the template — and says in its own docstring that it asserts
one direction only. It cannot see a value mismatch, and
:func:`~prometheus.config.shipped_defaults.resolve_max_tool_iterations` names
that same gap: "the config drift guard checks key PRESENCE and cannot see a
value mismatch". This module is the VALUE half, evaluated at runtime against
the config actually loaded.

WARN, DO NOT REFUSE
-------------------
A deliberate override is legitimate and common — the operator's config wins on
purpose. The failure being fixed is SILENCE, not divergence. Nothing here
changes a resolved value; every number in force is still whatever the
``resolve_*`` functions say it is.

WHY THIS IS NOT IN ``shipped_defaults``
---------------------------------------
That module documents itself as having "zero imports, no side effects, safe
from ``cli/init.py`` (the stdlib-only fast setup path)". It is imported by the
writer as well as the readers, and keeping it dependency-free is what makes
that safe. This module imports it, not the other way round.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prometheus.config.shipped_defaults import (
    SHIPPED_ALLOWED_AUDIO_TYPES,
    SHIPPED_ALLOWED_DOCUMENT_TYPES,
    SHIPPED_ALLOWED_IMAGE_TYPES,
    SHIPPED_ALWAYS_LOADED,
    SHIPPED_DENIED_PATHS,
    SHIPPED_MAX_TOOL_ITERATIONS,
    SHIPPED_MAX_TOOL_ITERATIONS_CLOUD,
    SHIPPED_TELEGRAM_ENABLED,
    SHIPPED_WORKSPACE_ROOT,
    resolve_denied_paths,
    resolve_max_tool_iterations,
    resolve_max_tool_iterations_cloud,
    resolve_media_allowlist,
    resolve_telegram_enabled,
    resolve_workspace_root,
)

logger = logging.getLogger(__name__)

#: Where a resolved value came from. Same discipline as ``LIMIT_SOURCES`` in
#: ``context/budget.py``: a value without one of these is not interpretable,
#: because 500 from the shipped default and 500 from a config file that
#: happens to agree are DIFFERENT FACTS — only the first survives a raise.
#:
#: ``shipped_default``  key ABSENT; the shipped constant is in force
#: ``config``           key present and AGREES with the shipped constant
#: ``config_override``  key present and DIFFERS; the operator's value wins
#: ``config_rejected``  key present but UNUSABLE (``0``, ``"abc"``, wrong
#:                      type); the resolver fell back, so the operator's
#:                      value does nothing at all
CONFIG_SOURCES: tuple[str, ...] = (
    "shipped_default",
    "config",
    "config_override",
    "config_rejected",
)

#: The two states worth interrupting a boot log for. ``config`` and
#: ``shipped_default`` are the quiet, correct states.
NOISY_SOURCES: frozenset[str] = frozenset({"config_override", "config_rejected"})


@dataclass(frozen=True)
class ConfigValue:
    """One governed key: what is in force, what ships, and which won."""

    key: str
    resolved: Any
    shipped: Any
    source: str

    @property
    def diverges(self) -> bool:
        return self.source in NOISY_SOURCES


@dataclass(frozen=True)
class _Governed:
    """A key whose shipped default a config can silently override."""

    key: str
    section: tuple[str, ...]
    name: str
    shipped: Any
    resolve: Callable[[dict], Any]
    admits: Callable[[Any], bool]


# ---------------------------------------------------------------------------
# Did the resolver ACCEPT the operator's value?
# ---------------------------------------------------------------------------
#
# "Rejected" cannot be decided from the outcome alone. A honoured value that
# happens to EQUAL the shipped default is indistinguishable, by result, from
# one the resolver threw away — both leave the shipped number in force. So the
# question is answered against each resolver's admission rule instead.
#
# Measured, because the docstrings disagree with the code:
# ``_positive_int`` says a string "100" is 'treated as "not configured" rather
# than obeyed', but ``int("100")`` parses, so it IS obeyed and resolves to 100.
# Classifying it as rejected would tell an operator their working line does
# nothing. There are four admission shapes across the nine governed keys.


def _admits_positive_int(raw: Any) -> bool:
    """``_positive_int``: anything ``int()`` parses to a positive number."""
    try:
        return int(raw) > 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _admits_list(raw: Any) -> bool:
    """``resolve_denied_paths`` / ``resolve_media_allowlist``: a list, or nothing.

    ``[]`` IS admitted — the empty list is a deliberate operator statement both
    resolvers document as load-bearing, and collapsing it with absence is the
    bug they were written to end.
    """
    return isinstance(raw, list)


def _admits_anything(raw: Any) -> bool:
    """``resolve_telegram_enabled`` (``bool(value)``) and the always_loaded rule.

    Any non-None value is used. Absence is handled before this is consulted,
    so these keys can never be reported as rejected.
    """
    del raw
    return True


def _admits_workspace_root(raw: Any) -> bool:
    """``resolve_workspace_root``: a non-blank string, or a list holding one."""
    if isinstance(raw, str):
        return bool(raw.strip())
    if isinstance(raw, list):
        return any(isinstance(v, str) and v.strip() for v in raw)
    return False


def _always_loaded(deferred_cfg: dict) -> list[str]:
    """Mirror of ``DynamicToolLoader.__init__``'s resolution rule.

    Deliberate duplication: the rule lives inline in a constructor
    (``context/dynamic_tools.py``), not behind a ``resolve_*`` this module
    could call. ``test_always_loaded_mirror_agrees_with_the_real_loader``
    pins the two together, so if the loader's rule changes this goes red
    rather than quietly reporting a value nothing enforces.
    """
    configured = deferred_cfg.get("always_loaded")
    return list(SHIPPED_ALWAYS_LOADED if configured is None else configured)


#: Every shipped default a live config can override. Adding a ``SHIPPED_*``
#: constant without adding it here is caught by
#: ``test_every_shipped_constant_is_governed``.
GOVERNED: tuple[_Governed, ...] = (
    _Governed(
        "model.max_tool_iterations", ("model",), "max_tool_iterations",
        SHIPPED_MAX_TOOL_ITERATIONS, resolve_max_tool_iterations,
        _admits_positive_int,
    ),
    _Governed(
        "model.max_tool_iterations_cloud", ("model",), "max_tool_iterations_cloud",
        SHIPPED_MAX_TOOL_ITERATIONS_CLOUD, resolve_max_tool_iterations_cloud,
        _admits_positive_int,
    ),
    _Governed(
        "gateway.telegram_enabled", ("gateway",), "telegram_enabled",
        SHIPPED_TELEGRAM_ENABLED, resolve_telegram_enabled, _admits_anything,
    ),
    _Governed(
        "security.workspace_root", ("security",), "workspace_root",
        SHIPPED_WORKSPACE_ROOT, resolve_workspace_root, _admits_workspace_root,
    ),
    _Governed(
        "security.denied_paths", ("security",), "denied_paths",
        SHIPPED_DENIED_PATHS, resolve_denied_paths, _admits_list,
    ),
    _Governed(
        "gateway.media.allowed_image_types", ("gateway", "media"),
        "allowed_image_types", SHIPPED_ALLOWED_IMAGE_TYPES,
        lambda cfg: resolve_media_allowlist(cfg, "allowed_image_types"),
        _admits_list,
    ),
    _Governed(
        "gateway.media.allowed_audio_types", ("gateway", "media"),
        "allowed_audio_types", SHIPPED_ALLOWED_AUDIO_TYPES,
        lambda cfg: resolve_media_allowlist(cfg, "allowed_audio_types"),
        _admits_list,
    ),
    _Governed(
        "gateway.media.allowed_document_types", ("gateway", "media"),
        "allowed_document_types", SHIPPED_ALLOWED_DOCUMENT_TYPES,
        lambda cfg: resolve_media_allowlist(cfg, "allowed_document_types"),
        _admits_list,
    ),
    _Governed(
        "tools.deferred_loading.always_loaded", ("tools", "deferred_loading"),
        "always_loaded", SHIPPED_ALWAYS_LOADED, _always_loaded, _admits_anything,
    ),
)

_MISSING = object()


def _section(config: dict | None, path: tuple[str, ...]) -> dict:
    """Walk to a config section, treating any non-dict on the way as absent."""
    node: Any = config or {}
    for part in path:
        if not isinstance(node, dict):
            return {}
        node = node.get(part)
        if node is None:
            return {}
    return node if isinstance(node, dict) else {}


def _same(left: Any, right: Any) -> bool:
    """Compare two config values without the list/tuple false positive.

    ``SHIPPED_DENIED_PATHS`` is a tuple and YAML yields a list, so comparing
    them with ``!=`` reports a divergence that does not exist. That exact
    mistake cost a wrong finding during the Phase B sweep, which is why the
    normalisation is a named function with a test rather than an inline
    ``list()`` someone can forget at the next call site.
    """
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return list(left) == list(right)
    if isinstance(left, (list, tuple)) != isinstance(right, (list, tuple)):
        return False
    return bool(left == right)


def describe(config: dict | None) -> tuple[ConfigValue, ...]:
    """Resolve every governed key and say where its value came from.

    Reads only. Nothing here decides a value — each entry calls the SAME
    ``resolve_*`` the daemon calls, so a reported number and an enforced
    number cannot drift. That drift is the failure ``resolve_effective_limit``
    documents as having "shipped twice", and reimplementing the precedence
    here would be the third.
    """
    out: list[ConfigValue] = []
    for entry in GOVERNED:
        section = _section(config, entry.section)
        resolved = entry.resolve(section)
        raw = section.get(entry.name, _MISSING)

        if raw is _MISSING or raw is None:
            source = "shipped_default"
        elif not entry.admits(raw):
            # The resolver refused it and fell back — a 0 ceiling, a blank
            # workspace root, a scalar where a list belongs. The operator's
            # line is inert and nothing has ever said so.
            source = "config_rejected"
        elif not _same(resolved, entry.shipped):
            source = "config_override"
        else:
            # Accepted, and it agrees with the shipped value. Note this is NOT
            # the same fact as absence, which is why they get different tags.
            source = "config"

        out.append(ConfigValue(entry.key, resolved, entry.shipped, source))
    return tuple(out)


def warn_on_divergence(config: dict | None, *, log: Any = None) -> tuple[ConfigValue, ...]:
    """Log one line per diverging key at boot. Returns every value described.

    Warns, never refuses — see the module docstring. The two states that stay
    silent are the two that are correct: a key absent (shipped default in
    force) and a key present that agrees with it.
    """
    emit = log if log is not None else logger
    values = describe(config)
    for value in values:
        if value.source == "config_override":
            emit.warning(
                "config OVERRIDES a shipped default: %s = %r (shipped: %r). "
                "THE CONFIG WINS. If you did not mean to pin this, remove the "
                "key and it will follow the shipped value across upgrades.",
                value.key, value.resolved, value.shipped,
            )
        elif value.source == "config_rejected":
            emit.warning(
                "config value IGNORED: %s is set but unusable, so the shipped "
                "default %r is in force. Your value does nothing — fix it or "
                "remove the key.",
                value.key, value.shipped,
            )
    return values
