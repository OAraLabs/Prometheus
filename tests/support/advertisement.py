"""Assert a tool is ADVERTISED to the model, not merely registered.

THE DISTINCTION THIS EXISTS FOR
-------------------------------
``create_tool_registry()`` returns every tool the process knows about. The
model is handed ``DynamicToolLoader.schemas_for_run()`` — a different and much
smaller set. With deferred loading active (``enabled: auto``, which resolves to
ON for every local provider) the live box advertised **8 of 52** tools.

So ``assert "x" in {t.name for t in registry.list_tools()}`` — the shape every
registration test in this suite used — answers a question no user has: *is the
object constructed?* The question that matters is *can the model call it?*

This was not theoretical. ``vault_search`` and ``vault_read`` shipped, were
verified registered by a test asserting exactly that, were confirmed in the
daemon's own startup log, and were **invisible to the model**. Asked to use the
brain vault, Prometheus correctly answered that it had no such capability. The
registration test passed the entire time. Cf. Standing-Principles §2b — a check
that answers cleanly, about a different subject — written into the vault hours
before being walked into.

WHICH CONFIG THIS READS
-----------------------
The **shipped template** (``config/prometheus.yaml.default``), not the operator's
live config. Three reasons: the template is what every fresh install gets, it is
the only one present in CI (the live config is gitignored), and a guard that
reads a machine-local file proves nothing about what ships. Divergence between
the two is a separate concern — ``test_config_drift.py`` owns key-level drift,
and :func:`live_always_loaded` below covers the value-level case when a live
config exists.

DEFERRED IS NOT THE SAME AS UNREACHABLE
---------------------------------------
A deferred tool is still callable IF the model finds it via ``tool_search``.
But ``tool_search`` returns only the **top 5** of 51 scored tools, so
"discoverable" is a real bar a tool can fail. :func:`assert_tool_discoverable`
tests it against the same scorer the live tool uses, which is what turns
"documented discovery path" from a comment into a check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parents[2]
TEMPLATE_CONFIG = _REPO / "config" / "prometheus.yaml.default"
LIVE_CONFIG = _REPO / "config" / "prometheus.yaml"


def _deferred_config(path: Path) -> dict[str, Any]:
    """The ``tools.deferred_loading`` sub-dict — the exact shape the daemon
    passes to ``DynamicToolLoader`` at ``daemon.py:355``.

    Passing one level higher (``tools``) silently yields an empty
    ``always_loaded`` and therefore zero advertised tools, which looks like a
    catastrophic product bug and is actually a harness bug. It cost a debugging
    round the first time; hence this helper rather than an inline ``.get``.
    """
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return (cfg.get("tools") or {}).get("deferred_loading") or {}


def template_always_loaded() -> set[str]:
    """Tool names the SHIPPED default advertises."""
    return set(_deferred_config(TEMPLATE_CONFIG).get("always_loaded") or [])


def live_always_loaded() -> set[str] | None:
    """The operator's ``always_loaded``, or ``None`` when there is no live config
    (CI, a fresh clone). Callers must handle ``None`` rather than skipping
    silently — an absent live config is a fact, not a reason to assert nothing.
    """
    if not LIVE_CONFIG.exists():
        return None
    return set(_deferred_config(LIVE_CONFIG).get("always_loaded") or [])


def build_registry():
    """The registry the daemon and CLI both build. Not a hand-assembled one."""
    from prometheus.__main__ import create_tool_registry

    return create_tool_registry({})


def advertised_names(config_path: Path = TEMPLATE_CONFIG) -> set[str]:
    """Names the model actually receives under deferred loading.

    Built through ``DynamicToolLoader`` — the real selector — rather than by
    reading ``always_loaded`` and trusting it. If the loader ever stops honouring
    the key, reading the key would still "pass" while the model saw something
    else entirely: the §1b failure (a config key with no reader) wearing a test's
    clothes.
    """
    from prometheus.context.dynamic_tools import DynamicToolLoader

    loader = DynamicToolLoader(build_registry(), _deferred_config(config_path))
    return {s.get("name") for s in loader.schemas_for_run(True)}


def registered_names() -> set[str]:
    return {t.name for t in build_registry().list_tools()}


def assert_tool_advertised(name: str) -> None:
    """The tool is in the set the model receives under the shipped default."""
    advertised = advertised_names()
    assert name in advertised, (
        f"{name!r} is NOT advertised to the model under the shipped default.\n"
        f"  advertised ({len(advertised)}): {sorted(advertised)}\n"
        f"  Being in create_tool_registry() is not enough — the model receives "
        f"schemas_for_run(), and a registered-but-unadvertised tool cannot be "
        f"called no matter how correct it is. Either add {name!r} to "
        f"tools.deferred_loading.always_loaded, or classify it in "
        f"DEFERRED_BY_DESIGN (tests/test_tool_advertisement.py) with a reason "
        f"and a discovery query."
    )


def tool_search_hits(query: str, limit: int = 5) -> list[str]:
    """Tool names ``tool_search`` would return for *query*, best first.

    Uses ``ToolSearchTool._score_tool`` — the same scorer the live tool runs —
    and the same top-N truncation, so this measures what the model would
    actually be shown rather than whether a match exists somewhere.
    """
    from prometheus.tools.tool_search import ToolSearchTool

    tools = build_registry().list_tools()
    scored = sorted(
        ((ToolSearchTool._score_tool(t, query.lower()), t.name) for t in tools),
        key=lambda pair: pair[0],
    )
    return [name for _score, name in scored[:limit]]


def assert_tool_discoverable(name: str, query: str) -> None:
    """A deferred tool is reachable: *query* surfaces it in ``tool_search``.

    The bar is the top 5 of 51, which a tool with a generic description can
    genuinely fail — at which point it is registered, unadvertised, and
    unfindable, i.e. dead code with a docstring.
    """
    hits = tool_search_hits(query)
    assert name in hits, (
        f"{name!r} is deferred AND not discoverable: tool_search({query!r}) "
        f"returns {hits} and does not include it.\n"
        f"  tool_search shows the model only its top 5 of "
        f"{len(registered_names())} tools, so a deferred tool that never places "
        f"is unreachable in practice. Either sharpen its description, choose a "
        f"query a user would actually prompt, or promote it to always_loaded."
    )
