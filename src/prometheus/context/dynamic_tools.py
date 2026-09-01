"""DynamicToolLoader — task-adaptive tool selection for Sprint 4.

Reduces the tool list sent to the model based on the current task:
  - Core tools are always included (bash, read_file, write_file).
  - Task-based: keyword matching adds relevant tools.
  - On-demand: if model requests an unknown tool, load it from the registry.

Usage:
    loader = DynamicToolLoader(registry)
    schemas = loader.active_schemas(task_description="read config and grep for errors")
    # → includes bash, read_file, write_file, grep (keyword matched)
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus.config.shipped_defaults import SHIPPED_ALWAYS_LOADED
from prometheus.tools.base import ToolRegistry

log = logging.getLogger(__name__)

# Tools always included regardless of task
CORE_TOOLS: frozenset[str] = frozenset({"bash", "read_file", "write_file"})

# Keyword → additional tool names to include
_KEYWORD_TOOL_MAP: dict[str, list[str]] = {
    "grep": ["grep"],
    "search": ["grep", "web_search"],
    "find": ["grep", "glob"],
    "glob": ["glob"],
    "pattern": ["glob"],
    "edit": ["edit_file"],
    "modify": ["edit_file"],
    "replace": ["edit_file"],
    "patch": ["edit_file"],
    "list": ["glob"],
    "files": ["glob"],
    # Web tools
    "web": ["web_search", "web_fetch"],
    "url": ["web_fetch"],
    "fetch": ["web_fetch"],
    "browse": ["browser"],
    "navigate": ["browser"],
    "website": ["web_fetch", "browser"],
    # Messaging
    "message": ["message"],
    "send": ["message"],
    "discord": ["message"],
    "slack": ["message"],
    "telegram": ["message"],
    # Audio / TTS
    "speak": ["tts"],
    "voice": ["tts"],
    "audio": ["tts"],
    "speech": ["tts"],
    # Dashboard / visualization
    "dashboard": ["dashboard"],
    "html": ["dashboard"],
    "visuali": ["dashboard"],
    "serve": ["dashboard"],
    # Notebooks
    "notebook": ["notebook_edit"],
    "jupyter": ["notebook_edit"],
    "ipynb": ["notebook_edit"],
    # Sessions
    "session": ["sessions_list", "sessions_send", "sessions_spawn"],
    "agent": ["sessions_list", "sessions_spawn"],
    # User interaction
    "ask": ["ask_user"],
    "clarify": ["ask_user"],
    "question": ["ask_user"],
}


def _normalize_enabled(value: object) -> object:
    """Normalize ``tools.deferred_loading.enabled`` to the tri-state
    ``True | False | "auto"``.

    Accepts YAML booleans, the string "auto" (any case), and — because hand
    edits happen — the strings "true"/"false". Anything unrecognizable
    degrades to "auto" with a warning rather than silently disabling: "auto"
    is the documented default, and a typo shouldn't flip behavior to a
    surprising fixed state.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return "auto"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return "auto"
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
    log.warning(
        "tools.deferred_loading.enabled=%r not recognized — using 'auto'", value
    )
    return "auto"


class DynamicToolLoader:
    """Select an appropriate subset of tools for a given task.

    Args:
        registry: Populated ToolRegistry (all available tools).
        deferred_config: Optional config dict from tools.deferred_loading.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        deferred_config: dict[str, Any] | None = None,
    ) -> None:
        self._registry = registry
        self._deferred = deferred_config or {}
        self._configured = _normalize_enabled(self._deferred.get("enabled", "auto"))
        # always_loaded is parsed unconditionally now: with mode "auto" the
        # decision isn't known until run start (resolve_deferred), so the set
        # must be ready either way. Do NOT mutate this set per-tier — it is
        # the frozen A/B baseline (see PR feat/deferred-tools-tier-aware).
        # FIRSTLIGHT FL-2u: the fallback is the SHIPPED set, not []. An
        # install that upgrades keeps its old config, which predates the
        # key — and [] with deferred loading active (mode "auto" resolves
        # ON for every local provider) means ADVERTISE NOTHING: the model
        # is handed no tools it can call, silently. Absence must be safe.
        # An operator who genuinely wants an empty set writes
        # ``always_loaded: []`` explicitly, which is honoured (the key is
        # present, so the fallback never fires).
        configured = self._deferred.get("always_loaded")
        # Runtime-registered (dynamic) names inside _always_loaded — MCP
        # today. Tracked apart so a registry change can replace them
        # without touching the operator's static list.
        self._dynamic_names: frozenset[str] = frozenset()
        self._always_loaded: frozenset[str] = frozenset(
            SHIPPED_ALWAYS_LOADED if configured is None else configured
        )

    def add_always_loaded(self, names: list[str] | set[str]) -> None:
        """Extend the advertised baseline with runtime-registered tools.

        For DYNAMIC tools (MCP today, packs tomorrow) whose names cannot
        appear in the static ``always_loaded`` YAML list because they only
        exist once a server is connected. FOUNDATION 2.3a: dynamic tools
        must get an explicit advertise-or-defer decision instead of
        silently landing on the invisible side — this is the advertise
        branch, called by mcp.bootstrap when
        ``tools.deferred_loading.mcp_always_deferred`` is false.

        Boot-time, or a registry change BETWEEN runs (an operator adding or
        removing an MCP server over REST, #369 — see
        ``sync_dynamic_always_loaded``). Never mid-run: resolve_deferred /
        schemas_for_run freeze the catalog at run start, so a change here
        lands on the next run of every session. That costs each session one
        prompt-prefix cache miss on its next turn — the price of the tool
        set actually changing, not the #120 mid-run mutation this module's
        docstrings warn about.
        """
        self._dynamic_names = frozenset(self._dynamic_names | set(names))
        self._always_loaded = frozenset(self._always_loaded | set(names))

    def sync_dynamic_always_loaded(self, names: set[str] | list[str]) -> None:
        """Make the dynamic part of the advertised baseline equal ``names``.

        Replaces whatever runtime-registered names were advertised before
        (so a removed server's tools leave the catalog) and leaves the
        operator's static ``always_loaded`` list untouched. Between runs
        only, same contract as ``add_always_loaded``.
        """
        static = self._always_loaded - self._dynamic_names
        self._dynamic_names = frozenset(names)
        self._always_loaded = frozenset(static | self._dynamic_names)

    @property
    def _deferred_enabled(self) -> bool:
        """Legacy view of the tri-state: True only when EXPLICITLY enabled.

        Kept for pre-tri-state call sites that have no adapter in scope and
        therefore cannot resolve "auto". The run path uses
        :meth:`resolve_deferred` instead.
        """
        return self._configured is True

    @property
    def configured_mode(self) -> object:
        """The configured tri-state: True | False | "auto"."""
        return self._configured

    def set_configured(self, value: object) -> None:
        """Update the tri-state at runtime (the Beacon toggle's write path).

        Takes effect at the NEXT run start — resolve_deferred reads this at
        that moment, and the advertised set is then frozen for that run. It
        never changes a run already in flight.
        """
        self._configured = _normalize_enabled(value)
        self._deferred["enabled"] = self._configured

    def resolve_deferred(self, adapter: Any | None = None) -> tuple[bool, str]:
        """Resolve the tri-state to an effective on/off, with a human-readable
        source string ("auto → enabled (local provider)" etc.).

        Called ONCE at run start by run_loop; the result — and therefore the
        advertised tool set — is frozen for the whole run. Changing the
        advertised set mid-run is the #120 prefix-mutation bug class: it
        invalidates the provider's cached prompt prefix.

        Auto resolution keys on ``adapter.tier`` as the local/cloud proxy —
        ``"off"`` means a cloud API provider (the adapter formats nothing and
        the server enforces structure natively); every other tier is a local
        backend. Providers expose neither an effective context-window size nor
        a prefix-caching capability flag, so tier is the cleanest signal that
        actually exists (checked 2026-07-31: no such attributes on any
        ModelProvider).

        Rationale: on a small local window, dropping ~8k tokens of schemas is
        the difference between a run surviving or not, and fewer choices helps
        small-model tool selection. On cloud, frontier models handle the full
        catalog fine and a byte-stable catalog is served from prefix cache
        after round 0 — deferral there saves little and costs discovery
        round-trips. Unknown provenance (no adapter) advertises everything:
        full schemas are the compatible status quo; deferral is the
        optimization, and optimizations shouldn't fire blind.
        """
        if self._configured is True:
            return True, "explicitly enabled"
        if self._configured is False:
            return False, "explicitly disabled"
        tier = getattr(adapter, "tier", None) if adapter is not None else None
        if tier is None:
            return False, "auto → disabled (provider tier unknown)"
        if tier == "off":
            return False, "auto → disabled (cloud provider)"
        return True, f"auto → enabled (local provider, tier {tier})"

    def schemas_for_run(self, deferred: bool) -> list[dict[str, Any]]:
        """Schemas to advertise for a run whose deferred decision is made.

        Stateless: the same inputs always produce the same set, so a run that
        computes this once at start has a byte-stable tool catalog for every
        round (cache-safe by construction — tool discovery happens via the
        ``tool_search`` TOOL RESULT, which lands in append-only history, never
        in this list).
        """
        if deferred:
            return self._deferred_schemas()
        return self._registry.to_api_schema()

    def active_schemas(
        self,
        task_description: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return tool schemas appropriate for *task_description*.

        Always includes CORE_TOOLS.  Additional tools are added by
        keyword-matching *task_description* against _KEYWORD_TOOL_MAP.
        Falls back to all registered tools if no task description is given.

        Args:
            task_description: Free-text description of the current task.

        Returns:
            List of tool schemas in Anthropic API format.
        """
        # Deferred loading: only include always_loaded tools in the prompt
        if self._deferred_enabled:
            return self._deferred_schemas()

        if task_description is None:
            return self._registry.to_api_schema()

        selected: set[str] = set(CORE_TOOLS)
        words = set(task_description.lower().split())

        for keyword, tools in _KEYWORD_TOOL_MAP.items():
            if keyword in words:
                selected.update(tools)

        schemas: list[dict[str, Any]] = []
        for tool in self._registry.list_tools():
            if tool.name in selected:
                schemas.append(tool.to_api_schema())

        # If nothing extra matched (only core), return all to avoid over-pruning
        if not schemas:
            return self._registry.to_api_schema()

        return schemas

    # NOTE: on_demand() was deleted here (feat/deferred-tools-tier-aware). It
    # returned a single tool's schema "when the model requests a tool not in
    # the active set" — but nothing ever called it (verified: zero production
    # call sites), and its existence read as live mid-run schema-injection
    # infrastructure, which would be the #120 prefix-mutation bug class if
    # anyone wired it up. Deferred tools are reachable without it: the model
    # either calls them by name anyway (the registry still executes them —
    # the "lucky guess" path) or discovers them via the tool_search TOOL
    # RESULT, which lands in append-only history and never touches the
    # advertised catalog.

    def _deferred_schemas(self) -> list[dict[str, Any]]:
        """Return only schemas for always_loaded tools (deferred mode)."""
        schemas = []
        for tool in self._registry.list_tools():
            if tool.name in self._always_loaded:
                schemas.append(tool.to_api_schema())
        return schemas

    @property
    def deferred_count(self) -> int:
        """Number of tools deferred (not in prompt) when deferred loading is on."""
        if not self._deferred_enabled:
            return 0
        total = len(self._registry.list_tools())
        loaded = len(self._always_loaded)
        return max(0, total - loaded)

    def all_schemas(self) -> list[dict[str, Any]]:
        """Return schemas for every registered tool."""
        return self._registry.to_api_schema()
