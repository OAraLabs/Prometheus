"""The adapter tier, decided in one place and explained wherever it is shown.

WHY THIS MODULE EXISTS
----------------------
The tier (``off`` / ``light`` / ``full``) used to be chosen by matching the
model's FILE NAME against ``config/model_registry.yaml``. A native tool-calling
model the registry did not know — Bonsai 2 27B, Ornith 1.5 9B, both Qwen-based
and writing Qwen's XML tool calls — landed at ``full``: its tools were taken out
of the native field, written into the prompt, and its replies read by a JSON
reader. Measured on the Bonsai tier sweep (611 runs): ``off`` and ``light``
level, ``full`` 25 points lower, 21 of them parse disagreements where the
model's XML calls were lost (WP-X.28).

The server can say what its template does. llama.cpp publishes the chat
template and its own analysis of it (``chat_template_caps``) at ``/props``;
Ollama lists ``tools`` under ``capabilities`` at ``/api/show`` and returns the
Go template. :func:`classify_template` turns either payload into a
:class:`ToolTemplate`; :func:`resolve_tier` decides the tier from every source
there is, in a fixed order, and records which one decided:

    override > provider_class > registry > template > fallback

The registry precedes the template on purpose (ruled 2026-09-25): a listed
model keeps exactly the tier it has today, and a registry entry is how an
operator pins a model. The template decides only for models the registry does
not list. The fallback — no entry and no readable template — stays ``full``,
loudly.

Nothing here does I/O. The probes that feed it live on the providers
(``detect_tool_template``) and in the backend registry's probe; the wiring
that reads the registry file is in ``__main__``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

TIERS = ("off", "light", "full")

#: Which source decided the tier. ``forced`` is not a resolver outcome: it is
#: what ``create_adapter`` records when the ``_get_adapter_tier`` seam was
#: replaced (the ladder's tier sweep forces a tier that way).
TIER_SOURCES = ("override", "provider_class", "registry", "template", "fallback", "forced")

#: How each source reads to a human — the same sentence on the boot line, in
#: /doctor and on the API, so no surface invents its own wording.
TIER_SOURCE_TEXT = {
    "override": "config — adapter.model_tiers names this model",
    "provider_class": "the API enforces structure (cloud provider)",
    "registry": "config/model_registry.yaml lists this model family",
    "template": "the backend's chat template reports native tool calls",
    "fallback": "nothing decided it: no registry entry and the template could not be read",
    "forced": "forced — the tier decision was replaced (a tier sweep)",
}

#: The call formats this module names. Only formats seen in committed evidence
#: (the 4090's Qwen3.8 template and the mini's qwen2.5 Ollama template) are
#: named; anything else is "unknown".
CALL_FORMATS = ("qwen-xml", "qwen-json", "unknown")


@dataclass(frozen=True)
class ToolTemplate:
    """What a backend's chat template says about tool calling.

    ``native`` is True when the template renders a tools list and tool calls,
    False when the server says it does not, and None when nothing readable was
    there — the three are different facts and the resolver treats them
    differently.
    """

    native: bool | None
    call_format: str | None      # one of CALL_FORMATS when native, else None
    evidence: str                # one sentence naming what was read

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TierDecision:
    """The tier, and why."""

    tier: str
    source: str                  # one of TIER_SOURCES
    detail: str                  # the sentence every surface prints
    call_format: str | None      # from the template, when one was read
    decided_for: str             # the model name the decision was made for
    disagreement: str | None = None   # set when the registry and the template disagree

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.tier} — {self.source}: {self.detail}"


# ---------------------------------------------------------------------------
# Reading a template
# ---------------------------------------------------------------------------

def classify_template(
    *,
    chat_template: str | None = None,
    caps: dict[str, Any] | None = None,
    ollama_capabilities: list[Any] | None = None,
    ollama_template: str | None = None,
) -> ToolTemplate:
    """What the served template does with tools, from whatever the server gave.

    Rules, first match wins, each naming its evidence:

    1. llama.cpp ``chat_template_caps`` — the server's own analysis of the
       template it renders: native when it ``supports_tools`` (renders a tools
       list) AND ``supports_tool_calls`` (renders tool calls in history).
    2. llama.cpp ``chat_template`` text: native when a Jinja conditional
       reads ``tools`` and the assistant branch renders ``tool_calls``.
    3. Ollama ``capabilities`` from ``/api/show``: native when it lists
       ``tools``; the Go ``template`` gives the format.
    4. Ollama ``template`` alone: native when it reads ``.Tools`` and
       ``.ToolCalls``.
    5. Nothing readable → ``native=None``.
    """
    if isinstance(caps, dict) and "supports_tools" in caps and "supports_tool_calls" in caps:
        native = bool(caps.get("supports_tools")) and bool(caps.get("supports_tool_calls"))
        return ToolTemplate(
            native=native,
            call_format=_call_format(chat_template) if native else None,
            evidence=(
                "llama.cpp /props chat_template_caps: "
                f"supports_tools={_yn(caps.get('supports_tools'))}, "
                f"supports_tool_calls={_yn(caps.get('supports_tool_calls'))}"
            ),
        )
    if isinstance(chat_template, str) and chat_template.strip():
        native = _jinja_renders_tools(chat_template)
        return ToolTemplate(
            native=native,
            call_format=_call_format(chat_template) if native else None,
            evidence=(
                "llama.cpp /props chat_template "
                + ("renders a tools list and tool calls" if native
                   else "renders neither a tools list nor tool calls")
            ),
        )
    if isinstance(ollama_capabilities, list):
        native = "tools" in ollama_capabilities
        return ToolTemplate(
            native=native,
            call_format=(_call_format(ollama_template) if native else None),
            evidence=(
                "ollama /api/show capabilities "
                + ("list tools" if native else f"do not list tools ({ollama_capabilities})")
            ),
        )
    if isinstance(ollama_template, str) and ollama_template.strip():
        native = ".Tools" in ollama_template and ".ToolCalls" in ollama_template
        return ToolTemplate(
            native=native,
            call_format=_call_format(ollama_template) if native else None,
            evidence=(
                "ollama /api/show template "
                + ("renders .Tools and .ToolCalls" if native else "renders no tool calls")
            ),
        )
    return ToolTemplate(
        native=None, call_format=None,
        evidence="no chat template or capability list was readable",
    )


def _yn(value: Any) -> str:
    return "true" if value else "false"


def _jinja_renders_tools(template: str) -> bool:
    """Does this Jinja template take a ``tools`` list and render ``tool_calls``?"""
    import re

    takes_tools = re.search(r"\{%-?\s*if\s+tools\b", template) is not None
    renders_calls = "tool_calls" in template
    return takes_tools and renders_calls


def _call_format(template: str | None) -> str:
    """The shape a call takes in this template's rendering."""
    if not template:
        return "unknown"
    if "<function=" in template and "<parameter=" in template:
        return "qwen-xml"
    if "<tool_call>" in template:
        return "qwen-json"
    return "unknown"


# ---------------------------------------------------------------------------
# Deciding the tier
# ---------------------------------------------------------------------------

def resolve_tier(
    *,
    provider_name: str,
    model_name: str,
    template: ToolTemplate | None = None,
    registry_entry: dict[str, Any] | None = None,
    registry_note: str | None = None,
    override: str | None = None,
) -> TierDecision:
    """Decide the tier from every source, in the fixed order, and say which one did.

    ``registry_entry`` is the matched ``config/model_registry.yaml`` family (or
    None); ``registry_note`` says why there is none when the file itself was
    the problem (missing, unreadable), so the fallback's sentence can name it.
    ``override`` is a tier from ``adapter.model_tiers`` (``auto`` and None mean
    absent).
    """
    name = model_name or ""
    shown = _display(name)
    fmt = template.call_format if template is not None else None

    if override is not None and override != "auto":
        if override not in TIERS:
            raise ValueError(
                f"adapter.model_tiers value {override!r} for {shown!r} is not one of "
                f"{', '.join(TIERS)} (or auto)"
            )
        return TierDecision(
            tier=override, source="override",
            detail=f"adapter.model_tiers names {shown}: {override}",
            call_format=fmt, decided_for=name,
        )

    from prometheus.providers.registry import ProviderRegistry

    if provider_name == "anthropic" or ProviderRegistry.is_cloud(provider_name):
        return TierDecision(
            tier="off", source="provider_class",
            detail=f"{provider_name} is a cloud provider; the API enforces structure",
            call_format=None, decided_for=name,
        )

    if registry_entry is not None:
        function_calling = (registry_entry.get("capabilities") or {}).get("function_calling") or {}
        native = bool(function_calling.get("supported", False)) and function_calling.get("requires") is None
        family = registry_entry.get("display_name") or "a listed family"
        detail = (
            f"config/model_registry.yaml lists {shown} as {family}, "
            + ("with native tool calling" if native else "without native tool calling")
        )
        disagreement = None
        if template is not None and template.native is not None and template.native != native:
            disagreement = (
                f"the chat template disagrees with the registry for {shown}: "
                f"{template.evidence}; the registry decides"
            )
        return TierDecision(
            tier="light" if native else "full", source="registry", detail=detail,
            call_format=fmt, decided_for=name, disagreement=disagreement,
        )

    if template is not None and template.native is not None:
        detail = template.evidence
        if template.native and template.call_format:
            detail += f"; calls as {template.call_format}"
        return TierDecision(
            tier="light" if template.native else "full", source="template",
            detail=detail, call_format=fmt, decided_for=name,
        )

    why = f"no registry entry matches {shown}"
    if registry_note:
        why += f" ({registry_note})"
    why += "; " + (template.evidence if template is not None else "the chat template was not read")
    return TierDecision(
        tier="full", source="fallback", detail=why, call_format=None, decided_for=name,
    )


def _display(model_name: str) -> str:
    """A model name as a person reads it: a .gguf path shows as its file name."""
    tail = model_name.rsplit("/", 1)[-1] if model_name else ""
    return tail or "(no model name)"
