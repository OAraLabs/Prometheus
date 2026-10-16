"""Per-model output caps: how many tokens one reply may spend (``max_tokens``).

Every route used to send 4096 whatever the model, from hard-coded
``max_tokens=4096`` sites in the daemon and the CLI. A report or a long file in
one write call needs more than that. On 0.9.10 a qwen3.8-max turn spent five
rounds writing a report into ``write_file``: each reply was cut at 4096, and
the call arrived as ``{}``. Models differ by an order of magnitude in what they
allow, and in whether their thinking spends the same budget, so the cap is
resolved per model, for each request:

1. ``context.model_overrides.<model>.max_output_tokens``: an operator said so
   for this model (matched by id or GGUF file name, context/model_keys.py).
2. The route's own ``max_tokens`` (``slash_commands.<name>``, a
   ``router.fallback`` entry, any OpenAI-compatible provider entry): an
   operator said so for this route. It reached the provider and was then
   ignored, because every request already carried 4096.
3. :data:`MODEL_OUTPUT_LIMITS`: the cap chosen for a model we know.
4. :data:`LOCAL_OUTPUT_CAP` for a llama.cpp backend.
5. The default: the loop's own ``max_tokens``.

Whatever wins is clamped to the model's documented maximum when one is known,
so no setting can ask a vendor for more than it serves. Last, on a window the
server MEASURED (a local box reports n_ctx), the cap never takes more than
:data:`LOCAL_ROOM_FRACTION` of it, so a small local window keeps most of its
room for the conversation. A 16384 window allows 4096; an 8192 one, 2048.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from prometheus.context.model_keys import override_key

log = logging.getLogger(__name__)

#: The cap for a llama.cpp backend, whatever GGUF it serves: 4096, what every
#: local request sent before per-model caps. 0.9.12 sent 8192, a quarter of the
#: 4090's 32768 window, and the compactor's projected-overflow rule (#773)
#: counts that as room the next round needs: with ~18.6k tokens of system
#: prompt and tool schemas, the line fell from 26,214 to ~22.3k and a FIRST
#: turn was summarised after two or three tool results (2026-10-16). A reply
#: cut at 4096 is told to send its work in smaller pieces. Ollama is not
#: included: its probe reports the model's TRAINED context length, not the
#: runtime ``num_ctx``, so the room a reply has there is not known and it keeps
#: the default.
LOCAL_OUTPUT_CAP = 4096

#: The most of a MEASURED window one reply may ask for. The compactor's
#: projected-overflow rule (#773) counts the request's ``max_tokens`` as room
#: the next round needs, so a cap that took half a small window would compact
#: the conversation down to almost nothing to make room for it.
LOCAL_ROOM_FRACTION = 0.25

#: Never clamp a measured window's cap below this: a reply needs room to say
#: anything at all, and a window this small is already a misconfiguration.
MIN_OUTPUT_CAP = 1024


@dataclass(frozen=True)
class ModelOutputLimit:
    """The cap we send for a model, and the most its vendor documents.

    ``maximum`` is None where the vendor publishes no figure; the cap is then
    kept at or below what the vendor sends when the field is omitted.
    """

    cap: int
    maximum: int | None


# Caps by EXACT model name, as the request names the model. Two sizes: 16384
# where the reply has the cap to itself, 32768 where the model thinks by
# default and its thinking spends the same budget (a 4096 cap there is mostly
# thinking). Every cap is at or below every documented maximum in the table,
# so a fallback served by another listed model never asks for more than it
# allows. Maxima checked against the vendors' pages on 2026-10-06. MiMo is not
# listed: its API wants max_completion_tokens (#688), so it keeps the default.
MODEL_OUTPUT_LIMITS: dict[str, ModelOutputLimit] = {
    # Alibaba Model Studio: 131,072 output. `max_tokens` caps the answer only;
    # thinking has its own budget (and is counted in usage on top of the cap).
    "qwen3.8-max": ModelOutputLimit(16384, 131072),
    "qwen3.8-flash": ModelOutputLimit(16384, 131072),
    # xAI publishes no output maximum (500K context; 128,000 when omitted).
    # Reasoning is outside the cap.
    "grok-4.5": ModelOutputLimit(16384, None),
    # Anthropic: Haiku 4.5 is 64K output and thinks only when asked.
    "claude-haiku-4-5": ModelOutputLimit(16384, 64000),
    "claude-haiku-4-5-20251001": ModelOutputLimit(16384, 64000),
    # Sonnet 5 and Opus 5 are 128K output and run adaptive thinking when the
    # request says nothing about it, which ours does not; that thinking counts
    # against max_tokens.
    "claude-sonnet-5": ModelOutputLimit(32768, 128000),
    "anthropic/claude-sonnet-5": ModelOutputLimit(32768, 128000),
    "claude-opus-5": ModelOutputLimit(32768, 128000),
    # DeepSeek: 384K output; thinking is the default mode and counts.
    "deepseek-flash": ModelOutputLimit(32768, 393216),
    "deepseek-v4-flash": ModelOutputLimit(32768, 393216),
    "deepseek-v4-pro": ModelOutputLimit(32768, 393216),
    # Moonshot publishes no output maximum (32,768 when omitted); reasoning
    # and answer must both fit.
    "kimi-k2.6": ModelOutputLimit(32768, None),
    # Z.ai: 128K output.
    "glm-5.3": ModelOutputLimit(16384, 128000),
    "glm-5.3-flash": ModelOutputLimit(16384, 128000),
    # OpenAI: 128,000 output, reasoning tokens included.
    "gpt-5.6-luna": ModelOutputLimit(32768, 128000),
    "gpt-5.6-terra": ModelOutputLimit(32768, 128000),
    "gpt-6-astra": ModelOutputLimit(32768, 128000),
    # Google: 65,536 output, thought tokens included.
    "gemini-2.5-flash": ModelOutputLimit(32768, 65536),
    "gemini-2.5-pro": ModelOutputLimit(32768, 65536),
}


@dataclass(frozen=True)
class OutputCaps:
    """Resolves the output cap for a request (see the module docstring)."""

    #: ``context.model_overrides.<model>.max_output_tokens``, flattened.
    model_overrides: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> OutputCaps:
        """Read the per-model overrides beside ``effective_limit`` in
        ``context.model_overrides``. A value that is not a positive integer
        is skipped with a warning, never guessed."""
        ctx = (config or {}).get("context") or {}
        overrides: dict[str, int] = {}
        for name, entry in (ctx.get("model_overrides") or {}).items():
            if not isinstance(entry, dict) or "max_output_tokens" not in entry:
                continue
            value = _positive_int(entry["max_output_tokens"])
            if value is None:
                log.warning(
                    "context.model_overrides.%s.max_output_tokens = %r is not a "
                    "positive integer; ignored", name, entry["max_output_tokens"],
                )
                continue
            overrides[str(name)] = value
        return cls(model_overrides=overrides)

    def resolve(
        self,
        model: str,
        *,
        provider: object = None,
        default: int,
        window: int = 0,
        window_measured: bool = False,
    ) -> tuple[int, str]:
        """The cap for a request to ``model`` served by ``provider``, and
        where it came from (``model_override``, ``route``, ``table``,
        ``local`` or ``default``, with ``+clamped`` when the model's maximum
        lowered it and ``+room`` when a measured window did)."""
        known = MODEL_OUTPUT_LIMITS.get(model)
        route = _positive_int(getattr(provider, "route_max_tokens", None))
        key = override_key(model, self.model_overrides)
        if key is not None:
            cap, source = self.model_overrides[key], "model_override"
        elif route is not None:
            cap, source = route, "route"
        elif known is not None:
            cap, source = known.cap, "table"
        elif getattr(provider, "provider_name", "") == "llama_cpp":
            cap, source = LOCAL_OUTPUT_CAP, "local"
        else:
            cap, source = default, "default"
        if known is not None and known.maximum is not None and cap > known.maximum:
            log.warning(
                "output cap %d for %s (%s) is above the model's maximum %d; sending %d",
                cap, model, source, known.maximum, known.maximum,
            )
            cap, source = known.maximum, f"{source}+clamped"
        room = room_for_window(window) if window_measured else None
        if room is not None and cap > room:
            cap, source = room, f"{source}+room"
        return cap, source


#: The loop's own ``max_tokens`` (daemon.py, __main__.py): step 5 above.
DEFAULT_REPLY_TOKENS = 4096


@dataclass(frozen=True)
class _ServedBy:
    """What :meth:`OutputCaps.resolve` reads off a provider, for a caller without one."""

    provider_name: str = ""
    route_max_tokens: int | None = None


def reply_room(
    config: Mapping[str, Any] | None,
    model: str,
    *,
    local: bool,
    window: int = 0,
    window_measured: bool = False,
    provider: object = None,
    default: int = DEFAULT_REPLY_TOKENS,
) -> tuple[int, str]:
    """The room a reply to ``model`` gets: the ``max_tokens`` its requests ask for.

    One rule for every reader (audit 2026-10-16). It used to be three numbers:
    the request asked for the output cap, which is also what the compactor's
    projected rule counts, while ``/context`` and document sizing subtracted
    ``context.reserved_output`` (2,000) and a call with no cap
    ``compaction.reserve_tokens``. ``context.reserved_output`` is no longer
    read. *local* says the model is served by llama.cpp (step 4). A caller
    with the serving *provider* passes it, which also brings a route's own
    ``max_tokens`` (step 2); without it that step is skipped.
    """
    served = provider if provider is not None else _ServedBy(provider_name="llama_cpp" if local else "")
    return OutputCaps.from_config(config).resolve(
        model, provider=served, default=default, window=window, window_measured=window_measured,
    )


def room_for_window(window: int) -> int | None:
    """The most one reply may ask for on a measured ``window``, or None when
    the window is unknown (0)."""
    if window <= 0:
        return None
    return max(MIN_OUTPUT_CAP, int(window * LOCAL_ROOM_FRACTION))


def _positive_int(value: object) -> int | None:
    # Numbers and numeric strings only: anything else (a bool, an object that
    # happens to define __int__) is not a setting.
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
