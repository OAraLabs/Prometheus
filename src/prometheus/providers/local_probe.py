"""What a local AI app at an address is, and whether Prometheus can think with it (setup's look and probe, Beacon's A2).

One definition of "usable", shared by ``GET /api/setup/detect`` (its ``local`` answer), ``POST /api/setup/probe``,
``POST /api/setup/configure`` for a local provider, and (later) the model routes. The contract is Beacon's
``docs/ONBOARDING-LOCAL-FIRST.md`` §4.1, mirrored in ``docs/contracts/setup-probe.md`` once built.

What a probe may do, so the route cannot be used to read pages it was not built for:

* GET (and Ollama's one ``POST /api/show``) to fixed paths only; no redirect followed; at most ``MAX_ANSWER_BYTES`` read;
* the app is identified by what it answers, never by its port;
* only parsed fields come back, never an answer's body.

RED: this module is a stub. It answers wrongly on purpose so ``tests/test_setup_local_probe.py`` fails until it is built.

Source: novel code for Prometheus, 2026-10-20.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The places looked at on a computer: the port each app listens on by default. A test or a recorder points these at stand-ins.
KNOWN_PLACES: list[dict[str, Any]] = [
    {"app": "llama_cpp", "port": 8080},
    {"app": "ollama", "port": 11434},
    {"app": "lm_studio", "port": 1234},
    {"app": "vllm", "port": 8000},
]
#: The host that is "this machine" for a look.
THIS_MACHINE = "localhost"

#: The most of one answer a probe reads.
MAX_ANSWER_BYTES = 256 * 1024

#: Why nothing usable was found, in the order the first that applies is reported.
REASONS = (
    "bad_target",
    "name_not_found",
    "no_answer",
    "refused",
    "needs_key",
    "not_ai",
    "short_context",
    "none_loaded",
    "no_chat",
    "no_models",
)

#: The smallest window Prometheus can hold a conversation in: round 0 of a fresh install + the local reply cap + 4,096 of
#: conversation, rounded up to 8,192 / 16,384 / 32,768. Set only from MIN_WINDOW_MEASURED (Beacon's spec §4.3).
MIN_WINDOW: int = 0
#: How MIN_WINDOW was measured: {"commit", "round0": {"llama_cpp": int, "ollama": int}, "method"}. None until measured.
MIN_WINDOW_MEASURED: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProbeModel:
    id: str
    loaded: bool | None
    chat: bool | None
    window: int | None
    window_source: str | None
    usable: bool
    problem: str | None


@dataclass(frozen=True)
class Found:
    app: str
    provider: str
    base_url: str
    version: str | None
    latency_ms: float
    models: list[ProbeModel] = field(default_factory=list)


@dataclass(frozen=True)
class ProbeAnswer:
    target: str | None
    found: list[Found]
    reason: str | None
    detail: str
    min_window: int

    def as_dict(self) -> dict[str, Any]:
        return {"target": self.target, "found": [], "reason": self.reason, "detail": self.detail, "min_window": self.min_window}


def parse_target(text: str) -> tuple[str | None, str, int | None] | None:
    """(scheme or None, host, port or None) for a typed computer; None when it is not one (``bad_target``)."""
    return None


def look_here(timeout: float = 1.0) -> ProbeAnswer:
    """This machine's known places, in parallel."""
    return ProbeAnswer(target=None, found=[], reason=None, detail="", min_window=MIN_WINDOW)


def probe_target(text: str, timeout: float = 3.0) -> ProbeAnswer:
    """The one computer someone typed: its port, or every known place on it, in parallel."""
    return ProbeAnswer(target=text, found=[], reason=None, detail="", min_window=MIN_WINDOW)


def probe_choice(provider: str, base_url: str, model: str, timeout: float = 3.0) -> tuple[bool, str | None, Found | None]:
    """Configure's re-probe: is ``model`` usable at ``base_url`` as ``provider``? (ok, reason, what was found)."""
    return True, None, None
