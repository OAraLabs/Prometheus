"""The ``computer`` block on ``GET /api/status``.

WHY THIS EXISTS — the answer had no surface at all
---------------------------------------------------
``check_preconditions`` decides whether a desktop action could possibly land.
Until this module, its result reached **nowhere**: not a log line, not a
metric, not an endpoint. It surfaced only as a blocked ``StepResult.reason``,
and only when a step was actually attempted. So "is computer use able to act
right now?" was a question with no answer short of trying it — which is the
worst possible way to discover that the answer is no, because the failing
case is the one that looks like success.

This is the first place that answer exists.

SHAPE — deliberately the SAME convention as the ``deployment`` block
---------------------------------------------------------------------
Both blocks are computed by a pure function in a non-web module, report each
axis as a STRING rather than a boolean, carry a rollup ``state``, and treat
``unknown`` as a third answer that never collapses into the healthy one. That
is not imitation for its own sake: an operator reading ``/api/status`` should
not have to learn a second vocabulary halfway down the payload, and two
conventions on one endpoint is how one of them ends up misread.

WHAT IS DELIBERATELY NOT HERE
------------------------------
No path, no display number, no runtime directory. ``HalfResult.component``
names the thing instead of locating it — the locations involved are a
uid-bearing runtime path and a display number, and neither belongs on an
endpoint whose entire audience is someone who is already worried. The paths
stay in the logs, where the reader is on the box already.
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus.computer.driver import (
    STATE_UNKNOWN,
    PreconditionResult,
    check_preconditions,
)

logger = logging.getLogger(__name__)

#: One sentence per rollup state, naming the REMEDY or the CONSEQUENCE.
#: Lifted verbatim as a convention from ``_FRESHNESS_DETAIL`` in
#: context/environment.py, whose comment says it best: the rollup is read by
#: people at 2am, and a state name without an action is a puzzle rather than a
#: signal. Two blocks on one endpoint should not answer that differently.
_SUBSTRATE_DETAIL: dict[str, str] = {
    "ready": (
        "Both halves answered: input can dispatch and observation would "
        "return a real tree."
    ),
    "act_only": (
        "Input would dispatch but observation would return an EMPTY TREE — "
        "actions would report success and do nothing. Steps are refused in "
        "this state; check the accessibility bus."
    ),
    "observe_only": (
        "The accessibility tree is readable but no display accepts input — "
        "restart the daemon from inside a graphical session."
    ),
    "unavailable": (
        "Neither half answered; this process has no usable desktop."
    ),
    "unknown": (
        "The substrate could NOT be determined; this is not the same as "
        "usable."
    ),
}

#: Prefix every wrapped desktop tool carries. Used only to COUNT what is
#: registered — never to decide whether a call is a computer action, which is
#: answered by the schema (``x-prometheus-computer-verb``). A name prefix
#: deciding a security question is the defect PR #515's own docstrings record
#: at length; counting for a status line is not that.
_TOOL_PREFIX = "computer_"


def _role_set_drops() -> dict[str, object]:
    """How many click grants were dropped because the role set changed."""
    try:
        from prometheus.permissions import checker as _c

        n = getattr(_c, "ROLE_SET_DROPPED_GRANTS", 0)
        return {
            "count": n,
            "detail": getattr(_c, "ROLE_SET_DROP_REASON", "") or (
                "no grants dropped; the clickable-element role set is "
                "unchanged since they were given."
            ),
        }
    except Exception:  # pragma: no cover
        return {"count": 0, "detail": "unknown — the gate could not be read."}


def computer_status(
    tool_registry: Any = None,
    target_registry: Any = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The ``computer`` block. Never raises; degrades to ``unknown``.

    ``registered`` is the operationally decisive field and it is first for
    that reason: with no wrapped tool in the registry, nothing can click no
    matter what the substrate says. Reporting a healthy substrate without it
    would read as "computer use is working" when the honest answer is
    "computer use is not wired in".
    """
    block: dict[str, Any] = {
        "registered": _registered_count(tool_registry),
        # A silent drop is the same defect wearing the other hat: the grant is
        # gone either way, and nobody re-grants what they were not told they
        # lost. Surfaced here, not only in a log line nobody tails.
        "grants_dropped_role_set_change": _role_set_drops(),
        "targets": _targets(target_registry),
        "substrate": substrate_block(env),
    }
    return block


def substrate_block(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Probe both halves and render them. The part that does I/O.

    Split from :func:`computer_status` so the caller can put THIS on a worker
    thread — it opens a socket with a timeout — while the registry reads stay
    cheap and synchronous.
    """
    try:
        result = check_preconditions(env)
    except Exception as exc:  # noqa: BLE001 — status must still render
        logger.warning("computer substrate could not be probed: %s", exc)
        return _unknown_substrate(f"probe failed: {exc.__class__.__name__}")
    return render(result)


def render(result: PreconditionResult) -> dict[str, Any]:
    """A PreconditionResult as the wire block. Pure; no I/O."""
    return {
        # BOTH HALVES, INDEPENDENTLY. They fail on different axes — input is
        # XTEST over X11, observation is AT-SPI over D-Bus — and the mixed
        # case is the dangerous one. A single boolean cannot express
        # "input would dispatch and observation would return an empty tree",
        # which is precisely the state that reports success having done
        # nothing.
        "act": _half(result.act),
        "observe": _half(result.observe),
        # ready | act_only | observe_only | unavailable | unknown.
        # `unknown` outranks everything — see PreconditionResult.state.
        "state": result.state,
        # The same affordance the `deployment` block gives: a state name plus
        # one sentence saying what to do about it.
        "detail": _SUBSTRATE_DETAIL.get(result.state),
    }


def _half(half: Any) -> dict[str, Any]:
    return {
        "state": getattr(half, "state", STATE_UNKNOWN),
        "component": getattr(half, "component", None),
        # Empty detail renders as None, not "" — an absent explanation and an
        # empty one are the same fact and should have one spelling.
        "detail": getattr(half, "detail", "") or None,
    }


def _unknown_substrate(detail: str) -> dict[str, Any]:
    """The degraded view. UNKNOWN, never a healthy-looking default.

    ``state`` is ``unknown`` rather than ``unavailable``: a probe that broke
    tells us nothing about the substrate, and claiming it is down is an
    assertion we did not establish — the same discipline the halves
    themselves follow.
    """
    unknown = {"state": STATE_UNKNOWN, "component": None, "detail": detail}
    return {"act": dict(unknown), "observe": dict(unknown),
            "state": STATE_UNKNOWN,
            "detail": _SUBSTRATE_DETAIL[STATE_UNKNOWN]}


def _registered_count(tool_registry: Any) -> Any:
    """How many wrapped desktop tools a model could actually call.

    ``None`` when there is no registry to ask — distinguishable from ``0``,
    which is a measured fact ("a registry exists and holds none"). Today the
    honest answer on a running daemon is ``0``: ``register_computer_tools``
    has no call site, by design.
    """
    if tool_registry is None:
        return None
    try:
        names = [t.name for t in tool_registry.list_tools()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("computer tool registry unreadable: %s", exc)
        return None
    return sum(1 for n in names if str(n).startswith(_TOOL_PREFIX))


def _targets(target_registry: Any) -> Any:
    """Declared targets and whether each has a driver bound.

    ``None`` when no registry is wired — again distinct from ``[]``, which
    means "a registry exists and declares no target". Only the NAME and kind
    are exposed: a target's connection settings are opaque by construction
    (``targets.Target.connection``) and must not reach an endpoint.
    """
    if target_registry is None:
        return None
    try:
        out = []
        for name in target_registry.names():
            target = target_registry.get(name)
            bound = True
            try:
                target_registry.resolve(name)
            except Exception:
                bound = False
            out.append({
                "name": name,
                "kind": getattr(target, "kind", None),
                "driver_bound": bound,
            })
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("computer target registry unreadable: %s", exc)
        return None
