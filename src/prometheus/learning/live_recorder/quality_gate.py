"""Quality gate — deterministic structural checks on generated skills.

No model needed. Zero cost. Instant. Catches pipeline failures that would
produce unusable skills before they reach skills/auto/.

Five checks:
1. App consistency: a clear primary app across steps (multi-app aware)
2. Action distribution: no single action type dominates (>70%)
3. No consecutive duplicates: no action repeats 3+ times in a row
4. Min viable steps: between 3 and 50 steps
5. Parameter sanity: all referenced parameters actually exist

Ported from skillforge-engine core/refinement/quality_gate.py. The
original operated on the vision pipeline's ActionSequence; this port
adds :func:`gate_actions` to run directly on the live recorder's action
dicts (application inferred from each action's URL).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger(__name__)


@dataclass
class GateCheck:
    """Result of a single quality check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class QualityGateResult:
    """Full quality gate results."""

    checks: list[GateCheck] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    total: int = 0
    overall: str = "unknown"  # pass, warn, fail

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "checks": [asdict(c) for c in self.checks],
        }


@dataclass
class _GateAction:
    """Minimal action representation the checks operate on."""

    action_type: str
    description: str
    target_element: str
    parameter_name: str = ""
    is_parameterizable: bool = False
    application: str = ""


def _detect_sandwich_pattern(actions: list[_GateAction]) -> bool:
    """Detect the App A -> App B -> App A shape of legitimate multi-app work.

    Covers the common case of a primary app with excursions to a secondary
    (popup, login, different tool) and back, including multiple round trips.
    """
    apps = [a.application for a in actions if a.application]
    if len(apps) < 3:
        return False

    if apps[0] == apps[-1]:
        return True

    for i in range(len(apps) - 2):
        if apps[i] == apps[i + 2] and apps[i] != apps[i + 1]:
            return True

    return False


def run_quality_gate(
    actions: list[_GateAction],
    parameters: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> QualityGateResult:
    """Run all enabled quality checks on an action sequence."""
    checks_config = (config or {}).get("checks", {})
    result = QualityGateResult()

    if not actions:
        result.checks.append(GateCheck("has_actions", False, "No actions extracted"))
        result.failed = 1
        result.total = 1
        result.overall = "fail"
        return result

    # Check 1: App consistency (multi-app aware)
    if checks_config.get("app_consistency", True):
        apps = [a.application for a in actions if a.application]
        if apps:
            app_counts = Counter(apps)
            unique_apps = len(app_counts)
            most_common_app = app_counts.most_common(1)[0]
            primary_ratio = most_common_app[1] / len(apps)

            if unique_apps == 1:
                passed = True
                detail = f"Single app: {most_common_app[0]} (100%)"
            elif unique_apps == 2:
                secondary_app = app_counts.most_common(2)[1]
                secondary_ratio = secondary_app[1] / len(apps)
                has_sandwich = _detect_sandwich_pattern(actions)

                if has_sandwich or primary_ratio >= 0.5:
                    passed = True
                    detail = (
                        f"Multi-app workflow: {most_common_app[0]} ({primary_ratio:.0%}) "
                        f"+ {secondary_app[0]} ({secondary_ratio:.0%})"
                    )
                else:
                    passed = primary_ratio >= 0.4
                    detail = (
                        f"Two apps detected: {most_common_app[0]} ({primary_ratio:.0%}), "
                        f"{secondary_app[0]} ({secondary_ratio:.0%}). "
                        f"{'Accepted as multi-app' if passed else 'No clear primary app'}"
                    )
            elif unique_apps <= 4:
                passed = primary_ratio >= 0.4
                apps_list = ", ".join(f"{app}({count})" for app, count in app_counts.most_common())
                detail = (
                    f"Multi-app ({unique_apps} apps): {apps_list}. "
                    f"{'Accepted' if passed else 'No clear primary'}"
                )
            else:
                passed = False
                detail = f"Too many apps ({unique_apps}): likely hallucination or very noisy recording"

            result.checks.append(GateCheck("app_consistency", passed, detail))
        else:
            result.checks.append(GateCheck("app_consistency", True, "No app data to check"))

    # Check 2: Action distribution (multi-app adjusted)
    if checks_config.get("action_distribution", True):
        action_types = [a.action_type for a in actions]
        type_counts = Counter(action_types)
        most_common = type_counts.most_common(1)[0]
        dominance = most_common[1] / len(action_types)

        # Multi-app workflows naturally have more clicks (app switching)
        unique_apps = len({a.application for a in actions if a.application})
        threshold = 0.7 if unique_apps <= 1 else 0.8

        passed = dominance <= threshold
        result.checks.append(GateCheck(
            "action_distribution", passed,
            f"'{most_common[0]}' is {dominance:.0%} of actions "
            f"(threshold: <{threshold:.0%}{', relaxed for multi-app' if unique_apps > 1 else ''}). "
            f"Distribution: {dict(type_counts)}",
        ))

    # Check 3: No consecutive duplicates
    if checks_config.get("no_consecutive_dupes", True):
        max_consecutive = 1
        current_run = 1
        for i in range(1, len(actions)):
            prev = (actions[i - 1].action_type, actions[i - 1].target_element, actions[i - 1].description)
            curr = (actions[i].action_type, actions[i].target_element, actions[i].description)
            if prev == curr:
                current_run += 1
                max_consecutive = max(max_consecutive, current_run)
            else:
                current_run = 1
        passed = max_consecutive < 3
        result.checks.append(GateCheck(
            "no_consecutive_dupes", passed,
            f"Max consecutive identical actions: {max_consecutive} (threshold: <3)",
        ))

    # Check 4: Min viable steps
    if checks_config.get("min_viable_steps", True):
        count = len(actions)
        passed = 3 <= count <= 50
        result.checks.append(GateCheck(
            "min_viable_steps", passed,
            f"{count} steps (expected: 3-50)",
        ))

    # Check 5: Parameter sanity
    if checks_config.get("parameter_sanity", True):
        defined_params = {p.get("name", "") for p in parameters}
        used_params = {a.parameter_name for a in actions if a.is_parameterizable and a.parameter_name}
        undefined_but_used = used_params - defined_params
        defined_but_unused = defined_params - used_params
        passed = len(undefined_but_used) == 0
        detail = f"Defined: {len(defined_params)}, Used: {len(used_params)}"
        if undefined_but_used:
            detail += f", UNDEFINED but used: {undefined_but_used}"
        if defined_but_unused:
            detail += f", Defined but unused: {defined_but_unused}"
        result.checks.append(GateCheck("parameter_sanity", passed, detail))

    result.total = len(result.checks)
    result.passed = sum(1 for c in result.checks if c.passed)
    result.failed = result.total - result.passed

    if result.failed == 0:
        result.overall = "pass"
    elif result.failed <= 1:
        result.overall = "warn"
    else:
        result.overall = "fail"

    return result


def _app_from_url(url: str) -> str:
    try:
        domain = urlparse(url).netloc
    except ValueError:
        return ""
    return domain.replace("www.", "")


def gate_actions(
    actions: list[dict[str, Any]],
    parameters: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> QualityGateResult:
    """Run the quality gate on live recorder action dicts.

    Adapter from the live pipeline's action shape (``action_type``,
    ``target``, ``url``, ``is_parameter`` ...) to the gate's internal
    representation. ``application`` is derived from each action's URL, so
    the app-consistency check reflects real page domains.
    """
    gate_actions_list = [
        _GateAction(
            action_type=a.get("action_type", ""),
            description=a.get("description", ""),
            target_element=str(a.get("target", "")),
            parameter_name=a.get("parameter_name") or "",
            is_parameterizable=bool(a.get("is_parameter")),
            application=_app_from_url(a.get("url", "")),
        )
        for a in actions
    ]
    return run_quality_gate(gate_actions_list, parameters, config)
