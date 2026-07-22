"""Golden-skill diff scoring — hallucination-penalized accuracy.

Scores a generated SKILL.md against a hand-annotated golden SKILL.md.
Both missing AND hallucinated steps reduce the score:

    accuracy = matched / (expected + hallucinated)

Ported from skillforge-engine run_daily_validation.py (score_output). The
original compared step *counts* only; this port adds deterministic
step-text matching (stdlib difflib, greedy best-match with a similarity
floor) so a generated skill with the right count but wrong steps no
longer scores 100%. That was the failure mode behind SkillForge's
curated-demo numbers: the final nightly ground-truth run averaged 56.1%
with hallucinated steps dominant, which count-based scoring alone
understated.

Parses both golden formats in the annotated corpus (``**App:**`` /
``**Task:**`` headers, bold numbered steps) and Prometheus's generated
format (frontmatter + ``## Steps``).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# A generated step must reach this similarity against a golden step to
# count as a match. Deliberately forgiving: step titles are short and
# phrasing varies ("Click Save" vs "Click the Save button").
MATCH_THRESHOLD = 0.45


@dataclass
class SkillScore:
    """Result of scoring one generated skill against its golden skill."""

    accuracy: float = 0.0
    expected_steps: int = 0
    actual_steps: int = 0
    matched_steps: int = 0
    missing_steps: int = 0
    hallucinated_steps: int = 0
    matches: list[dict] = field(default_factory=list)
    unmatched_golden: list[str] = field(default_factory=list)
    unmatched_generated: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_skill_md(content: str) -> dict[str, Any]:
    """Extract app/task metadata and the step list from SKILL.md text.

    Handles the golden-corpus format (``**App:**``, ``1. **Step title**``)
    and the Prometheus generated format (frontmatter ``name:``/
    ``description:``, plain ``1. Step`` lines under ``## Steps``).
    """
    app_match = re.search(r"\*\*App:\*\*\s*(.+)", content)
    task_match = re.search(r"\*\*Task:\*\*\s*(.+)", content)

    name = ""
    in_fm = False
    for raw in content.splitlines():
        line = raw.strip()
        if line == "---":
            in_fm = not in_fm
            continue
        if in_fm and line.startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
            break

    # Bold numbered steps first ("1. **Step title**"), then plain
    # numbered lines. Sub-bullets ("   - detail") are never steps.
    steps = [s.strip() for s in re.findall(r"^\d+\.\s+\*\*(.+?)\*\*", content, re.MULTILINE)]
    if not steps:
        steps = [
            s.strip()
            for s in re.findall(r"^\d+\.\s+(.+?)$", content, re.MULTILINE)
            if not s.strip().startswith("**")
        ]

    return {
        "name": name,
        "app_name": app_match.group(1).strip() if app_match else "Unknown",
        "task": task_match.group(1).strip() if task_match else "Unknown",
        "steps": steps,
        "step_count": len(steps),
    }


def _normalize_step(text: str) -> str:
    text = re.sub(r"[`*_']", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_step(a), _normalize_step(b)).ratio()


def match_steps(
    golden_steps: list[str],
    generated_steps: list[str],
    threshold: float = MATCH_THRESHOLD,
) -> tuple[list[dict], list[str], list[str]]:
    """Greedy best-first matching of generated steps onto golden steps.

    Each golden step matches at most one generated step and vice versa.
    Returns (matches, unmatched_golden, unmatched_generated).
    """
    pairs = [
        (_similarity(g, o), gi, oi)
        for gi, g in enumerate(golden_steps)
        for oi, o in enumerate(generated_steps)
    ]
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

    matched_golden: set[int] = set()
    matched_generated: set[int] = set()
    matches: list[dict] = []

    for score, gi, oi in pairs:
        if score < threshold:
            break
        if gi in matched_golden or oi in matched_generated:
            continue
        matched_golden.add(gi)
        matched_generated.add(oi)
        matches.append({
            "golden": golden_steps[gi],
            "generated": generated_steps[oi],
            "similarity": round(score, 3),
        })

    unmatched_golden = [s for i, s in enumerate(golden_steps) if i not in matched_golden]
    unmatched_generated = [s for i, s in enumerate(generated_steps) if i not in matched_generated]
    return matches, unmatched_golden, unmatched_generated


def score_skill(generated_content: str, golden_content: str) -> SkillScore:
    """Score generated SKILL.md text against golden SKILL.md text."""
    gen = parse_skill_md(generated_content)
    gold = parse_skill_md(golden_content)

    if not gold["steps"]:
        return SkillScore(error="golden skill has no parseable steps")
    if not gen["steps"]:
        return SkillScore(
            expected_steps=gold["step_count"],
            missing_steps=gold["step_count"],
            error="generated skill has no parseable steps",
        )

    matches, unmatched_golden, unmatched_generated = match_steps(gold["steps"], gen["steps"])

    expected = gold["step_count"]
    matched = len(matches)
    hallucinated = len(unmatched_generated)
    # Hallucination-penalized: extra steps grow the denominator, so
    # inventing steps can never buy accuracy.
    accuracy = matched / (expected + hallucinated) if (expected + hallucinated) else 0.0

    return SkillScore(
        accuracy=round(accuracy, 4),
        expected_steps=expected,
        actual_steps=gen["step_count"],
        matched_steps=matched,
        missing_steps=len(unmatched_golden),
        hallucinated_steps=hallucinated,
        matches=matches,
        unmatched_golden=unmatched_golden,
        unmatched_generated=unmatched_generated,
    )


def score_skill_files(generated_path: Path, golden_path: Path) -> SkillScore:
    """Score a generated SKILL.md file against a golden SKILL.md file."""
    if not golden_path.exists():
        return SkillScore(error=f"golden skill not found: {golden_path}")
    if not generated_path.exists():
        return SkillScore(error=f"generated skill not found: {generated_path}")
    return score_skill(
        generated_path.read_text(encoding="utf-8"),
        golden_path.read_text(encoding="utf-8"),
    )
