"""Tests for golden-skill diff scoring (hallucination-penalized)."""

from __future__ import annotations

from prometheus.learning.skill_scoring import (
    match_steps,
    parse_skill_md,
    score_skill,
)

GOLDEN = """# TrackerRMS: Repost a Job

**App:** TrackerRMS
**Task:** Repost a job listing

## Steps

1. **Navigate to Jobs section**
   - Click on "Jobs" in the main navigation menu

2. **Find the job to repost**
   - Search or browse to locate the existing job listing

3. **Open job details**
   - Click on the job title to view full details

4. **Click Repost button**
   - Click the "Repost" or "Post Again" button

5. **Confirm posting**
   - Confirm the job board selection and submit
"""

PROMETHEUS_STYLE = """---
name: trackerrms-repost-job
description: Repost a job listing in TrackerRMS
---

# TrackerRMS - Repost Job

## Steps

1. Navigate to the Jobs section
2. Find the job to repost
3. Open the job details
4. Click the Repost button
5. Confirm the posting
"""


def test_parse_golden_format():
    parsed = parse_skill_md(GOLDEN)
    assert parsed["app_name"] == "TrackerRMS"
    assert parsed["task"] == "Repost a job listing"
    assert parsed["step_count"] == 5
    assert parsed["steps"][0] == "Navigate to Jobs section"
    # Sub-bullets must not leak into steps
    assert all("Click on" not in s for s in parsed["steps"])


def test_parse_prometheus_format():
    parsed = parse_skill_md(PROMETHEUS_STYLE)
    assert parsed["name"] == "trackerrms-repost-job"
    assert parsed["step_count"] == 5


def test_perfect_match_scores_full():
    score = score_skill(PROMETHEUS_STYLE, GOLDEN)
    assert score.matched_steps == 5
    assert score.hallucinated_steps == 0
    assert score.missing_steps == 0
    assert score.accuracy == 1.0


def test_hallucinated_steps_penalize():
    generated = PROMETHEUS_STYLE + "6. Open the settings panel\n7. Delete the account\n"
    score = score_skill(generated, GOLDEN)
    assert score.matched_steps == 5
    assert score.hallucinated_steps == 2
    # 5 / (5 + 2)
    assert abs(score.accuracy - 5 / 7) < 0.001


def test_missing_steps_penalize():
    generated = """---
name: partial
description: partial skill
---

## Steps

1. Navigate to the Jobs section
2. Click the Repost button
"""
    score = score_skill(generated, GOLDEN)
    assert score.matched_steps == 2
    assert score.missing_steps == 3
    assert score.hallucinated_steps == 0
    assert abs(score.accuracy - 2 / 5) < 0.001


def test_right_count_wrong_steps_does_not_score_full():
    """The failure mode count-based scoring hid: 5 invented steps != 100%."""
    generated = """---
name: wrong
description: totally different workflow
---

## Steps

1. Compose a new email message
2. Attach the quarterly report
3. Add recipients from the address book
4. Set the priority flag to high
5. Schedule delivery for Monday morning
"""
    score = score_skill(generated, GOLDEN)
    assert score.accuracy < 0.5
    assert score.hallucinated_steps > 0


def test_empty_generated_skill():
    score = score_skill("# Nothing here\n\nno steps at all\n", GOLDEN)
    assert score.accuracy == 0.0
    assert score.missing_steps == 5
    assert score.error


def test_match_steps_greedy_one_to_one():
    matches, ug, un = match_steps(
        ["Click Save", "Open settings"],
        ["Click the Save button", "Click the Save button again"],
    )
    # Each golden step matches at most one generated step
    assert len(matches) == 1
    assert ug == ["Open settings"]
    assert len(un) == 1
