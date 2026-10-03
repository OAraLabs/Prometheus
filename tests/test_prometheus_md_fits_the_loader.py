"""The repo's PROMETHEUS.md fits the project-file cap, and its links resolve.

A session bound to this checkout loads PROMETHEUS.md as project instructions,
and the loader keeps only the first ``context.project_file_max_chars`` of it
(default 12,000; see context/prompt_assembler.py). At 39,217 characters the
file was cut inside SUNRISE, so every rule after that point, the loud-failure
law included, never reached a session. The rules now live in the file and the
detail in docs/project/; this keeps the file under 11,000 characters, with
headroom below the cap, and keeps its links pointing at real files.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "PROMETHEUS.md"
CEILING = 11_000


def test_prometheus_md_is_under_the_ceiling():
    text = DOC.read_text(encoding="utf-8")
    assert len(text) < CEILING, (
        f"PROMETHEUS.md is {len(text)} characters; the loader keeps 12,000 and this "
        f"file must stay under {CEILING}. Move detail to docs/project/, not rules.")


def test_prometheus_md_links_resolve():
    text = DOC.read_text(encoding="utf-8")
    targets = [t for t in re.findall(r"\]\(([^)#\s]+)\)", text) if not t.startswith("http")]
    missing = [t for t in targets if not (REPO / t).exists()]
    assert not missing, f"PROMETHEUS.md links to files that do not exist: {missing}"
