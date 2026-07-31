"""Live recorder — deterministic "record a skill" pipeline for browser traces.

Transplanted from the shelved SkillForge project's Live DOM engine.
A Chrome extension captures DOM events (clicks, inputs, navigations) with
full element context; this package filters and groups those events,
maps them to structured workflow actions, and synthesizes a SKILL.md —
all without a single model call. An optional model-backed step verifier
and a deterministic quality gate sit between synthesis and persistence.

Entry point: :class:`prometheus.learning.live_recorder.service.LiveRecorderService`.
"""

from prometheus.learning.live_recorder.event_processor import process_events
from prometheus.learning.live_recorder.event_to_actions import events_to_actions
from prometheus.learning.live_recorder.synthesizer import build_skill_content

__all__ = [
    "process_events",
    "events_to_actions",
    "build_skill_content",
]
