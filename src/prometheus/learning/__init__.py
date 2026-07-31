"""Learning loop — periodic nudges, autonomous skill creation, and skill refinement.

The record-a-skill pipeline (browser demonstration traces -> SKILL.md)
lives in the :mod:`prometheus.learning.live_recorder` sub-package.
"""

from prometheus.learning.nudge import PeriodicNudge
from prometheus.learning.skill_creator import SkillCreator
from prometheus.learning.skill_refiner import SkillRefiner

__all__ = [
    "PeriodicNudge",
    "SkillCreator",
    "SkillRefiner",
]
