"""Computer use — bounded desktop actions through the consent gate.

Milestone 1: the candidate table, the extent kind, the gate rule and the
validation path, provable against recorded fixtures with no display, no Cua
install, no TypeSafe credential and no cloud account.

NOT wired into the agent's tool registry. The models here declare schemas the
gate reads (``permissions/computer_schema.py``) and the loop drives them, but
registering them so a chat model can call a click directly is a separate
decision with its own blast radius — see the milestone note in the survey at
``audits/20260919T040248Z-cua-jev-use-computer-use-survey.md``.
"""

from prometheus.computer.types import (
    Candidate,
    Choice,
    ChoiceRequest,
    Element,
    Observation,
)

__all__ = [
    "Candidate",
    "Choice",
    "ChoiceRequest",
    "Element",
    "Observation",
]
