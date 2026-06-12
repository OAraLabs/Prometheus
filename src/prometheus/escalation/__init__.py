"""Teacher escalation — Tier-1 failure detection + cloud-teacher recovery.

When the local model fails an agent turn (deterministic Tier-1 signals only,
no LLM judging), the failed context can be escalated to a configured cloud
teacher model which produces a corrective reply for the user and a SKILL.md
draft so the local model can handle that task class next time. Every
escalation records a golden trace into telemetry.db for the LoRA flywheel.

Provenance: clean-room reimplementation inspired by the teacher-escalation
design in the Odysseus project (MIT). Design knowledge only — tiered failure
detection, skill-persisted-only-if-teacher-passes-detector. No source copied.

``detector`` is deliberately stdlib-only (no prometheus imports) so external
harnesses (BAKEOFF-harness.md) can load it straight from a checkout.
"""

from prometheus.escalation.detector import FailureVerdict, detect_failure

__all__ = ["FailureVerdict", "detect_failure"]
