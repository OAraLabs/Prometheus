"""Teacher escalation engine — cloud-teacher call + skill writer.

When the Tier-1 detector (``escalation/detector.py``) flags a local agent
turn as failed, this module escalates the full failed context to a configured
cloud teacher model. The teacher produces (a) a corrective reply delivered to
the user and (b) a SKILL.md draft persisted through the existing SkillCreator
path so the local model can handle that task class next time. Every
escalation — fired, refused, or failed — records a golden trace
(``telemetry.signal_events`` row, ``signal_type="teacher_escalation"``) for
the LoRA flywheel.

Trigger conditions (ALL must hold, checked in this order, each logged when
it blocks):

  1. agent mode (tool-capable turn), not plain chat
  2. the provider that served the turn is local/self-hosted
     (``ProviderRegistry.is_cloud(...)`` is False)
  3. ``escalation.teacher_model`` is set in config (unset → feature inert)
  4. the Tier-1 detector returned ``failed=True``
  5. the per-session escalation budget is not exhausted
     (``escalation.max_per_session``, default 3 — prevents loops and cost
     runaway; in-memory per daemon lifetime, resets on restart)

Config (``escalation:`` section of prometheus.yaml; ABSENT by default —
the feature ships inert):

    escalation:
      teacher_model: claude-sonnet-4-6   # REQUIRED to arm the feature
      teacher_provider: anthropic        # optional (default anthropic)
      api_key_env: ANTHROPIC_API_KEY     # optional (provider default applies)
      max_per_session: 3                 # optional

Failure policy: fail LOUD, never silently. A teacher call/parse failure
writes ``silent_failures`` + a golden-trace row and falls through to the
local model's original reply plus a visible system note. A teacher whose
corrective reply itself fails the detector persists NOTHING (the skill gate)
and is recorded the same way.

Provenance: clean-room reimplementation inspired by the teacher-escalation
design in the Odysseus project (MIT). Design knowledge only — tiered failure
detection, skill-persisted-only-if-teacher-passes-detector. No source copied.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from prometheus.escalation.detector import FailureVerdict, detect_failure

if TYPE_CHECKING:
    from pathlib import Path

    from prometheus.learning.skill_creator import SkillCreator
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

DEFAULT_TEACHER_PROVIDER = "anthropic"
DEFAULT_MAX_PER_SESSION = 3
DEFAULT_MAX_TOKENS = 4096

# Payload caps for the golden-trace row: "full exchange" with sane bounds so
# a pathological transcript can't bloat telemetry.db. Documented, not hidden.
_CAP_REQUEST = 4_000
_CAP_REPLY = 8_000
_CAP_RAW = 16_000
_CAP_RESULT_EACH = 2_000
_CAP_RESULTS = 25


# ---------------------------------------------------------------------------
# Teacher prompt + deterministic section parsing
# ---------------------------------------------------------------------------

TEACHER_PROMPT_TEMPLATE = """\
You are the teacher model for a smaller local agent that just failed a turn.
A deterministic failure detector flagged the transcript below.

Produce BOTH sections, each in its own fenced code block with EXACTLY these
labels, CORRECTIVE_REPLY first, SKILL_DRAFT last, nothing after the final
closing fence:

```CORRECTIVE_REPLY
<the reply that should have been given to the user — direct, complete,
actionable; address the user, not the local model; no code fences inside>
```

```SKILL_DRAFT
---
name: <short-kebab-case-name>
description: <one line — what task class this procedure handles>
---

# <Skill Name>

## When to use
<one sentence>

## Steps
1. <step>
2. <step>

## Notes
- <caveats>
```

USER REQUEST:
{user_request}

FAILED TRANSCRIPT (tool calls and results, in order):
{transcript}

FINAL REPLY THE USER SAW:
{final_reply}

DETECTOR REASONS:
{reasons}
"""

# CORRECTIVE_REPLY: non-greedy to the next fence line (the prompt forbids
# fences inside it). SKILL_DRAFT: GREEDY to the LAST fence so markdown code
# blocks inside the skill body survive — the prompt pins SKILL_DRAFT as the
# final section. Both deterministic; label case-insensitive for robustness.
_CORRECTIVE_RE = re.compile(
    r"```CORRECTIVE_REPLY[ \t]*\r?\n(.*?)\r?\n```", re.DOTALL | re.IGNORECASE)
_SKILL_RE = re.compile(
    r"```SKILL_DRAFT[ \t]*\r?\n(.*)\r?\n```", re.DOTALL | re.IGNORECASE)


def parse_teacher_sections(raw: str) -> tuple[str | None, str | None, list[str]]:
    """Extract (corrective_reply, skill_draft, problems) from teacher output.

    A missing OR empty section returns ``None`` for that slot and a
    human-readable problem string — the caller treats any ``None`` as
    teacher failure (spec: missing section = log, count, persist nothing).
    """
    problems: list[str] = []
    corrective: str | None = None
    skill: str | None = None

    m = _CORRECTIVE_RE.search(raw or "")
    if m and m.group(1).strip():
        corrective = m.group(1).strip()
    else:
        problems.append("CORRECTIVE_REPLY section missing or empty")

    m = _SKILL_RE.search(raw or "")
    if m and m.group(1).strip():
        skill = m.group(1).strip()
    else:
        problems.append("SKILL_DRAFT section missing or empty")

    return corrective, skill, problems


def _cap(text: object, limit: int) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= limit else s[:limit] + f"…[capped at {limit} chars]"


def build_trace_from_messages(messages: list) -> list[dict]:
    """Build the detector/teacher trace from one turn's new ConversationMessages.

    Pairs each ToolResultBlock with its ToolUseBlock by ``tool_use_id`` and
    emits dicts in the detector's input shape, in result order (the order
    the model saw outcomes). Duck-typed on block attributes (``type`` /
    ``name`` / ``input`` / ``tool_use_id`` / ``content`` / ``is_error``) so
    gateways can pass engine messages without this module importing their
    concrete types.
    """
    uses: dict[str, tuple[str, object]] = {}
    trace: list[dict] = []
    for msg in messages or ():
        for block in getattr(msg, "content", None) or ():
            btype = getattr(block, "type", "")
            if btype == "tool_use":
                uses[str(getattr(block, "id", ""))] = (
                    str(getattr(block, "name", "?")),
                    getattr(block, "input", {}),
                )
            elif btype == "tool_result":
                name, args = uses.get(
                    str(getattr(block, "tool_use_id", "")), ("?", {}))
                trace.append({
                    "tool_name": name,
                    "arguments": args,
                    "result": str(getattr(block, "content", "")),
                    "is_error": bool(getattr(block, "is_error", False)),
                })
    return trace


def _format_transcript(tool_results: list[dict]) -> str:
    if not tool_results:
        return "(no tool calls this turn)"
    lines = []
    for i, tr in enumerate(tool_results[:_CAP_RESULTS], 1):
        flag = " [ERROR]" if tr.get("is_error") else ""
        lines.append(
            f"{i}. {tr.get('tool_name', '?')}({tr.get('arguments', {})})"
            f"{flag} → {_cap(tr.get('result', ''), 400)}"
        )
    if len(tool_results) > _CAP_RESULTS:
        lines.append(f"… {len(tool_results) - _CAP_RESULTS} more call(s) omitted")
    return "\n".join(lines)


def build_teacher_prompt(
    user_request: str,
    tool_results: list[dict],
    final_reply: str,
    verdict: FailureVerdict,
) -> str:
    return TEACHER_PROMPT_TEMPLATE.format(
        user_request=_cap(user_request, _CAP_REQUEST),
        transcript=_format_transcript(tool_results),
        final_reply=_cap(final_reply, _CAP_REPLY) or "(empty reply)",
        reasons="; ".join(verdict.reasons) or "(none)",
    )


# ---------------------------------------------------------------------------
# Outcome + engine
# ---------------------------------------------------------------------------

@dataclass
class EscalationOutcome:
    """Result of an escalation attempt (``None`` from maybe_escalate means
    trigger conditions 1–4 were simply not met — nothing happened)."""

    status: str  # "escalated" | "teacher_failed" | "refused_budget"
    corrective_reply: str | None = None
    skill_path: str | None = None
    skill_rejected_reasons: list[str] = field(default_factory=list)
    detector_reasons: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    # User-visible system note ("" = nothing to show). Never silent: any
    # status that changes or fails to change the reply explains itself.
    note: str = ""


class TeacherEscalation:
    """Escalate failed local agent turns to a configured cloud teacher.

    Construct via :meth:`from_config`; with ``escalation.teacher_model``
    unset the instance is inert (every ``maybe_escalate`` returns ``None``
    at condition 3). The provider is built lazily on first real escalation
    so an unarmed daemon never needs cloud credentials.
    """

    def __init__(
        self,
        *,
        teacher_model: str | None,
        teacher_provider: str = DEFAULT_TEACHER_PROVIDER,
        api_key_env: str | None = None,
        max_per_session: int = DEFAULT_MAX_PER_SESSION,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        telemetry: Any | None = None,
        signal_bus: Any | None = None,
        provider: "ModelProvider | None" = None,
        skill_creator: "SkillCreator | None" = None,
    ) -> None:
        from prometheus.learning.llm_envelope import LLMCallEnvelope

        self._teacher_model = teacher_model or None
        self._teacher_provider_name = teacher_provider
        self._api_key_env = api_key_env
        self._max_per_session = int(max_per_session)
        self._max_tokens = int(max_tokens)
        self._telemetry = telemetry
        self._signal_bus = signal_bus
        self._provider = provider  # lazy unless injected (tests)
        self._skill_creator = skill_creator  # lazy unless injected (tests)
        self._session_counts: dict[str, int] = {}
        self._stats = {
            "fired": 0,
            "skills_written": 0,
            "teacher_failed": 0,
            "refused_budget": 0,
        }
        # return_none: a failed teacher call is already loud (the envelope
        # writes silent_failures + a failed subsystem_runs row); the caller
        # falls back to the local reply per spec.
        self._envelope = LLMCallEnvelope(
            subsystem="teacher_escalation",
            telemetry=telemetry,
            on_failure="return_none",
        )

    # -- construction ---------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        *,
        telemetry: Any | None = None,
        signal_bus: Any | None = None,
        skill_creator: "SkillCreator | None" = None,
    ) -> "TeacherEscalation":
        """Build from the loaded prometheus.yaml dict. Always returns an
        engine; with no ``escalation.teacher_model`` it is inert."""
        esc = (config or {}).get("escalation") or {}
        return cls(
            teacher_model=esc.get("teacher_model"),
            teacher_provider=esc.get("teacher_provider", DEFAULT_TEACHER_PROVIDER),
            api_key_env=esc.get("api_key_env"),
            max_per_session=esc.get("max_per_session", DEFAULT_MAX_PER_SESSION),
            max_tokens=esc.get("max_tokens", DEFAULT_MAX_TOKENS),
            telemetry=telemetry,
            signal_bus=signal_bus,
            skill_creator=skill_creator,
        )

    @property
    def is_armed(self) -> bool:
        return bool(self._teacher_model)

    @property
    def signal_bus(self) -> Any | None:
        return self._signal_bus

    @signal_bus.setter
    def signal_bus(self, bus: Any) -> None:
        """Late wiring — the daemon constructs SignalBus after this engine
        (same setter pattern as SkillCreator). Propagates to an already-built
        skill creator so its skill_created events reach the bus too."""
        self._signal_bus = bus
        if self._skill_creator is not None:
            self._skill_creator.signal_bus = bus

    def stats(self) -> dict[str, Any]:
        """Snapshot for /escalations: counters + per-session budget state."""
        return {
            **self._stats,
            "max_per_session": self._max_per_session,
            "sessions": dict(self._session_counts),
            "armed": self.is_armed,
            "teacher": (
                f"{self._teacher_provider_name}/{self._teacher_model}"
                if self.is_armed else None
            ),
        }

    # -- internals --------------------------------------------------------

    def _ensure_provider(self) -> "ModelProvider":
        if self._provider is None:
            from prometheus.providers.registry import ProviderRegistry

            cfg: dict[str, Any] = {
                "provider": self._teacher_provider_name,
                "model": self._teacher_model,
            }
            if self._api_key_env:
                cfg["api_key_env"] = self._api_key_env
            self._provider = ProviderRegistry.create(cfg)
        return self._provider

    def _ensure_skill_creator(self) -> "SkillCreator | None":
        if self._skill_creator is None:
            try:
                from prometheus.learning.skill_creator import SkillCreator

                # The persist path makes no model calls; the provider handle
                # is only required by SkillCreator's constructor contract.
                self._skill_creator = SkillCreator(
                    self._ensure_provider(),
                    model=self._teacher_model or "default",
                    telemetry=self._telemetry,
                )
                if self._signal_bus is not None:
                    self._skill_creator.signal_bus = self._signal_bus
            except Exception:
                log.exception("teacher escalation: SkillCreator init failed")
                return None
        return self._skill_creator

    def _record_failure(self, operation: str, message: str, context: dict) -> None:
        """Loud failure: log at error level + silent_failures row."""
        log.error("teacher escalation %s failed: %s", operation, message)
        if self._telemetry is None:
            return
        try:
            self._telemetry.record_silent_failure(
                subsystem="teacher_escalation",
                operation=operation,
                exc=RuntimeError(message),
                context=context,
            )
        except Exception:
            log.warning("teacher escalation: telemetry write failed", exc_info=True)

    async def _record_trace(self, payload: dict[str, Any]) -> None:
        """Golden trace — exactly ONE durable write path.

        Prefer the SignalBus: its ``emit`` persists to
        ``telemetry.signal_events`` BEFORE broadcasting
        (sentinel/signals.py), so subscribers and the durable row stay in
        lockstep. Without a bus (tests, busless construction) write the
        telemetry row directly. Never both — no duplicate rows.
        """
        try:
            if self._signal_bus is not None:
                from prometheus.sentinel.signals import ActivitySignal

                await self._signal_bus.emit(ActivitySignal(
                    kind="teacher_escalation",
                    payload=payload,
                    source="teacher_escalation",
                ))
                return
            if self._telemetry is not None:
                self._telemetry.record_signal_event(
                    signal_type="teacher_escalation",
                    payload=payload,
                    source_subsystem="teacher_escalation",
                )
        except Exception:
            # The trace must never break the reply path — but its failure
            # must still be visible somewhere.
            log.exception("teacher escalation: golden-trace write failed")

    def _trace_payload(
        self,
        *,
        status: str,
        session_id: str,
        user_request: str,
        tool_results: list[dict],
        final_reply: str,
        verdict: FailureVerdict,
        teacher_raw: str | None = None,
        corrective: str | None = None,
        skill_path: str | None = None,
        skill_rejected_reasons: list[str] | None = None,
        failure: str | None = None,
    ) -> dict[str, Any]:
        return {
            "source": "teacher_escalation",
            "status": status,
            "session_id": session_id,
            "teacher_provider": self._teacher_provider_name,
            "teacher_model": self._teacher_model,
            "user_request": _cap(user_request, _CAP_REQUEST),
            "tool_results": [
                {
                    "tool_name": tr.get("tool_name"),
                    "arguments": _cap(tr.get("arguments"), 1_000),
                    "result": _cap(tr.get("result"), _CAP_RESULT_EACH),
                    "is_error": bool(tr.get("is_error")),
                }
                for tr in (tool_results or [])[:_CAP_RESULTS]
            ],
            "local_reply": _cap(final_reply, _CAP_REPLY),
            "detector_reasons": list(verdict.reasons),
            "matched_patterns": list(verdict.matched_patterns),
            "teacher_raw": _cap(teacher_raw, _CAP_RAW) if teacher_raw else None,
            "corrective_reply": _cap(corrective, _CAP_REPLY) if corrective else None,
            "skill_persisted": bool(skill_path),
            "skill_path": skill_path,
            "skill_rejected_reasons": list(skill_rejected_reasons or []),
            "failure": failure,
            "budget": {
                "used": self._session_counts.get(session_id, 0),
                "max": self._max_per_session,
            },
        }

    @staticmethod
    def _failed_note(why: str) -> str:
        return (
            f"(System note: this turn was flagged as failed and teacher "
            f"escalation was attempted but did not succeed ({why}) — "
            f"showing the local model's original reply.)"
        )

    # -- the escalation ---------------------------------------------------

    async def maybe_escalate(
        self,
        *,
        session_id: str,
        user_request: str,
        tool_results: list[dict],
        final_reply: str,
        agent_mode: bool,
        primary_provider: str,
    ) -> EscalationOutcome | None:
        """Run the five trigger conditions in order; escalate when all hold.

        Returns ``None`` when conditions 1–4 don't hold (normal, nothing
        happened); an :class:`EscalationOutcome` for anything that should be
        visible (escalated / teacher_failed / refused_budget).
        """
        # 1. agent mode (tool-capable turn)
        if not agent_mode:
            log.debug("teacher escalation blocked: not an agent-mode turn")
            return None

        # 2. primary must be local/self-hosted
        from prometheus.providers.registry import ProviderRegistry

        if ProviderRegistry.is_cloud(primary_provider):
            log.debug(
                "teacher escalation blocked: serving provider %r is cloud",
                primary_provider,
            )
            return None

        # 3. teacher configured
        if not self._teacher_model:
            log.debug(
                "teacher escalation blocked: escalation.teacher_model not set")
            return None

        # 4. detector verdict
        verdict = detect_failure(tool_results, final_reply)
        if not verdict.failed:
            log.debug("teacher escalation blocked: turn passed the detector")
            return None

        # 5. per-session budget
        used = self._session_counts.get(session_id, 0)
        if used >= self._max_per_session:
            log.warning(
                "teacher escalation refused: budget exhausted for %s (%d/%d)",
                session_id, used, self._max_per_session,
            )
            self._stats["refused_budget"] += 1
            await self._record_trace(self._trace_payload(
                status="refused_budget", session_id=session_id,
                user_request=user_request, tool_results=tool_results,
                final_reply=final_reply, verdict=verdict,
                failure=f"budget exhausted ({used}/{self._max_per_session})",
            ))
            return EscalationOutcome(
                status="refused_budget",
                detector_reasons=verdict.reasons,
                matched_patterns=verdict.matched_patterns,
            )

        # All five hold — this attempt consumes budget whether or not the
        # teacher succeeds (spec: "log, count, do not persist anything").
        self._session_counts[session_id] = used + 1
        self._stats["fired"] += 1
        log.info(
            "teacher escalation fired for %s (%d/%d): %s",
            session_id, used + 1, self._max_per_session,
            "; ".join(verdict.matched_patterns),
        )

        # Teacher call — existing provider layer via LLMCallEnvelope only.
        try:
            provider = self._ensure_provider()
        except Exception as exc:
            self._stats["teacher_failed"] += 1
            self._record_failure(
                "provider_init", f"{type(exc).__name__}: {exc}",
                {"session_id": session_id},
            )
            await self._record_trace(self._trace_payload(
                status="teacher_failed", session_id=session_id,
                user_request=user_request, tool_results=tool_results,
                final_reply=final_reply, verdict=verdict,
                failure=f"provider_init: {exc}",
            ))
            return EscalationOutcome(
                status="teacher_failed",
                detector_reasons=verdict.reasons,
                matched_patterns=verdict.matched_patterns,
                note=self._failed_note("teacher provider unavailable"),
            )

        prompt = build_teacher_prompt(user_request, tool_results, final_reply, verdict)
        raw = await self._envelope.call(
            provider=provider,
            model=self._teacher_model,
            prompt=prompt,
            max_tokens=self._max_tokens,
            operation="teacher_call",
            context={"session_id": session_id},
        )

        if raw is None or not str(raw).strip():
            # Call failed (envelope already recorded it) or returned nothing.
            self._stats["teacher_failed"] += 1
            if raw is not None:
                self._record_failure(
                    "teacher_call", "teacher returned an empty response",
                    {"session_id": session_id},
                )
            await self._record_trace(self._trace_payload(
                status="teacher_failed", session_id=session_id,
                user_request=user_request, tool_results=tool_results,
                final_reply=final_reply, verdict=verdict,
                failure="teacher call failed or returned empty",
            ))
            return EscalationOutcome(
                status="teacher_failed",
                detector_reasons=verdict.reasons,
                matched_patterns=verdict.matched_patterns,
                note=self._failed_note("the teacher call failed"),
            )

        corrective, skill_draft, problems = parse_teacher_sections(str(raw))
        if corrective is None or skill_draft is None:
            self._stats["teacher_failed"] += 1
            self._record_failure(
                "parse_sections", "; ".join(problems),
                {"session_id": session_id, "raw_preview": str(raw)[:300]},
            )
            await self._record_trace(self._trace_payload(
                status="teacher_failed", session_id=session_id,
                user_request=user_request, tool_results=tool_results,
                final_reply=final_reply, verdict=verdict,
                teacher_raw=str(raw),
                failure="missing sections: " + "; ".join(problems),
            ))
            return EscalationOutcome(
                status="teacher_failed",
                detector_reasons=verdict.reasons,
                matched_patterns=verdict.matched_patterns,
                note=self._failed_note("the teacher reply was malformed"),
            )

        # Skill persistence gate: the SAME Tier-1 detector judges the
        # teacher's corrective reply. A teacher that also stalled or denied
        # capability writes nothing — and we don't substitute its reply
        # either (replacing one failure with another helps nobody).
        teacher_verdict = detect_failure([], corrective)
        if teacher_verdict.failed:
            self._stats["teacher_failed"] += 1
            log.warning(
                "teacher escalation: corrective reply failed the detector "
                "(%s) — persisting nothing",
                "; ".join(teacher_verdict.matched_patterns),
            )
            await self._record_trace(self._trace_payload(
                status="teacher_failed", session_id=session_id,
                user_request=user_request, tool_results=tool_results,
                final_reply=final_reply, verdict=verdict,
                teacher_raw=str(raw), corrective=corrective,
                skill_rejected_reasons=teacher_verdict.reasons,
                failure="teacher reply failed the detector",
            ))
            return EscalationOutcome(
                status="teacher_failed",
                detector_reasons=verdict.reasons,
                matched_patterns=verdict.matched_patterns,
                skill_rejected_reasons=teacher_verdict.reasons,
                note=self._failed_note("the teacher's reply failed the same checks"),
            )

        # Persist the skill through the existing SkillCreator path (its
        # validation + slug confinement + no-overwrite + signal apply).
        skill_path: "Path | None" = None
        creator = self._ensure_skill_creator()
        if creator is not None:
            try:
                skill_path = await creator.persist_skill_content(
                    skill_draft, trigger=user_request)
            except Exception as exc:
                self._record_failure(
                    "persist_skill", f"{type(exc).__name__}: {exc}",
                    {"session_id": session_id},
                )
        if skill_path is not None:
            self._stats["skills_written"] += 1

        await self._record_trace(self._trace_payload(
            status="escalated", session_id=session_id,
            user_request=user_request, tool_results=tool_results,
            final_reply=final_reply, verdict=verdict,
            teacher_raw=str(raw), corrective=corrective,
            skill_path=str(skill_path) if skill_path else None,
            skill_rejected_reasons=(
                [] if skill_path else ["skill content failed SkillCreator validation"]
            ) if creator is not None else ["skill creator unavailable"],
        ))

        short = "; ".join(verdict.matched_patterns) or "failed turn"
        return EscalationOutcome(
            status="escalated",
            corrective_reply=corrective,
            skill_path=str(skill_path) if skill_path else None,
            detector_reasons=verdict.reasons,
            matched_patterns=verdict.matched_patterns,
            note=(
                f"(System note: the local model's turn failed [{short}] — "
                f"this reply came from the teacher model "
                f"{self._teacher_provider_name}/{self._teacher_model}."
                + (" A skill was saved for next time.)" if skill_path else ")")
            ),
        )
