"""LiveRecorderService — orchestrates the record-a-skill pipeline.

Upload (events + screenshots) -> deterministic processing -> quality gate
-> optional model step-verification -> SKILL.md synthesis -> persistence
through ``SkillCreator.persist_skill_content()``. Every upload is archived
under ``~/.prometheus/data/recordings/<id>/`` (events, screenshots,
extracted actions, gate + verification results) for provenance and future
flywheel use.

Trust policy (two-tier): DOM-recorded traces are deterministic ground
truth, so they may auto-persist once the quality gate passes. The step
verifier is advisory unless it reports ``poor`` quality with critical
issues, which blocks persistence. Vision-derived skills (phase 2) never
take this path — they land as drafts requiring human review.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prometheus.config.paths import get_data_dir
from prometheus.learning.live_recorder.event_processor import process_events
from prometheus.learning.live_recorder.event_to_actions import events_to_actions
from prometheus.learning.live_recorder.quality_gate import gate_actions
from prometheus.learning.live_recorder.step_verifier import StepVerifier
from prometheus.learning.live_recorder.synthesizer import build_skill_content

if TYPE_CHECKING:
    from prometheus.learning.skill_creator import SkillCreator

log = logging.getLogger(__name__)

_RECORDINGS_DIR_NAME = "recordings"


class LiveRecorderService:
    """Turn uploaded browser traces into persisted auto-skills.

    Args:
        skill_creator: The daemon's SkillCreator — THE write path for
            machine-generated skills (validation, no-overwrite,
            ``skill_created`` signal).
        step_verifier: Optional model-backed reviewer. ``None`` disables
            verification entirely (config ``learning.live_recorder.verify_steps``).
        skill_registry: Optional live SkillRegistry; when present,
            ``reload_user_skills()`` is called after a successful persist so
            the running agent picks the skill up without a restart.
        recordings_dir: Override the provenance archive location.
    """

    def __init__(
        self,
        skill_creator: SkillCreator,
        *,
        step_verifier: StepVerifier | None = None,
        skill_registry: Any | None = None,
        recordings_dir: Path | None = None,
    ) -> None:
        self._skill_creator = skill_creator
        self._step_verifier = step_verifier
        self._skill_registry = skill_registry
        self._recordings_dir = recordings_dir or (get_data_dir() / _RECORDINGS_DIR_NAME)

    async def persist_content(self, content: str, *, trigger: str) -> Path | None:
        """Persist finished skill markdown through the standard auto-skill path.

        Wraps ``SkillCreator.persist_skill_content`` and reloads the live
        SkillRegistry on success. Used by the upload pipeline below and by
        the skill-drafts accept flow (vision-derived skills approved in
        Beacon land through the exact same write path).
        """
        skill_path = await self._skill_creator.persist_skill_content(content, trigger=trigger)
        if skill_path is not None:
            self._reload_registry()
        return skill_path

    async def handle_upload(
        self,
        events: list[dict[str, Any]],
        metadata: dict[str, Any],
        screenshots: list[bytes] | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline on one uploaded recording.

        Returns a JSON-serializable result dict. ``status`` is one of:
        ``created`` (skill persisted), ``rejected`` (quality gate or
        verifier refused it), or ``error`` (pipeline failure).
        """
        recording_id = f"rec-{int(time.time())}-{abs(hash(str(metadata))) % 10000:04d}"
        rec_dir = self._recordings_dir / recording_id
        screenshots = screenshots or []

        norm_meta = _normalize_metadata(metadata, event_count=len(events))
        screenshot_paths = self._archive_upload(rec_dir, events, norm_meta, screenshots)

        processed = process_events(events, screenshots=screenshot_paths)
        if not processed.get("success"):
            return {"status": "error", "error": processed.get("error", "event processing failed"),
                    "recording_id": recording_id}

        actions_data = events_to_actions(processed)
        if not actions_data.get("success"):
            return {"status": "error", "error": actions_data.get("error", "action mapping failed"),
                    "recording_id": recording_id}

        actions = actions_data["actions"]
        parameters = actions_data["parameters"]
        self._write_json(rec_dir / "actions.json", {"actions": actions, "parameters": parameters})

        # Deterministic quality gate — authoritative
        gate = gate_actions(actions, parameters)
        self._write_json(rec_dir / "quality_gate.json", gate.to_dict())
        if gate.overall == "fail":
            log.info("Live recorder: quality gate rejected recording %s (%d/%d checks failed)",
                     recording_id, gate.failed, gate.total)
            return {
                "status": "rejected",
                "reason": "quality_gate",
                "quality_gate": gate.to_dict(),
                "recording_id": recording_id,
            }

        # Model step verification — advisory unless clearly poor
        verification = None
        if self._step_verifier is not None:
            result = await self._step_verifier.verify(actions, parameters, norm_meta)
            verification = result.to_dict()
            self._write_json(rec_dir / "verification.json", verification)
            if result.overall_quality == "poor" and result.critical_count > 0:
                log.info("Live recorder: step verifier rejected recording %s (%d critical issues)",
                         recording_id, result.critical_count)
                return {
                    "status": "rejected",
                    "reason": "step_verifier",
                    "quality_gate": gate.to_dict(),
                    "verification": verification,
                    "recording_id": recording_id,
                }

        draft = build_skill_content(actions, parameters, norm_meta)

        trigger = f"browser recording of {norm_meta.get('start_url') or 'unknown site'} ({recording_id})"
        skill_path = await self.persist_content(draft.content, trigger=trigger)
        if skill_path is None:
            return {
                "status": "error",
                "error": "skill persistence rejected the generated content",
                "recording_id": recording_id,
            }

        log.info("Live recorder: created skill %s from recording %s", skill_path.name, recording_id)
        return {
            "status": "created",
            "skill_name": skill_path.stem,
            "skill_path": str(skill_path),
            "title": draft.title,
            "description": draft.description,
            "step_count": draft.step_count,
            "parameter_count": draft.parameter_count,
            "quality_gate": gate.to_dict(),
            "verification": verification,
            "recording_id": recording_id,
        }

    # ------------------------------------------------------------------

    def _archive_upload(
        self,
        rec_dir: Path,
        events: list[dict[str, Any]],
        metadata: dict[str, Any],
        screenshots: list[bytes],
    ) -> dict[int, str]:
        """Write the raw upload to the provenance archive. Best-effort."""
        screenshot_paths: dict[int, str] = {}
        try:
            shots_dir = rec_dir / "screenshots"
            shots_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(rec_dir / "events.json", events)
            self._write_json(rec_dir / "metadata.json", metadata)
            for i, blob in enumerate(screenshots):
                path = shots_dir / f"{i}.jpg"
                path.write_bytes(blob)
                screenshot_paths[i] = str(path)
        except OSError:
            log.warning("Live recorder: failed to archive recording to %s", rec_dir, exc_info=True)
        return screenshot_paths

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except OSError:
            log.warning("Live recorder: failed to write %s", path, exc_info=True)

    def _reload_registry(self) -> None:
        """Ask the live SkillRegistry to re-scan skills/auto/ (pull-based reload)."""
        if self._skill_registry is None:
            return
        try:
            reload_fn = getattr(self._skill_registry, "reload_user_skills", None)
            if callable(reload_fn):
                reload_fn()
                log.info("Live recorder: skill registry reloaded")
        except Exception:
            log.warning("Live recorder: skill registry reload failed", exc_info=True)


def _normalize_metadata(metadata: dict[str, Any], *, event_count: int) -> dict[str, Any]:
    """Map extension camelCase metadata to the pipeline's snake_case fields."""
    duration_ms = metadata.get("duration") or 0
    return {
        "start_url": metadata.get("startUrl") or metadata.get("start_url") or "",
        "duration_seconds": int(round(duration_ms / 1000)) if duration_ms else
        int(metadata.get("duration_seconds") or 0),
        "event_count": event_count,
        "extension_version": metadata.get("extensionVersion") or metadata.get("extension_version") or "unknown",
        "captured_at": metadata.get("capturedAt") or metadata.get("captured_at") or "",
        "title": metadata.get("title") or "",
        "app": metadata.get("app") or "",
        "description": metadata.get("description") or "",
        "source": "live_dom_recording",
    }
