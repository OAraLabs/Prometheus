"""Vision digest — VLM keyframe analysis for the video-ingestion funnel.

Ported from skillforge-engine ``core/vision_digest.py`` (v2). Kept: the
:class:`FrameDigest` / :class:`DigestResult` dataclass shapes, the frame
analysis and comparison prompt texts, the JSON fence-strip parsing, and
the per-frame checkpoint/resume logic (a crashed run picks up at the
last completed frame). NOT ported: SkillForge's bespoke vision clients
(Anthropic/Local/Gemini) and the Ollama VRAM flush/restart machinery —
model calls go through Prometheus's provider layer instead
(``ProviderRegistry.create`` once, then ``provider.stream_message`` per
frame), so any configured provider that accepts OpenAI-style
``image_url`` content blocks works: llama.cpp with ``--mmproj``, Ollama,
or a cloud provider.

Image messages follow ``prometheus.tools.builtin.vision``: raw
OpenAI-style payloads (``{"type": "image_url", "image_url": {"url":
"data:image/png;base64,..."}}``) passed as
``ApiMessageRequest(messages=<raw list>)``, bypassing the typed
``ConversationMessage`` (which has no image block type).

The registry gate (:func:`validate_vision_model`) checks the configured
model against ``config/model_registry.yaml`` so we refuse to burn a
whole digest run on a model the registry says is text-only.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from prometheus.providers.registry import ProviderRegistry

log = logging.getLogger(__name__)

try:  # Reuse the vision tool's data-url helper when available
    from prometheus.tools.builtin.vision import _image_to_base64_data_url as _to_data_url
except ImportError:  # pragma: no cover — minimal replication
    def _to_data_url(path: str) -> str:
        data = Path(path).read_bytes()
        mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


# =====================================================================
# Dataclasses (shapes kept from the SkillForge original)
# =====================================================================


@dataclass
class FrameDigest:
    """Structured analysis of a single keyframe."""

    frame_id: int
    timestamp_seconds: float
    filepath: str

    # What's on screen
    application: str = ""           # e.g., "TrackerRMS", "Chrome - Gmail", "Excel"
    page_or_view: str = ""          # e.g., "Candidate Search", "Inbox", "Sheet1"
    url_if_visible: str = ""        # URL bar contents if visible

    # UI State — each element: {"type", "label", "state", "location"}
    ui_elements: list[dict] = field(default_factory=list)

    # What's happening
    description: str = ""           # Natural language description of the frame
    user_action_visible: str = ""   # What action the user appears to be taking
    data_on_screen: list[str] = field(default_factory=list)

    # Context for action extraction
    notable_changes: str = ""       # What changed vs previous frame
    cursor_context: str = ""        # Where cursor is, what it's near

    # Model metadata
    model_used: str = ""
    tokens_used: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DigestResult:
    """Complete vision digest for all frames."""

    digests: list[FrameDigest] = field(default_factory=list)
    model_provider: str = ""
    model_name: str = ""
    total_tokens_used: int = 0
    total_api_calls: int = 0
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "total_tokens_used": self.total_tokens_used,
            "total_api_calls": self.total_api_calls,
            "processing_time_seconds": self.processing_time_seconds,
            "total_frames_digested": len(self.digests),
            "digests": [d.to_dict() for d in self.digests],
        }

    def save(self, output_path: Path | str) -> None:
        Path(output_path).write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8",
        )
        log.info("Vision digest saved: %s", output_path)


# =====================================================================
# Prompts (kept verbatim from the SkillForge original)
# =====================================================================

FRAME_ANALYSIS_PROMPT = """You are analyzing a screenshot from a screen recording of a user performing a workflow task. Your job is to produce a structured analysis of this frame that will be used to reconstruct the workflow as an automatable skill.

Analyze this screenshot and respond with a JSON object (no markdown fencing, just pure JSON):

{
  "application": "Name of the application or website visible",
  "page_or_view": "Specific page, tab, or view within the application",
  "url_if_visible": "URL from the address bar if visible, otherwise empty string",
  "description": "2-3 sentence natural language description of what's shown on screen",
  "user_action_visible": "What the user appears to be doing RIGHT NOW (e.g., 'clicking the Search button', 'typing in the name field', 'scrolling through results')",
  "ui_elements": [
    {
      "type": "button|input|dropdown|link|menu|tab|modal|table|form|checkbox|radio",
      "label": "Text label or description of the element",
      "state": "active|disabled|focused|selected|filled|empty|highlighted",
      "location": "Approximate screen location: top-left|top-center|top-right|center-left|center|center-right|bottom-left|bottom-center|bottom-right|sidebar|header|footer|modal"
    }
  ],
  "data_on_screen": ["List of notable data values visible (names, numbers, search terms, etc.)"],
  "cursor_context": "Where the mouse cursor is positioned and what it's near",
  "confidence": 0.95
}

Focus on:
1. INTERACTIVE elements (buttons, inputs, links) — these drive the workflow
2. DATA visible on screen — search terms, form values, results
3. NAVIGATION context — where we are in the application
4. The specific ACTION the user is taking or just took

Be precise about UI element labels — use the exact text shown on screen.
If you can't determine something, use an empty string rather than guessing."""

FRAME_COMPARISON_PROMPT = """You are analyzing two consecutive screenshots from a workflow recording. Describe what changed between Frame A (previous) and Frame B (current).

Respond with a JSON object (no markdown fencing):

{
  "notable_changes": "Concise description of what changed between the frames",
  "action_that_caused_change": "The user action that most likely caused this change (e.g., 'clicked Submit button', 'typed search query', 'selected dropdown option')",
  "change_type": "navigation|form_input|button_click|data_load|modal_open|modal_close|scroll|selection|other",
  "change_significance": "high|medium|low",
  "new_state": "Brief description of the new state after the change"
}

Focus on MEANINGFUL changes — ignore minor rendering differences, cursor movements without clicks, or loading state changes."""


# =====================================================================
# JSON parsing (fence-strip logic kept from the original)
# =====================================================================


def parse_json_response(text: str) -> dict[str, Any]:
    """Parse JSON from a model response, handling markdown fencing."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


# =====================================================================
# Checkpoint helpers (resume logic kept from the original)
# =====================================================================


def _checkpoint_path(output_dir: Path | str) -> Path:
    return Path(output_dir) / "digest_checkpoint.json"


def load_checkpoint(output_dir: Path | str) -> dict[str, Any]:
    """Load an existing checkpoint.

    Returns ``{"completed_frame_ids": set, "digests": list}`` — empty
    when no checkpoint exists (fresh run).
    """
    cp = _checkpoint_path(output_dir)
    if cp.is_file():
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            completed = set(data.get("completed_frame_ids", []))
            log.info("Digest checkpoint: resuming — %d frames already done", len(completed))
            return {
                "completed_frame_ids": completed,
                "digests": data.get("digests", []),
            }
        except Exception as exc:  # noqa: BLE001 — corrupt checkpoint: start fresh
            log.warning("Could not load digest checkpoint (starting fresh): %s", exc)

    return {"completed_frame_ids": set(), "digests": []}


def save_checkpoint(
    output_dir: Path | str,
    completed_frame_ids: set[int],
    digests: list[dict[str, Any]],
) -> None:
    """Save progress after each frame — lets a crashed run resume."""
    cp = _checkpoint_path(output_dir)
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(
            json.dumps({
                "completed_frame_ids": sorted(completed_frame_ids),
                "digests": digests,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        log.warning("Digest checkpoint save failed (non-fatal): %s", exc)


def clear_checkpoint(output_dir: Path | str) -> None:
    """Remove the checkpoint after a successful complete run."""
    cp = _checkpoint_path(output_dir)
    if cp.is_file():
        cp.unlink()
        log.info("Digest checkpoint cleared (run complete)")


# =====================================================================
# Registry gate
# =====================================================================


def _load_model_registry(registry_path: Path | str | None = None) -> dict[str, Any]:
    """Load ``config/model_registry.yaml`` (same file /doctor reads)."""
    import yaml

    if registry_path is not None:
        candidates = [Path(registry_path)]
    else:
        candidates = [
            # repo root: src/prometheus/learning/video_ingest/ -> up 4
            Path(__file__).resolve().parents[4] / "config" / "model_registry.yaml",
            Path("config/model_registry.yaml"),
        ]
    for path in candidates:
        if path.exists():
            try:
                return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception as exc:  # noqa: BLE001 — unreadable registry
                log.warning("Failed to load model registry from %s: %s", path, exc)
                return {}
    return {}


def _match_model(model_name: str, registry: dict[str, Any]) -> dict[str, Any] | None:
    """Fuzzy-match a model name against registry match_patterns.

    Replicated from ``prometheus.infra.doctor.match_model``:
    case-insensitive substring match, first hit in file order wins.
    """
    if not model_name:
        return None
    model_lower = model_name.lower()
    for _family_id, family in registry.get("models", {}).items():
        for pattern in family.get("match_patterns", []):
            if pattern.lower() in model_lower:
                return family
    return None


def validate_vision_model(
    vision_model_cfg: dict[str, Any],
    *,
    registry_path: Path | str | None = None,
) -> tuple[bool, str]:
    """Check the configured model against the capability registry.

    Returns ``(supported, explanation)``. Unknown models and a missing
    registry are permissive (``True``) — the registry can only vouch for
    families it knows about. A known text-only family returns ``False``
    so the pipeline refuses before burning a whole digest run.
    """
    model_name = str(vision_model_cfg.get("model") or "")
    registry = _load_model_registry(registry_path)

    if not registry.get("models"):
        return True, "model registry not found — skipping vision capability check"
    if not model_name:
        return True, "no model name configured — cannot check registry; assuming vision support"

    family = _match_model(model_name, registry)
    if family is None:
        return True, (
            f"model '{model_name}' is not in the capability registry — "
            "assuming vision support"
        )

    display = family.get("display_name", model_name)
    vision_cap = family.get("capabilities", {}).get("vision", {})
    if vision_cap.get("supported"):
        msg = f"{display} supports vision"
        requires = vision_cap.get("requires")
        if requires:
            msg += f" (requires {requires})"
        hint = vision_cap.get("setup_hint")
        if hint:
            msg += f". Setup: {str(hint).strip()}"
        return True, msg

    msg = f"{display} does not support vision"
    detail = vision_cap.get("description")
    if detail:
        msg += f" ({detail})"
    msg += " — configure a vision-capable model (e.g. Gemma 4 with mmproj) for video ingestion"
    return False, msg


# =====================================================================
# Provider call
# =====================================================================


def _image_part(path: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": _to_data_url(path)}}


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


async def _call_vision(
    provider: Any,
    model: str,
    content: list[dict[str, Any]],
    max_tokens: int,
) -> str:
    """One multimodal call: raw OpenAI-style content blocks, text out."""
    from prometheus.providers.base import ApiMessageRequest, ApiTextDeltaEvent

    request = ApiMessageRequest(
        model=model,
        messages=[{"role": "user", "content": content}],  # raw payload, see module docstring
        max_tokens=max_tokens,
    )
    text_parts: list[str] = []
    async for event in provider.stream_message(request):
        if isinstance(event, ApiTextDeltaEvent):
            text_parts.append(event.text)
    return "".join(text_parts)


# =====================================================================
# Analysis -> action derivation
# =====================================================================

_CHANGE_TYPE_TO_ACTION = {
    "navigation": "navigate",
    "form_input": "type",
    "button_click": "click",
    "selection": "select",
    "modal_open": "click",
    "modal_close": "click",
}

_ACTIVE_STATES = ("focused", "active", "selected", "highlighted")


def _infer_action_type(text: str) -> str:
    """Keyword-infer a lowercase action type from a model action phrase."""
    t = text.lower()
    if not t:
        return "unknown"
    if "click" in t or "press" in t or "tap" in t:
        return "click"
    if "submit" in t:
        return "submit"
    if "typ" in t or "enter" in t or "fill" in t:
        return "type"
    if "select" in t or "choos" in t or "dropdown" in t:
        return "select"
    if "navigat" in t or "go to" in t or "open" in t or "visit" in t:
        return "navigate"
    return "unknown"


def _quoted_fragment(text: str) -> str:
    """First quoted fragment in a phrase — the likely target label."""
    m = re.search(r"['\"]([^'\"]{1,60})['\"]", text)
    return m.group(1) if m else ""


def _focused_target(analysis: dict[str, Any]) -> str:
    """Label of the first focused/active UI element, if any."""
    for element in analysis.get("ui_elements") or []:
        if str(element.get("state", "")).lower() in _ACTIVE_STATES:
            label = str(element.get("label", "")).strip()
            if label:
                return label
    return ""


def _derive_actions(
    analysis: dict[str, Any],
    comparison: dict[str, Any],
) -> list[dict[str, Any]]:
    """Derive ``{type, target, description}`` actions from a frame's analysis.

    Prefers the frame comparison (structured ``change_type`` +
    ``action_that_caused_change``) when the change is significant;
    falls back to the analysis' ``user_action_visible``.
    """
    actions: list[dict[str, Any]] = []

    caused = str((comparison or {}).get("action_that_caused_change") or "").strip()
    significance = str((comparison or {}).get("change_significance") or "").lower()
    change_type = str((comparison or {}).get("change_type") or "").lower()
    if caused and significance in ("high", "medium"):
        action_type = _CHANGE_TYPE_TO_ACTION.get(change_type) or _infer_action_type(caused)
        if action_type != "unknown":
            actions.append({
                "type": action_type,
                "target": _quoted_fragment(caused) or _focused_target(analysis),
                "description": caused,
            })

    if not actions:
        visible = str((analysis or {}).get("user_action_visible") or "").strip()
        if visible:
            action_type = _infer_action_type(visible)
            if action_type != "unknown":
                actions.append({
                    "type": action_type,
                    "target": _quoted_fragment(visible) or _focused_target(analysis),
                    "description": visible,
                })

    return actions


def _context_from_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "application": analysis.get("application", ""),
        "page_or_view": analysis.get("page_or_view", ""),
        "url": analysis.get("url_if_visible", ""),
        "page_title": analysis.get("page_or_view", ""),
    }


# =====================================================================
# Public API
# =====================================================================


async def digest_keyframes(
    keyframes: list[dict[str, Any]],
    narration: list[dict[str, Any]],
    vision_model_cfg: dict[str, Any],
    output_dir: Path,
    *,
    prompt_context: str = "",
) -> list[dict[str, Any]]:
    """Digest every keyframe with the configured vision model.

    Args:
        keyframes: Manifest entries from ``keyframes.extract_keyframes``
            (``{index, path, timestamp, ...}``).
        narration: Aligned narration from ``ingest.align_narration``
            (``{frame_index, narration}``); may be empty.
        vision_model_cfg: A Prometheus ``model:``-shaped config block
            (``{provider, model, base_url?, ...}``) — handed to
            ``ProviderRegistry.create`` exactly once.
        output_dir: Where ``digests.json`` and the resume checkpoint live.
        prompt_context: Optional workflow hint prepended to the analysis
            prompt (e.g. the inferred recording title).

    Returns:
        Per-frame digests:
        ``{keyframe_index, vision: {actions: [{type, target, description}],
        context: {...}}, narration, analysis}`` — the shape
        ``action_extractor.extract_actions`` consumes. Also written to
        ``output_dir/digests.json``. A per-frame checkpoint means a
        crashed run resumes at the last completed frame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provider = ProviderRegistry.create(vision_model_cfg)
    model = str(vision_model_cfg.get("model") or "")
    max_tokens = int(vision_model_cfg.get("max_tokens") or 2048)

    narration_by_index = {
        n.get("frame_index"): str(n.get("narration") or "") for n in narration
    }

    checkpoint = load_checkpoint(output_dir)
    completed: set[int] = checkpoint["completed_frame_ids"]
    digests: list[dict[str, Any]] = list(checkpoint["digests"])

    started = time.time()
    prev_path: str | None = None

    for i, kf in enumerate(keyframes):
        idx = int(kf.get("index", i))
        path = str(kf.get("path") or "")

        if idx in completed:
            prev_path = path
            continue

        narr = narration_by_index.get(idx, "")

        analysis_prompt = FRAME_ANALYSIS_PROMPT
        extras = []
        if prompt_context:
            extras.append(f"Workflow context: {prompt_context}")
        if narr:
            extras.append(f'The user narrated while performing this step: "{narr}"')
        if extras:
            analysis_prompt = analysis_prompt + "\n\n" + "\n".join(extras)

        analysis: dict[str, Any] = {}
        comparison: dict[str, Any] = {}
        try:
            raw = await _call_vision(
                provider, model,
                [_image_part(path), _text_part(analysis_prompt)],
                max_tokens,
            )
            analysis = parse_json_response(raw)
        except Exception as exc:  # noqa: BLE001 — one bad frame must not kill the run
            log.error("Frame %d analysis failed: %s", idx, exc)

        if analysis and prev_path:
            try:
                raw = await _call_vision(
                    provider, model,
                    [
                        _text_part("Frame A (previous):"),
                        _image_part(prev_path),
                        _text_part("Frame B (current):"),
                        _image_part(path),
                        _text_part(FRAME_COMPARISON_PROMPT),
                    ],
                    max_tokens,
                )
                comparison = parse_json_response(raw)
            except Exception as exc:  # noqa: BLE001 — comparison is enrichment only
                log.warning("Frame %d comparison failed (non-fatal): %s", idx, exc)

        frame_digest = FrameDigest(
            frame_id=idx,
            timestamp_seconds=float(kf.get("timestamp") or 0.0),
            filepath=path,
            application=analysis.get("application", ""),
            page_or_view=analysis.get("page_or_view", ""),
            url_if_visible=analysis.get("url_if_visible", ""),
            description=analysis.get("description", ""),
            user_action_visible=analysis.get("user_action_visible", ""),
            ui_elements=analysis.get("ui_elements", []),
            data_on_screen=analysis.get("data_on_screen", []),
            cursor_context=analysis.get("cursor_context", ""),
            notable_changes=comparison.get("notable_changes", ""),
            model_used=model,
            confidence=float(analysis.get("confidence") or 0.0),
        )

        digests.append({
            "keyframe_index": idx,
            "vision": {
                "actions": _derive_actions(analysis, comparison),
                "context": _context_from_analysis(analysis),
            },
            "narration": narr,
            "analysis": frame_digest.to_dict(),
        })

        completed.add(idx)
        save_checkpoint(output_dir, completed, digests)
        log.info("Digested keyframe %d (%d/%d)", idx, len(completed), len(keyframes))

        prev_path = path

    digests.sort(key=lambda d: d.get("keyframe_index", 0))

    digests_path = output_dir / "digests.json"
    try:
        digests_path.write_text(json.dumps(digests, indent=2), encoding="utf-8")
        log.info(
            "Vision digest complete: %d frames in %.1fs -> %s",
            len(digests), time.time() - started, digests_path,
        )
    except OSError:
        log.warning("Failed to write %s", digests_path, exc_info=True)

    clear_checkpoint(output_dir)
    return digests
