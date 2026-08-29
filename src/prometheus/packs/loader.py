"""Pack loader — discover ~/.prometheus/packs/<name>/ at startup.

FOUNDATION Part 2: a pack is a directory with a ``pack.yaml`` manifest
declaring a ``pack_api`` contract version and a ``provides`` block. Two
registration points, no more: **skills** (markdown knowledge — quarantined
into the existing SkillDraftStore, promoted only by the explicit human
ACCEPT) and **panels** (declared here, served read-only over ``/api/packs``;
Beacon-side loading is deferred until its sandbox is named). Tools are
deliberately NOT a registration point — a third party who wants to give
Prometheus a tool writes an MCP server (spec 2.3a), and a ``tools/``
directory in a pack is refused by name.

Refusal semantics: a bad pack is refused — loudly, with the pack name and
the violation — but the DAEMON boots. Third-party content must never be
able to keep the daemon down; the refusal is the boundary error the spec
demands, recorded on the registry so ``/api/packs`` shows it.

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from prometheus.config.paths import get_config_dir

logger = logging.getLogger(__name__)

# The pack contract version this daemon supports. Moves independently of
# the Prometheus version. Bumping it is a FOUNDATION changelog event.
PACK_API_CURRENT = 1

_INGESTED_STATE_FILENAME = "_ingested.json"


def get_packs_dir() -> Path:
    """Return ~/.prometheus/packs (created on first read, like the other
    config-dir helpers)."""
    packs_dir = get_config_dir() / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    return packs_dir


@dataclass
class PanelDeclaration:
    """A Beacon panel a pack declares. Declaration only in this version —
    the daemon never loads or executes it."""

    name: str
    route: str
    title: str
    icon: str = ""
    component: str = ""


@dataclass
class PackRecord:
    """One discovered pack directory, loaded or refused."""

    name: str
    path: str
    state: str  # "loaded" | "refused"
    detail: str = ""
    version: str = ""
    pack_api: int | None = None
    skills: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)  # draft ids created this boot
    panels: list[PanelDeclaration] = field(default_factory=list)


@dataclass
class PackRegistry:
    """The outcome of one discovery pass, pinned for /api/packs."""

    records: list[PackRecord] = field(default_factory=list)

    def loaded(self) -> list[PackRecord]:
        return [r for r in self.records if r.state == "loaded"]

    def refused(self) -> list[PackRecord]:
        return [r for r in self.records if r.state == "refused"]

    def to_status(self) -> list[dict[str, Any]]:
        return [
            {
                "name": r.name,
                "state": r.state,
                "detail": r.detail,
                "version": r.version,
                "pack_api": r.pack_api,
                "skills": r.skills,
                "quarantined_drafts": r.quarantined,
                "panels": [
                    {
                        "name": p.name,
                        "route": p.route,
                        "title": p.title,
                        "icon": p.icon,
                        "component": p.component,
                    }
                    for p in r.panels
                ],
            }
            for r in self.records
        ]


# Process-pinned registry, same pattern as set_vault_root/set_instance_id:
# the daemon discovers once at boot; /api/packs reads it back.
_registry: PackRegistry | None = None


def set_pack_registry(registry: PackRegistry | None) -> None:
    global _registry
    _registry = registry


def get_pack_registry() -> PackRegistry | None:
    return _registry


def _refuse(records: list[PackRecord], name: str, path: Path, reason: str) -> None:
    # The spec's wording is the contract: fail loudly AT LOAD with the
    # pack name and the violation.
    logger.error("Pack %r REFUSED: %s", name, reason)
    records.append(PackRecord(
        name=name, path=str(path), state="refused", detail=reason,
    ))


def _check_pack_api(value: Any) -> str | None:
    """Return a refusal reason, or None when the contract version is ours.

    Mirrors the vault marker's mismatch table (spec 2.2) deliberately —
    same silent-misread class, same shape of refusal.
    """
    if value is None:
        return (
            "declares no pack_api — a pack that does not declare a "
            "contract version has not agreed to one"
        )
    if not isinstance(value, int) or isinstance(value, bool):
        return f"pack_api must be an integer, got {value!r}"
    if value > PACK_API_CURRENT:
        return (
            f"declares pack_api {value}; this daemon supports "
            f"pack_api {PACK_API_CURRENT} — newer pack, older daemon. "
            "Upgrade Prometheus or use a pack built for this contract"
        )
    if value < PACK_API_CURRENT:
        return (
            f"declares pack_api {value}; this daemon supports "
            f"pack_api {PACK_API_CURRENT} and no older contract exists "
            "to fall back to"
        )
    return None


def _parse_panels(pack_dir: Path, declared: list[str]) -> tuple[list[PanelDeclaration], str | None]:
    """Validate declared panels against panels/ on disk. Returns
    (declarations, refusal_reason)."""
    panels_dir = pack_dir / "panels"
    on_disk = sorted(p.stem for p in panels_dir.glob("*.yaml")) if panels_dir.is_dir() else []
    declared_sorted = sorted(str(p) for p in declared)
    if declared_sorted != on_disk:
        undeclared = [p for p in on_disk if p not in declared_sorted]
        missing = [p for p in declared_sorted if p not in on_disk]
        return [], (
            "provides.panels disagrees with panels/ on disk"
            + (f" — claims {missing} it does not ship" if missing else "")
            + (f" — ships {undeclared} it does not declare" if undeclared else "")
        )
    declarations: list[PanelDeclaration] = []
    for stem in on_disk:
        panel_path = panels_dir / f"{stem}.yaml"
        try:
            raw = yaml.safe_load(panel_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            return [], f"panel {stem!r} is unreadable: {exc}"
        if not isinstance(raw, dict) or not raw.get("route") or not raw.get("title"):
            return [], (
                f"panel {stem!r} must declare at least route and title "
                "(spec 2.3: a route, a title, an icon, and a component)"
            )
        declarations.append(PanelDeclaration(
            name=stem,
            route=str(raw["route"]),
            title=str(raw["title"]),
            icon=str(raw.get("icon", "")),
            component=str(raw.get("component", "")),
        ))
    return declarations, None


def _load_ingested_state(packs_dir: Path) -> dict[str, Any]:
    state_path = packs_dir / _INGESTED_STATE_FILENAME
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        logger.warning("Packs: unreadable %s — re-quarantining is the safe "
                       "direction, continuing with empty state",
                       _INGESTED_STATE_FILENAME)
        return {}


def _save_ingested_state(packs_dir: Path, state: dict[str, Any]) -> None:
    state_path = packs_dir / _INGESTED_STATE_FILENAME
    tmp = state_path.with_name(state_path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(state_path)


def load_packs(
    packs_dir: Path | None = None,
    draft_store: Any | None = None,
) -> PackRegistry:
    """Discover packs, enforce the contract, quarantine their skills.

    Every skill a loaded pack ships lands in the SkillDraftStore — the
    same quarantine→human-ACCEPT lifecycle vision-derived skills use
    (spec 2.3: a skill arriving from a pack is untrusted; this is a
    security rule and it does not change). Ingestion is deduplicated by
    content hash in ``packs/_ingested.json`` so a boot does not re-draft
    what a human already accepted or rejected.
    """
    packs_dir = packs_dir or get_packs_dir()
    records: list[PackRecord] = []
    ingested = _load_ingested_state(packs_dir)
    ingested_dirty = False

    for pack_dir in sorted(p for p in packs_dir.iterdir() if p.is_dir()):
        dirname = pack_dir.name
        if dirname.startswith((".", "_")):
            continue

        manifest_path = pack_dir / "pack.yaml"
        if not manifest_path.exists():
            _refuse(records, dirname, pack_dir, "no pack.yaml manifest")
            continue
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            _refuse(records, dirname, pack_dir, f"pack.yaml unreadable: {exc}")
            continue
        if not isinstance(manifest, dict):
            _refuse(records, dirname, pack_dir, "pack.yaml is not a mapping")
            continue

        name = str(manifest.get("name") or dirname)

        api_reason = _check_pack_api(manifest.get("pack_api"))
        if api_reason:
            _refuse(records, name, pack_dir, api_reason)
            continue

        # Tools are MCP, not packs (2.3a). Refused by NAME so the error
        # teaches the contract rather than silently ignoring the dir.
        if (pack_dir / "tools").exists():
            _refuse(records, name, pack_dir, (
                "ships a tools/ directory — tools are not a pack "
                "registration point; write an MCP server (FOUNDATION 2.3a)"
            ))
            continue

        provides = manifest.get("provides") or {}
        if not isinstance(provides, dict):
            _refuse(records, name, pack_dir, "provides is not a mapping")
            continue

        # Manifest integrity (2.4): the provides block must match the
        # disk, both directions — absence in the manifest is not
        # permission.
        skills_dir = pack_dir / "skills"
        disk_skills = sorted(p.stem for p in skills_dir.glob("*.md")) if skills_dir.is_dir() else []
        declared_skills = sorted(str(s) for s in (provides.get("skills") or []))
        if declared_skills != disk_skills:
            undeclared = [s for s in disk_skills if s not in declared_skills]
            missing = [s for s in declared_skills if s not in disk_skills]
            _refuse(records, name, pack_dir, (
                "provides.skills disagrees with skills/ on disk"
                + (f" — claims {missing} it does not ship" if missing else "")
                + (f" — ships {undeclared} it does not declare" if undeclared else "")
            ))
            continue

        panels, panel_reason = _parse_panels(
            pack_dir, provides.get("panels") or []
        )
        if panel_reason:
            _refuse(records, name, pack_dir, panel_reason)
            continue

        record = PackRecord(
            name=name,
            path=str(pack_dir),
            state="loaded",
            version=str(manifest.get("version", "")),
            pack_api=int(manifest["pack_api"]),
            skills=disk_skills,
            panels=panels,
        )

        # Quarantine: every shipped skill becomes a DRAFT. Drafts live in
        # skills/drafts/, which the skill loader's globs never match — an
        # unpromoted pack skill is unreachable by the loop by
        # construction, and the reachability test drives the loop to
        # prove it.
        if disk_skills and draft_store is not None:
            for stem in disk_skills:
                content = (skills_dir / f"{stem}.md").read_text(encoding="utf-8")
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                if digest in ingested:
                    continue
                sidecar = draft_store.create(
                    content,
                    source=f"pack:{name}",
                    provenance={
                        "pack": name,
                        "pack_version": record.version,
                        "file": f"skills/{stem}.md",
                        "sha256": digest,
                    },
                )
                record.quarantined.append(str(sidecar.get("draft_id", "")))
                ingested[digest] = {
                    "pack": name,
                    "skill": stem,
                    "draft_id": sidecar.get("draft_id"),
                    "drafted_at": datetime.now(timezone.utc).isoformat(),
                }
                ingested_dirty = True
            if record.quarantined:
                logger.info(
                    "Pack %r: %d skill(s) QUARANTINED as drafts %s — "
                    "review via /api/learning/skill-drafts; none are "
                    "reachable until accepted",
                    name, len(record.quarantined), record.quarantined,
                )

        logger.info(
            "Pack %r loaded: pack_api %d, %d skill(s), %d panel(s)",
            name, record.pack_api, len(record.skills), len(record.panels),
        )
        records.append(record)

    if ingested_dirty:
        _save_ingested_state(packs_dir, ingested)

    registry = PackRegistry(records=records)
    if records:
        logger.info(
            "Packs: %d loaded, %d refused",
            len(registry.loaded()), len(registry.refused()),
        )
    return registry
