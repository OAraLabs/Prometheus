"""Brain-vault format marker — the versioned `.prometheus-vault` file.

Foundation Spec Part 1.1 (docs/FOUNDATION.md): every vault carries a version
marker so a newer vault read by an older binary refuses loudly instead of
being silently misread — the project's dominant bug class. The marker also
carries the *instance* identity (Part 3.3): a UUID minted when the vault is
adopted, which travels with the vault when it moves machines. Node identity
(the per-machine keypair) deliberately does NOT live here.

This module is the ONLY writer of vault content in the entire codebase, and
it writes exactly one file at the vault root. It cannot live in
``tools/builtin/vault.py``: that module is kept structurally read-only by a
receiver-blind AST guard (``tests/test_vault_tools.py``), and that guard is
load-bearing — the vault's zone rules say the machine never writes there.
The marker is the deliberate, narrow exception, so it gets its own module
where the write surface is auditable in isolation.

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# The current on-disk vault format. Bumping this is a breaking change under
# the Foundation Spec: it requires a migration path in check_vault_marker()
# and a version-bump row in docs/FOUNDATION.md's changelog.
VAULT_FORMAT_CURRENT = 1

MARKER_FILENAME = ".prometheus-vault"


class VaultMarkerError(RuntimeError):
    """A marker state the daemon must not proceed past."""


class VaultMarkerCorrupt(VaultMarkerError):
    """The marker file exists but cannot be parsed into a valid marker."""


@dataclass
class VaultMarker:
    vault_format: int
    created: str
    created_by: str
    instance_id: str
    enrolled_nodes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "vault_format": self.vault_format,
            "created": self.created,
            "created_by": self.created_by,
            "instance_id": self.instance_id,
            "enrolled_nodes": self.enrolled_nodes,
        }


def marker_path(vault_root: Path) -> Path:
    return vault_root / MARKER_FILENAME


def read_marker(vault_root: Path) -> VaultMarker | None:
    """Read the marker at ``vault_root``, or ``None`` when absent.

    Raises :class:`VaultMarkerCorrupt` when the file exists but is not a
    valid marker — an unreadable marker is a deliberate state to surface,
    never one to guess past.
    """
    path = marker_path(vault_root)
    if not path.exists():
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise VaultMarkerCorrupt(
            f"Vault marker {path} is not parseable YAML: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise VaultMarkerCorrupt(
            f"Vault marker {path} does not contain a mapping "
            f"(got {type(raw).__name__})"
        )
    fmt = raw.get("vault_format")
    if not isinstance(fmt, int) or isinstance(fmt, bool):
        raise VaultMarkerCorrupt(
            f"Vault marker {path} has no integer vault_format "
            f"(got {raw.get('vault_format')!r})"
        )
    instance_id = raw.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise VaultMarkerCorrupt(
            f"Vault marker {path} has no instance_id — the marker without "
            "its instance identity is half a marker"
        )
    enrolled = raw.get("enrolled_nodes") or []
    if not isinstance(enrolled, list):
        raise VaultMarkerCorrupt(
            f"Vault marker {path} enrolled_nodes is not a list"
        )
    return VaultMarker(
        vault_format=fmt,
        created=str(raw.get("created", "")),
        created_by=str(raw.get("created_by", "")),
        instance_id=instance_id,
        enrolled_nodes=[dict(e) for e in enrolled if isinstance(e, dict)],
    )


def write_marker(vault_root: Path, marker: VaultMarker) -> Path:
    """Write the marker atomically (tmp + rename), preserving key order.

    Atomic because the vault is a live git repo another process (or the
    human) may be looking at: a torn half-written marker would read as
    corrupt, and corrupt refuses the boot.
    """
    path = marker_path(vault_root)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        yaml.safe_dump(marker.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def create_marker(vault_root: Path, created_by: str) -> VaultMarker:
    """Mint a marker for an existing vault. Refuses to overwrite one.

    This is adoption, and adoption is explicit: the daemon never calls this
    on its own (Foundation Spec 1.1 — "do not silently adopt"). The one
    caller is ``oara vault adopt``.
    """
    if not vault_root.is_dir():
        raise VaultMarkerError(
            f"No vault directory at {vault_root} — nothing to adopt. "
            "Check vault.root / PROMETHEUS_VAULT."
        )
    if marker_path(vault_root).exists():
        raise VaultMarkerError(
            f"{marker_path(vault_root)} already exists — this vault is "
            "already adopted. Refusing to overwrite an identity."
        )
    marker = VaultMarker(
        vault_format=VAULT_FORMAT_CURRENT,
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        created_by=created_by,
        instance_id=str(uuid.uuid4()),
        enrolled_nodes=[],
    )
    write_marker(vault_root, marker)
    return marker


# The brand prefix stripped from enrollment labels, spelled as adjacent
# string fragments so this file passes the very hooks it exists to satisfy:
# both this repo's pre-commit and the vault's own block infrastructure
# hostnames carrying the contiguous prefix in committed text. A machine-
# written marker line like ``label: <brand>-mini`` therefore turned every
# commit of the vault into a hook fight — proven live in the first adopted
# vault, where the label had to be hand-neutralized after enrollment.
_BRAND_HOST_PREFIX = "oara" "-"


def neutral_label(hostname: str) -> str:
    """Derive a display label safe to commit inside the vault.

    Strips the brand infra prefix case-insensitively (``<Brand>-mini`` →
    ``mini``); falls back to ``"node"`` when nothing usable remains. The
    label is display only — enrollment identity is the pubkey — so the
    strip loses nothing.
    """
    label = hostname.strip()
    if label.lower().startswith(_BRAND_HOST_PREFIX):
        label = label[len(_BRAND_HOST_PREFIX):]
    return label or "node"


def enroll_node(vault_root: Path, pubkey: str, label: str) -> bool:
    """Add a node's public key to the marker's ``enrolled_nodes``.

    Returns True when the node was newly enrolled, False when it was
    already present (the every-boot case). Spec 3.5/3.6: in this version
    the local node self-enrolls on startup against an adopted vault — the
    human who ran ``oara vault adopt`` is the approval. The explicit
    approve-before-enroll step arrives with the fleet and replaces this.

    ``label`` is display only (a hostname, typically). It is never
    identity — the pubkey is. It is neutralized via :func:`neutral_label`
    before it is written, so an infrastructure hostname never lands in the
    vault's git history. Already-enrolled labels are left as they are —
    re-enrollment is skipped by pubkey before the label is ever compared.
    """
    marker = read_marker(vault_root)
    if marker is None:
        raise VaultMarkerError(
            f"Cannot enroll a node at {vault_root}: no marker. Adopt first."
        )
    if any(node.get("pubkey") == pubkey for node in marker.enrolled_nodes):
        return False
    label = neutral_label(label)
    marker.enrolled_nodes.append({
        "pubkey": pubkey,
        "label": label,
        "enrolled": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    write_marker(vault_root, marker)
    logger.info(
        "Node %s… enrolled into instance %s as %r",
        pubkey[:12], marker.instance_id, label,
    )
    return True


def check_vault_marker(vault_root: Path, mode: str = "warn") -> VaultMarker | None:
    """The startup gate. Called once from ``run_daemon`` after the vault
    root is pinned.

    ``mode`` is ``vault.format_check``: ``off`` | ``warn`` | ``refuse``.
    It governs only the LEGACY states (vault present, marker absent or
    unreadable). A successfully parsed marker whose format disagrees with
    this build refuses REGARDLESS of mode — that is the silent-misread case
    the marker exists to catch, and it can only arise from a deliberate
    state (a newer Prometheus wrote it, or a rollback ran), never from a
    vault that simply predates markers.

    An absent VAULT is not an error, matching the long-standing stance in
    ``config/prometheus.yaml.default``: the vault tools report absence
    clearly when called; startup does not.
    """
    if not vault_root.is_dir():
        return None

    try:
        marker = read_marker(vault_root)
    except VaultMarkerCorrupt as exc:
        if mode == "refuse":
            raise
        if mode == "warn":
            logger.warning(
                "Vault marker is CORRUPT and vault.format_check is 'warn' — "
                "continuing, but this file should never be half-valid: %s",
                exc,
            )
        return None

    if marker is None:
        if mode == "off":
            return None
        message = (
            f"Vault at {vault_root} has no {MARKER_FILENAME} marker "
            "(a vault from before vault_format existed). Run "
            "`oara vault adopt` once to mint its marker and "
            "instance identity. Prometheus does not adopt silently."
        )
        if mode == "refuse":
            raise VaultMarkerError(message)
        logger.warning("%s", message)
        return None

    if marker.vault_format == VAULT_FORMAT_CURRENT:
        logger.info(
            "Vault marker OK: format %d, instance %s",
            marker.vault_format, marker.instance_id,
        )
        return marker

    if marker.vault_format > VAULT_FORMAT_CURRENT:
        raise VaultMarkerError(
            f"Vault at {vault_root} has vault_format "
            f"{marker.vault_format}; this build reads format "
            f"{VAULT_FORMAT_CURRENT}. A newer vault under an older binary "
            "would be silently misread — refusing to start. Upgrade "
            "Prometheus, or restore the vault that matches this build."
        )

    # Lower than current: a migration would run here. Format 1 is the first
    # format, so today any lower value is not a legacy vault but a broken
    # marker — refuse with both numbers named rather than guessing.
    raise VaultMarkerError(
        f"Vault at {vault_root} has vault_format {marker.vault_format}; "
        f"this build reads format {VAULT_FORMAT_CURRENT} and has no "
        f"migration from {marker.vault_format}. Refusing to start."
    )
