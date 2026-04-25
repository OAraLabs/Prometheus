"""SYMBIOTE — GitHub research, code assimilation, and graft pipeline.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT (Prometheus)

The SYMBIOTE package implements the Scout → Harvest → Graft pipeline:
  • Scout — search GitHub, license-gate candidates, rank by relevance
  • Harvest — clone candidate, scan with DangerousCodeScanner, extract modules
  • Graft — adapt modules into Prometheus with provenance headers + tests

MORPH (blue-green deployment) and BackupVault are Session B scope and are
NOT part of this package yet.
"""

from prometheus.symbiote.license_gate import (
    LicenseCheck,
    LicenseGate,
    LicenseVerdict,
)
from prometheus.symbiote.code_scanner import (
    DangerousCodeScanner,
    ScanFinding,
    ScanResult,
    ScanVerdict,
)

# Coordinator singleton — daemon sets this at startup so the agent-facing
# tools and Telegram /symbiote command can find the live instance.
_coordinator: object | None = None


def set_coordinator(coordinator: object | None) -> None:
    """Daemon calls this once after constructing the SymbioteCoordinator."""
    global _coordinator
    _coordinator = coordinator


def get_coordinator() -> object | None:
    """Returns the active SymbioteCoordinator, or None if SYMBIOTE is disabled."""
    return _coordinator


__all__ = [
    "LicenseGate",
    "LicenseCheck",
    "LicenseVerdict",
    "DangerousCodeScanner",
    "ScanFinding",
    "ScanResult",
    "ScanVerdict",
    "set_coordinator",
    "get_coordinator",
]
