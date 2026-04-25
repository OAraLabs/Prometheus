"""Backward-compatibility shim — DangerousCodeScanner moved to ``prometheus.security``.

The scanner was originally introduced under ``prometheus.symbiote`` during
the GRAFT-SYMBIOTE sprint. It was promoted to a shared component on
2026-04-25 because its applicability is broader than the SYMBIOTE harvest
pipeline. This module re-exports the public symbols so existing imports
keep working.

New code should import from ``prometheus.security.code_scanner`` directly.
"""

from prometheus.security.code_scanner import (  # noqa: F401
    DangerousCodeScanner,
    ScanFinding,
    ScanResult,
    ScanVerdict,
)

__all__ = [
    "DangerousCodeScanner",
    "ScanFinding",
    "ScanResult",
    "ScanVerdict",
]
