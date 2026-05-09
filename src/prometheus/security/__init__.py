"""Shared security utilities used across multiple Prometheus subsystems.

Exposes:
  - ``DangerousCodeScanner`` — AST-based static-analysis scanner.
    Has a ``scan_markdown_content`` method that extracts Python from
    fenced code blocks for skill-file scanning.
  - ``assert_path_under_roots`` / ``is_path_under_roots`` — write-boundary
    helpers used by autonomous components (MemoryExtractor and friends)
    that should not be allowed to write outside an allow-list.

The DangerousCodeScanner was originally introduced under
``prometheus.symbiote`` for the GRAFT-SYMBIOTE sprint and promoted here
once GEPA and SkillRefiner needed to scan AI-generated skill variants
before promoting them.
"""

from prometheus.security.code_scanner import (
    DangerousCodeScanner,
    ScanFinding,
    ScanResult,
    ScanVerdict,
)
from prometheus.security.path_guard import (
    assert_path_under_roots,
    is_path_under_roots,
)

__all__ = [
    "DangerousCodeScanner",
    "ScanFinding",
    "ScanResult",
    "ScanVerdict",
    "assert_path_under_roots",
    "is_path_under_roots",
]
