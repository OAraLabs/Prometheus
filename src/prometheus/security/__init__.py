"""Shared security utilities used across multiple Prometheus subsystems.

Exposes:
  - ``DangerousCodeScanner`` — AST-based static-analysis scanner.
    Has a ``scan_markdown_content`` method that extracts Python from
    fenced code blocks for skill-file scanning.
  - ``assert_path_under_roots`` / ``is_path_under_roots`` — write-boundary
    helpers used by autonomous components (MemoryExtractor and friends)
    that should not be allowed to write outside an allow-list.
  - ``install_log_redaction`` — arms gateway-token redaction on the
    logging handlers. Called from every entry point that configures
    logging, so a bot token cannot reach a log file or the journal.

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
from prometheus.security.log_redaction import (
    REDACTED,
    RedactingFilter,
    RedactingFormatter,
    install_log_redaction,
    redact_capture,
    redact_secrets,
)
from prometheus.security.path_guard import (
    assert_path_under_roots,
    is_path_under_roots,
)

__all__ = [
    "REDACTED",
    "DangerousCodeScanner",
    "RedactingFilter",
    "RedactingFormatter",
    "ScanFinding",
    "ScanResult",
    "ScanVerdict",
    "assert_path_under_roots",
    "install_log_redaction",
    "is_path_under_roots",
    "redact_capture",
    "redact_secrets",
]
