"""LicenseGate — hard block on incompatible licenses.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT

Prometheus is MIT-licensed. SYMBIOTE only grafts code with compatible
licenses. GPL/AGPL/SSPL/BUSL are viral or source-available restrictive —
BLOCKED, not warned. Unknown licenses are also BLOCKED (assume worst case).

Detection priority:
  1. GitHub API license field (SPDX from repo metadata)
  2. LICENSE / LICENSE.md / COPYING file content
  3. license field in package.json / setup.py / pyproject.toml
  4. SPDX header comments in individual source files
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

log = logging.getLogger(__name__)


class LicenseVerdict(str, Enum):
    """Outcome of a license check."""

    ALLOW = "allow"      # MIT, Apache-2.0, BSD, ISC, etc.
    WARN = "warn"        # LGPL, MPL — usable but flag obligations
    BLOCK = "block"      # GPL, AGPL, SSPL, BUSL — incompatible
    UNKNOWN = "unknown"  # No license detected (also treated as BLOCK by callers)


@dataclass
class LicenseCheck:
    """Result of a license inspection."""

    spdx_id: str | None
    verdict: LicenseVerdict
    source: str  # "github_api" | "license_file" | "package_metadata" | "spdx_header" | "none"
    obligations: list[str] = field(default_factory=list)
    raw_text: str | None = None


# ---------------------------------------------------------------------------
# License classification tables
# ---------------------------------------------------------------------------

_ALLOW_SPDX: frozenset[str] = frozenset({
    "MIT",
    "MIT-0",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unlicense",
    "0BSD",
    "CC0-1.0",
    "Zlib",
    "Python-2.0",
})

_WARN_SPDX: frozenset[str] = frozenset({
    "LGPL-2.1-only",
    "LGPL-2.1-or-later",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "MPL-2.0",
    "EPL-2.0",
    "CDDL-1.0",
    "CDDL-1.1",
})

_BLOCK_SPDX: frozenset[str] = frozenset({
    "GPL-2.0-only",
    "GPL-2.0-or-later",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",
    "SSPL-1.0",
    "BSL-1.1",
    "BUSL-1.1",
    "Elastic-2.0",
    "CC-BY-NC-4.0",
    "CC-BY-NC-SA-4.0",
})

_OBLIGATIONS: dict[str, list[str]] = {
    "MIT": ["Include copyright notice", "Include license text"],
    "MIT-0": ["No obligations"],
    "Apache-2.0": [
        "Include copyright notice",
        "Include license text",
        "State changes made",
        "Include NOTICE file if present",
    ],
    "BSD-2-Clause": ["Include copyright notice", "Include license text"],
    "BSD-3-Clause": [
        "Include copyright notice",
        "Include license text",
        "Do not endorse using contributor names",
    ],
    "ISC": ["Include copyright notice", "Include license text"],
    "Unlicense": ["No obligations"],
    "0BSD": ["No obligations"],
    "CC0-1.0": ["No obligations"],
    "LGPL-2.1-only": [
        "Include copyright notice",
        "Include license text",
        "Allow relinking",
        "Disclose LGPL-covered source",
    ],
    "LGPL-2.1-or-later": [
        "Include copyright notice",
        "Include license text",
        "Allow relinking",
        "Disclose LGPL-covered source",
    ],
    "LGPL-3.0-only": [
        "Include copyright notice",
        "Include license text",
        "Allow relinking",
        "Disclose LGPL-covered source",
    ],
    "LGPL-3.0-or-later": [
        "Include copyright notice",
        "Include license text",
        "Allow relinking",
        "Disclose LGPL-covered source",
    ],
    "MPL-2.0": [
        "Include copyright notice",
        "Include license text",
        "Disclose MPL-covered source files",
    ],
}


# Common short forms / aliases users may encounter.
_ALIASES: dict[str, str] = {
    "MIT LICENSE": "MIT",
    "THE MIT LICENSE": "MIT",
    "APACHE 2.0": "Apache-2.0",
    "APACHE-2.0": "Apache-2.0",
    "APACHE LICENSE 2.0": "Apache-2.0",
    "BSD-3": "BSD-3-Clause",
    "BSD 3-CLAUSE": "BSD-3-Clause",
    "BSD-2": "BSD-2-Clause",
    "BSD 2-CLAUSE": "BSD-2-Clause",
    "GPL-2": "GPL-2.0-only",
    "GPL2": "GPL-2.0-only",
    "GPL-3": "GPL-3.0-only",
    "GPL3": "GPL-3.0-only",
    "GPLV3": "GPL-3.0-only",
    "AGPL-3": "AGPL-3.0-only",
    "AGPL3": "AGPL-3.0-only",
    "AGPLV3": "AGPL-3.0-only",
    "LGPL-3": "LGPL-3.0-only",
    "LGPL3": "LGPL-3.0-only",
    "LGPL-2.1": "LGPL-2.1-only",
}


def _normalize_spdx(spdx: str | None) -> str | None:
    """Return the canonical SPDX ID for a license string, or None."""
    if not spdx:
        return None
    s = spdx.strip()
    if not s:
        return None
    upper = s.upper()
    if upper in _ALIASES:
        return _ALIASES[upper]
    # Match against allow/warn/block sets case-insensitively.
    for known in _ALLOW_SPDX | _WARN_SPDX | _BLOCK_SPDX:
        if known.upper() == upper:
            return known
    return s  # return as-is so callers see what was detected


# ---------------------------------------------------------------------------
# LicenseGate
# ---------------------------------------------------------------------------


class LicenseGate:
    """Hard gate on license compatibility for SYMBIOTE harvests.

    GPL/AGPL/SSPL/BUSL are BLOCKED — not warned. Unknown licenses are also
    BLOCKED (callers should treat ``UNKNOWN`` as ``BLOCK``).
    """

    # Re-exported for easy override / inspection in tests.
    ALLOW: frozenset[str] = _ALLOW_SPDX
    WARN: frozenset[str] = _WARN_SPDX
    BLOCK: frozenset[str] = _BLOCK_SPDX

    def __init__(
        self,
        *,
        extra_allow: list[str] | None = None,
        extra_block: list[str] | None = None,
    ) -> None:
        self._extra_allow: frozenset[str] = frozenset(extra_allow or [])
        self._extra_block: frozenset[str] = frozenset(extra_block or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_spdx(self, spdx: str | None) -> LicenseVerdict:
        """Map an SPDX identifier to a verdict."""
        if not spdx:
            return LicenseVerdict.UNKNOWN
        normalized = _normalize_spdx(spdx)
        if normalized in self._extra_block or normalized in self.BLOCK:
            return LicenseVerdict.BLOCK
        if normalized in self._extra_allow or normalized in self.ALLOW:
            return LicenseVerdict.ALLOW
        if normalized in self.WARN:
            return LicenseVerdict.WARN
        return LicenseVerdict.UNKNOWN

    def check(self, repo_data: dict | None) -> LicenseCheck:
        """Check a license from GitHub API repo metadata.

        Expects ``repo_data`` shaped like the GitHub Search API response,
        i.e. ``{"license": {"spdx_id": "MIT"}, ...}`` or ``{"license": None}``.
        """
        if not repo_data:
            return LicenseCheck(
                spdx_id=None,
                verdict=LicenseVerdict.UNKNOWN,
                source="none",
                obligations=[],
                raw_text=None,
            )
        license_obj = repo_data.get("license") if isinstance(repo_data, dict) else None
        spdx = None
        if isinstance(license_obj, dict):
            spdx = license_obj.get("spdx_id") or license_obj.get("key")
        elif isinstance(license_obj, str):
            spdx = license_obj

        normalized = _normalize_spdx(spdx)
        verdict = self.classify_spdx(normalized)
        return LicenseCheck(
            spdx_id=normalized,
            verdict=verdict,
            source="github_api" if license_obj else "none",
            obligations=self.format_obligations(normalized) if verdict != LicenseVerdict.UNKNOWN else [],
            raw_text=None,
        )

    def check_file(self, file_path: Path) -> LicenseCheck:
        """Scan a LICENSE / COPYING file and return the verdict.

        Best-effort: looks for SPDX-License-Identifier headers first, then
        falls back to keyword matching (MIT, Apache, GPL, etc.).
        """
        if not file_path.exists() or not file_path.is_file():
            return LicenseCheck(
                spdx_id=None,
                verdict=LicenseVerdict.UNKNOWN,
                source="none",
                obligations=[],
                raw_text=None,
            )
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return LicenseCheck(
                spdx_id=None,
                verdict=LicenseVerdict.UNKNOWN,
                source="license_file",
                obligations=[],
                raw_text=None,
            )

        # SPDX header takes priority.
        spdx_match = re.search(
            r"SPDX-License-Identifier:\s*([A-Za-z0-9\.\-\+]+)",
            text,
        )
        if spdx_match:
            spdx = _normalize_spdx(spdx_match.group(1))
            verdict = self.classify_spdx(spdx)
            return LicenseCheck(
                spdx_id=spdx,
                verdict=verdict,
                source="spdx_header",
                obligations=self.format_obligations(spdx),
                raw_text=text[:8000],
            )

        spdx = self._guess_spdx_from_text(text)
        verdict = self.classify_spdx(spdx)
        return LicenseCheck(
            spdx_id=spdx,
            verdict=verdict,
            source="license_file",
            obligations=self.format_obligations(spdx) if spdx else [],
            raw_text=text[:8000],
        )

    def format_obligations(self, spdx_id: str | None) -> list[str]:
        """Return the obligations list for a given SPDX ID."""
        if not spdx_id:
            return []
        return list(_OBLIGATIONS.get(spdx_id, []))

    # ------------------------------------------------------------------
    # Internal heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_spdx_from_text(text: str) -> str | None:
        """Heuristic SPDX detection from license file content."""
        upper = text.upper()
        # Order matters — more specific licenses first.
        markers: list[tuple[str, str]] = [
            ("AGPL-3.0-only", "GNU AFFERO GENERAL PUBLIC LICENSE"),
            ("LGPL-3.0-only", "GNU LESSER GENERAL PUBLIC LICENSE"),
            ("GPL-3.0-only", "GNU GENERAL PUBLIC LICENSE"),
            ("Apache-2.0", "APACHE LICENSE"),
            ("MIT", "MIT LICENSE"),
            ("MIT", "PERMISSION IS HEREBY GRANTED, FREE OF CHARGE"),
            ("BSD-3-Clause", "REDISTRIBUTION AND USE IN SOURCE AND BINARY FORMS"),
            ("ISC", "PERMISSION TO USE, COPY, MODIFY, AND/OR DISTRIBUTE"),
            ("MPL-2.0", "MOZILLA PUBLIC LICENSE"),
            ("BUSL-1.1", "BUSINESS SOURCE LICENSE"),
            ("SSPL-1.0", "SERVER SIDE PUBLIC LICENSE"),
            ("Unlicense", "THIS IS FREE AND UNENCUMBERED SOFTWARE"),
        ]
        for spdx, marker in markers:
            if marker in upper:
                return spdx
        return None
