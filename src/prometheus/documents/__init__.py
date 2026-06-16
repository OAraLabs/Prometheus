"""Documents Editor — confined, disk-backed read/save/edit service.

See :mod:`prometheus.documents.service`. ADDITIVE by construction: reuses the
coding-mode path-confinement (``ProcessSandbox.resolve``) and the existing
``CodeStrReplaceTool`` edit primitive against a GENERAL documents root, rather
than building a parallel edit-tool family.
"""

from __future__ import annotations

from prometheus.documents.service import (
    DocumentEditResult,
    DocumentEntry,
    DocumentsError,
    DocumentsService,
    SuggestedEdit,
)

__all__ = [
    "DocumentsService",
    "DocumentEntry",
    "DocumentEditResult",
    "SuggestedEdit",
    "DocumentsError",
]
