"""Structural entity validation for the memory extractor + wiki linker.

SPRINT MEMORY-1, 2c. A purely STRUCTURAL, domain-agnostic gate: it rejects
candidates that are not plausibly an entity in ANY domain — filenames/paths,
code identifiers, shell/code syntax, and over-long phrases/task-strings —
WITHOUT a content denylist. There is deliberately no hard-coded ``sed`` /
``uv`` / ``pytest`` list: that would bake one user's (coding) domain into
every install and be useless for a non-coding agent. Bare command words are
structurally indistinguishable from valid short names and are intentionally
left for the human to triage.

Rejections are appended to an inspectable quarantine log (never silently
dropped) so the gate is auditable and tunable. The thresholds below are
module constants on purpose — they are the per-install tuning surface.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from prometheus.config.paths import get_config_dir

log = logging.getLogger(__name__)

# --- Tunables (per-install tuning surface) ---------------------------------
MAX_ENTITY_WORDS = 6   # entity names are short; > this ⇒ a phrase / task string
MAX_ENTITY_CHARS = 80

# File extensions that mark a token as a filename. These are FILE extensions,
# NOT domain TLDs — ``.ai`` / ``.io`` / ``.com`` are intentionally absent so
# domains like ``claude.ai`` stay valid entities.
_FILE_EXT_RE = re.compile(
    r"\.(py|pyc|js|mjs|cjs|ts|tsx|jsx|md|markdown|txt|rst|json|ya?ml|toml|ini|cfg|"
    r"conf|sh|bash|zsh|fish|rs|go|c|cc|cpp|cxx|h|hpp|java|rb|php|pl|swift|kt|scala|"
    r"sql|csv|tsv|parquet|log|db|sqlite|srt|vtt|html?|css|scss|sass|xml|svg|lock|"
    r"gguf|safetensors|bin|png|jpe?g|gif|webp|pdf|zip|tar|gz|whl)$",
    re.IGNORECASE,
)

# A lowercase snake_case token (e.g. ``timestamp_ms``, ``ensure_text_type``).
# Mixed-case ``en_GB`` and uppercase ``COVID_19`` deliberately do NOT match.
_CODE_IDENT_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")

# Shell / code metacharacters. ``#``/``@``/``%``/``^``/``+``/``~`` are excluded
# so legitimate names like ``C#``, ``C++``, ``@handle`` survive.
_METACHAR_RE = re.compile(r"[|&;$<>`*(){}\[\]=\\]")


# --- Per-install allow-list (the rescue hatch) -----------------------------
# Ships EMPTY in the repo — no domain baked into a public checkout. The user
# seeds their own gitignored ``~/.prometheus/entity_allowlist.txt`` (one name
# per line; ``#`` comments allowed) with the handful of REAL entities that are
# structurally indistinguishable from junk (e.g. ``llama.cpp``). The structural
# gate drops aggressively; the quarantine log surfaces what dropped; the user
# allow-lists back the few genuine ones. Repo ships machinery, never contents.
_allowlist_cache: set[str] | None = None


def _default_allowlist_path() -> Path:
    return get_config_dir() / "entity_allowlist.txt"


def load_allowlist(path: Path | None = None) -> set[str]:
    """Load the per-install entity allow-list (lowercased). Absent → empty.

    Failures are logged, never raised — a missing/unreadable allow-list must
    not break extraction.
    """
    target = path or _default_allowlist_path()
    names: set[str] = set()
    try:
        if target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    names.add(s.lower())
    except OSError:
        log.warning("Could not read entity allow-list at %s", target, exc_info=True)
    return names


def get_allowlist() -> set[str]:
    """Process-cached allow-list. Call :func:`reset_allowlist_cache` after edits."""
    global _allowlist_cache
    if _allowlist_cache is None:
        _allowlist_cache = load_allowlist()
    return _allowlist_cache


def reset_allowlist_cache() -> None:
    """Drop the cached allow-list (for tests, or after editing the file)."""
    global _allowlist_cache
    _allowlist_cache = None


def classify_entity(name: str | None, *, allowlist: set[str] | None = None) -> str | None:
    """Return a rejection reason if *name* is structurally not a valid entity.

    Returns ``None`` when the candidate is acceptable. Deterministic — no
    network, no embeddings, no content denylist. Reasons (checked in order):
    ``empty``, ``too_long``, ``path``, ``filename``, ``code_syntax``,
    ``code_identifier``, ``phrase_too_long``.
    """
    if name is None:
        return "empty"
    s = name.strip()
    if not s:
        return "empty"
    al = allowlist if allowlist is not None else get_allowlist()
    if s.lower() in al:
        return None  # explicitly rescued by the per-install allow-list
    if len(s) > MAX_ENTITY_CHARS:
        return "too_long"
    if "/" in s or "\\" in s:
        return "path"
    if _FILE_EXT_RE.search(s):
        return "filename"
    if _METACHAR_RE.search(s):
        return "code_syntax"
    if _CODE_IDENT_RE.search(s):
        return "code_identifier"
    if len(s.split()) > MAX_ENTITY_WORDS:
        return "phrase_too_long"
    return None


def is_valid_entity(name: str | None) -> bool:
    """True iff *name* passes the structural entity gate."""
    return classify_entity(name) is None


def _default_quarantine_path() -> Path:
    return get_config_dir() / "memory-quarantine.log"


def quarantine(
    name: str,
    reason: str,
    *,
    context: str = "",
    path: Path | None = None,
) -> None:
    """Append a rejected entity candidate to the inspectable quarantine log.

    Tab-separated: ``<iso-utc>\\t<reason>\\t<name>\\t<context>``. Failure to
    write is logged loudly but never raises — quarantining must not break the
    extraction pass.
    """
    target = path or _default_quarantine_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"{ts}\t{reason}\t{name!r}"
        if context:
            line += f"\t{context}"
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        log.warning("Could not write entity quarantine log at %s", target, exc_info=True)
