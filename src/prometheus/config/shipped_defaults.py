"""Values a FRESH INSTALL ships with — imported by the readers AND the writer.

WHY THIS MODULE EXISTS (FIRSTLIGHT FL-2u)
-----------------------------------------
``config/prometheus.yaml.default`` is the documented shape of a config, but
it is **not installed**: ``pyproject.toml`` packages only ``src/prometheus``
and the template lives at the repo root, so a ``pip install`` has no copy of
it. Anything that needs a shipped default AT RUNTIME therefore cannot read
the template — it must read a constant, and that constant must live where
both the reader and the writer can import it without either depending on
the other.

That is this module: zero imports, no side effects, safe from
``cli/init.py`` (the stdlib-only fast setup path) and from the runtime
readers alike. ``tests/test_setup_advertised_defaults.py`` pins every
value here equal to the template, so "the constant" and "the documented
default" cannot drift.

THE RULE THIS ENCODES
---------------------
A key's fallback must be a WORKING value, not a degenerate one. FL-2u:
``setup`` learned to write ``tools.deferred_loading.always_loaded``, but an
install that UPGRADES keeps its old config — and the reader's fallback was
``[]``, which with deferred loading active means *advertise nothing*. The
fix is not to write config into an existing install; it is to make absence
safe (CROSS-CUTTING §5 — a property that cannot be violated beats a check
that must remember to run). See ``tests/test_absence_hostile_keys.py``.
"""

from __future__ import annotations

# tools.deferred_loading.always_loaded — the tool set a fresh install
# advertises. Absent from an upgraded config, and the old fallback of []
# meant the model was handed nothing it could call.
SHIPPED_ALWAYS_LOADED: tuple[str, ...] = (
    "bash", "task_create", "read_file", "write_file", "edit_file",
    "grep", "glob", "tool_search", "web_search", "web_fetch", "memory",
)

# security.workspace_root — the directory file operations are confined to.
# Absent from an upgraded config, and every reader's fallback was None, which
# ``SecurityGate._within_workspace`` reads as "no boundary at all".
SHIPPED_WORKSPACE_ROOT: str = "~/.prometheus/workspace"

# gateway.media.allowed_*_types — inbound MIME allowlists on the Telegram
# surface, the one exposed to the public internet by design. Absent from a
# config predating PR #141, and the readers' fallback of [] means
# "no restriction" (``media_guard``'s ``if allowed and ...``), so the MIME
# check was skipped entirely.
#
# ⚠ Spelled as the SNIFFER emits, not as a human would write it: "audio/mp3"
# looks right and is wrong — real MP3s sniff as "audio/mpeg", and that one
# entry silently refused every MP3 (#140 → #141).
SHIPPED_ALLOWED_IMAGE_TYPES: tuple[str, ...] = (
    "image/jpeg", "image/png", "image/gif", "image/webp",
)
SHIPPED_ALLOWED_AUDIO_TYPES: tuple[str, ...] = (
    "audio/ogg", "audio/mpeg", "audio/wav",
)
SHIPPED_ALLOWED_DOCUMENT_TYPES: tuple[str, ...] = (
    "application/pdf", "text/plain", "text/markdown", "text/csv",
    "text/html", "text/javascript", "text/x-python", "text/x-shellscript",
    "text/typescript", "application/json", "application/sql",
    "application/toml", "application/xml", "application/x-yaml",
)

_SHIPPED_MEDIA_ALLOWLISTS: dict[str, tuple[str, ...]] = {
    "allowed_image_types": SHIPPED_ALLOWED_IMAGE_TYPES,
    "allowed_audio_types": SHIPPED_ALLOWED_AUDIO_TYPES,
    "allowed_document_types": SHIPPED_ALLOWED_DOCUMENT_TYPES,
}


# ---------------------------------------------------------------------------
# Resolvers — THE single reader for each absence-hostile key.
#
# One function per key, not a fallback repeated at each call site. Four
# separate ``sec.get("workspace_root")`` calls is the shape §1b warns about:
# "six edits and a seventh next quarter". ``tests/test_absence_hostile_keys``
# asserts nothing outside this module reads these keys directly.
# ---------------------------------------------------------------------------

def resolve_workspace_root(security_cfg: dict | None) -> str:
    """The confinement root for a security config section.

    Absent or blank -> :data:`SHIPPED_WORKSPACE_ROOT`. Never None: a None
    reaching ``SecurityGate`` disables confinement entirely, which is the
    defect this exists to close.

    ``SecurityGate(workspace_root=None)`` still means "no confinement" — that
    is a deliberate API choice used by 39 of its 43 construction sites (tests,
    and callers confined another way). What must never happen is a *config*
    that merely omits the key landing on it.
    """
    value = (security_cfg or {}).get("workspace_root")
    if isinstance(value, str) and value.strip():
        return value
    return SHIPPED_WORKSPACE_ROOT


def resolve_media_allowlist(media_cfg: dict | None, key: str) -> list[str]:
    """One inbound MIME allowlist for the gateway media section.

    * key ABSENT (or null/blank) -> the shipped allowlist.
    * key present as a LIST -> that list verbatim, **including ``[]``**.

    The empty-list case is load-bearing and deliberate: the template documents
    ``[]`` as "no restriction for that kind", and an operator who wrote it
    meant it. Absence is not that statement — it is a config written before
    the key existed. Collapsing the two with ``cfg.get(key) or []`` is exactly
    what left every pre-#141 install with no type filtering.

    A blank/``null`` value resolves to the SHIPPED list, not to "no
    restriction": ``allowed_image_types:`` with nothing after it is a
    half-written key, and the fail-closed reading is the safe one on a
    surface exposed to the public internet.
    """
    shipped = _SHIPPED_MEDIA_ALLOWLISTS[key]
    value = (media_cfg or {}).get(key)
    if isinstance(value, list):
        return list(value)
    return list(shipped)
