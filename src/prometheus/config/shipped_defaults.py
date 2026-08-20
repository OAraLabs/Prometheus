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

# security.workspace_root — where write_file/edit_file may write without
# asking. NOT confinement: bash is gated on its command string, not on the
# paths a command touches. See config/prometheus.yaml.default.
# Absent from an upgraded config, and every reader's fallback was None, which
# ``SecurityGate._within_workspace`` reads as "no boundary at all".
SHIPPED_WORKSPACE_ROOT: str = "~/.prometheus/workspace"

#: Tool-call ceiling per user message, local (llama.cpp) routing.
#:
#: RAISED 50 -> 500 (LONGHAUL-1b) once the progress-aware repeat detector
#: shipped (#263). Round count alone cannot tell a long productive run from a
#: stuck one, so the flat cap was killing real work: of 38 recorded
#: ``max_iterations_hit`` events, 24 show near-total tool-signature diversity
#: (0.93-1.00) -- agents covering ground, cut off mid-task. 22 of those 38 came
#: AFTER the earlier 25 -> 50 raise, so 50 was still binding.
#:
#: The loops the cap used to contain are now contained far earlier and on the
#: right signal: the worst recorded case, ``bash: echo "Paris"`` issued 15
#: times, trips the detector at occurrence 3.
#:
#: THIS NUMBER IS NOT THE SAFETY MECHANISM. Progress is. The cap is only a
#: backstop for the shape the detector cannot see -- a flail with DIFFERENT
#: arguments every round (see _ProgressRepeatDetector's blind-spot note). Do
#: not read a high value here as permission to remove it.
SHIPPED_MAX_TOOL_ITERATIONS: int = 500

#: Tool-call ceiling for cloud routing (tier=off).
#:
#: ⚠ THIS IS NOW *BELOW* THE LOCAL CAP, INVERTING THE HISTORICAL ORDERING, AND
#: THAT IS DELIBERATE (Will's call, LONGHAUL-1b). Cloud sat ABOVE local for the
#: whole prior life of the key -- 50 vs 25 -- on the rationale that Claude plans
#: longer multi-step sequences than local models do. The ordering flipped
#: because the constraint that matters changed: cloud rounds cost money per
#: call, local rounds cost only wall-clock on hardware already paid for, so the
#: budget the operator actually wants to be generous with is the local one.
#:
#: ⚠ KNOWN RISK, RECORDED RATHER THAN MITIGATED: this hands the LARGEST budget
#: to the path most prone to the one failure the progress detector cannot see.
#: The detector keys on the exact (name, input) signature, so a flail with
#: DIFFERENT arguments every round is invisible to it, and a weaker local model
#: is likelier to produce exactly that shape than Claude is. A varied-argument
#: runaway on local can now burn 500 rounds where it previously burned 50.
#: If unattended local runs start costing hours rather than minutes, THIS
#: number is the first thing to look at -- not the detector.
SHIPPED_MAX_TOOL_ITERATIONS_CLOUD: int = 100

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

# gateway.telegram_enabled — whether the Telegram gateway starts at all.
#
# The behaviour site (daemon.py) defaulted this to True while the template,
# both setup-wizard display surfaces and every sibling gateway said False.
# Absence therefore meant "start the public gateway", and it only took a token
# in the environment to trigger it.
#
# ⚠ It never failed alone. `allowed_chat_ids` is the adjacent key in the same
# section, and empty/absent there meant "allow EVERY chat" — so the config
# that omitted one omitted the other, and the compound outcome was a bot live
# to anyone who found it while the status panel reported it off. They are
# fixed together because they fail together.
SHIPPED_TELEGRAM_ENABLED: bool = False

# security.denied_paths — the file boundary every path gate resolves against.
#
# ⚠ THE ABSENT KEY MEANT NO BOUNDARY AT ALL. `checker.py` did
# `[... for p in (denied_paths or [])]`, so a config omitting the key produced
# an EMPTY list and `read_file ~/.ssh/id_rsa` was permitted. Verified by
# outcome, not inferred: with the key present both `~/.ssh/id_rsa` and
# `/etc/shadow` are denied; with it absent, or with an empty `security:`
# section, both are ALLOWED.
#
# That mattered more than a single key normally would, because every path gate
# merged this week — grep/glob roots (#214), cron cwd (#215), task watch_dir
# (#216) — resolves against this list. The gating work was correct and rested
# on a key that need not exist.
#
# The command side never had this problem, and the reason is the model for the
# fix: `checker._ALWAYS_BLOCKED_PATTERNS` is a hardcoded floor applied BEFORE
# the config list, so `rm -rf /` is refused whether or not anyone wrote
# `denied_commands`. Paths now have both halves — this resolver for absence,
# and `checker._ALWAYS_DENIED_PATHS` as the structural floor beneath any
# config at all.
# ⚠ The three ``~``-relative entries covered ONLY the daemon user's home —
# `~` expands to whoever the process runs as, so another user's keys and env
# files were outside the boundary. All three are now home-agnostic globs.
# `/etc`, `/sys` and `/boot` are absolute and never had the gap.
#
# `.ssh` and `.gnupg` ALSO sit in `checker._ALWAYS_DENIED_PATHS`, so they hold
# beneath any config. `.config/*/*env` stays policy-only — env files are
# credential-bearing but an operator may have a legitimate reason to read one,
# and the floor should stay small enough to deserve being unoverridable.
SHIPPED_DENIED_PATHS: tuple[str, ...] = (
    "/etc", "/sys", "/boot", "/*/.ssh", "/*/.gnupg", "/*/.config/*/*env",
)


def resolve_denied_paths(security_cfg: dict | None) -> list[str]:
    """The denied-path list for a security config section.

    * key ABSENT (or null) -> :data:`SHIPPED_DENIED_PATHS`.
    * key present as a LIST -> that list verbatim, **including ``[]``**.

    The empty-list case is honoured deliberately, and it is safe here in a way
    it would not have been before: `checker._ALWAYS_DENIED_PATHS` still denies
    the credential locations regardless. So an operator who writes
    ``denied_paths: []`` gets what they asked for — no *policy* denials — and
    still cannot hand an agent their private keys.

    Absence is NOT that statement. It is a config written before the key
    existed, or trimmed by hand, and collapsing the two with
    ``cfg.get(k) or []`` is exactly what made the boundary vanish.
    """
    value = (security_cfg or {}).get("denied_paths")
    if isinstance(value, list):
        return [str(v) for v in value]
    return list(SHIPPED_DENIED_PATHS)


def resolve_telegram_enabled(gateway_cfg: dict | None) -> bool:
    """Whether the Telegram gateway may start.

    Absent -> :data:`SHIPPED_TELEGRAM_ENABLED` (False). Every writer of a
    config writes this key explicitly (``setup_wizard``, ``cli/init``,
    ``web/setup_server``, ``cli/migrate``), so an absent key means a
    hand-written or hand-trimmed config — not an operator's decision to run a
    public gateway.
    """
    value = (gateway_cfg or {}).get("telegram_enabled")
    if value is None:
        return SHIPPED_TELEGRAM_ENABLED
    return bool(value)


def resolve_allowed_chat_ids(gateway_cfg: dict | None) -> list[int]:
    """The chats permitted to drive the agent over Telegram.

    Returns the configured ids, or an EMPTY list — and empty is refused by the
    caller rather than treated as "no restriction". ``daemon`` will not start
    the gateway on an empty result, and ``PlatformConfig.chat_allowed`` denies
    on empty, so the two layers disagree about nothing.

    ⚠ THIS DELIBERATELY DIVERGES FROM ``resolve_media_allowlist`` ABOVE, and
    the divergence is the point. There, an explicit ``[]`` is honoured as the
    operator's opt-out, because "accept any file type" is a coherent thing to
    want. Here it is not: an unrestricted chat allowlist hands an agent with
    shell access to anyone who finds the bot, and there is no configuration in
    which that is the intent.

    The second reason is sharper. The shipped template writes
    ``allowed_chat_ids: []`` as its PLACEHOLDER. Honouring ``[]`` verbatim —
    the #141 rule — would mean every fresh install is open to the world the
    moment someone flips ``telegram_enabled: true``. The #141 analogy holds
    for media types and breaks exactly here, so absent and ``[]`` are treated
    identically and both are refused upstream.
    """
    value = (gateway_cfg or {}).get("allowed_chat_ids")
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for v in value:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            # A malformed entry is neither an allow nor a crash: dropping it
            # shrinks the allowlist, which is the fail-closed direction, and
            # an allowlist that ends up empty is refused by the caller
            # (CROSS-CUTTING §8 — a control must not fail by exception).
            continue
    return out


# ---------------------------------------------------------------------------
# Resolvers — THE single reader for each absence-hostile key.
#
# One function per key, not a fallback repeated at each call site. Four
# separate ``sec.get("workspace_root")`` calls is the shape §1b warns about:
# "six edits and a seventh next quarter". ``tests/test_absence_hostile_keys``
# asserts nothing outside this module reads these keys directly.
# ---------------------------------------------------------------------------

def resolve_max_tool_iterations(model_cfg: dict | None) -> int:
    """The local tool-call ceiling for a ``model:`` config section.

    ONE source of truth on purpose. This value was previously written out at
    eight separate sites (four code defaults of 25, the LoopContext dataclass,
    two AgentLoop signatures, the shipped template) and had ALREADY drifted --
    live ran 50 against a template of 25, a divergence
    ``docs/sprints/SPRINT-CONSENT.md`` names explicitly because the config
    drift guard checks key PRESENCE and cannot see a value mismatch. Raising
    the number by editing eight sites would have guaranteed a ninth.

    Absent, blank or non-integer -> :data:`SHIPPED_MAX_TOOL_ITERATIONS`.
    """
    return _positive_int(
        (model_cfg or {}).get("max_tool_iterations"), SHIPPED_MAX_TOOL_ITERATIONS
    )


def resolve_max_tool_iterations_cloud(model_cfg: dict | None) -> int:
    """The cloud tool-call ceiling. See :func:`resolve_max_tool_iterations`."""
    return _positive_int(
        (model_cfg or {}).get("max_tool_iterations_cloud"),
        SHIPPED_MAX_TOOL_ITERATIONS_CLOUD,
    )


def _positive_int(value: object, fallback: int) -> int:
    """Coerce a config value to a POSITIVE int, else the shipped default.

    A zero or negative ceiling would halt every turn on its first tool batch;
    a string "100" is what a hand-edited YAML actually produces. Both are
    treated as "not configured" rather than obeyed.
    """
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return n if n > 0 else fallback


def resolve_workspace_root(security_cfg: dict | None) -> str | list[str]:
    """The workspace root(s) for a security config section.

    Names where write_file/edit_file may write unprompted — not a
    filesystem confinement; see the module comment above.

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
    if isinstance(value, list):
        roots = [v for v in value if isinstance(v, str) and v.strip()]
        if roots:
            return roots
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
