"""Config drift guard — every live key must exist in prometheus.yaml.default.

WHY THIS EXISTS
---------------
On 2026-08-02 the live config carried ``web.ws_auth: false`` — a key **no code
in src/ has ever read**. The feature that would have read it was written
2026-06-15 and never merged. So the config asserted the opposite of live
behaviour for seven weeks: an operator reading it believed WS first-frame auth
was disabled while it was in fact enabled on every connection.

It was found by accident, while triaging an unrelated stash.

This guard catches that class *structurally*: a key present in the live config
but absent from ``prometheus.yaml.default`` is either a typo, a setting whose
feature never landed, or a knob someone removed from the code and forgot to
remove from the config. All three are drift, and all three are silent.

DIRECTION, AND WHAT THIS DELIBERATELY DOES NOT CHECK
----------------------------------------------------
This asserts **live ⊆ default**, one direction only.

The complementary check — every key in ``.default`` has a *reader* in ``src/`` —
is NOT here. A survey on 2026-08-02 found **26 such keys, 25 of them set in the
live config**, including the entire ``web_tools`` section (the live config says
``fetch_timeout_seconds: 30`` while ``web_fetch.py`` hardcodes ``timeout=20.0``)
and both ``gateway.rate_limits.*`` entries. Shipping that direction today would
require a 26-entry allowlist, and an allowlist that large *is* the hiding place
the guard exists to eliminate. It is queued as its own change, after triage.

There is deliberately **no allowlist here.** If a key legitimately belongs in
the live config, it belongs in ``.default`` too — documented, with its default
value. That is the whole point.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT = _REPO / "config" / "prometheus.yaml.default"
_LIVE = _REPO / "config" / "prometheus.yaml"


def _flatten(node: object, prefix: tuple[str, ...] = ()) -> set[str]:
    """Every dotted key path in a nested mapping, sections included."""
    keys: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            path = prefix + (str(k),)
            keys.add(".".join(path))
            keys |= _flatten(v, path)
    return keys


def _open_maps(node: object, prefix: tuple[str, ...] = ()) -> set[str]:
    """Paths the template declares as OPEN MAPS by giving them ``{}``.

    Some settings are keyed by names the user chooses — model names under
    ``context.model_overrides``, server names under ``mcp_servers``, language
    ids under ``lsp.servers``. A template cannot enumerate those, so requiring
    each live child to appear in ``.default`` would be incoherent, not strict.

    Writing ``model_overrides: {}`` in the template IS the declaration that the
    children are data rather than schema. This is deliberately NOT an allowlist:
    nothing is exempted by name in this file, and the exemption is visible in
    the template itself where a reviewer of the config will see it.
    """
    opens: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            path = prefix + (str(k),)
            if isinstance(v, dict) and not v:
                opens.add(".".join(path))
            else:
                opens |= _open_maps(v, path)
    return opens


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def test_default_config_parses():
    """The template must be valid YAML — everything else depends on it."""
    assert _load(_DEFAULT), f"{_DEFAULT} is empty or not a mapping"


def test_every_live_config_key_exists_in_the_default_template():
    """Live ⊆ default. A live-only key is drift, and drift is silent.

    Skipped when there is no live config (fresh clone, CI). It is a guard for
    a real deployment, not a property of the repo.
    """
    if not _LIVE.exists():
        pytest.skip("no live config/prometheus.yaml — nothing to compare")

    default = _load(_DEFAULT)
    default_keys = _flatten(default)
    open_maps = _open_maps(default)
    live_keys = _flatten(_load(_LIVE))

    def under_open_map(key: str) -> bool:
        return any(key.startswith(m + ".") for m in open_maps)

    orphans = sorted(k for k in live_keys - default_keys if not under_open_map(k))

    assert not orphans, (
        f"{len(orphans)} key(s) in config/prometheus.yaml do not exist in "
        f"config/prometheus.yaml.default.\n\n"
        f"Each is one of: (a) a setting whose feature never landed — the "
        f"ws_auth case, where the config asserted the opposite of live "
        f"behaviour for seven weeks; (b) a knob deleted from the code but not "
        f"from the config; (c) a typo, silently ignored.\n\n"
        f"Fix by adding it to the template with its default and a comment, or "
        f"by deleting it from the live config. Do NOT bulk-add to make this "
        f"pass — that ratifies the drift this guard exists to catch.\n\n  "
        + "\n  ".join(orphans)
    )


# ---------------------------------------------------------------------------
# Reader direction: every template key should be read by some code.
#
# This is the complementary half of the check above, and it ships as a RATCHET
# rather than a plain assertion, because a plain assertion would need a
# 35-entry allowlist on day one — and an allowlist that large is the hiding
# place the guard exists to remove.
#
# The register below is a DEBT LIST, not an exemption list. The difference is
# that it is enforced in BOTH directions:
#
#   * a key with no reader that is NOT registered  -> FAIL (new drift)
#   * a registered key that HAS gained a reader    -> FAIL (register is stale)
#
# So it cannot grow quietly, and it cannot rot: fixing a key forces its removal
# from the list, and the count only goes down. Each entry carries a disposition.
# ---------------------------------------------------------------------------

# Scanned for readers across src/, scripts/ and tests/ — a config key consumed
# by a script is legitimately consumed. (Checked 2026-08-02: none of the below
# are read anywhere in the repo, so the scope is not the reason they are here.)
_READER_ROOTS = ("src", "scripts", "tests")

# key -> disposition. WIRE = build the control; DELETE = drop the key;
# DECIDE = needs a product call. Nothing may sit here without one.
KNOWN_UNREAD: dict[str, str] = {
    # (The nine gateway.rate_limits.* / gateway.media.* entries were removed
    # when PR 4b built the controls the config had been claiming. The register
    # shrank, which is the only direction it may move.)
    # ── WIRE: the config lies about live behaviour.
    "web_tools": "WIRE — section; web_fetch hardcodes timeout=20.0",
    "web_tools.fetch_timeout_seconds": "WIRE — config says 30, code hardcodes 20.0",
    "web_tools.fetch_max_chars": "WIRE — tool uses its input-model Field default",
    "web_tools.search_max_results": "WIRE — tool uses its input-model Field default",
    "web_tools.download_dir": "DECIDE — no downloader reads it",
    "web_tools.download_max_mb": "DECIDE — no downloader reads it",
    "web_tools.youtube_transcript_language": "DECIDE — transcript path ignores it",
    # ── DECIDE: knobs that may be abandoned rather than pending.
    "symbiote.language_default": "DECIDE — symbiote may be dormant",
    "symbiote.min_stars_default": "DECIDE — symbiote may be dormant",
    "symbiote.morph.auto_rollback": "DECIDE — symbiote may be dormant",
    "symbiote.backup.pre_graft_backup": "DECIDE — symbiote may be dormant",
    "printing_press.auto_suggest": "DECIDE",
    "profiles.custom_dir": "DECIDE — profiles load from a fixed dir",
    "whisper.device": "DECIDE — engine picks its own device",
    "learning.auto_skill_creation": "DECIDE — SkillCreator has its own gate",
    "learning.curator_telegram_summary": "DECIDE",
    "sentinel.idle_threshold_minutes": "DECIDE — observer uses its own constant",
    "security.audit.retention_days": "DECIDE — no pruner exists",
    "evals.skip_network_tasks": "DECIDE — runner does not branch on it",
    "tools.deferred_loading.search_mcp": "DECIDE — deferred loading is off",
    "tools.deferred_loading.mcp_always_deferred": "DECIDE — deferred loading is off",
    "gateway.heartbeat_interval": "DECIDE — heartbeat uses its own interval",
    "infrastructure.archive_enabled": "DECIDE — the archive lives in the voice stack, not here",
    # ── DOCUMENTED-ONLY: recorded so the topology is not lost. Values live
    # only in the local config; the template carries empty placeholders.
    "infrastructure.gpu_host": "DOCUMENTED-ONLY — local topology, empty in template",
    "infrastructure.mini_host": "DOCUMENTED-ONLY — local topology, empty in template",
    "infrastructure.mini_port": "DOCUMENTED-ONLY — local topology",
}


def _reader_blob() -> str:
    """Source of everything that could legitimately consume a config key.

    Excludes THIS file. The register below quotes the very key names it tracks,
    so scanning it would make single-segment entries look like their own
    readers — ``"web_tools"`` in KNOWN_UNREAD matched the reader pattern and
    the ratchet reported the register stale. The register is bookkeeping, not
    a consumer.
    """
    self_path = Path(__file__).resolve()
    parts: list[str] = []
    for root in _READER_ROOTS:
        for py in (_REPO / root).rglob("*.py"):
            if py.resolve() == self_path:
                continue
            parts.append(py.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _keys_without_readers() -> set[str]:
    import re as _re

    blob = _reader_blob()
    unread: set[str] = set()

    def walk(node: object, prefix: tuple[str, ...] = ()) -> None:
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            path = prefix + (str(k),)
            if not _re.search(rf'["\']{_re.escape(str(k))}["\']', blob):
                unread.add(".".join(path))
            walk(v, path)

    walk(_load(_DEFAULT))
    return unread


def test_no_new_config_key_without_a_reader():
    """A template key nothing reads is a promise the code does not keep."""
    unread = _keys_without_readers()
    new = sorted(unread - set(KNOWN_UNREAD))
    assert not new, (
        f"{len(new)} config key(s) in prometheus.yaml.default have no reader in "
        f"{'/, '.join(_READER_ROOTS)}/. A key nothing reads is a promise the "
        f"code does not keep — web.ws_auth sat inert for seven weeks asserting "
        f"the opposite of live behaviour.\n\nEither wire it, delete it, or add "
        f"it to KNOWN_UNREAD with a disposition.\n\n  " + "\n  ".join(new)
    )


def test_known_unread_register_is_not_stale():
    """The ratchet: a registered key that gained a reader must be de-registered.

    Without this the register would be an allowlist — write once, hide forever.
    With it the list can only shrink, and fixing a key forces the bookkeeping.
    """
    unread = _keys_without_readers()
    now_read = sorted(set(KNOWN_UNREAD) - unread)
    assert not now_read, (
        f"{len(now_read)} key(s) in KNOWN_UNREAD now HAVE readers. Remove them "
        f"from the register — it is a shrinking debt list, not an allowlist.\n\n  "
        + "\n  ".join(f"{k}  ({KNOWN_UNREAD[k]})" for k in now_read)
    )


def test_every_registered_key_carries_a_disposition():
    """No silent entries. An unexplained entry is the next hiding place."""
    valid = ("WIRE", "DELETE", "DECIDE", "DOCUMENTED-ONLY")
    bad = sorted(
        k for k, v in KNOWN_UNREAD.items()
        if not v.strip() or not v.strip().startswith(valid)
    )
    assert not bad, (
        f"KNOWN_UNREAD entries must start with one of {valid}:\n  "
        + "\n  ".join(bad)
    )
