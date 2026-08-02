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
