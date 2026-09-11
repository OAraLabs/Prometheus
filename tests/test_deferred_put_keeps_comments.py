"""`PUT /api/tools/deferred` must not wipe the config file's comments.

THE DEFECT
----------
The route persisted by round-tripping the whole file::

    on_disk = yaml.safe_load(fh)
    on_disk.setdefault("tools", {}).setdefault("deferred_loading", {})["enabled"] = v
    yaml.dump(on_disk, fh, default_flow_style=False, sort_keys=False)

`safe_load` returns plain dicts and lists. Comments are not part of that
representation, so they were not lost at dump time — they were already gone at
load time, and `yaml.dump` faithfully wrote back everything it had been given.

Measured against the shipped template, from one toggle of one boolean:

    comment lines  713 -> 0
    file bytes     61,504 -> 9,937

The existing `test_put_persists_surgically_without_leaking_runtime_secrets`
calls the same route and passed throughout, because its fixture config is built
with `yaml.dump` and therefore has no comments to lose. A guard written against
a fixture that cannot exhibit the defect measures nothing about it — which is
why the fixture here is the REAL shipped template, comments and all.

WHAT IS ASSERTED
----------------
Not "some comments survive" but "the file is byte-identical except the one
line that changed". Counting comments alone would pass a rewrite that preserved
comment COUNT while reflowing every value, reordering keys, or restyling
quotes — all of which a `safe_load`/`dump` round-trip does.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.context.dynamic_tools import DynamicToolLoader  # noqa: E402
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult  # noqa: E402
from prometheus.web.server import (  # noqa: E402
    _set_yaml_scalar_preserving_comments,
    create_app,
)
from pydantic import BaseModel  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "config" / "prometheus.yaml.default"


class _EmptyInput(BaseModel):
    pass


def _registry() -> ToolRegistry:
    reg = ToolRegistry()

    class _T(BaseTool):
        name = "bash"
        description = "t"
        input_model = _EmptyInput

        async def execute(self, arguments, context):  # noqa: ANN001
            return ToolResult(output="ok")

    reg.register(_T())
    return reg


def _client(config: dict):
    app = create_app(config)
    loader = DynamicToolLoader(
        _registry(), config.setdefault("tools", {}).setdefault("deferred_loading", {})
    )
    app.state.ws_bridge = SimpleNamespace(
        loop_context=SimpleNamespace(tool_loader=loader, adapter=SimpleNamespace(tier="off"))
    )
    return TestClient(app)


def _comments(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("#"))


# ── through the real endpoint, against the real template ────────────────────

def test_the_put_changes_exactly_one_line_of_the_real_template(tmp_path, monkeypatch):
    """The whole assertion, on the file this defect actually damages."""
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg = cfg_dir / "prometheus.yaml"
    original = TEMPLATE.read_text(encoding="utf-8")
    cfg.write_text(original, encoding="utf-8")

    assert _comments(original) > 500, (
        "the template fixture has almost no comments — this test cannot "
        "detect the defect it exists for"
    )

    client = _client(yaml.safe_load(original) or {})
    body = client.put("/api/tools/deferred", json={"enabled": True}).json()
    assert body["persisted"] is True, body

    updated = cfg.read_text(encoding="utf-8")

    assert _comments(updated) == _comments(original), (
        f"comment lines {_comments(original)} -> {_comments(updated)}"
    )

    changed = [
        line for line in difflib.unified_diff(
            original.splitlines(), updated.splitlines(), lineterm="", n=0
        )
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    assert len(changed) == 2, (  # one removed line, one added line
        f"the PUT rewrote {len(changed)} lines; it must change exactly one:\n"
        + "\n".join(changed[:40])
    )
    # `changed[n][0]` is the diff marker; the indentation follows it.
    assert changed[0][1:].strip() == "enabled: auto", changed[0]
    assert changed[1][1:].strip() == "enabled: true", changed[1]

    assert yaml.safe_load(updated)["tools"]["deferred_loading"]["enabled"] is True


def test_the_value_actually_round_trips_for_every_tri_state(tmp_path, monkeypatch):
    """Preserving the file is worthless if the setting stops persisting."""
    monkeypatch.chdir(tmp_path)
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg = cfg_dir / "prometheus.yaml"

    for sent, expected in ((True, True), (False, False), ("auto", "auto")):
        cfg.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
        client = _client(yaml.safe_load(cfg.read_text()) or {})
        body = client.put("/api/tools/deferred", json={"enabled": sent}).json()
        assert body["persisted"] is True, body
        got = yaml.safe_load(cfg.read_text())["tools"]["deferred_loading"]["enabled"]
        assert got == expected, f"sent {sent!r}, file says {got!r}"


# ── the writer itself ───────────────────────────────────────────────────────

def test_a_trailing_comment_on_the_edited_line_survives():
    text = "tools:\n  deferred_loading:\n    enabled: auto  # why it is auto\n"
    out = _set_yaml_scalar_preserving_comments(
        text, ["tools", "deferred_loading", "enabled"], "false"
    )
    assert out == "tools:\n  deferred_loading:\n    enabled: false  # why it is auto\n", out


def test_a_missing_key_is_appended_without_touching_what_exists():
    text = "# a leading comment\nmodel:\n  provider: llama_cpp  # keep me\n"
    out = _set_yaml_scalar_preserving_comments(
        text, ["tools", "deferred_loading", "enabled"], "true"
    )
    assert out.startswith(text), "existing content was modified while appending"
    assert yaml.safe_load(out)["tools"]["deferred_loading"]["enabled"] is True
    assert yaml.safe_load(out)["model"]["provider"] == "llama_cpp"
    assert "# keep me" in out and "# a leading comment" in out


def test_a_same_named_key_in_another_section_is_not_touched():
    """`enabled:` appears all over a real config. Only the one under the
    requested path may change."""
    text = (
        "web:\n  enabled: true\n"
        "tools:\n  deferred_loading:\n    enabled: auto\n"
        "push:\n  enabled: false\n"
    )
    out = _set_yaml_scalar_preserving_comments(
        text, ["tools", "deferred_loading", "enabled"], "false"
    )
    parsed = yaml.safe_load(out)
    assert parsed["tools"]["deferred_loading"]["enabled"] is False
    assert parsed["web"]["enabled"] is True, "a different section's key changed"
    assert parsed["push"]["enabled"] is False


def test_the_whole_template_is_byte_identical_apart_from_the_target():
    original = TEMPLATE.read_text(encoding="utf-8")
    out = _set_yaml_scalar_preserving_comments(
        original, ["tools", "deferred_loading", "enabled"], "false"
    )
    a, b = original.splitlines(), out.splitlines()
    assert len(a) == len(b), f"line count changed {len(a)} -> {len(b)}"
    differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(differing) == 1, f"{len(differing)} lines differ, expected 1"
    assert b[differing[0]].strip() == "enabled: false"
