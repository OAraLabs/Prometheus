"""The CLI — and so every spawned worker — compacts the way the daemon does.

WHAT HAPPENED (2026-10-09, #704)
--------------------------------
Both workers in the incident were ``python -m prometheus`` children. The CLI
built its LoopContext with NO ContextCompactor and never read the box's
window, so a worker had none of what kept its parent conversation alive:

* no tier-2 compaction — a 53-round turn grew to 32,766 input tokens and the
  next request (33,094) was over the 32,768 window;
* no #356 pre-flight — with ``compactor=None`` the loop's window resolver
  returns ``(0, False)`` and the estimate returns 0 (``needed=0 window=0`` in
  the fallback's log line), so the over-window prompt was SENT and refused;
* only microcompaction, which trims old tool results and never the file
  bodies a coding worker writes.

THE RULE THESE TESTS PIN
------------------------
* The CLI builds the daemon's ContextCompactor (``compaction:`` config, the
  same switch) with the window MEASURED from the server — llama.cpp's
  ``/props`` n_ctx — not a configured ``effective_limit`` that outlived a
  model swap (72000 on the box where #704 reproduced).
* A long turn stays under that window.
* A turn compaction cannot get under is refused by the pre-flight, with the
  reason, before anything is sent — whether or not a fallback is configured.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests.fixtures.fake_llama_server import WORKER_MARK, FakeLlamaServer


def _argv(*args: str) -> list[str]:
    import sys

    return [sys.executable, "-m", "prometheus", *args]


@pytest.fixture
def work(tmp_path, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    w = tmp_path / "work"
    w.mkdir()
    return w


def _config(local_url: str, work: Path, model_extra: str = "") -> Path:
    cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "prometheus.yaml"
    path.write_text(
        "model:\n  provider: llama_cpp\n"
        f"  base_url: {local_url}\n  model: fake-local-27b\n"
        + model_extra +
        "security:\n  permission_mode: autonomous\n"
        f"  workspace_root: {work}\n"
        "compaction:\n  enabled: true\n"
        # The stale number from the box where #704 reproduced: the probe must win.
        "context:\n  effective_limit: 72000\n",
        encoding="utf-8",
    )
    return path


def test_a_long_cli_turn_stays_under_the_measured_window(work):
    with FakeLlamaServer(work_dir=work, writes_per_turn=16) as box:
        cfg = _config(box.url, work)
        subprocess.run(_argv("--config", str(cfg), "--once", f"{WORKER_MARK} Q. Write files."),
                       cwd=work, capture_output=True, text=True, timeout=180)
        rows = box.chat_rows()
    over = [r for r in rows if r["outcome"] == "400"]
    peak = max((r["est_prompt_tokens"] for r in rows), default=0)
    assert rows, "nothing reached the box"
    assert not over, (f"{len(over)} request(s) sent over n_ctx=32768 (peak estimate "
                      f"{peak}); the CLI did not compact")
    assert any(r["outcome"] == "summary" for r in rows), "no compaction summary was ever asked for"


@pytest.mark.parametrize("fallback", ["configured", "disabled"])
def test_a_turn_compaction_cannot_fit_is_refused_before_it_is_sent(work, fallback):
    # With no fallback target the window used to count as UNMEASURED, so the
    # pre-flight was off even for a local box (model.fallback.enabled: false).
    extra = "" if fallback == "configured" else "  fallback:\n    enabled: false\n"
    with FakeLlamaServer(work_dir=work, n_ctx=300) as box:
        cfg = _config(box.url, work, extra)
        r = subprocess.run(_argv("--config", str(cfg), "--once", f"{WORKER_MARK} R."),
                           cwd=work, capture_output=True, text=True, timeout=120)
        sent = [row for row in box.chat_rows() if row["has_tools"]]
    assert not sent, f"{len(sent)} over-window request(s) were sent; the pre-flight never ran"
    assert "too long" in r.stdout, r.stdout[-600:]
