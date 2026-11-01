"""Tests for scripts/distill/disk_preflight.py: the disk check every 4090 job runs before it starts."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_F = Path(__file__).resolve().parent.parent / "scripts" / "distill" / "disk_preflight.py"


def _load() -> Any:
    name = "distill_disk_preflight"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _F)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DP = _load()
GIB = DP.GIB


def test_items_parse_with_and_without_a_path() -> None:
    assert DP.parse_item("BF16:16.7:/home/t/a:b.gguf") == DP.Item("BF16", 16.7, "/home/t/a:b.gguf")
    assert DP.parse_item("logs:1") == DP.Item("logs", 1.0, None)
    for bad in ("nolabel", ":3", "x:abc", "x:-1"):
        with pytest.raises(ValueError):
            DP.parse_item(bad)


def test_only_what_a_resumed_job_still_has_to_write_counts(tmp_path: Path) -> None:
    """The 2026-11-01 failure: the merged weights and the BF16 GGUF were already on the 4090, yet the
    fixed check counted them again (37 GB free, 'needs ~60')."""
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "a.safetensors").write_bytes(b"x" * 300)
    (merged / "b.safetensors").write_bytes(b"x" * 700)
    sizes = {str(merged): 1000, str(tmp_path / "bf16.gguf"): 0}
    items = [DP.Item("merged", 1000 / GIB, str(merged)), DP.Item("BF16", 2.0, str(tmp_path / "bf16.gguf")),
             DP.Item("Q4_K_M", 0.5)]
    need, lines = DP.plan(items, need_gib=1.0, floor_gib=10.0, size_of=lambda p: sizes.get(p, 0))
    assert need == pytest.approx(1.0 + 0 + 2.0 + 0.5 + 10.0)
    assert lines[0] == "fixed 1.0" and lines[-1] == "floor 10.0"
    assert lines[1].startswith("merged 0.0 of") and "present" in lines[1]
    # the real sizes on disk, directories summed, symlinks not followed
    (tmp_path / "view").mkdir()
    (tmp_path / "view" / "link").symlink_to(merged / "b.safetensors")
    assert DP.path_bytes(str(merged)) == 1000
    assert DP.path_bytes(str(tmp_path / "view")) < 700
    assert DP.path_bytes(str(tmp_path / "missing")) == 0


def test_a_partly_written_output_counts_for_its_rest() -> None:
    it = DP.Item("BF16", 16.0, "/x")
    assert DP.remaining_gib(it, lambda p: int(6 * GIB)) == pytest.approx(10.0)
    assert DP.remaining_gib(it, lambda p: int(20 * GIB)) == 0.0


def test_refusal_text_is_at_most_five_lines_and_says_what_to_do() -> None:
    v = DP.verdict("Phase 6 run", 28.0, 60.0, ["fixed 1.0", "BF16 0.0 of 17.4 (present: 17.4)", "floor 10.0"])
    text = DP.refusal_text(v, host="4090", path="~", detail="d" * 400)
    lines = text.splitlines()
    assert len(lines) <= 5 and lines[0].startswith("4090: Phase 6 run NOT started: 28.0 GiB free")
    assert "60.0 GiB" in lines[0] and "Plan (GiB)" in lines[1] and "Nothing was changed" in lines[-1]
    assert all(len(ln) <= 310 for ln in lines)


def _notify(tmp_path: Path, rc: int = 0) -> Path:
    out = tmp_path / "sent.txt"
    n = tmp_path / "notify.sh"
    n.write_text(f"#!/bin/sh\n{{ echo \"--- $1\"; cat; }} >> {out}\nexit {rc}\n")
    n.chmod(0o755)
    return n


def test_cli_refuses_with_exit_28_logs_and_telegrams_once(tmp_path: Path, capsys: Any) -> None:
    n = _notify(tmp_path)
    log = tmp_path / "logs" / "disk-preflight.log"
    argv = ["check", "--job", "teacher serve", "--need-gib", "1e6", "--path", str(tmp_path),
            "--notify", str(n), "--log", str(log)]
    assert DP.main(argv) == DP.EXIT_REFUSED
    assert "disk REFUSED: teacher serve needs" in capsys.readouterr().out
    sent = (tmp_path / "sent.txt").read_text()
    assert sent.startswith("--- 5\n") and "teacher serve NOT started" in sent
    rec = json.loads(log.read_text().splitlines()[-1])
    assert rec["ok"] is False and rec["job"] == "teacher serve" and rec["note"] == "telegram sent"
    # a retry minutes later: logged again, not Telegrammed again
    assert DP.main(argv) == DP.EXIT_REFUSED
    assert (tmp_path / "sent.txt").read_text() == sent
    assert json.loads(log.read_text().splitlines()[-1])["note"] == "telegram recent"


def test_cli_ok_logs_without_a_telegram(tmp_path: Path) -> None:
    n = _notify(tmp_path)
    log = tmp_path / "pf.log"
    assert DP.main(["check", "--job", "x", "--need-gib", "0", "--floor-gib", "0", "--path", str(tmp_path),
                    "--notify", str(n), "--log", str(log), "--item", f"out:0.001:{tmp_path / 'o'}"]) == 0
    assert not (tmp_path / "sent.txt").exists()
    assert json.loads(log.read_text())["ok"] is True


def test_no_notify_and_bad_usage(tmp_path: Path) -> None:
    n = _notify(tmp_path)
    assert DP.main(["check", "--job", "x", "--need-gib", "1e6", "--path", str(tmp_path), "--notify", str(n),
                    "--log", str(tmp_path / "l.log"), "--no-notify"]) == DP.EXIT_REFUSED
    assert not (tmp_path / "sent.txt").exists()
    assert DP.main(["check", "--job", "x", "--item", "bad"]) == 2
    assert DP.main(["check", "--job", "x", "--need-gib", "-1"]) == 2


def test_a_failed_telegram_is_tried_again_next_time(tmp_path: Path) -> None:
    n = _notify(tmp_path, rc=1)
    log = str(tmp_path / "l.log")
    assert DP.send("a\nb", str(n), job="j", log=log) == "failed"
    assert DP.send("a\nb", str(n), job="j", log=log) == "failed"  # no stamp after a failure
    assert DP.send("a", str(tmp_path / "none.sh"), job="j", log=log) == "no-notify"
    ok = _notify(tmp_path / "ok") if (tmp_path / "ok").mkdir() is None else None
    assert DP.send("a", str(ok), job="j", log=log) == "sent"
    assert DP.send("a", str(ok), job="j", log=log, now=os.path.getmtime(DP._stamp(log, "j")) + DP.RESEND_S + 1) \
        == "sent"
