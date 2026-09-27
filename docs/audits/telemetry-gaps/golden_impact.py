"""Which parity goldens hold rows each telemetry fix would change (the static half).

    python3 docs/audits/telemetry-gaps/golden_impact.py [tests/fixtures/parity]

The parity harness dumps every table of every store, so a fix changes a golden
exactly when a scenario writes a row the fix would write differently. This
scan reads the committed ``*.expected.json`` dumps and, per fix in
docs/audits/TELEMETRY-GAPS.md, lists the goldens holding such a row. It is a
prediction; the replays in §3 of the report are the evidence (they agree).
Runs on any checkout; prints scenario names and row counts only.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

FAILURE_TYPES = {"hook_blocked", "no_registry", "validation_failed", "unknown_tool", "template_markup",
                 "input_validation", "permission_denied", "tool_timeout", "tool_exception", "malformed_empty"}
DROPS_INPUT = {"permission_denied", "hook_blocked", "tool_timeout", "tool_exception", "unknown_tool",
               "no_registry"}


def rows(dump: dict, store_suffix: str, table: str) -> list[dict]:
    for path, store in dump["stores"].items():
        if path.endswith(store_suffix) and "sqlite" in store and table in store["sqlite"]:
            t = store["sqlite"][table]
            return [dict(zip(t["columns"], r)) for r in t["rows"]]
    return []


def tel(dump: dict, table: str) -> list[dict]:
    return rows(dump, ".prometheus/telemetry.db", table)


def loop_rounds(dump: dict) -> list[dict]:
    return [r for r in tel(dump, "subsystem_runs")
            if r["subsystem"] == "agent_loop" and r["operation"] == "loop_round"]


# fix -> (what it changes, predicate over one golden's dump returning the number of rows it changes)
FIXES: dict[str, tuple[str, Callable[[dict], int]]] = {
    "T1": ("failure-path tool_calls rows gain a session id",
           lambda d: sum(1 for r in tel(d, "tool_calls")
                         if r["error_type"] in FAILURE_TYPES and r["session_id"] is None)),
    "T2": ("lucky_guess markers leave tool_calls",
           lambda d: sum(1 for r in tel(d, "tool_calls") if r["error_type"] == "lucky_guess")),
    "T3": ("_loop_transition rows gain a session id",
           lambda d: sum(1 for r in tel(d, "tool_calls")
                         if r["tool_name"] == "_loop_transition" and r["session_id"] is None)),
    "T4": ("session-less agent_loop rows gain a session id",
           lambda d: sum(1 for r in tel(d, "subsystem_runs")
                         if r["subsystem"] == "agent_loop" and r["session_id"] is None)),
    "T5": ("microcompact rows change session", lambda d: sum(
        1 for r in tel(d, "subsystem_runs") if r["operation"] == "microcompact")),
    "T6": ("cache counts the parser already reads reach the row (Anthropic, OpenAI-compatible)",
           lambda d: sum(1 for r in loop_rounds(d) if "cloud" in C.provider_of(r["model"]))),
    "T7 (with T6)": ("llama.cpp rounds gain the cache count",
                     lambda d: sum(1 for r in loop_rounds(d) if C.provider_of(r["model"]) == "llama.cpp (local)")),
    "T8": ("Anthropic rounds' input_tokens becomes the whole prompt",
           lambda d: sum(1 for r in loop_rounds(d) if C.provider_of(r["model"]) == "anthropic (cloud)")),
    "T9": ("local rounds stamped 'unknown' become 'local'",
           lambda d: sum(1 for r in loop_rounds(d) if r["billing_mode"] == "unknown"
                         and C.provider_of(r["model"]).endswith("(local)"))),
    "T10": ("compactor rows gain tokens and a session",
            lambda d: sum(1 for r in tel(d, "subsystem_runs") if r["subsystem"] == "context_compactor")),
    "T11": ("a titled session or an LCM summary gains a row",
            lambda d: len(rows(d, "lcm.db", "session_titles")) + len(rows(d, "lcm.db", "lcm_summaries"))),
    "T12a": ("tool calls served by qwen/xai/Anthropic gain served_model",
             lambda d: sum(1 for r in tel(d, "tool_calls") if "cloud" in C.provider_of(r["model"])
                           and r["tool_name"] != "_loop_transition")),
    "T12b": ("tool calls served by Ollama gain served_model",
             lambda d: sum(1 for r in tel(d, "tool_calls") if C.provider_of(r["model"]) == "ollama (local)"
                           and r["tool_name"] != "_loop_transition" and r["served_model"] is None)),
    # T13 records what the provider reports, so alone it reaches only llama.cpp's
    # rounds (the one provider that reports it today); after T12 every round.
    "T13": ("a round's summary gains the served model (llama.cpp reports it today)",
            lambda d: sum(1 for r in loop_rounds(d) if C.provider_of(r["model"]) == "llama.cpp (local)")),
    "T13 after T12": ("... and every other provider's rounds once T12 lands", lambda d: len(loop_rounds(d))),
    "T14": ("a degraded round's tool rows change model/provider",
            lambda d: sum(1 for r in tel(d, "tool_calls") if r.get("served_model")
                          and "cloud" in C.provider_of(r["model"]) and C.is_path_model(r["served_model"]))),
    "T15": ("failure rows that dropped the call's input gain it",
            lambda d: sum(1 for r in tel(d, "tool_calls")
                          if r["error_type"] in DROPS_INPUT and r["parsed_tool_call"] is None)),
}


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/parity")
    goldens = {p.name.removesuffix(".expected.json"): json.loads(p.read_text())
               for p in sorted(root.glob("*.expected.json"))}
    print(f"{len(goldens)} goldens: {', '.join(goldens)}")
    for fix, (what, count) in FIXES.items():
        hit = {name: n for name, d in goldens.items() if (n := count(d))}
        verdict = f"CHANGES {len(hit)}: " + ", ".join(f"{k} ({v})" for k, v in hit.items()) if hit else "changes none"
        print(f"{fix:14} {what}\n{'':14} -> {verdict}")


if __name__ == "__main__":
    main()
