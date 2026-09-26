"""What the model is shown about skills, and what listing them would cost.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=~/prometheus-deploy/src \\
        python3 seen_by_model.py <snapshot-dir> <qwen tokenizer.json> [--chat-only]

Renders the "# Available Skills" section with the daemon's own
``build_runtime_system_prompt`` + ``skills_for_prompt`` (bootstrap files off,
an empty cwd, memory stubbed — only the skills section is printed), prints the
``skill`` and ``tool_search`` schemas as advertised, the mini's deferred
``always_loaded`` list, and token counts with a Qwen3 tokenizer and with
Prometheus's own 4-chars-per-token estimator. Core-skill lines are shown with
names redacted unless ``--chat-only`` is given (the three core skills are the
repo's builtins, so nothing private is involved today either way).
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import load_catalog, registry_view  # noqa: E402

DEPLOY_CONFIG = Path.home() / "prometheus-deploy" / "config" / "prometheus.yaml"


def main() -> None:
    from tokenizers import Tokenizer

    from prometheus.context.prompt_assembler import build_runtime_system_prompt
    from prometheus.context.token_estimation import estimate_tokens
    from prometheus.skills.loader import skills_for_prompt
    from prometheus.tools.builtin.skill import SkillTool
    from prometheus.tools.tool_search import ToolSearchTool

    snap, tok_path = Path(sys.argv[1]), sys.argv[2]
    chat_only = "--chat-only" in sys.argv
    tok = Tokenizer.from_file(tok_path)

    def qt(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False).ids)

    skills = skills_for_prompt() or []
    with tempfile.TemporaryDirectory() as cwd:
        prompt = build_runtime_system_prompt(
            cwd=cwd,
            config={"bootstrap": {"load_soul": False, "load_agents": False},
                    "anatomy": {"include_in_system_prompt": False}},
            memory_content="(omitted)",
            skills=skills,
        )
    start = prompt.index("# Available Skills")
    end = prompt.find("\n\n# ", start + 1)
    section = prompt[start:end if end > 0 else None]
    shown = section
    if not chat_only:
        shown = "\n".join(
            ("- **<core skill>**: <description>" if line.startswith("- **") else line)
            for line in section.splitlines()
        )
    print("== rendered '# Available Skills' section (as the daemon builds it at boot)")
    print(shown)
    print(f"   tokens: qwen3 {qt(section)}, estimator {estimate_tokens(section)}")

    print("\n== tool schemas as advertised (to_api_schema)")
    for tool in (SkillTool(), ToolSearchTool()):
        schema = tool.to_api_schema()
        openai = {"type": "function", "function": {
            "name": schema["name"], "description": schema["description"],
            "parameters": schema["input_schema"]}}
        text = json.dumps(openai)
        print(json.dumps(schema, indent=1))
        print(f"   tokens (OpenAI-format JSON): qwen3 {qt(text)}, estimator {estimate_tokens(text)}")

    cfg = yaml.safe_load(DEPLOY_CONFIG.read_text())
    dl = ((cfg.get("tools") or {}).get("deferred_loading") or {})
    al = dl.get("always_loaded")
    print("\n== mini config tools.deferred_loading")
    print(f"   enabled: {dl.get('enabled')!r}; always_loaded ({len(al or [])}): {al}")
    print(f"   'skill' in always_loaded: {'skill' in (al or [])}; "
          f"'tool_search' in always_loaded: {'tool_search' in (al or [])}")

    print("\n== token cost of listing skills by name + description (the core-line format)")
    reg = registry_view(load_catalog(snap))
    line = lambda s: f"- **{s['name']}**: {s['description']}"  # noqa: E731
    header = "## Skills (load one with the skill tool when its description fits the task)"

    def cost(rows, cut=None):
        lines = [line({**s, "description": s["description"][:cut] if cut else s["description"]})
                 for s in rows]
        text = header + "\n" + "\n".join(lines)
        return qt(text), estimate_tokens(text)

    everything = sorted(reg.values(), key=lambda s: s["name"])
    for label, rows in (
        ("all served skills", everything),
        ("builtin + auto only", [s for s in everything if s["source"] in ("builtin", "auto")]),
        ("user skills only", [s for s in everything if s["source"] == "user"]),
    ):
        full, est = cost(rows)
        cut, _ = cost(rows, 120)
        print(f"   {label:22} n={len(rows):3}: qwen3 {full:5} (estimator {est:5}); "
              f"descriptions cut to 120 chars: {cut}")
    per = [qt(line(s)) for s in everything]
    print(f"   one line: median {statistics.median(per):.0f}, p90 {sorted(per)[int(len(per) * 0.9)]}, "
          f"max {max(per)} tokens; header {qt(header)}")


if __name__ == "__main__":
    main()
