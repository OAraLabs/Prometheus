"""Retire the 2026-10-16 local workaround in a Prometheus config, as a targeted edit.

usage: retire_workaround.py <config> [--write]
Removes exactly:
  * the 27B's per-model output cap (its comment, key and max_output_tokens: 4096
    lines under context.model_overrides): 0.9.13's llama.cpp default is 4096;
  * "- agent" from tools.deferred_loading.always_loaded: under local_catalog
    "stubs" (the 0.9.13 default) the list is not consulted, and agent always
    carries its full schema.
Refuses unless each block occurs exactly once. Without --write, prints the diff.
"""
import difflib, sys

path, write = sys.argv[1], "--write" in sys.argv[2:]
text = open(path, encoding="utf-8").read()
lines = text.splitlines(keepends=True)

def find_block(pred_lines):
    hits = [i for i in range(len(lines) - len(pred_lines) + 1)
            if all(p(lines[i + j]) for j, p in enumerate(pred_lines))]
    if len(hits) != 1:
        sys.exit(f"refused: expected one match, found {len(hits)}")
    return hits[0]

cap = find_block([
    lambda l: l.strip().startswith("# 2026-10-16: 0.9.12's 8192 local output cap"),
    lambda l: l.strip().endswith('Qwen3.8-27B-UD-Q4_K_XL.gguf":') and l.startswith("    \""),
    lambda l: l.strip() == "max_output_tokens: 4096",
])
anchor = find_block([lambda l: l.strip() == "always_loaded:"])
end = anchor + 1
while end < len(lines) and lines[end].startswith("    - "):
    end += 1
agent = [i for i in range(anchor + 1, end) if lines[i] == "    - agent\n"]
if len(agent) != 1:
    sys.exit(f"refused: expected one '- agent' in always_loaded, found {len(agent)}")
drop = {cap, cap + 1, cap + 2, agent[0]}
new = [l for i, l in enumerate(lines) if i not in drop]
if write:
    open(path, "w", encoding="utf-8").write("".join(new))
    print(f"wrote {path}: removed {len(drop)} lines")
else:
    sys.stdout.writelines(difflib.unified_diff(lines, new, "before", "after"))
