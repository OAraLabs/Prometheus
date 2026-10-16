"""Day-after cleanup of keys 0.9.13 warns about at boot: a targeted edit.

usage: cleanup_dead_config.py <config> [--write]
Removes exactly:
  * context.reserved_output: 2000  (no longer read: the reply room is each model's output cap)
  * these context.model_overrides entries, which match no model the daemon serves or routes
    to (0.9.13 boot WARNING): gpt-4o, qwen3.5-32b, gemma4-26b-moe,
    google_gemma-4-26B-A4B-it-Q4_K_M, gemma4-31b
Each must occur exactly once, uncommented, where named, with exactly the child line expected;
anything else refuses and writes nothing. Without --write, prints the diff.
"""
import difflib, sys

path, write = sys.argv[1], "--write" in sys.argv[2:]
lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
DEAD = {"gpt-4o": 128000, "qwen3.5-32b": 24000, "gemma4-26b-moe": 32000,
        "google_gemma-4-26B-A4B-it-Q4_K_M": 72000, "gemma4-31b": 48000}

def refuse(msg):
    sys.exit(f"refused: {msg}; nothing written")

def section(name, indent):
    """[start, end) of the block under the one uncommented `name:` at *indent*."""
    hits = [i for i, l in enumerate(lines) if l == " " * indent + name + ":\n"]
    if len(hits) != 1:
        refuse(f"expected one '{name}:' at indent {indent}, found {len(hits)}")
    end = hits[0] + 1
    while end < len(lines) and (not lines[end].strip() or lines[end].startswith(" " * (indent + 1))):
        end += 1
    return hits[0], end

ctx_start, ctx_end = section("context", 0)
drop = set()
res = [i for i in range(ctx_start + 1, ctx_end) if lines[i] == "  reserved_output: 2000\n"]
if len(res) != 1:
    refuse(f"expected one '  reserved_output: 2000' under context, found {len(res)}")
drop.add(res[0])
mo = [i for i in range(ctx_start + 1, ctx_end) if lines[i] == "  model_overrides:\n"]
if len(mo) != 1:
    refuse(f"expected one '  model_overrides:' under context, found {len(mo)}")
mo_end = mo[0] + 1
while mo_end < ctx_end and (lines[mo_end].startswith("    ") or not lines[mo_end].strip()):
    mo_end += 1
for key, limit in DEAD.items():
    hits = [i for i in range(mo[0] + 1, mo_end) if lines[i] == f"    {key}:\n"]
    if len(hits) != 1:
        refuse(f"expected one model_overrides entry '{key}', found {len(hits)}")
    i = hits[0]
    children = []
    j = i + 1
    while j < mo_end and lines[j].startswith("      "):
        children.append(j)
        j += 1
    if [lines[c] for c in children] != [f"      effective_limit: {limit}\n"]:
        refuse(f"'{key}' does not hold exactly 'effective_limit: {limit}'")
    drop.update([i, *children])
new = [l for i, l in enumerate(lines) if i not in drop]
if write:
    open(path, "w", encoding="utf-8").write("".join(new))
    print(f"wrote {path}: removed {len(drop)} lines")
else:
    sys.stdout.writelines(difflib.unified_diff(lines, new, "before", "after"))
