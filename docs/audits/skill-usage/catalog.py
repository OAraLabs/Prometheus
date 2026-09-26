"""Skill catalog on the mini: sources, core tier, sizes, creation and archive history.

Usage (on the mini, deploy tree on the path so skills are parsed by the
daemon's own loader):

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=~/prometheus-deploy/src \\
        python3 catalog.py <snapshot-dir> [--tokenizer <qwen tokenizer.json>]

Prints aggregates only. ``load_catalog`` is also imported by the other scans.
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

HOME = Path.home()
SKILLS = HOME / ".prometheus" / "skills"
AUTO = SKILLS / "auto"
ARCHIVE = AUTO / ".archive"
CURATOR = HOME / ".prometheus" / "curator"


def _parse(path: Path) -> tuple[str, str, bool, str]:
    """(name, description, core, how the description was found) via the daemon's loader."""
    from prometheus.skills.loader import _parse_frontmatter, _parse_skill_markdown, _skill_is_core

    content = path.read_text(encoding="utf-8")
    name, desc = _parse_skill_markdown(path.stem, content)
    how = "fallback 'Skill: <name>'" if desc == f"Skill: {name}" else "first paragraph"
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                if _parse_frontmatter("\n".join(lines[1:i]))[1]:
                    how = "frontmatter"
                break
    return name, desc, _skill_is_core(content), how


def load_catalog(snap: Path) -> list[dict[str, Any]]:
    """Every skill the loader serves today, plus archived auto skills (served: False).

    ``born`` is the earliest of: the auto state file's first_seen_at, the
    skill_created signal for that name, and the file's mtime. ``archived_at``
    comes from the Curator run records that moved the file.
    """
    from prometheus.skills.loader import get_builtin_skills

    state = {}
    try:
        state = json.loads((AUTO / "_state.json").read_text()).get("skills", {})
    except (OSError, json.JSONDecodeError):
        pass
    created: dict[str, float] = {}
    tel = C.telemetry(snap)
    for row in tel.execute(
        "SELECT payload, timestamp FROM signal_events WHERE signal_type='skill_created'"
    ):
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            continue
        stem = Path(str(payload.get("skill_path", ""))).stem or str(payload.get("skill_name", ""))
        ts = C.dt.datetime.fromisoformat(row["timestamp"]).timestamp()
        for key in {stem, str(payload.get("skill_name", ""))}:
            if key:
                created[key] = min(created.get(key, ts), ts)
    archived: dict[str, float] = {}
    if CURATOR.is_dir():
        for run in sorted(CURATOR.glob("*/run.json")):
            try:
                rec = json.loads(run.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for p in rec.get("prunings") or []:
                dest = p.get("archived_to") or ""
                if dest:
                    archived[Path(dest).stem] = float(rec.get("ended_at") or rec.get("started_at") or 0)

    out: list[dict[str, Any]] = []

    def add(path: Path, source: str, served: bool) -> None:
        name, desc, core, how = _parse(path)
        st = path.stat()
        births = [st.st_mtime]
        rec = state.get(path.stem) or state.get(name)
        if rec and rec.get("first_seen_at"):
            births.append(float(rec["first_seen_at"]))
        for key in (path.stem, name):
            if key in created:
                births.append(created[key])
        out.append({
            "name": name, "stem": path.stem, "source": source, "served": served,
            "core": core, "description": desc, "desc_how": how,
            "bytes": st.st_size, "mtime": st.st_mtime, "born": min(births),
            "created_signal": created.get(path.stem) or created.get(name),
            "archived_at": archived.get(path.stem) if not served else None,
            "is_backup": ".bak-" in path.name,
            "pinned": bool(rec and rec.get("pinned")),
            "state": (rec or {}).get("state"),
            "content": path.read_text(encoding="utf-8"),
        })

    for s in get_builtin_skills():
        add(Path(s.path), "builtin", True)
    for p in sorted(SKILLS.glob("*.md")):
        add(p, "user", True)
    for p in sorted(AUTO.glob("*.md")):
        add(p, "auto", True)
    for p in sorted(ARCHIVE.glob("*.md")) if ARCHIVE.is_dir() else []:
        add(p, "auto (archived)", False)
    return out


def registry_view(cat: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """name -> the entry the registry would serve (load order builtin, user, auto; last wins)."""
    reg: dict[str, dict[str, Any]] = {}
    for s in cat:
        if s["served"]:
            reg[s["name"]] = s
    return reg


def _tok_counter(path: str | None):
    if not path:
        return None
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def main() -> None:
    snap = Path(sys.argv[1])
    tok_path = sys.argv[sys.argv.index("--tokenizer") + 1] if "--tokenizer" in sys.argv else None
    count = _tok_counter(tok_path)
    cat = load_catalog(snap)
    reg = registry_view(cat)
    served = [s for s in cat if s["served"]]

    print("== files by source (served = loaded by the daemon's registry)")
    by = collections.Counter((s["source"], s["served"]) for s in cat)
    for k, v in sorted(by.items()):
        print(f"  {k[0]:18} served={k[1]!s:5}  files={v}")
    print(f"  distinct names the registry serves: {len(reg)} "
          f"(files {len(served)}; same-name collisions {len(served) - len(reg)})")
    core = collections.Counter(s["source"] for s in reg.values() if s["core"])
    print(f"  core (tier: core / autoload: true): {sum(core.values())} {dict(core)}")
    print(f"  backups (*.bak-*.md): served {sum(s['is_backup'] for s in served)}, "
          f"archived {sum(s['is_backup'] for s in cat if not s['served'])}")

    print("== description source (registry view)")
    print("  ", dict(collections.Counter((s["source"], s["desc_how"]) for s in reg.values())))
    lens = [len(s["description"]) for s in reg.values()]
    print(f"  description chars: median {statistics.median(lens):.0f}, p90 "
          f"{sorted(lens)[int(len(lens) * 0.9)]}, max {max(lens)}")

    print("== body size by source (registry view)")
    for src in ("builtin", "user", "auto"):
        rows = [s for s in reg.values() if s["source"] == src]
        if not rows:
            continue
        b = [s["bytes"] for s in rows]
        line = (f"  {src:8} n={len(rows):3} bytes median {statistics.median(b):.0f} "
                f"max {max(b)} total {sum(b)}")
        if count:
            t = [count(s["content"]) for s in rows]
            line += f" | qwen3 tokens median {statistics.median(t):.0f} max {max(t)} total {sum(t)}"
        print(line)

    print("== user skills by file mtime month")
    print("  ", sorted(collections.Counter(C.month(s["mtime"]) for s in cat if s["source"] == "user").items()))

    print("== auto skills: creation month (live / archived)")
    for label, rows in (("live", [s for s in cat if s["source"] == "auto"]),
                        ("archived", [s for s in cat if s["source"] == "auto (archived)"])):
        print(f"  {label:8} n={len(rows)} by birth month "
              f"{sorted(collections.Counter(C.month(s['born']) for s in rows).items())}")
        print(f"           with a skill_created signal: {sum(1 for s in rows if s['created_signal'])}; "
              f"pinned {sum(s['pinned'] for s in rows)}; states "
              f"{dict(collections.Counter(s['state'] for s in rows))}")
        if rows:
            b = [s["bytes"] for s in rows]
            print(f"           bytes median {statistics.median(b):.0f} min {min(b)} max {max(b)}")
    arch = [s for s in cat if s["source"] == "auto (archived)"]
    print("  archived: Curator run record found for", sum(1 for s in arch if s["archived_at"]),
          "of", len(arch), "| by archive month",
          sorted(collections.Counter(C.month(s["archived_at"]) for s in arch if s["archived_at"]).items()))


def chat_only_auto_listing(snap: Path) -> None:
    """Names + descriptions of every auto skill (live and archived), for the
    operator's terminal only — the quality classification is done by reading
    these. Never commit this output."""
    print("#### CHAT-ONLY OUTPUT — do not commit ####")
    rows = [s for s in load_catalog(snap) if s["source"].startswith("auto") and not s["is_backup"]]
    for n, s in enumerate(sorted(rows, key=lambda s: s["born"]), 1):
        print(f"{n:2}. [{'live' if s['served'] else 'archived'} {C.month(s['born'])} {s['bytes']}B] "
              f"{s['name']} — {s['description'][:220]!r}")


if __name__ == "__main__":
    if "--chat-only" in sys.argv:
        chat_only_auto_listing(Path(sys.argv[1]))
    else:
        main()
