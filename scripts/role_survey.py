"""Which AT-SPI roles appear in real windows, and which does the table offer?

DIAGNOSE ONLY. Widening `_CLICKABLE_ROLES` is a new CONSENT SURFACE — each
added role is a class of thing the system may be asked to click — so this
reports the gap and changes nothing.
"""
from __future__ import annotations
import sys, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import gi
gi.require_version("Atspi", "2.0")
from gi.repository import Atspi
from prometheus.computer.candidates import _CLICKABLE_ROLES, _EDITABLE_ROLES

def walk(node, out, depth=0):
    if depth > 16 or len(out) > 6000: return
    try:
        st = node.get_state_set()
        if st.contains(Atspi.StateType.SHOWING):
            out.append((node.get_role_name(), node.get_name() or ""))
        n = node.get_child_count()
    except Exception: return
    for i in range(min(n, 250)):
        try: walk(node.get_child_at_index(i), out, depth+1)
        except Exception: pass

def survey(app_filter=None):
    d = Atspi.get_desktop(0)
    per_app = {}
    for i in range(d.get_child_count()):
        try:
            a = d.get_child_at_index(i); name = a.get_name() or "?"
            if a.get_child_count() == 0: continue
            if app_filter and name not in app_filter: continue
        except Exception: continue
        nodes = []
        for w in range(min(a.get_child_count(), 6)):
            try: walk(a.get_child_at_index(w), nodes)
            except Exception: pass
        if nodes: per_app[name] = Counter(r for r, _ in nodes)
    return per_app

if __name__ == "__main__":
    per_app = survey(sys.argv[1:] or None)
    known = _CLICKABLE_ROLES | _EDITABLE_ROLES
    total = Counter()
    for app, c in sorted(per_app.items()):
        total.update(c)
        gap = {r: n for r, n in c.items() if r not in known}
        print(f"\n{app}  ({sum(c.values())} showing nodes, {len(c)} distinct roles)")
        offered = {r: n for r, n in c.items() if r in known}
        print(f"  OFFERED  : {sum(offered.values()):>4} nodes  {sorted(offered)}")
        top = sorted(gap.items(), key=lambda kv: -kv[1])[:12]
        print(f"  NOT offered: {sum(gap.values()):>3} nodes  " +
              ", ".join(f"{r}×{n}" for r, n in top))
    print("\n" + "=" * 78)
    print("AGGREGATE — roles present in these windows but NOT in the offered set")
    print("=" * 78)
    gap = {r: n for r, n in total.items() if r not in known}
    for r, n in sorted(gap.items(), key=lambda kv: -kv[1]):
        print(f"  {r:<28} {n:>5} nodes")
    print(f"\n  offered set today: {sorted(known)}")
