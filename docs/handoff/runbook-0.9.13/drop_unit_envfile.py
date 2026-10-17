"""Remove the one `EnvironmentFile=` line from an installed Prometheus systemd user unit: a targeted edit.

usage: drop_unit_envfile.py [--write] [--unit PATH]
Default unit: ~/.config/systemd/user/prometheus.service (drop-ins in prometheus.service.d/ are checked too).

For a HAND-WRITTEN unit. A unit written by `oara install-service` is regenerated with
`oara install-service --force` instead. That rewrites the whole file, which on a hand-written unit
drops its WorkingDirectory=, ExecStartPre= and Environment= lines.

0.9.13's daemon loads the env file itself at startup, so the unit no longer needs to pass it.
Refuses unless the main unit has exactly one `EnvironmentFile=` line and no drop-in has any; writes
nothing then. Without --write, prints the diff. Afterwards: `systemctl --user daemon-reload`, restart.
"""
import difflib, pathlib, sys

args = sys.argv[1:]
write = "--write" in args
unit = pathlib.Path(args[args.index("--unit") + 1]) if "--unit" in args else \
    pathlib.Path.home() / ".config/systemd/user/prometheus.service"
lines = unit.read_text(encoding="utf-8").splitlines(keepends=True)
hits = [i for i, l in enumerate(lines) if l.lstrip().startswith("EnvironmentFile=")]
if len(hits) != 1:
    sys.exit(f"refused: expected one EnvironmentFile= line in {unit.name}, found {len(hits)}; nothing written")
dropins = sorted((unit.parent / (unit.name + ".d")).glob("*.conf"))
in_dropins = [d.name for d in dropins if any(l.lstrip().startswith("EnvironmentFile=")
                                             for l in d.read_text(encoding="utf-8").splitlines())]
if in_dropins:
    sys.exit(f"refused: drop-ins also set EnvironmentFile= ({', '.join(in_dropins)}); nothing written")
new = [l for i, l in enumerate(lines) if i != hits[0]]
if write:
    unit.write_text("".join(new), encoding="utf-8")
    print(f"wrote {unit}: removed 1 line. Next: systemctl --user daemon-reload, then restart.")
else:
    sys.stdout.writelines(difflib.unified_diff(lines, new, unit.name, unit.name + " (after)"))
