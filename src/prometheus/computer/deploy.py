"""deploy.sh's two questions about computer use, answered by the daemon's rule.

1. **Which extras?** ``computer`` (cua-driver, a native desktop driver) is
   installed ONLY when the live config says ``computer_use.enabled: true`` —
   the same literal-true reader the daemon uses
   (``integration.computer_use_enabled``). A box that has not switched it on
   does not carry the driver at all. Named in ``PROMETHEUS_DEPLOY_EXTRAS``
   while off, it is dropped, and the deploy says so.
2. **Is the venv right?** (gate G4) Off: cua-driver must be ABSENT. On: it
   must be exactly a version the adapter was validated against
   (``integration.SUPPORTED_DRIVER_VERSIONS``).

Run as a module (``python -m prometheus.computer.deploy``) from the new
venv, so deploy.sh parses no YAML of its own. A config that cannot be read
answers "off": the safe direction for a desktop driver.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

EXTRA = "computer"


def _load(path: str | Path) -> Any:
    import yaml

    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - unreadable or invalid means off
        return {}


def computer_extra_wanted(path: str | Path) -> bool:
    """Is computer use on in the config at *path*? Only a literal true."""
    from prometheus.computer.integration import computer_use_enabled

    cfg = _load(path)
    return computer_use_enabled(cfg if isinstance(cfg, dict) else {})


def rewrite_extras(extras: str, wanted: bool) -> list[str]:
    """The extras with ``computer`` present exactly when *wanted*. Order is
    kept; it is appended when added."""
    words = [w for w in (extras or "").split() if w]
    if wanted:
        return words if EXTRA in words else [*words, EXTRA]
    return [w for w in words if w != EXTRA]


def driver_check(*, enabled: bool, installed: str | None) -> tuple[bool, str]:
    """G4: does the venv's cua-driver match the switch?"""
    from prometheus.computer.integration import SUPPORTED_DRIVER_VERSIONS

    supported = ", ".join(sorted(SUPPORTED_DRIVER_VERSIONS))
    if not enabled:
        if installed is None:
            return True, "computer use is off; cua-driver is not installed"
        return False, (f"cua-driver {installed} is installed although computer "
                       f"use is off — the extra must follow the config")
    if installed is None:
        return False, ("computer use is on but cua-driver is not installed — "
                       "the computer extra did not install")
    if installed not in SUPPORTED_DRIVER_VERSIONS:
        return False, (f"cua-driver {installed} is installed; the adapter was "
                       f"validated against {supported} only")
    return True, f"computer use is on; cua-driver {installed}"


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "extras":
        wanted = computer_extra_wanted(argv[1])
        before = (argv[2] or "").split()
        after = rewrite_extras(argv[2], wanted)
        if EXTRA in before and EXTRA not in after:
            print("deploy:   computer use is off in the live config — the "
                  "computer extra is not installed", file=sys.stderr)
        elif EXTRA in after and EXTRA not in before:
            print("deploy:   computer use is on in the live config — "
                  "installing the computer extra", file=sys.stderr)
        print(" ".join(after))
        return 0
    if len(argv) >= 2 and argv[0] == "driver":
        from prometheus.computer.integration import installed_driver_version

        ok, why = driver_check(enabled=computer_extra_wanted(argv[1]),
                               installed=installed_driver_version())
        print(f"deploy:   {why}", file=sys.stderr)
        return 0 if ok else 1
    print("usage: python -m prometheus.computer.deploy "
          "extras <config> '<extras>' | driver <config>", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised through deploy.sh
    raise SystemExit(main(sys.argv[1:]))
