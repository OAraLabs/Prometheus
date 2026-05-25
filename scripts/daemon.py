#!/usr/bin/env python3
"""Thin shim — delegates to ``prometheus.daemon``.

The daemon implementation now lives inside the installed package at
``src/prometheus/daemon.py`` so PyPI wheels can ship it. This file
stays around for the editable / source-checkout path:

    python3 scripts/daemon.py --debug

Both forms resolve to the same ``main()``.
"""

from __future__ import annotations


def main() -> None:
    from prometheus.daemon import main as _main
    _main()


if __name__ == "__main__":
    main()
