#!/usr/bin/env python3
"""PARITY harness CLI — record, replay, diff, benchmark (WP-1.2).

    uv run python scripts/parity_harness.py record  --upstream-primary URL --upstream-alt URL
    uv run python scripts/parity_harness.py replay  [--scenario NAME ...]
    uv run python scripts/parity_harness.py stability
    uv run python scripts/parity_harness.py bench   --runs 10
    uv run python scripts/parity_harness.py normalizations
    uv run python scripts/parity_harness.py rebaseline-from-exchanges [--dry-run]

See scripts/parity/__init__.py for how it works and tests/fixtures/parity/
for the recorded traces. Exit status: 0 = parity, 1 = a diff, 2 = the harness
itself could not run (which is never reported as a pass).
``rebaseline-from-exchanges`` is only for a change to what the daemon records;
it exits 1 and writes nothing if any model request changed (see
parity/rebaseline.py and tests/fixtures/parity/README.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
