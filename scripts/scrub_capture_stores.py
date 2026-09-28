#!/usr/bin/env python3
"""Scrub secrets already sitting in the capture stores: now ``oara scrub``.

The scrub moved into the package (prometheus.security.scrub_capture_stores)
because scripts/ ships in neither the wheel nor the sdist. This path stays so
the commands already written against it keep working from a checkout; the
flags and the output are the same.

    python3 scripts/scrub_capture_stores.py            # DRY RUN: counts only, touches nothing
    python3 scripts/scrub_capture_stores.py --apply    # rewrite in place, after a backup
"""

import sys
from pathlib import Path

# From a checkout, the checkout's own package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prometheus.security.scrub_capture_stores import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
