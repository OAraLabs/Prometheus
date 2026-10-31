"""Stand-in for scripts/distill/taskset_gen.py, for the generation-runner tests.

``exclude --check ...`` logs its arguments to $FAKE_STATE/ts-calls.log and exits FAKE_TS_EXCL_RC (default 0:
the taskset holds no row of a listed session; 1: it does, with the summary line the real one prints).
"""

import json
import os
import sys
from pathlib import Path

state = Path(os.environ["FAKE_STATE"])
with open(state / "ts-calls.log", "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
rc = int(os.environ.get("FAKE_TS_EXCL_RC", "0"))
if rc == 1:
    print(json.dumps({"check": "holds rows of listed sessions", "rows_dropped": 9, "task_groups_dropped": 7}))
elif rc:
    print("taskset_gen: lcm snapshot not found", file=sys.stderr)
else:
    print(json.dumps({"check": "clean", "rows_dropped": 0}))
sys.exit(rc)
