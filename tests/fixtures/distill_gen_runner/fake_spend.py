"""Stand-in for scripts/distill/spend.py: ``preflight`` exits FAKE_SPEND_RC (default 0); calls are logged."""

import json
import os
import sys
from pathlib import Path

state = Path(os.environ["FAKE_STATE"])
with open(state / "spend-calls.log", "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
rc = int(os.environ.get("FAKE_SPEND_RC", "0"))
if rc:
    print("refused: STOP is active (the $140 alert); the operator was notified", file=sys.stderr)
else:
    print(json.dumps({"committed_usd": 0.0}))
sys.exit(rc)
