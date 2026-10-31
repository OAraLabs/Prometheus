#!/usr/bin/env bash
# GPU tower job runner for the nightly flywheel window (logic in job_runner.py beside this file).
#
#   job_runner.sh                        # dry run (default): plan + checks, starts nothing
#   job_runner.sh --execute --stop-at <epoch>
#   job_runner.sh --summary              # <= 5 lines from the last progress.json
#   job_runner.sh --status               # progress.json
#
# --execute is launched ONLY by the window start (window_start.sh), as the transient unit
# distill-jobs-<stamp>.service with the window's DJR_* settings (its NO-SWAPS-TONIGHT flag,
# hours, disk floor, production unit, rehearsal result and dead-man timer), after the
# rehearsal PASS, the dead-man and `apply flywheel`. A hand launch refuses without a
# rehearsal PASS and an armed dead-man. The window end stops it with SIGTERM (systemctl
# --user stop of that unit; KillMode=mixed). Stdlib python only. Core dumps are off for
# the runner and every child (a crashed GPU process must not fill the home disk).
set -euo pipefail
ulimit -c 0
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${DJR_RUNNER_PY:-/usr/bin/python3}"
exec "$py" -I "$here/job_runner.py" "$@"
