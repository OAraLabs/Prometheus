#!/usr/bin/env bash
# gen_runner.sh run [--execute] [--until ISO|HH:MM] | harvest [--execute] | status   (3090 Ti host, ~/distill/window/)
#
# Teacher generation in a flywheel window: probe guard -> local teacher_gen (concurrency 1), plus Max on the
# pay-per-token key when QWEN_PAYG_KEY is present (judge_run.sh first, then teacher_gen --teacher max-payg),
# and the harvest into the pool. Dry run unless --execute. Logic and safety rules: gen_runner.py (its
# docstring). Settings: window.conf beside this file (GEN_* keys).
# mini_window.py window-start / slot-exit start it as a transient unit NAMED distill-gen-<stamp>, so a
# business restore (apply business stops every distill-* unit) and slot-enter stop it:
#   systemd-run --user --unit=distill-gen-$(date +%Y%m%d-%H%M%S) --collect -p KillMode=mixed -p TimeoutStopSec=90 ~/distill/window/gen_runner.sh run --execute
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DISTILL_WINDOW_PY:-/usr/bin/python3}" -I "$here/gen_runner.py" --conf "${DISTILL_WINDOW_CONF:-$here/window.conf}" "$@"
