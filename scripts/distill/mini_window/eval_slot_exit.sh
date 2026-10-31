#!/usr/bin/env bash
# eval_slot_exit.sh [--execute]  (3090 Ti host, ~/distill/window/)
#
# 03:15 / 06:15 on a Q4 night: once the nightly eval is done, back to the Q4 teacher (Ollama stopped).
# Dry run unless --execute: read-only checks run, every change is printed. Logic and safety rules:
# mini_window.py slot-exit (its docstring). Settings: window.conf beside this file.
# Run it from a transient unit NOT named distill-* (apply business stops every distill-* unit), with a
# stop timeout long enough for the restore a SIGTERM starts (the default 90 s would SIGKILL it), e.g.
#   systemd-run --user --unit=window-mini-slot-exit-$(date +%H%M) -p TimeoutStopSec=20min -p KillMode=mixed --collect ~/distill/window/eval_slot_exit.sh --execute
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DISTILL_WINDOW_PY:-/usr/bin/python3}" -I "$here/mini_window.py" --conf "${DISTILL_WINDOW_CONF:-$here/window.conf}" slot-exit "$@"
