#!/usr/bin/env bash
# mini_rehearsal.sh [--execute]  (3090 Ti host, ~/distill/window/)
#
# the first-swap round trip: business -> teacher (healthy, next voice probe OK) -> business (verified), dead-man armed first.
# Dry run unless --execute: read-only checks run, every change is printed. Logic and safety rules:
# mini_window.py rehearsal (its docstring). Settings: window.conf beside this file.
# Run it from a transient unit NOT named distill-* (apply business stops every distill-* unit), with a
# stop timeout long enough for the restore a SIGTERM starts (the default 90 s would SIGKILL it), e.g.
#   systemd-run --user --unit=window-mini-rehearsal-$(date +%H%M) -p TimeoutStopSec=20min -p KillMode=mixed --collect ~/distill/window/mini_rehearsal.sh --execute
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${DISTILL_WINDOW_PY:-/usr/bin/python3}" -I "$here/mini_window.py" --conf "${DISTILL_WINDOW_CONF:-$here/window.conf}" rehearsal "$@"
