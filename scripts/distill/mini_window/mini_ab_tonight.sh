#!/usr/bin/env bash
# mini_ab_tonight.sh [run] [--execute] | restore [--execute] | ngl | arm --date D [--execute]   (3090 Ti host, ~/distill/window/)
#
# The teacher quant A/B night: pause local generation, arm A on the serving IQ3 teacher, swap the local model
# port to a partly offloaded Q4_K_M (distill-teacher-q4-partial.service), arm B and the judge there, the report,
# and ALWAYS the IQ3 teacher back + generation restarted. Without --execute it is the FAKE night (nothing
# changes; arms simulated on a virtual clock). Logic and safety rules: mini_ab.py (its docstring).
# Settings: window.conf beside this file (AB_* keys; defaults in mini_ab.py).
# Armed by `mini_ab.py arm --date YYYY-MM-DD --execute` as the one-shot transient timer
# window-mini-ab-<date> (NOT distill-*: apply business stops every distill-* unit), e.g.
#   systemd-run --user --unit=window-mini-ab-20261101 --on-calendar='2026-11-01 05:00:00 UTC' \
#     -p KillMode=mixed -p TimeoutStopSec=20min -p RuntimeMaxSec=25800 ~/distill/window/mini_ab_tonight.sh --execute
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in
  ""|-*) set -- run "$@" ;;
esac
exec "${DISTILL_WINDOW_PY:-/usr/bin/python3}" -I "$here/mini_ab.py" --conf "${DISTILL_WINDOW_CONF:-$here/window.conf}" "$@"
