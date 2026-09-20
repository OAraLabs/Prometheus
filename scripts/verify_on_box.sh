#!/usr/bin/env bash
# On-box verification — the guards CI structurally cannot run.
#
# CI has no AT-SPI bindings and no display, so a test that interrogates the
# live accessibility vocabulary SKIPS there. Skipped is not passed: a guard
# that only runs when someone remembers is not a guard. These run here, under
# system python, and are named in the output so a silent skip is visible.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
echo "══ ON-BOX VERIFICATION — guards CI cannot run ══"
echo
FAIL=0
run() {
  echo "── $1"
  shift
  if python3 -m pytest "$@" -q 2>&1 | tail -3; then :; else FAIL=1; fi
  echo
}
run "clickable-role vocabulary is live (needs AT-SPI bindings)" \
    tests/test_clickable_roles_are_live.py
echo "══ NAMED BECAUSE THEY SKIP ELSEWHERE ══"
echo "  tests/test_clickable_roles_are_live.py  — needs gi/Atspi; skips in CI and the venv"
echo
[ "$FAIL" = 0 ] && echo "on-box verification PASSED" || echo "on-box verification FAILED"
exit "$FAIL"
