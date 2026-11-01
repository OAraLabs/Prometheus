#!/usr/bin/env bash
# gen_local.sh run [--execute] [--until T]   (3090 Ti host, ~/distill/round2/bin/)
#
# GEN_RUNNER of the 3090 Ti's own nightly windows (round2_chain.py mini-night start writes it into the conf
# mini_window_start.sh reads; mini_window.py start_gen runs "<this> run --execute" as distill-gen-<stamp>):
# gen_runner.py run from the round-2 code with the round-2 taskset, flags and run directory
# (round2_chain.py gen-local; the window's own runner when that code is not staged).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${R2_PY:-/usr/bin/python3}" -I "$here/round2_chain.py" --conf "${R2_CONF:-$HOME/distill/round2/round2.conf}" gen-local "$@"
