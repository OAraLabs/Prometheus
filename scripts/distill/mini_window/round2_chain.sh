#!/usr/bin/env bash
# round2_chain.sh <command> [options]   (3090 Ti host, ~/distill/round2/bin/; round2_chain.py beside it)
#
# Round 2 generation automation: the Sun -> Mon 4090 chain, the nightly 4090 teacher + remote generation, the
# 3090 Ti's own nightly windows, the 09:00 heartbeat. Logic and safety rules: round2_chain.py (its docstring).
# Settings: ~/distill/round2/round2.conf (R2_CONF overrides; round2.conf.example beside this file).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${R2_PY:-/usr/bin/python3}" -I "$here/round2_chain.py" --conf "${R2_CONF:-$HOME/distill/round2/round2.conf}" "$@"
