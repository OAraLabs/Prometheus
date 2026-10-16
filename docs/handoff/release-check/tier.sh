#!/usr/bin/env bash
# tier.sh <label> [extra isolate.py --set ...]: the 12-request real-model tier on the operator's config, as live
# runs it (scratch daemon behind the logging proxy; never the live daemon). Waits for the live daemon to be idle.
#   CHECKOUT  the checkout under test (with a synced .venv)                       required
#   WORK      where results go: $WORK/measure-<label>.json and measure-<label>/   required
#   SRC       the source config (its model.base_url is the real model server; read by the proxy, never printed)
#             default: the 10-16 backup, i.e. the live config without the workaround
#   PROJECT   the deploy checkout's PROMETHEUS.md (the daemon's cwd file, as live)
# Score it with: python3 tier_score.py "$WORK" <label>
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
C=${CHECKOUT:?set CHECKOUT}; W0=${WORK:?set WORK}
SRC=${SRC:-$HOME/prometheus-deploy/config/prometheus.yaml.bak-20261016T122703Z-pre-agent-cap4096}
PROJECT=${PROJECT:-$HOME/prometheus-deploy/PROMETHEUS.md}
label=$1; shift; SCEN=${SCEN:-r_cron,r_agent,r_history,r_wiki,r_tasks,r_sessions,r_todo,r_cron_list,r_memory,r_mcp,r_tts,r_youtube}
API=18705; WS=18710; PX=18790
W=$W0/measure-$label
rm -rf "$W"; mkdir -p "$W"
$C/.venv/bin/python -I $C/scripts/local_gate/isolate.py --source-config "$SRC" --source-home $HOME/.prometheus \
  --out "$W" --api-port $API --ws-port $WS --proxy-port $PX --python $C/.venv/bin/python \
  --mcp-tools $HERE/context7_stub_tools.json "$@" > /dev/null
$C/.venv/bin/python -I $C/scripts/local_gate/run_gate.py --checkout $C --work "$W" --label m-$label \
  --upstream-config "$SRC" --api-port $API --ws-port $WS --proxy-port $PX --scenarios "$SCEN" \
  --idle-cmd $HERE/idle.sh --results $W0/measure-$label.json --timeout 240 --overhead \
  --project-file "$PROJECT"
