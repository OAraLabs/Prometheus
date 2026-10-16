#!/usr/bin/env bash
# exit 0 when the live daemon shows no model activity in the last 45 s
for i in $(seq 1 20); do
  n=$(timeout 20 journalctl --user -u prometheus.service --since "-45 s" --no-pager -o cat -g "agent_loop|ModelRouter|context.compactor|lcm_summar|memory_extractor" 2>/dev/null | grep -vc "^-- No entries")
  [ "$n" = "0" ] && exit 0
  sleep 10
done
exit 1
