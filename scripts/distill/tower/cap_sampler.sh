#!/usr/bin/env bash
# cap_sampler.sh OUT.csv   (the 4090; run by capacity.py's block as the unit distill-cap-sampler)
#
# One line a second until stopped: epoch, VRAM used MiB, VRAM free MiB, host MemAvailable kB. The block
# reads the minimum free VRAM (the probe ladder's headroom), the peak VRAM and the lowest MemAvailable
# (the QLoRA benchmark's host RAM) over each measurement's time window. Reads only; writes only OUT.
set -u
out=${1:?usage: cap_sampler.sh OUT.csv}
umask 077
while :; do
  g=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
  m=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)
  printf '%s,%s,%s\n' "$(date +%s)" "${g:-,}" "${m:-}" >> "$out"
  sleep 1
done
