#!/usr/bin/env bash
# Install the local-teacher user units on the GPU host (never starts or enables them).
#
#   scripts/distill/units/install_teacher_units.sh <ssh-target>             # print the commands (default)
#   scripts/distill/units/install_teacher_units.sh <ssh-target> --execute   # run them
#
# Copies distill-teacher.service, distill-teacher-q4.service and their OnFailure= handler
# window-mini-teacher-failed.service into ~/.config/systemd/user/ on the
# host (0644, through a temporary name and an atomic rename), checks them with
# `systemd-analyze --user verify`, then `systemctl --user daemon-reload` so the user manager knows them.
# It refuses to replace a unit that is running (stop it with `setups.py apply business` first) and
# refuses when the host lacks the model file or the llama-server binary a unit names, or the staged
# window scripts the handler runs (mini_window.py stage first).
# No unit has an [Install] section: nothing here makes them start, now or at boot. setups.py
# starts the teachers (apply flywheel / flywheel-night-q4) and stops them (apply business); only a
# teacher's failure starts the handler.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
units=(distill-teacher.service distill-teacher-q4.service window-mini-teacher-failed.service)
target=""
execute=0
for a in "$@"; do
  case "$a" in
    --execute) execute=1 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $a" >&2; exit 64 ;;
    *) [ -z "$target" ] || { echo "one ssh target only" >&2; exit 64; }; target="$a" ;;
  esac
done
[ -n "$target" ] || { echo "usage: $0 <ssh-target> [--execute]" >&2; exit 64; }

for u in "${units[@]}"; do
  [ -r "$here/$u" ] || { echo "missing $here/$u" >&2; exit 1; }
done

ssh_cmd=(ssh -o BatchMode=yes -o ConnectTimeout=10 "$target")
dir='$HOME/.config/systemd/user'

# Host-side preconditions, read-only: the binary and models exist, no teacher unit is running.
check='set -e
for u in '"${units[*]}"'; do
  s=$(systemctl --user show -p ActiveState --value "$u" 2>/dev/null || true)
  case "$s" in active|activating|deactivating|reloading) echo "$u is $s: not replacing a running unit"; exit 1;; esac
done
for f in "$HOME/ladder/src/llama.cpp-9d57ce456/build/bin/llama-server" \
         "$HOME/distill/models/teacher/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf" \
         "$HOME/distill/models/teacher-q4/Qwen3.8-27B-Q4_K_M.gguf" \
         "$HOME/distill/window/mini_window.py" "$HOME/distill/window/window.conf"; do
  [ -r "$f" ] || { echo "missing on the host: $f"; exit 1; }
done
echo "preconditions ok"'

steps=()
steps+=("ssh:$check")
steps+=("ssh:mkdir -p \"$dir\"")
for u in "${units[@]}"; do
  steps+=("put:$u")
done
verify="systemd-analyze --user verify"
for u in "${units[@]}"; do verify+=" \"$dir/$u\""; done
steps+=("ssh:$verify")
steps+=("ssh:systemctl --user daemon-reload")
steps+=("ssh:for u in ${units[*]}; do printf '%s ' \"\$u\"; systemctl --user show -p LoadState -p UnitFileState -p ActiveState --value \"\$u\" | tr '\\n' ' '; echo; done")

# The remote command as one single-quoted word, for the printout.
sq() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

for step in "${steps[@]}"; do
  kind="${step%%:*}"
  body="${step#*:}"
  if [ "$kind" = put ]; then
    remote="cat > \"$dir/.$body.tmp\" && chmod 0644 \"$dir/.$body.tmp\" && mv -f \"$dir/.$body.tmp\" \"$dir/$body\""
    if [ "$execute" = 0 ]; then
      echo "[dry run] ${ssh_cmd[*]} $(sq "$remote") < $here/$body"
      continue
    fi
    "${ssh_cmd[@]}" "$remote" < "$here/$body"
    echo "installed $body"
    continue
  fi
  if [ "$execute" = 0 ]; then
    echo "[dry run] ${ssh_cmd[*]} $(sq "$body")"
    continue
  fi
  "${ssh_cmd[@]}" "$body"
done
[ "$execute" = 1 ] || echo "dry run: nothing was copied or reloaded (pass --execute)"
