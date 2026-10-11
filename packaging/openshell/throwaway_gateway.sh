#!/usr/bin/env bash
# A throwaway OpenShell gateway for testing the sandbox image, the way docs/guide/openshell.md was tested.
# NOT for real use: plaintext, no client auth, loopback only. For a gateway you keep, install OpenShell
# the way its own documentation says.
#
#   packaging/openshell/throwaway_gateway.sh start DIR [PORT]   start one; prints the two lines to export
#   packaging/openshell/throwaway_gateway.sh stop DIR           stop it and delete DIR
#
# Everything lives in DIR: the release binaries (checksums verified), HOME for the CLI and the gateway
# (config, state, TLS/JWT keys, the database), the log and the PID. It needs Docker 28+, a Linux kernel
# with Landlock, and no root. The gateway pulls its supervisor and sandbox images on the first sandbox;
# `stop` removes those only if they were not on this machine before `start`.
#
# The config mirrors OpenShell's own dev script (tasks/scripts/gateway-docker.sh at the same tag),
# including its app_armor_profile = "Unconfined" for the sandbox containers.
set -euo pipefail

VERSION=0.1.3
NAME=prom-throwaway
cmd=${1:-}; dir=${2:-}
[ -n "$cmd" ] && [ -n "$dir" ] || { echo "usage: $0 start|stop DIR [PORT]" >&2; exit 2; }

case "$(uname -m)" in
  x86_64) arch=x86_64 ;;
  aarch64|arm64) arch=aarch64 ;;
  *) echo "unsupported architecture $(uname -m)" >&2; exit 2 ;;
esac
images=("ghcr.io/nvidia/openshell/supervisor:${VERSION}" "ghcr.io/nvidia/openshell/sandbox:${VERSION}")

if [ "$cmd" = start ]; then
  port=${3:-17670}
  [ ! -e "$dir" ] || { echo "$dir exists; pick a new directory" >&2; exit 2; }
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then echo "port $port is in use" >&2; exit 2; fi
  mkdir -p "$dir"/{bin,home,state}
  dir=$(cd "$dir" && pwd)
  for image in "${images[@]}"; do
    docker image inspect "$image" >/dev/null 2>&1 && echo "$image" >> "$dir/images-present-before"
  done
  touch "$dir/images-present-before"

  base="https://github.com/NVIDIA/OpenShell/releases/download/v${VERSION}"
  cd "$dir/bin"
  for f in "openshell-${arch}-unknown-linux-musl.tar.gz" openshell-checksums-sha256.txt \
           "openshell-gateway-${arch}-unknown-linux-gnu.tar.gz" openshell-gateway-checksums-sha256.txt; do
    curl -fsSLO "$base/$f"
  done
  cat openshell-checksums-sha256.txt openshell-gateway-checksums-sha256.txt | sha256sum -c --ignore-missing
  tar -xzf "openshell-${arch}-unknown-linux-musl.tar.gz"
  tar -xzf "openshell-gateway-${arch}-unknown-linux-gnu.tar.gz"
  cd "$dir"

  export HOME="$dir/home"
  bin="$dir/bin"
  "$bin/openshell-gateway" generate-certs --output-dir "$dir/state/tls" \
    --server-san 127.0.0.1 --server-san localhost --server-san host.openshell.internal >/dev/null
  cat > "$dir/state/gateway.toml" <<EOF
[openshell]
version = 2

[openshell.gateway]
name = "${NAME}"
compute_driver = "docker"
disable_tls = true

[openshell.gateway.auth]
allow_unauthenticated_users = true

[openshell.gateway.gateway_jwt]
signing_key_path = "${dir}/state/tls/jwt/signing.pem"
public_key_path = "${dir}/state/tls/jwt/public.pem"
kid_path = "${dir}/state/tls/jwt/kid"
gateway_id = "${NAME}"

[openshell.drivers.docker]
supervisor_image = "${images[0]}"
sandbox_runtime_image = "${images[1]}"
image_pull_policy = "if_not_present"
sandbox_label = "${NAME}"
grpc_endpoint = "http://127.0.0.1:${port}"
app_armor_profile = "Unconfined"
EOF
  mkdir -p "$HOME/.config/openshell/gateways/${NAME}"
  cat > "$HOME/.config/openshell/gateways/${NAME}/metadata.json" <<EOF
{
  "name": "${NAME}",
  "gateway_endpoint": "http://127.0.0.1:${port}",
  "is_remote": false,
  "gateway_port": ${port},
  "auth_mode": "plaintext"
}
EOF
  nohup "$bin/openshell-gateway" --config "$dir/state/gateway.toml" --port "$port" --log-level info \
    --compute-driver docker --disable-tls --db-url "sqlite:${dir}/state/gateway.db?mode=rwc" \
    > "$dir/gateway.log" 2>&1 &
  echo $! > "$dir/gateway.pid"
  for _ in $(seq 1 60); do
    "$bin/openshell" --gateway "$NAME" status >/dev/null 2>&1 && break
    sleep 1
  done
  "$bin/openshell" --gateway "$NAME" status
  echo "export HOME=$HOME PATH=$bin:\$PATH"
  echo "gateway: ${NAME} (pid $(cat "$dir/gateway.pid"), 127.0.0.1:${port}, log $dir/gateway.log)"
  exit 0
fi

if [ "$cmd" = stop ]; then
  [ -f "$dir/gateway.pid" ] || { echo "$dir is not a throwaway gateway directory" >&2; exit 2; }
  dir=$(cd "$dir" && pwd)
  export HOME="$dir/home"
  "$dir/bin/openshell" --gateway "$NAME" sandbox delete --all >/dev/null 2>&1 || true
  pid=$(cat "$dir/gateway.pid")
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
  for image in "${images[@]}"; do
    grep -qxF "$image" "$dir/images-present-before" || docker image rm "$image" >/dev/null 2>&1 || true
  done
  rm -rf "$dir"
  echo "stopped ${NAME}; removed $dir"
  exit 0
fi

echo "usage: $0 start|stop DIR [PORT]" >&2
exit 2
