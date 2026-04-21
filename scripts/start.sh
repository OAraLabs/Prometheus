#!/usr/bin/env bash
# Prometheus startup script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting Prometheus..."
cd "$PROJECT_ROOT"

source "$HOME/.local/bin/env" 2>/dev/null || true

exec uv run prometheus "$@"
