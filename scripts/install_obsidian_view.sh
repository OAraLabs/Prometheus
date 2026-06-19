#!/usr/bin/env bash
#
# Install the Prometheus Obsidian "read-only view" config into the runtime wiki
# vault (~/.prometheus/wiki/.obsidian/).
#
# Config-as-code: the source of truth is config/obsidian/ in the repo. This just
# copies it into the live vault so the manual-node color group is present BEFORE
# Obsidian ever opens the vault. Idempotent — re-running overwrites the installed
# config with the repo template.
#
# Safe to run anytime: WikiCompiler.regenerate_all() only rewrites *.md inside
# people/clients/projects/topics and never touches .obsidian/, so this config
# rides through every dream-cycle recompile. See docs/OBSIDIAN-VIEW.md.
#
# Override the vault location with PROMETHEUS_WIKI=/path/to/vault.
set -euo pipefail

VAULT="${PROMETHEUS_WIKI:-$HOME/.prometheus/wiki}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/config/obsidian"
DEST="$VAULT/.obsidian"

if [ ! -d "$VAULT" ]; then
  echo "error: vault not found at $VAULT — run a wiki_compile first, or set PROMETHEUS_WIKI." >&2
  exit 1
fi
if [ ! -d "$TEMPLATE_DIR" ]; then
  echo "error: template dir not found at $TEMPLATE_DIR." >&2
  exit 1
fi

mkdir -p "$DEST"
cp -r "$TEMPLATE_DIR/." "$DEST/"

echo "installed Obsidian view config -> $DEST"
echo "  $(cd "$DEST" && ls -1 | sed 's/^/  /')"
echo
echo "The 'manual: true' color group is now present. Open ~/.prometheus/wiki as an"
echo "Obsidian vault (read-only mount) and the Graph view will color /note nodes"
echo "distinctly. This is a READ-ONLY view — capture via /note, never hand-edit pages."
