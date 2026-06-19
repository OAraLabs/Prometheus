# Obsidian read-only view of the Prometheus wiki

SPRINT MEMORY-3, Phase 5. A way to *read* the self-maintaining wiki
(`~/.prometheus/wiki/`) as an Obsidian vault — graph view, backlinks, search —
without ever writing to it by hand.

## The one rule: it is READ-ONLY

The wiki is **compiled, not authored.** `WikiCompiler.regenerate_all()` rebuilds
every entity page from `memory.db` on each dream cycle — it wipes and rewrites
`*.md` under `people/`, `clients/`, `projects/`, `topics/`. Anything you type
into one of those pages in Obsidian is **gone on the next compile.**

So:

- **Capture goes through `/note`**, never the editor. `/note [@entity] <text>`
  (Telegram today; the Beacon web chat once that build is deployed) writes a
  durable, max-trust fact to `memory.db` — and `compile` projects it to a page.
  That is the *only* writer of your notes.
- **Never hand-edit a page.** A hand-edit makes the page a two-writer artifact
  that compile then fights and overwrites — the same drift class the wiki team
  has fixed twice (query-time index appends, lint-vs-compile link churn). Don't
  reintroduce it from the Obsidian side.
- `queries/` is preserved across compiles (filed-back synthesized answers), but
  it is still compile-owned. Treat the whole vault as output.

If you want something in the wiki, say it to Prometheus. `/note` it.

## The manual layer in the graph

Facts captured via `/note` are flagged `manual=1` in `memory.db`, and the
compiler stamps their pages with `manual: true` frontmatter (Phase 4b render
marker). The Obsidian graph config colors those nodes distinctly so your
deliberately-pinned knowledge stands out from the ambient, auto-extracted mass.

- Config: [`config/obsidian/graph.json`](../config/obsidian/graph.json) — a
  graph **color group** with the search query `["manual":true]`.
- This is config-as-code: the repo is the source of truth; the installer copies
  it into the live vault. Because `regenerate_all()` never touches `.obsidian/`
  (it is not in `_SUBDIRS`), the config survives every recompile — install once.

## Install (mini side)

```bash
scripts/install_obsidian_view.sh
```

Copies `config/obsidian/` into `~/.prometheus/wiki/.obsidian/`. Idempotent.
Run it on the box that hosts the wiki (the mini), so the color group is present
before Obsidian first opens the vault. Override the vault path with
`PROMETHEUS_WIKI=/path/to/vault` if needed.

## Viewing from the Mac (interactive — not covered by the installer)

Mount `~/.prometheus/wiki/` from the mini over Tailscale/SSHFS and open it as a
vault in Obsidian on the Mac. Two things to know:

1. **Read-only vs Obsidian's own state.** Obsidian wants to write *its* state
   (workspace layout, caches) into `.obsidian/` every session. A fully
   read-only mount may make it unhappy. The fix keeps the guarantee intact:
   mount the **pages** read-only but give **`.obsidian/` a writable path** —
   Obsidian gets its scratch space, your actual notes (`*.md`) stay
   un-hand-editable. Validate this live at the Mac.
2. **The graph query syntax.** `["manual":true]` is the intended frontmatter
   search for the color group; only Obsidian's graph actually rendering the
   manual nodes distinctly confirms the string. If the group comes up empty,
   adjust the query in `config/obsidian/graph.json`, re-run the installer.

## Acceptance

`tests/test_obsidian_view.py` proves the data the color group relies on: a
freshly compiled manual page carries `manual: true` frontmatter (the
`["manual":true]` predicate matches it) and an ambient page does not (the
predicate skips it). The final confirmation is visual, in Obsidian, at the Mac.
