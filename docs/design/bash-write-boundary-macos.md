# The bash write boundary on hosts without a kernel write floor

Status: **accepted gap, documented** (owner decision 2026-11-25). No code
change. This note records what holds, what does not, and why a command-string
pre-check was considered and rejected.

## The layers, as they exist

A model-written `bash` command passes three layers, in this order:

1. **The permission gate, before the command runs**
   (`permissions/checker.py`, `SecurityGate.evaluate`). The workspace write
   boundary keys on a **declared `file_path`** — the path a tool's own schema
   says it writes (`permissions/tool_paths.gate_path_for`). `bash` declares a
   `command` string and no path, so the boundary has nothing to rule on and
   the call falls through to ALLOW (default mode, user origin). What the gate
   DOES see for bash: always-blocked patterns, `denied_commands`, the
   recursive-`rm`-at-a-protected-root speed bump
   (`_rm_targets_a_protected_root`), and — at system origin or in cautious
   mode — the network/install approve patterns.

2. **The kernel floors, around the command**
   (`permissions/confinement.py`, `apply_floors`). The read floor (AppArmor
   on Linux, Seatbelt on macOS) confines reads of the credential families.
   The **write floor is bubblewrap only**: a mount namespace where everything
   outside the workspace roots is read-only. It is verified by outcome at
   boot and per composition. On a host without bwrap — **every Mac** —
   `bash_write_confinement: auto` degrades to an unfloored shell and says so
   (`bash_write_floor.state: unavailable` in the floor report, ERROR in the
   journal). `bash_write_confinement: required` refuses every bash call instead.

3. **The post-turn escape check, after the command ran**
   (`engine/agent_loop.py`, `_boundary_escapes`; the file-mutation verifier
   snapshots the filesystem). Every path that actually changed is re-asked of
   the gate as a declared write; anything the gate would not have permitted
   ends the turn with an explicit "these writes ALREADY HAPPENED and CANNOT
   BE UNDONE" message. This layer survives tool substitution — it watches the
   disk, not the toolset — but it is **detection, not containment**: the
   verifier holds size/mtime/mode, no content, and nothing is rolled back.

## What this means in default mode on macOS

Layers 1 and 2 do not bound bash writes. `echo x > ~/outside.txt`,
`mv`, `cp`, `rm`, `tee`, `dd of=`, `sed -i`, `python -c`, heredocs — all land
on disk unprompted, and layer 3 catches them only after the fact. This is not
a hypothetical: on 2026-11-24 a live conversation bound to `~/projects` ran
`cp ~/.prometheus/prometheus.yaml ~/.prometheus/prometheus.yaml.bak-…` (a
backup the owner had asked for). The gate never saw the path, the copy
landed, and the post-turn escape check ended the turn naming the file. The
system behaved exactly as designed — detection worked, containment did not
exist — and the owner then ruled the gap accepted-with-documentation rather
than forcing a fix.

The repo already litigated the shape of this hole:
`tests/test_bash_write_floor.py` enumerates the write syntaxes ("not seven
bugs; one bug seen seven ways") and states the ruling — close it **below the
syntax, where the kernel resolves the path** — and its skip marker says
plainly that on a host without bwrap "SKIPPED IS NOT PASSED … these are live
holes, not covered ground."

## Why not pre-check redirections / mv / rm against the boundary

A string-level pre-check was the first candidate fix. It was rejected for
reasons the codebase already records:

* **A parser has to win every round; the shell has to win once.**
  `checker.py`'s own docstring: any check on the command string is defeated
  by ordinary shell — `cd ~/.ssh && cat id_*`, `$HOME` indirection, globs,
  `sh -c`, `$(...)`. The existing `rm` speed bump is explicitly labelled "a
  SPEED BUMP, not the confinement" and deliberately does not follow `cd`,
  because tracking shell state is the losing game.
* **The enumeration is unbounded.** `>` `>>` `>|` `tee` `dd of=` `sed -i`
  `cp` `mv` `install` `ln` `python -c` heredocs, and every interpreter and
  wrapper not yet thought of. A pre-check that catches the listed forms
  invites the reading "bash writes are now bounded", which would be false —
  the same overclaim PR #177 removed from the escape-check wording.
* **Ambiguity has no safe default.** A conservative pre-check must either
  approve-prompt every command it cannot parse (most real commands: pipes,
  substitutions, variables — degrading default mode toward cautious for
  every shell call) or pass them (the hole remains, now with a false sense
  of coverage). Neither is a small, safe change.
* **The boundary is a property of paths, and bash's paths are only knowable
  at execution.** That is precisely what the kernel floor provides and what
  the post-turn check measures after the fact.

## What an operator can do today

* `security.bash_write_confinement: required` — every model-written bash call is
  refused on a host that cannot contain it. Fail-closed, loud, and the
  honest setting for a Mac that must not write outside the workspace.
  (The read floor's `bash_confinement: required` behaves the same way where
  Seatbelt cannot run.)
* Keep `auto` (the shipped default) and read the escape check as what it is:
  an alarm after the fact, plus the per-turn file checkpoints for sessions
  with a workspace (where the checkpoint budget allows — a workspace with
  more files than `checkpoints.max_files` is not checkpointed, and the
  journal says so).
* Watch the floor report's `dark` flag (`/api/status` security block,
  `oara doctor`): it is true when a floor was asked for and is not happening
  while bash runs anyway.

## The real fix, if the gap is ever closed

A kernel write floor for macOS — a Seatbelt profile that denies file writes
outside the workspace roots, composed and probed by outcome exactly like the
read floor and the Linux write floor. Seatbelt already runs on every Mac
command for reads; extending it to a workspace write boundary is design work
with real questions (dynamic per-session roots vs. a per-command profile,
`mv` across the boundary, symlink resolution, the deprecated `sandbox-exec`
binary's future). Until then, the layers above are the contract, and this
note is its statement.