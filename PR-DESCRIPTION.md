# feat(coding): /api/project-file — narrow daemon endpoint for Loop Manager project files

Adds a **narrow, deliberate** daemon endpoint so a **remote** Beacon (e.g. on a Mac) can read/write a
Loop Manager project's `TASKS.md`/`LOOP.md`/`PROGRESS.md` that live on the daemon host. Fixes the
macOS-walk bug where "Create TASKS.md" failed: Loop Manager's file editing was Beacon-host-local, so
a Mac-Beacon couldn't reach a mini-resident repo path (`/home/will/…` doesn't exist on the Mac).

## The narrow scope (this is the whole point)
`GET` / `PUT /api/project-file` permit writes **only**:
- to one of `{TASKS.md, LOOP.md, PROGRESS.md}` (basename allowlist),
- **at the top level** of the repo path (a sub-path `name` is rejected),
- when the path passes the **SAME `repo/.git` validation `POST /api/code` already uses** —
  `_resolve_coding_repo`, the single validator, now **reused** by both (not a second validator).

Not "let the daemon write anywhere." Every rejection returns its **own specific reason** — the
swallowed-ENOENT that made the original bug hard to diagnose does not recur:
- wrong basename → `filename not permitted (only TASKS.md, LOOP.md, PROGRESS.md)`
- sub-path name → `name must be a top-level filename, not a path`
- non-git path → `not a git repository (no .git)`
- missing path → `path does not exist`
- read of an absent file → `404 file not found`
- defense-in-depth: a symlink at `<repo>/TASKS.md` that resolves outside the repo root is rejected.

## No other write surface broadened
`/api/files` and its `~/.prometheus/workspace` sandbox are **untouched** — confirmed by
`test_api_files` + `test_api_files_security` passing unchanged. The only refactor to existing code is
extracting `/api/code`'s inline `repo/.git` check into `_resolve_coding_repo` (behavior-preserving;
`test_api_code` green; the message just got more specific).

## Tests — 150 web+coding passed
`tests/test_api_project_file.py` (side effects against a real git-repo fixture): the three files
write + read back **byte-identical**; the five rejection cases each assert their **distinct** reason;
404 on a missing read; blank-name + non-string-content rejected. Full web + coding suite green
(`test_api_files`/`security`, `test_api_documents`, `test_api_code`, all `test_coding_*`).

## Beacon side
Routes Loop Manager's read/write through this endpoint (PR stacked on Loop-Manager-Sprint-1 #9):
`OAraLabs/beacon-desktop` `feat/loop-remote-files`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
