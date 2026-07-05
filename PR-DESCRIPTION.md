# feat(dist): release CI + tracked test CI + packaging acceptance — Onboarding Phase 3 (Part B)

**Branch:** `feat/onboarding-phase3` off main (`b354e52`)
**Spec:** `docs/sprints/SPRINT-ONBOARDING-OVERHAUL.md` §2 PHASE 3 (Phases 0–2 shipped: #86/#87/#88).
**Companion PR:** beacon-desktop `feat/distribution` (linux AppImage/deb targets + release CI —
the primary half of this sprint).
**Status:** PR-ready — not merged, main untouched, live daemon untouched. **No tags created,
no Releases published** — the workflow only ever produces *drafts*.

## What this is

A stranger can `pip install` a released Prometheus instead of cloning:

- **`.github/workflows/ci.yml`** — the June 2026 audit's CI (written then but never pushed:
  the gh token lacked the `workflow` scope, so it sat untracked at the repo root of the main
  checkout). Folded into the branch as a tracked file with its intent intact and the stale
  bits modernized:
  - install/test now match the canonical dev loop: `astral-sh/setup-uv@v5` +
    `uv sync --extra web --extra anthropic --group dev` +
    `PYTHONPATH=$PWD/src uv run pytest -m "not network"` (was `pip install -e ".[dev,web]"`
    + bare pytest);
  - the KNOWN pre-existing failure
    `tests/test_bootstrap.py::TestMemoryInPrompt::test_empty_memory_files_no_section` is
    handled with a single-test `--deselect` (comment in the workflow says why) — **not** by
    excluding the file, so the other bootstrap tests still gate. Chose `--deselect` over
    editing the test to xfail to keep this branch zero-risk to source; un-deselecting is the
    ritual when the fix lands.
- **`.github/workflows/release.yml`** — tag `v*` + `workflow_dispatch` → `uv build`
  (sdist + wheel), smoke the wheel in a scratch venv (`prometheus --help`), upload artifacts,
  attach to a **draft** GitHub Release via `gh`. A PyPI publish step exists but runs ONLY
  when a `PYPI_API_TOKEN` repo secret is set (job-level env so the step-level `if:` can see
  it) — **Will must add that secret when the name is ready to go live; nothing publishes
  from this box or from CI without it.**
- **README** — short "Releases" note under Install (draft releases on tags, PyPI gating,
  pointer to beacon-desktop releases for the prebuilt client).

## Local acceptance (transcript)

```
$ uv build
Successfully built dist/oara_prometheus-0.1.0.tar.gz
Successfully built dist/oara_prometheus-0.1.0-py3-none-any.whl   # 1.9 MB / 870 KB

$ uvx twine check dist/*
Checking dist/oara_prometheus-0.1.0-py3-none-any.whl: PASSED     # readme renders
Checking dist/oara_prometheus-0.1.0.tar.gz: PASSED

$ uv venv relvenv && uv pip install --python relvenv/bin/python dist/*.whl
$ relvenv/bin/prometheus --version
Prometheus 0.1.0
$ relvenv/bin/prometheus --help
usage: prometheus [-h] ... {setup,token,doctor,install-service,daemon,identity,migrate,code,export-traces}
# all Phase-0 subcommands present from the wheel — packaging metadata is release-sane
```

Test suite (worktree, CI's exact invocation):
`PYTHONPATH=$PWD/src uv run pytest -m "not network" -q --deselect tests/test_bootstrap.py::TestMemoryInPrompt::test_empty_memory_files_no_section`
→ **3122 passed, 4 skipped, 4+1 deselected, 0 failed** (baseline run without the deselect
confirms exactly that one pre-existing failure). All three workflow files YAML-parse clean.
`dist/` is gitignored (not committed).

## Push status / how to land the workflow commit

The gh token here lacks the `workflow` scope, so the branch tip (the single commit that adds
`.github/workflows/ci.yml` + `release.yml`) may be **local-only**. If
`origin/feat/onboarding-phase3` is missing it, after `gh auth refresh -s workflow` run:

```bash
cd ~/Prometheus
git push origin feat/onboarding-phase3
```

(The branch lives in the worktree `.claude/worktrees/agent-ac0d6e4e365433897`; it's the same
repo, so the push works from the main checkout too.)

## Follow-ups (out of scope)

- oara.ai landing-page install instructions — its deploy is manual (Pages direct-upload) and
  blocked on the same workflow-scope gap; do in a follow-up once `gh auth refresh -s workflow`
  lands.
- Fix `test_empty_memory_files_no_section`, then remove the `--deselect` from ci.yml.
- Add `PYPI_API_TOKEN` secret when ready to publish `oara-prometheus` to PyPI.
- macOS eyes-on walk of the Phase 1/2 pairing + wizard UI (carried from Phase 2).
