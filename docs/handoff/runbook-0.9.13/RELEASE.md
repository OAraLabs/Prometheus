# 0.9.13 runbook: release

Every merge, approval and publish here is Will's go. The live config steps for the deploy are in `LIVE-CONFIG.md`, next to this file.

## 1. Version bump

- #821 (`chore(release): version 0.9.13`) holds the version strings and the 0.9.13 CHANGELOG section, which is the release notes.
- Will merges it. It is not tagged.

## 2. Tag and publish

1. Tag #821's merge commit `v0.9.13`.
2. Reject the CI macOS job, then build and notarize on the Mac.
3. The `release` and `pypi` environment approvals are Will's.
4. Publish the GitHub release with the label **Latest**, only after the three Mac files are attached.
5. Check that `releases/latest/download/prometheus-mac.json` says 0.9.13, and that PyPI lists `oara-prometheus` 0.9.13 with an sdist.

## 3. Homebrew tap: 0.9.8 → 0.9.13 (after PyPI publishes)

A PR on `OAraLabs/homebrew-tap`, which Will approves and merges. Never merge it yourself.

**This is not only `url` and `sha256`.** The package's base dependencies changed in 0.9.9, and the formula was last bumped at 0.9.8:
- PyMuPDF (AGPL-3.0) was replaced by `pypdfium2` and `Pillow`.
- `distro` was added (Linux only).

The formula still has `depends_on "pymupdf"` and no resource for any of the three. Homebrew installs resources without dependency resolution, so a url+sha-only bump would build a 0.9.13 that is missing them. Checked 2026-10-16:
- `pillow` is in homebrew/core at 12.3.0 (MIT-CMU), which meets the package's `>=12.3.0`;
- `pypdfium2` is not in homebrew/core;
- PyPI has sdists for `pypdfium2` 5.14.0, `pillow` and `distro`.

Steps:

1. **The main package.** Download the 0.9.13 sdist and hash it locally; don't copy the digest from PyPI's metadata. Then:
   - check that the size and hash match PyPI's published values;
   - check that `PKG-INFO` and `pyproject.toml` inside the tarball say 0.9.13;
   - set the formula's `url` and `sha256`.
2. **Dependencies.**
   - Remove `depends_on "pymupdf"`, and remove `pymupdf` from `exclude_packages`.
   - Add `depends_on "pillow"`, and add `pillow` to `exclude_packages`. This is the same treatment as `cryptography` and `pydantic`. `depends_on` lines must stay above `pypi_packages`, which `brew audit` enforces.
   - Regenerate the resource blocks with `brew update-python-resources --ignore-main-package-cooldown oaralabs/tap/oara`. Don't edit them by hand.
     - The flag exempts only the formula's own package from Homebrew's 24-hour release cooldown, which is hard-coded and has no override. Don't patch Homebrew.
     - Resources come from Homebrew's resolver (the newest versions the constraints allow), not from `uv.lock`, so expect them to move.
3. **`pypdfium2` is the risk.** It has no core formula, so it becomes a resource built from its sdist. That build may try to download PDFium binaries, which a Homebrew build may not be allowed to do. If the build fails, or the formula needs anything beyond a plain resource, stop and bring it to Will. Don't vendor binaries or work around the sandbox.
4. **Check the PR branch on a clean install,** on a Linux host with Homebrew. A macOS install is the real gate for Mac users, so run it on the Mac too before Will merges.

       brew uninstall --force oaralabs/tap/oara; brew untap oaralabs/tap
       brew tap oaralabs/tap
       git -C "$(brew --repository oaralabs/tap)" fetch origin <pr-branch>
       git -C "$(brew --repository oaralabs/tap)" checkout <pr-branch>
       brew audit --strict oaralabs/tap/oara
       brew install oaralabs/tap/oara       # fully qualified: a bare tap is untrusted
       oara --version                       # must print 0.9.13
       brew test oaralabs/tap/oara          # the formula's test boots the daemon into setup mode

   Also check that a PDF extracts and that the daemon starts without an import error for `pypdfium2`, `PIL` or `distro`.
5. **Open the PR.** Say what changed and why: the 0.9.9 dependency swap, the hash checked locally, and the checks above with their output. List what was not run. Will approves the merge.
6. **After Will merges, check again from the tap's `main`:**

       brew update
       brew uninstall --force oaralabs/tap/oara; brew untap oaralabs/tap
       brew tap oaralabs/tap
       brew install oaralabs/tap/oara
       oara --version                       # must print 0.9.13

## 4. Deploy (Will's go)

- Use `scripts/deploy.sh <sha> --prepare-only` first, and arm the maintenance window before the restart.
- Inside the window, before the restart, make the live config edit in `LIVE-CONFIG.md`.
- After the restart, check the running SHA, the model choices and `oara doctor`. Print check names and statuses only.

## 5. Day after

The config cleanup in `LIVE-CONFIG.md`: `reserved_output` and the 5 unmatched overrides.
