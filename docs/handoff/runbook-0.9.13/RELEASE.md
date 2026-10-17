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
     - **Run it on Linux** (the deploy box's linuxbrew). A Mac regenerate drops `distro`, a Linux-only dependency. If it has to run on a Mac, add `--extra-packages distro --exclude-packages cryptography,pillow,pydantic`, which reproduces the Linux set exactly.
     - The flag exempts only the formula's own package from Homebrew's 24-hour release cooldown, which is hard-coded and has no override. Don't patch Homebrew.
     - Resources come from Homebrew's resolver (the newest versions the constraints allow), not from `uv.lock`, so expect them to move.
3. **`pypdfium2` builds on Linux and macOS (rehearsed 2026-10-16).** Both rehearsals used 0.9.12, which has the same dependencies as 0.9.13. **For 0.9.13, ship the formula unpinned, exactly as trialled.**
   - **Linux** (Homebrew 7.0.6) was the full trial below.
   - **macOS arm64** was Will's rehearsal: pypdfium2 builds, and all checks pass.
   - **What passed on Linux:**
     - The resource regenerate gave 30 resources: `pypdfium2` 5.14.0 and `distro` 1.9.0 added, and patch bumps to fastapi, opentelemetry-api, starlette and websockets.
     - `brew audit --strict` and `--strict --online` were clean.
     - The clean install took about 4 minutes. `oara --version` printed 0.9.12, and `brew test` passed.
     - Both PDF fixtures extracted through the daemon's own code. Pillow downscaled a 6000×4000 image to 2576×1717.
   - **How it builds:** pypdfium2's sdist downloads PDFium (chromium/8076) from `github.com/bblanchon/pdfium-binaries` during the build. Its log says "Verification is off", so nothing in the formula checksums those bytes.
     - Homebrew allows network access during a build by default, for a formula without a `fetch` step, on Linux and macOS alike, so this works on both.
   - **PDFium pinning is a post-launch formula revision, not part of 0.9.13.**
     - The plan: make PDFium a checksummed resource per platform and stage it into pypdfium2's `data/<platform>/` before its build. That means the library, its version file and the headers. pypdfium2's own cache check then uses the staged copy instead of downloading. Build offline to prove it, and test on Linux and the Mac.
     - `brew update-python-resources` refuses a formula that carries the `pdfium-binaries` resources. Once PDFium is pinned, regenerate with `--print-only` and paste in the printed blocks, or remove the PDFium resources, regenerate, and put them back.
   - **Contingency only:** if a later build ever fails on one platform, exclude `pypdfium2` (add it to `exclude_packages`, with no resource). The daemon imports it only to read PDFs. Without it a PDF comes back as "[PDF file: … — PDF text extraction is unavailable on this install]", and everything else works. Say so in the tap PR and the release notes.
   - **The trial formula** is on the local branch `throwaway/deps-0.9.12` in the tap clone (`brew --repository oaralabs/tap`), commit `8ef1eb0`. It is not pushed.
4. **Check the PR branch on a clean install,** on a Linux host with Homebrew. `brew uninstall` autoremoves dependencies nothing else needs. Set `HOMEBREW_NO_AUTOREMOVE=1` to keep them, and `HOMEBREW_NO_AUTO_UPDATE=1` so the tap clone stays on the PR branch. A macOS install is the real gate for Mac users, so run it on the Mac too before Will merges.

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
6. **After Will merges, check again from the tap's `main`.**
   - **A clean install on Linux:**

         brew update
         brew uninstall --force oaralabs/tap/oara; brew untap oaralabs/tap
         brew tap oaralabs/tap
         brew install oaralabs/tap/oara
         oara --version                     # must print 0.9.13

   - **An upgrade on the Mac,** which has oara 0.9.8 installed from the tap. This is the path existing users take:

         brew update
         brew upgrade oara
         oara --version                     # must print 0.9.13

## 4. Deploy (Will's go)

- Use `scripts/deploy.sh <sha> --prepare-only` first, and arm the maintenance window before the restart.
- Inside the window, before the restart, make the live config edit in `LIVE-CONFIG.md`.
- After the 0.9.13 restart is verified, and still inside the window, remove `EnvironmentFile=` from the installed service unit, then `daemon-reload`, restart and verify. The steps are in `LIVE-CONFIG.md`, using `drop_unit_envfile.py`, not `oara install-service --force`, which would replace this box's hand-written unit.
- After the restart, check the running SHA, the model choices and `oara doctor`. Print check names and statuses only.

## 5. Day after

The config cleanup in `LIVE-CONFIG.md`: `reserved_output` and the 5 unmatched overrides.
