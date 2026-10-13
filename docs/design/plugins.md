# Plugins for Prometheus: optional features, signed, pinned, installed on demand

**Status: design approved by Will on 2026-10-13, with the decisions recorded below. Nothing is built.** The
plugin core waits until 0.9.9 is released. The PDF swap waits for an extraction comparison on real PDFs, which
Will approves first.

Nothing in this note has been run except the size arithmetic, which comes from `uv.lock`'s recorded wheel
sizes, not from a build. Every claim about how Prometheus works today cites the file it was read from, at
`origin/main` 44c2c41. Claims about Beacon are made without file paths, because its repository is private;
the internal market note cites them.

This note follows Will's decisions of 2026-10-12:

- the word is **plugins**;
- a plugin may carry code, skills and MCP config;
- v1 is official plugins only;
- voice is the **Speech plugin**;
- there are four shelves.

Where reading the repo turned up a real problem with one of those decisions, it is flagged, not re-opened.

## For Will (one page)

### What this is

A **plugin** adds an optional feature to Prometheus itself, on demand, without making the app download bigger
for everyone. In v1 every plugin is official:

- built by the same release job as the app, from the same `uv.lock`;
- signed by the same team and notarized;
- unpacked outside the signed bundle, under `~/Library/Application Support/Prometheus/plugins/`.

The app accepts a plugin only if its SHA-256 is one the signed app itself carries. Something other than the
installer then has to prove it worked: the code signature, then a smoke test in a fresh process, before
anything says "installed". Each plugin has one id that is the same everywhere: in the CLI, in chat, in
Beacon, on the market page and in pip.

### My recommendation, per question

1. **Which extras.**
   - **Plugins:** Browser and Speech. Computer use later, once the macOS permission prompts are proven for
     the app.
   - **Bundled in the app:** Push (0.1 MB of pure Python), and PDF reading, through the library swap in
     item 2.
   - **Never shown as a plugin:** the skill-duplicate check. Rename its extra to `skill-dedupe` and keep
     `skills` as a quiet alias.
2. **PDF.** Swap PyMuPDF for **pypdfium2** (text) and **Pillow** (image downscaling), in every install, and put
   PDF back in the app with no plugin. That is about 8 MB of wheels instead of 23 MB. Measure extraction
   quality on a corpus first. This is a reading of licence labels, not legal advice.
3. **Format.**
   - A plugin is laid out as a Claude-style bundle: `.claude-plugin/plugin.json`, `skills/<name>/SKILL.md`,
     `.mcp.json`. Our build data goes in `metadata.prometheus`, the free-form field Claude Code ignores.
   - Code (installed wheels) sits beside them in `site-packages/`.
   - Our own plugins are therefore valid Claude-style bundles. Later, a third-party bundle could be accepted
     for its skills and MCP config only, never its hooks, commands or binaries.
4. **Build and sign.** CI builds the plugins **before** the app in the same release job. It signs every
   Mach-O, notarizes each zip and hashes it. The app embeds those hashes and is then signed. So the signed app
   is the trust root, and `plugins.json` beside `prometheus-mac.json` is a copy for people and the website.
5. **Install.**
   - The daemon owns the work behind `oara plugin add|remove|list|update`, `/plugin` in chat, and a REST route
     for Beacon.
   - The steps: stage, check the hash, check the signature, smoke-test, then switch on with one atomic rename.
   - A plugin takes effect on the next start (`POST /api/restart` exists).
   - A plugin must match the app's version and lock exactly. After an app update, the old one sits inactive and
     shows "update available"; nothing downloads until someone asks. The previous version is kept until its
     replacement passes.
   - It can install offline from a zip.
6. **Linux and pip.** Same ids, same status words. `oara plugin add speech` prints the right
   `pip install 'oara-prometheus[speech]'` (or uv, brew, Docker line) and does not run it.
7. **Messages.** One helper, `fix_for("speech")`, replaces the ~30 hand-written `pip install` lines. In the app
   it says "Add the Speech plugin"; elsewhere it gives the pip line.
8. **Beacon.**
   - Reads `GET /api/plugins`, the same function `oara doctor` prints.
   - Only owner devices may press Add. The dialog shows size, licences, and exactly what is downloaded from
     where.
   - Then it follows progress frames and offers a restart. The contract is in section 8.
9. **Market.**
   - One `oara.ai/market` page with the four shelves.
   - Plugins are generated from `plugins.json` at deploy time, so the shelf cannot drift from the release.
   - Connectors and Integrations are short, curated lists.
   - No accounts, no analytics, no tracking parameters, no script on load.
   - The design is in `oara-site/internal/market.md`.
10. **Security.**
    - Only an operator credential (global token or owner device) may add or remove a plugin. Never a scoped
      device, never the agent.
    - A plugin's tools and MCP servers pass the same permission gate as everything else.
    - The plugin directory is write-denied to agent shells by one more line in the existing macOS sandbox
      profile.
11. **First slice.** In order:
    1. the PDF swap, which needs no plugin;
    2. Push in the app, the extra renames and the `install_kind()` helper;
    3. the plugin core, with `oara plugin`, `/plugin` and `fix_for`;
    4. the Browser plugin in CI;
    5. `/api/plugins` for Beacon.

    The Speech plugin (without piper) follows the Browser plugin. All of this waits until 0.9.9 is out.

### Will's decisions (2026-10-13)

1. **Third-party plugins: never for code.**
   - No plugin code that was not built by this repository's release job is ever loaded into the daemon.
   - Third-party native code would need the `disable-library-validation` entitlement on the app (cost in
     10.1).
   - Pure-Python third-party code would load, but no mechanism can police it in-process (FOUNDATION 2.3a).
   - Third-party *skills* and *MCP config* are not decided. The shapes for them are in 3.3, and nothing is
     built for them.
2. **The Speech plugin ships without piper.** Speech-to-text by faster-whisper (MIT), and spoken replies by
   macOS's built-in `say`. No GPL code is distributed.
3. **PDF: pypdfium2 + Pillow, after a comparison.** Extraction is first tested against PyMuPDF on at least
   five real PDFs (scanned, multi-column, forms), and Will approves that comparison before the swap PR.
4. **No pricing.**
5. **Plugin auto-update is off.** After an app update, a plugin built for the previous version shows
   **"update available"** (5.4). Nothing downloads until someone asks.
6. **Packs stay working for now.** Plugins are the future user-facing format. The FOUNDATION Part 2
   amendment lands with the plugin core.
7. **Order.** The plugin core waits until 0.9.9 is out.

### Real problems found in the decisions (flagged, not re-opened)

1. **"Packs" already exist, and overlap with plugins.** FOUNDATION Part 2 packs are skills plus panels from
   `~/.prometheus/packs/`, served at `GET /api/packs` (finding 1). With plugins carrying skills, there would
   be two formats for the same thing.
   - *Decided:* packs keep working for now; plugins become the user-facing format; and FOUNDATION Part 2 is
     amended with the plugin core.
   - The amendment names section 10.1 as the mechanism its 2.4 asks for, for first-party code only.
   - It also makes the pack loader's rule (quarantine until accepted) the rule for any later third-party plugin
     skills.
2. **The Speech plugin's voice output was GPL.** The locked `piper-tts` 1.4.2 is **GPL-3.0-or-later** (PyPI
   metadata; the upstream repository is named `piper1-gpl`).
   - *Decided:* Speech ships without piper. Input is faster-whisper (MIT), and output is macOS's built-in
     `say`.
   - That needs a `say` path in `tts.py`, which today looks only for espeak and piper
     (`tools/builtin/tts.py:86-91`).
   - Whether `say`'s audio needs converting for Telegram voice notes is still to be checked.
   - `piper-tts` leaves the `speech` extra's plugin build. Whether it also leaves the pip extra for terminal
     installs, where pip installs it from PyPI and we distribute nothing, is a separate question for the
     Speech PR.
3. **"Voice" is still a user-facing word outside the plugin.** `oara`'s interactive chat has a `--voice` flag
   and `:voice on` (`__main__.py:970-979`), and the config section is `gateway.voice` (`cli/voice.py:297-302`).
   Decision 4 covers the plugin's name. Whether those also become "speech" is a later, separate rename.
4. **The `plugin` skill source is a placeholder.** `SkillDefinition.source` is documented as `"builtin" |
   "user" | "plugin"` (`skills/types.py:15`), but no loader produces `"plugin"`. The loader produces
   `"builtin"`, `"user"` and an undocumented `"auto"` (`skills/loader.py:36`, `:60`), and the `cwd` argument
   is "reserved for future plugin loading" (`:76`). The clash is with the pip extra `skills`, which would have
   shown as a "Skills plugin". Resolution (section 1): the extra is renamed and hidden. `"plugin"` becomes a
   real source, used only for skills that come from an installed plugin, with the plugin's id beside it.

## What I found that changes the plan

1. **Packs.** `src/prometheus/packs/loader.py:1-16`: a FOUNDATION Part 2 pack is a directory under
   `~/.prometheus/packs/<name>/` (`loader.py:45-50`) with a `pack.yaml` and a `pack_api` version (`:38-40`).
   - It registers **skills**, quarantined in the skill-draft flow until a person accepts them, and **panels**
     (declarations only). A `tools/` directory is refused by name.
   - It loads at boot (`daemon.py:998-1008`) and is served at `GET /api/packs` (`web/server.py:2597-2614`),
     which is operator-only (`web/route_access.py:213-214`). The user guide documents it
     (`docs/guide/features.md:302`).
   - FOUNDATION 2.1: third-party code "does not import Prometheus internals" (`docs/FOUNDATION.md:140-144`).
   - 2.3a moves tools to MCP, because in-process third-party code cannot be policed (`:204-215`).
   - 2.4 ends: a future version that "reintroduces in-process third-party code ... must name a real mechanism
     before that lands" (`:318-320`).
   - Plugins with code ARE in-process code: Playwright, faster-whisper, onnxruntime. Section 10.1 names the
     mechanism that makes first-party plugins acceptable. It does not extend to third parties.
2. **The app has no `oara` command.** The app install is "no terminal" (`docs/guide/install.md:34-50`). The
   launcher runs the bundled Python with `-I -B -m prometheus daemon` (`packaging/macos/launcher/main.swift:397`),
   and nothing puts `oara` on `PATH`. So for app users the way in is `/plugin` in chat and Beacon's button.
   `oara plugin` is the same code for terminal installs.
3. **`-I` decides how plugins are found.** Isolated mode ignores `PYTHONPATH` and the user site
   (`main.swift:389-393`). Only the daemon's own code at start can put a plugin on the path, and that is
   better anyway: it can refuse a stale or failed plugin before anything imports it.
4. **Two plugins are not only wheels.**
   - **Browser.** The browser tool launches Chromium (`tools/builtin/browser.py:134`), which Playwright
     downloads separately from Microsoft's CDN into `~/Library/Caches/ms-playwright` by default (Playwright's
     documentation). Prometheus sets no `PLAYWRIGHT_BROWSERS_PATH` and never runs `playwright install`. The
     only hint is "Run: pip install playwright && playwright install chromium" (`browser.py:48-51`).
   - **Speech input.** It loads faster-whisper's `"base"` model by name (`tools/builtin/whisper_stt.py:45`,
     `:207`), which the library fetches from Hugging Face on first use, unpinned.
   - **Speech output.** It finds piper with `shutil.which("piper")` (`tts.py:86-91`, `cli/voice.py:287`) and
     needs a model path set by hand (`cli/voice.py:297-302`).
   - **The pattern to reuse.** The skills encoder already fetches a pinned revision and checks each file's
     SHA-256 (`skills/similarity.py:10-20`, `:140-150`). Every runtime download a plugin needs reuses it.
5. **Licences inside wheels.** `av` (PyAV, BSD-3-Clause in its metadata) bundles FFmpeg libraries, whose
   licence depends on how they were built. The app build's licence scan reads package metadata only
   (`packaging/macos/build_app.py:530-553`), so it cannot see a native library's own licence. The plugin
   build needs a check for each bundled `.dylib` (section 4).
6. **No install hint is right for the app.** About 30 places tell the user how to install a missing extra, in
   three forms (`pip install 'oara-prometheus[x]'`, `uv add`, bare package names). They come from ad-hoc
   constants: `VOICE_EXTRA_FIX` (`cli/voice.py:56`), `_INSTALL_HINT` (`web/discovery.py:60`) and
   `_NOT_INSTALLED` (`browser.py:48-51`).
   - Two name a package that does not exist, `prometheus[...]`: `daemon.py:1798-1799` and
     `tracing/phoenix.py:66`.
   - There is no install-kind helper. Three places test `PROMETHEUS_INSTALL_KIND == "app"` inline
     (`config/local_pairing.py:86-89`, `web/bind.py:154-157`, `permissions/modes.py:43`).
7. **Push can stop the app booting.** With `push.enabled` and no `h2`, boot raises "pip install
   'oara-prometheus[push]'" (`web/launcher.py:310-319`). An app user cannot do that.
8. **Skills do not load from folders today.** The user and builtin loaders glob flat `*.md` files
   (`skills/loader.py:28`, `:53-56`). A Claude-style `skills/<name>/SKILL.md` is not loaded. Plugin skills
   need a folder-aware loader. That is a small change, and the same one a third-party bundle would need.
9. **MCP servers come from two sources today**, the YAML map and the daemon's REST store, merged at boot
   with YAML winning (`mcp/store.py:1-14`, `:293-318`). Plugin MCP config would be a third source.
10. **The `anthropic` extra is unused by the code.** No `anthropic` SDK import exists under `src/prometheus`.
    The provider uses `httpx` (`providers/anthropic.py:20`). The app ships the extra (`build_app.py:81`). This
    is a separate, small cleanup.
11. **Beacon installs the app once and never updates it.** An app found already in place that passes the
    checks is used as it is, with no version comparison. So an app version can change by hand, under old
    plugins. The design checks for a mismatch at every start, not only during a managed update.

## 1. Which extras become plugins, which are bundled, which are hidden

**Rule:** a code plugin is one `[project.optional-dependencies]` extra, with the same id. Its code is the
extra's locked closure **minus everything the app already ships**. It contains no Prometheus code and
registers nothing. It only makes `import playwright` (or `faster_whisper`) succeed for code already in the
signed app.

Sizes are each package's arm64/cp312 (or universal2) wheel size recorded in `uv.lock`, summed: compressed,
before thinning, not a build, not installed size.

| User-facing name | Id = extra | Beyond the app | Wheels | Also downloads | Recommendation |
|---|---|---|---|---|---|
| **Browser plugin** | `browser` (`pyproject.toml:96`) | playwright 1.60.0, greenlet, pyee | **42.6 MB** | Chromium from Playwright's CDN, about 150 MB (measure the headless shell) | **Plugin, first** |
| **Speech plugin** | `speech` (new; `voice` kept as an alias, `:129-134`) | 21 packages: av, onnxruntime, scipy, numpy, piper-tts, faster-whisper, ctranslate2, tokenizers, huggingface-hub… | **104.7 MB**; about 91 MB without piper-tts | a Whisper model from Hugging Face | **Plugin, without piper-tts** (decided); spoken replies use macOS `say` |
| **Computer-use plugin** | `computer` (`:112`, exact pin, not in `full`) | cua-driver 0.28.2, one universal2 wheel | 40.9 MB before thinning | none | **Plugin later**: macOS Accessibility and Screen Recording grants for the app's identity are unproven |
| *(none: hidden)* | `skill-dedupe` (new; `skills` kept as an alias, `:140`) | numpy, onnxruntime, tokenizers, hub client: 13 packages | 41.2 MB | the encoder, about 134 MB (`similarity.py:12`) | **Not a plugin.** An internal duplicate check, not a feature anyone asks for. Never listed |
| *(built in)* | `push` (`:95`) | h2, hpack, hyperframe | 0.1 MB | none | **Bundle in the app** (finding 7) |
| *(built in)* | base (`:46`, PyMuPDF) | n/a | n/a | none | **Swap the library** (section 2). The name "PDF plugin" stays unused unless Will picks the plugin route |
| *(built in)* | anthropic, mcp, slack, discord, discovery | already in the app (`build_app.py:81`) | n/a | n/a | n/a |
| n/a | evals, dev | developer tools | n/a | n/a | never |

**The extra renames.** Add `speech` and `skill-dedupe` with the same packages. Keep `voice` and `skills` as
aliases, so every existing `pip install 'oara-prometheus[voice]'` keeps working. The `web` extra is the
precedent: it stays defined although its packages moved to base (`pyproject.toml:172-176`). A test pins
that each alias equals its new name. `full` is unchanged.

**Overlap.** Speech and the hidden dedupe extra share onnxruntime, numpy and tokenizers. Plugins are
self-contained, so two may carry the same wheel twice. One lock means identical versions, and the first on
the path wins. The overlap costs disk, never correctness.

## 2. PDF: change the library, put it back in the app

**What PyMuPDF does today.** Exactly two things:

1. **Text extraction**, `utils/file_extract.py:101-126`: `page.get_text()` per page up to `MAX_EXTRACT_BYTES`
   (30,000, `:18`), with a truncation note, and a placeholder for image-only PDFs. In the app it returns
   "[PDF file: name — install PyMuPDF to extract text]" (`:107`), which an app user cannot act on.
2. **Image downscaling** before a picture goes to a model, `gateway/image_prep.py:103-151`: `fitz.Pixmap`
   shrinks by powers of two above `2 × 2576` px. PyMuPDF was chosen because it was already a dependency
   (`:15-18`). The power-of-two limit is documented as costing legibility (`:42-53`).

**Options.** Licences are from each project's PyPI metadata, read 2026-10-12. **This is a reading of licence
labels, not legal advice.**

| | Licence | arm64 wheel | Text | Images | Native code |
|---|---|---|---|---|---|
| PyMuPDF 1.27.2.3 (today) | AGPL-3.0 or Artifex commercial | 23.3 MB (52 MB installed, `macos-app-installer.md:37`) | yes | yes, powers of two | yes |
| **pypdfium2 5.14.0** | BSD-3-Clause or Apache-2.0, plus "dependency licenses" for PDFium's bundled third-party code | 3.5 MB | yes (PDFium, Chrome's PDF engine) | renders PDF pages; cannot decode PNG/JPEG | yes |
| pypdf 6.20.1 | BSD-3-Clause | 0.4 MB, pure Python | yes; weaker on complex layouts, slower | no | no |
| **Pillow 12.3.0** | MIT-CMU | 4.8 MB | n/a | yes, any ratio (`thumbnail`, Lanczos) | yes |

**Recommendation: pypdfium2 for text, Pillow for images, as base dependencies for every install, and
PyMuPDF removed.**

- One code path everywhere, and PDF back in the app with no plugin: about 8 MB of wheels instead of 23.
- Pillow resamples at any ratio, so `image_prep.py` drops its power-of-two workaround (`:42-53`) and sends
  exactly 2576 px.
- pypdf is the fallback if PDFium's third-party licence set is a problem. It has nothing to sign, but its
  quality must be measured, and it does nothing for images.

**Before switching, measure.** Run both extractors over real PDFs (forms, two-column papers, scans, non-Latin
text) and compare. Keep the image-only placeholder. Two things to verify, not assume:

- pypdfium2's documentation describes PDFium as not thread-safe, and uploads can arrive concurrently, so
  extraction should take a lock.
- PDFium's third-party licence file must reach the app's licence directory, because the metadata scan will
  not find it (finding 5).

Then `--without pymupdf` (`build_app.py:86-90`, `:757-777`) and the "ships without PyMuPDF" lines
(`install.md:46`, README) go.

## 3. The plugin format

### 3.1 Layout: a Claude-style bundle, plus code

```
speech/                                  one plugin, as unpacked
  .claude-plugin/
    plugin.json                          the manifest (3.2)
  site-packages/                         CODE: installed wheels, first-party only
  skills/
    transcribe-a-recording/SKILL.md      SKILLS: Claude-style folders (finding 8)
  .mcp.json                              MCP CONFIG: servers keyed by name; ${PLUGIN_ROOT} substituted
  licenses/                              each copyleft package's own text, plus THIRD-PARTY-LICENSES.txt
  RECORD                                 SHA-256 of every file above
```

This follows the standard Claude Code layout: the manifest at `.claude-plugin/plugin.json`, one
`skills/<name>/SKILL.md` per skill, MCP servers in `.mcp.json`, and `${CLAUDE_PLUGIN_ROOT}` for the plugin's own
path (Claude Code plugin reference). `site-packages/` is our addition. Other tools ignore it.

### 3.2 `plugin.json`

```json
{
  "name": "speech",
  "displayName": "Speech",
  "version": "0.9.9",
  "description": "Talk to Prometheus and hear it answer.",
  "license": "MIT",
  "metadata": {
    "prometheus": {
      "format": 1,
      "app_version": "0.9.9",
      "lock_sha256": "…",
      "extra": "speech",
      "team_id": "53JM8W47RL",
      "adds": ["speech-to-text (transcribe)", "spoken replies"],
      "code": {"site_packages": "./site-packages", "imports": ["faster_whisper", "ctranslate2", "onnxruntime"]},
      "licenses": [{"package": "faster-whisper", "version": "1.2.1", "license": "MIT"}],
      "downloads": [{"what": "Whisper base model", "host": "huggingface.co", "revision": "…", "files": []}],
      "smoke": {"argv": ["-m", "prometheus.plugins.smoke", "speech"], "expect": "speech ok"},
      "record_sha256": "…"
    }
  }
}
```

The standard fields mean what they mean in Claude Code (`name`, `displayName`, `version`, `description`,
`license`). Everything of ours goes under `metadata.prometheus`, which the Claude Code reference describes as
"Free-form object for your own data. Claude Code doesn't read it". An unknown *top-level* key would be stripped
with a warning there. `format` is our contract version, separate from the app version, in the way `pack_api`
already is (`loader.py:38-40`).

### 3.3 What each component does in Prometheus

| Component | v1 (official) | Later (third-party bundle) |
|---|---|---|
| **Code** (`site-packages/`) | Appended to `sys.path` at start (5.2). Registers nothing | **Never** (10.1) |
| **Skills** (`skills/<name>/SKILL.md`) | Loaded with `source="plugin"` and `plugin="<id>"`, trusted like builtins, because their bytes are pinned by the signed app. Listed on the Skills shelf as "from the Speech plugin", and each can be switched off | Quarantined in the existing skill-draft flow until a person accepts them, the FOUNDATION 2.3 rule (`FOUNDATION.md:194-196`) |
| **MCP config** (`.mcp.json`) | A third source in `merged_server_configs` (`mcp/store.py:293-318`), below YAML and the REST store on a name clash, with server names prefixed by the plugin id. Every call is gated like any MCP call | Allowed only with the **global token**. A bundle's `.mcp.json` chooses a command to spawn, which is the reason MCP definitions are global-token-only today (`web/server.py:2638-2651`) |
| Hooks, commands, agents, `bin/`, LSP, monitors, output styles | **Not accepted**: the build refuses them | **Not accepted.** Hooks would run shell commands at turn events. Prometheus's own extension points for turns are operator hooks and the pillar contract (`docs/contracts/hooks.md`) |

**Compatibility, stated plainly.** Our official plugins load in Claude-style tools as skills and MCP config.
Their code part is ignored there. A Claude-style third-party bundle could later load in Prometheus as skills
and MCP config, never as code.

## 4. Building and signing plugins in the release job

This extends `.github/workflows/release-macos.yml` and `packaging/macos/build_app.py`, and inherits their
rules:

- nothing resolves fresh;
- the wheel is built before the certificate reaches the runner (`release-macos.yml:144-153`);
- secrets come from the `release` environment, behind a required reviewer (`:70-71`);
- everything is signed by fingerprint (`build_app.py:470-485`).

**Plugins first, app last:**

1. For each plugin in `PLUGINS = ("browser", "speech")`:
   1. `uv export --frozen --no-dev --no-emit-project --hashes` with the app's extras plus the plugin's extra,
      minus the app's own export, using the same block filter as `filter_requirements` (`build_app.py:277`).
      Refuse if the difference is empty or contains `oara-prometheus`.
   2. `uv pip install --no-config --python <bundled python> --target <plugin>/site-packages --no-deps
      --require-hashes --only-binary :all:`, as the app does (`build_app.py:638-639`).
   3. Thin to arm64, prune `__pycache__`, and compile `unchecked-hash` bytecode (`:652-654`).
   4. Copy the plugin's skills and `.mcp.json` from the repo: `plugins/<id>/` in the tree, reviewed like any
      source.
   5. Sign every Mach-O with `codesign_argv` (`:470-485`): hardened runtime, timestamp, our fingerprint, **no
      entitlements**. One planned exception: Playwright's bundled `node` is a separate program run as a child,
      not loaded into Python. If, as expected for a JIT engine, it needs `com.apple.security.cs.allow-jit`,
      that one executable gets that one entitlement, read from its upstream signature. The app's Python never
      does. Prove it; don't assume it.
   6. Write `RECORD` and `plugin.json`. Then run the checks:
      - Every bundled `.dylib`/`.so` maps to a licence text (finding 5).
      - There is no `prometheus/` package, `.pth`, `sitecustomize`/`usercustomize` or console script.
      - There is no hooks, commands, agents or `bin/` directory.
      - Every module named in `imports` imports in the signed environment under the hardened runtime.
   7. `ditto -c -k --keepParent` to `Prometheus-plugin-<id>-<version>-arm64.zip`, then take its SHA-256.
      Notarize with the existing `notarize()` flow minus stapling (`build_app.py:728-746`). A zip cannot be
      stapled, and notarization does not change it, so the hash stands. A refused plugin fails the run; none is
      published unnotarized.
2. **Runtime downloads are pinned in CI.**
   - **Browser:** download the Chromium headless-shell build this Playwright version names, from the URL
     Playwright would use. Record its URL, size and SHA-256, and launch it once on the runner.
   - **Speech:** pin the Whisper model as a revision plus per-file SHA-256s, like `similarity.py`.
   - Prometheus's own downloader fetches them on the user's Mac. Playwright's documentation says nothing about
     verifying its downloads.
3. **Then the app.** The assembler writes `Contents/Resources/plugins.lock.json` (each plugin's asset, size,
   SHA-256, `RECORD` hash and pinned downloads) **before** signing. Then the app is signed, notarized and
   stapled as today. The sealed app names the exact bytes of every plugin for its version.
4. **Publish** in the same draft-release upload (`release-macos.yml:244-262`): one zip per plugin, plus
   **`plugins.json`**, which is the `plugin.json` entries, each zip's `size`, `installed_size` and `sha256`,
   and the release's `app_version` and `lock_sha256`. `team_id` there is display only, as in
   `prometheus-mac.json` (`build_app.py:192`).

**Every release rebuilds every plugin.** A plugin is bound to one app version and one lock (5.4). Matching
plugins across versions would need a compatibility rule nobody can test.

## 5. Install, find, update, remove

### 5.1 Where plugins live

```
~/Library/Application Support/Prometheus/plugins/
  state.json                     installed, version, last result; atomic writes
  browser/0.9.9/                 one directory per app version; never written after activation
    .claude-plugin/plugin.json
    site-packages/
    ms-playwright/               PLAYWRIGHT_BROWSERS_PATH points here, so remove takes Chromium too
  speech/0.9.9/
    …
    models/                      the pinned Whisper model, fetched at add time
  .staging/                      same volume, so activation is a rename
```

It sits beside the app's pairing directory (`config/local_pairing.py:96-97`). It is outside `~/.prometheus`,
so the pack loader never sees it (`loader.py:45-50`). It is outside the sealed bundle
(`macos-app-installer.md:135-137`). It is not in `~/Library/Caches`, which the system may purge.

### 5.2 How the app's Python finds them

At the top of `prometheus/__main__.py:main` (`:1459`), before any dispatch, `prometheus.plugins.activate()`
runs for `install_kind() == "app"` (5.6):

1. For each plugin marked `installed`, its `app_version` and `lock_sha256` must equal the app's, and its
   `RECORD` hash must equal the one in the app's own `plugins.lock.json`.
2. **Append**, never prepend, its `site-packages` to `sys.path`. Everything in the signed bundle wins any name
   collision, so a plugin can never replace `prometheus`, `ssl` or anything else the app imports.
3. Set what its code needs: `PLAYWRIGHT_BROWSERS_PATH` for Browser, and the model directory for Speech.
4. Register its skills and MCP servers with the loaders (3.3).
5. Record what was activated and what was skipped and why. That record is what `GET /api/plugins` and
   `oara doctor` read.

Children spawned as `python -m prometheus` pass through the same `main`. A console script installed with
`--target` names the build machine's Python, so any CLI a plugin needs runs as `sys.executable -m <module>`.

**A crash at import must not become a restart loop.** The agent restarts on failure every 10 s
(`macos-app-installer.md:159-161`). Before importing from a newly activated plugin, the daemon writes `booting:
<id>@<ver>` to `state.json`, and clears it once it has served `/api/status`. If a start finds the marker still
set, it marks that plugin `failed: crashed during start` and does not activate it.

### 5.3 `add`, step by step

One function serves `oara plugin add`, `/plugin add` and `POST /api/plugins/{id}`:

1. **Source.**
   - **By id:** `https://github.com/OAraLabs/Prometheus/releases/download/v<app version>/<asset>`, taken from the
     app's own `plugins.lock.json`. The running version's tag, never `latest`. Redirects are followed one hop
     at a time, to https on `github.com` (this repository's `/releases/` path),
     `objects.githubusercontent.com` or `release-assets.githubusercontent.com`.
   - **By file:** a path the operator gives (`/plugin add ~/Downloads/Prometheus-plugin-speech-0.9.9-arm64.zip`)
     or a zip Beacon uploads. No network needed.
2. **Show it and get a yes.** Size, installed size, licences, and every runtime download with host and size,
   from the embedded table.
   - In chat this is an approval through the existing queue, the pattern `/press install` uses
     (`gateway/commands.py:3900-3946`).
   - Beacon shows its own dialog.
   - The CLI asks unless `--yes` is given.
3. **Download** into `.staging/<id>-<random>/`, hashing as it streams. Stop past the embedded size, and refuse
   unless the digest equals the embedded one, naming both.
4. **Unpack** with `ditto -x -k`, after checking the entry list: no absolute paths, no `..`, no links out of the
   tree. Check every file against `RECORD`: nothing extra, nothing missing.
5. **Signature.** `codesign --verify --strict -R='anchor apple generic and certificate leaf[subject.OU] =
   "53JM8W47RL"'` on every Mach-O. `verify_app.py:228-231` builds the same requirement.
6. **Runtime downloads**, by the pinned downloader into the staging tree. Hash-checked, one retry, and `.part`
   files are never kept (`similarity.py:140-150`).
7. **Smoke test** in a child process: the bundled Python, `-I -B`, the staged plugin activated as in 5.2, and a
   120 s limit. Browser runs with no network and opens `about:blank`. It needs exit 0 and the expected line,
   or it fails with the child's last stderr lines as the reason.
8. **Activate.** `rename(.staging/<id>-<random>, <id>/<version>)`, then write `state.json` atomically.
   - Rename gives exactly one winner where `unlink` does not (`macos-app-installer.md:82-85`).
   - A concurrent second `add` loses the rename and reports `already_installing`.
9. **Report** `installed` with `restart_required: true`. A failure at any step deletes the staging directory,
   leaves the previous version and `state.json` alone, and records `last_attempt: {step, reason, at}`. The
   status then reads "not installed (last attempt failed at signature: …)".

The plugin takes effect at the next start: tools register at boot (`__main__.py:385-387`). Beacon calls `POST
/api/restart`, which re-execs the same command line (`web/server.py:558-579`). Chat offers `/restart`. No
hot-loading in v1.

### 5.4 Versions: a plugin matches the app exactly

A plugin is active only when its `app_version` and `lock_sha256` equal the app's. `prometheus-mac.json`
already records `lock_sha256` (`build_app.py:206`). After an app update:

- **The old plugin shows "update available"** (status `update_available`). It stays on disk, inactive, reported
  as "The Speech plugin is for 0.9.8; this app is 0.9.9. Update available: 91 MB". Nothing downloads by itself.
- **Updating.** `oara plugin update`, `/plugin update` or Beacon's Update fetches the matching version by the
  steps above. Once the new version has started cleanly, versions older than the previous one are deleted.
  Moving back to the previous app version reactivates its plugin with no download.
- **No automatic update** (decided). Nothing downloads at the first start of a new version. A download the
  person did not ask for at that moment is the kind of traffic the first pillar says to name. A setting for it
  can come later, if anyone asks.

### 5.5 Remove

`remove <id>` marks it `removing`. At the next start, before activation, its directory is deleted and it is
marked `not_installed`. Nothing is deleted live from a running process.

- Its runtime downloads go with it, because they live inside its directory.
- Its skills and MCP servers disappear with it, because they were never copied elsewhere.
- `Prometheus --uninstall --purge-data` removes `plugins/`. A plain `--uninstall` keeps it, the same rule as
  `~/.prometheus`.

### 5.6 One install-kind function

`prometheus.install_kind()` returns `app`, `docker`, `brew`, `uv_tool`, `pip` or `source`.

- `app` is the launcher's `PROMETHEUS_INSTALL_KIND=app` (`main.swift:383`).
- Docker gets an `ENV` line.
- The others are recognised from `sys.prefix` and from whether this is a git checkout.

The three inline checks (finding 6) call it, and code plugins only ever activate for `app`.

## 6. Linux, pip, Homebrew, Docker: one vocabulary

The ids are the extras, everywhere. `oara plugin list` shows the same rows and status words on every install.
Only where the bytes come from differs:

| Install kind | `oara plugin add speech` does | Status comes from |
|---|---|---|
| app | section 5.3 | `state.json` plus activation |
| pip / source | prints `pip install 'oara-prometheus[speech]'` (`-e '.[speech]'` for source) and exits 0, having changed nothing, and says so | whether each module the extra provides is importable (`importlib.util.find_spec`, as `cli/doctor.py:546-552` does) |
| uv tool | prints `uv tool install --reinstall 'oara-prometheus[<current extras>,speech]'` | same |
| brew | says the formula has no extras and the feature needs a pip or uv install | same |
| docker | prints the `PROMETHEUS_EXTRAS` build argument (`Dockerfile:38`, `:60`) | same |

It never runs pip. A package manager owns those environments, and changing them from the daemon would leave a
record that no longer matches the disk.

## 7. The "missing feature" messages

One helper, `prometheus.plugins.fix_for(extra) -> str`:

- **app:**
  - not installed: "Add the Speech plugin: `/plugin add speech`, or Plugins in Beacon";
  - update available: "An update to the Speech plugin is available for this version: `/plugin update`";
  - failed: the recorded reason and the retry.
- **elsewhere:** the exact line from section 6.

All ~30 call sites move to it, including:

- `cli/voice.py:56` and its uses;
- `tts.py:56-67`, `whisper_stt.py:112-120`, `gateway/telegram.py:3406-3407`;
- `browser.py:48-51` and `:107-110`;
- `gateway/discord.py:381-382`;
- `daemon.py:1798-1799` and `:1870-1871`;
- `cli/doctor.py:694`, `:714`, `:1300-1301` and `:1398-1399`;
- `web/discovery.py:60`, `web/launcher.py:317-318`;
- `computer/cua.py:137-141`, `computer/integration.py:371-372`;
- `skills/similarity.py:233-234`, which names the hidden extra and never says "plugin";
- `utils/file_extract.py:106-107`, which goes away with section 2.

A test fails on any new literal `pip install 'oara-prometheus[` outside the helper. The two wrong package
names (finding 6) are fixed by the move.

## 8. Beacon: the contract

What a client may rely on. Beacon designs its own UI.

**Routes.** All are in `OPERATOR_ONLY` (`web/route_access.py:127-291`), so a scoped device gets `403
operator_only` from the default-deny middleware (`web/server.py:495-501`). None is in `SCOPED_ALLOWED` or
`PUBLIC_ROUTES`. A request with an `Origin` header gets 400, as `/api/restart` does (`server.py:567-571`).

| Route | Answers |
|---|---|
| `GET /api/plugins` | `{install_kind, app_version, plugins: [row]}`: every plugin this version knows, installed or not |
| `POST /api/plugins/{id}` | 202 `{job}`. 409 `already_installing`. 404 for an id this version does not know. On a non-app install: 200 `{action: "command", command}`, and nothing runs |
| `POST /api/plugins/{id}/file` | the same, from an uploaded zip (multipart; `python-multipart` is base, `pyproject.toml:73`). Refused unless its hash is in the app's table |
| `DELETE /api/plugins/{id}` | 202 `{state: "removing", restart_required: true}` |
| `POST /api/plugins/update` | 202; updates every plugin whose status is `update_available` |

**A row.** `oara doctor` prints the same function's output.

```json
{
  "id": "speech", "name": "Speech", "adds": "Talk to Prometheus and hear it answer.",
  "status": "installed", "version": "0.9.9", "source": "plugin",
  "size": 0, "installed_size": 0, "licenses": [], "downloads": [],
  "components": {"code": true, "skills": ["transcribe-a-recording"], "mcp_servers": []},
  "verified": {"sha256": true, "signature": true, "smoke": "ok", "at": "2026-…Z"},
  "reason": null, "restart_required": false, "last_attempt": null
}
```

**`status`** is exactly one of the following, shown as given and never inferred:

- `not_installed`;
- `installing`, with a `phase` (`downloading`, `verifying`, `fetching_runtime`, `smoke_test` or `activating`),
  plus `bytes` and `total`;
- `installed`;
- `pending_restart`;
- `update_available`, with `for_version` (the plugin on disk was built for another app version);
- `failed`, with `reason` and `step`;
- `removing`;
- `unavailable`, with the reason: not built for this platform or install kind.

`source` is `plugin` in the app and `extra` elsewhere.

**Progress** is sent over the existing WebSocket as `plugin_progress {id, phase, bytes, total}` to operator
sockets, ending in `plugin_status {row}`. A client that missed frames reads `GET /api/plugins`.

**The Add flow:**

1. `GET`.
2. Show the size, licences, and each download's host and size.
3. Confirm.
4. `POST`, then follow the frames to `installed` or `failed`.
5. "Restart Prometheus to finish" → `POST /api/restart`.
6. Read the row again. It says `installed` only when that start activated it.

**Where it sits in Beacon:** Plugins is its own place beside Config › Connectors (MCP servers, today) and
Workspace › Integrations (other apps, today). A plugin's MCP servers also appear under Connectors, labelled
with the plugin's name and removable only by removing the plugin.

## 9. The four shelves in the product

| Shelf | What the user hears | Where it lives today | What plugins change |
|---|---|---|---|
| **Skills** | what it knows how to do | `~/.prometheus/skills/` plus 3 builtins (`skills/loader.py:13-71`; `features.md:291`); `GET /api/skills/list` | plugin skills appear here, labelled with their plugin |
| **Connectors** | other tools it can use (MCP servers) | YAML plus the REST store (`mcp/store.py:1-14`); Beacon Config › Connectors | plugin MCP servers appear here, labelled |
| **Integrations** | other apps it connects to | Beacon-side, a registry of control surfaces (contract v1) with entries pointing at another app's base URL; nothing in Prometheus | none. OAra Voice is the first integration, and it is not a plugin |
| **Plugins** | optional features of Prometheus itself | new | this note |

The rule in user copy: **plugins go into Prometheus; integrations are other apps it connects to.** "Pillars"
and "hook contract" never appear in user copy. Instinct and Cognition get their own shelf, under a
user-facing name, when they ship. OAra Voice shipping as a plugin that also installs its service is a later
decision, not designed here.

## 10. Security

### 10.1 Why first-party plugin code is acceptable, and third-party code is not

FOUNDATION 2.4 asks for a named mechanism before in-process third-party code lands (`FOUNDATION.md:318-320`).
For official plugins, the mechanism has four parts:

- **Who built it.** Only this repository's release job, from `uv.lock`, with `--require-hashes --only-binary
  :all:` (`build_app.py:20-27`), behind the `release` environment's reviewer (`release-macos.yml:17-21`).
- **What it may be.** Only the bytes whose hash the signed app carries (section 4, step 3). `plugins.json` and
  the website are copies.
- **What it can do.** Nothing new by itself. Its code satisfies imports the app already makes. Its skills and
  MCP servers come from reviewed files in this repository and pass the same gate as any other.
- **What checks it at load.** Library validation (hardened runtime, no entitlements:
  `macos-app-installer.md:369-372`) refuses native code another team signed.

A third-party plugin with code fails the first two points by definition. Loading its native code would need the
`com.apple.security.cs.disable-library-validation` entitlement on the app. **Cost:** the daemon holds the API
token, every provider key and every conversation. With validation off, any dylib that any process of this user
can drop where the daemon imports from runs inside it, signed by anyone or ad hoc. "Write a file" becomes "act
as the daemon". Recommendation: never.

### 10.2 Who may add or remove

**Operator credentials only:** the global token or an owner device (`config/api_token.py:451-461`;
`docs/contracts/device-scoping.md` 6.2).

- A scoped device cannot, because of default-deny.
- An owner device is enough for an *official* plugin. MCP definitions are global-token-only because the caller
  chooses a command to spawn (`web/server.py:2638-2651`). An official plugin request chooses only which entry
  of a fixed, hash-pinned list. That puts it at the tier of `PUT /api/model`, not `POST /api/devices`.
- A *third-party* bundle's `.mcp.json` would choose commands, so it follows today's MCP rule: the global token
  only.
- `/plugin add` and `/plugin remove` are operator chat commands. They are not in `SCOPED_SLASH_COMMANDS`
  (`route_access.py:115-123`), and in chat they go through the approval queue (5.3, step 2).

**Never the agent.** There is no plugin tool. The agent may *suggest* one, through `fix_for`. On the app there
is no `oara` on `PATH`, but an agent shell could call the bundled Python with `-m prometheus plugin add`. So:

- `plugin\s+(add|update)` joins the bash patterns that ask in every mode but autonomous
  (`permissions/checker.py:199-210`, beside `pip\s+install`);
- under `cautious`, the app's default, every shell command asks anyway (`install.md:45`).

### 10.3 Plugins pass the gate like everything else

Installing Browser makes the browser tool available. It does not change what the tool may do. The permission
mode, the approval prompts and the tool's own navigation limits (`browser.py:28` onward) are unchanged. A
plugin's MCP servers are gated like any MCP server: every call asks unless the mode is autonomous. Beacon and
`/plugin list` say so in one line: "Its actions still ask under your permission mode."

### 10.4 Supply chain and tampering

| Risk | Mitigation | Remaining |
|---|---|---|
| A compromised upstream wheel | `uv.lock` hashes, `--only-binary`; a lock bump is a reviewed PR | a malicious release locked on purpose; same as the app |
| A compromised CI run | the `release` environment's reviewer; the wheel is built before the certificate is present (`release-macos.yml:144-153`) | same as the app |
| A swapped zip on GitHub, an interception, a wrong offline file | the SHA-256 from the signed app's table, then the team signature | none known |
| A swapped Chromium or model on a CDN | a CI-pinned SHA-256 checked by our downloader | the CDN can make it unavailable; that shows as `failed: download` |
| An old plugin replayed | it must match this app's version and lock | none |
| Resolving from PyPI on the Mac | impossible: the app has no pip (`build_app.py:422-452`) | n/a |
| Tampering after install (a `.py` file or skill is not code-signed) | `RECORD` checked at activation and by `oara doctor`; one `(deny file-write* (subpath …/plugins))` added to the macOS Seatbelt profile for agent shells (`permissions/confinement.py:294-301`); the path added to `security.denied_paths` for file tools | another process of this user can still write there, as it can write `~/.prometheus` today |
| A malicious skill or MCP config (the ClawHub class) | v1 is official only. Later third-party skills are quarantined until accepted, and third-party MCP needs the global token | a person can still accept a bad skill. The market's review checklist exists to make that rarer |
| A spoofed market page | static, and links only to this repository's release assets; nothing installs from the website | n/a |

**The cost of the `RECORD` check at every start** (about 100 MB hashed) needs measuring. The fallback is to hash
`RECORD` itself at start, and every file in `oara doctor` and at `update`.

## 11. Rollout

Each step is a draft PR. The test that fails on today's behaviour is written and run first. Nothing merges
without Will's go.

1. **PDF without AGPL** (section 2). First, the comparison against PyMuPDF on at least five real PDFs
   (scanned, multi-column, forms), shown to Will. After he approves it, the swap goes in as its own PR. The app
   gets PDF and better image downscaling back, **with no plugin.** Beside it goes a small PR that removes the
   unused `anthropic` extra (finding 10).
2. **Push in the app, the extra renames (`speech`, `skill-dedupe`, with aliases), and `install_kind()`.**
3. **The plugin core and the messages.**
   - `prometheus.plugins`: state, activation, verify, smoke, fetch, `fix_for`.
   - The folder-aware skill loader with `source="plugin"`, and the third MCP source.
   - `oara plugin`, `/plugin`, the doctor section, and the ~30 hint sites.
   - Tested with a fixture plugin.
4. **The Browser plugin in CI.** Build, sign, notarize, embed the hashes, publish, and pin Chromium. Proven on
   this Mac, then on the clean MacBook.
5. **`/api/plugins` and the frames**, handed to Beacon.
6. **The oara.ai market page**, after Will's OK (market note).

**Waits:**

- the Speech plugin, after the Browser plugin;
- the Computer-use plugin (prove the macOS prompts first);
- any third-party format;
- **everything from step 2 on, until 0.9.9 is out** (Will, 2026-10-13). The FOUNDATION Part 2 amendment lands
  with step 3.

## What has been proven, and what has not

**Read, not run:** every statement about today's code above, at the cited lines. Licences and wheel sizes as
declared by PyPI and recorded in `uv.lock` on 2026-10-12. The Claude plugin format as published in the Claude
Code plugin reference on 2026-10-12.

**Not proven, and not claimed:**

- Playwright's `node` re-signed under our team with the hardened runtime, and which entitlement it needs.
- onnxruntime, ctranslate2 and PyAV's FFmpeg loading under library validation after re-signing. The app's
  proven list did not include them (`macos-app-installer.md:371-373`).
- A dylib from a notarized, unstapled zip downloaded by a browser (quarantined), installed offline. This needs
  the clean machine.
- Installed sizes, Chromium headless shell's real size, and the `RECORD` check's cost at start.
- pypdfium2's extraction quality against PyMuPDF's.
