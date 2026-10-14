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
   - Its design is a separate note kept with the website, which is not public.
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

    The Speech plugin (the OAra Voice stack, section 1a) comes right after Browser. All of this waits until
    0.9.9 is out.

### Will's decisions (2026-10-13)

1. **Third-party plugins: never for code.**
   - No plugin code that was not built by this repository's release job is ever loaded into the daemon.
   - Third-party native code would need the `disable-library-validation` entitlement on the app (cost in
     10.1).
   - Pure-Python third-party code would load, but no mechanism can police it in-process (FOUNDATION 2.3a).
   - Third-party *skills* and *MCP config* are not decided. The shapes for them are in 3.3, and nothing is
     built for them.
2. **The Speech plugin is the OAra Voice plugin stack, the first plugin after Browser** (revised
   2026-10-13).
   - The stack is Kokoro TTS, whisper.cpp STT and Silero VAD. It is not Prometheus's piper-based `voice`
     extra, and piper is removed from the plan.
   - With nothing downloaded, spoken replies fall back to macOS's built-in `say`.
   - Voice must be ready to ship right after launch. Section 1a has the stack, its licence audit and the
     "voice ready to ship" checklist.
3. **PDF: pypdfium2 + Pillow, after a comparison.** Extraction is first tested against PyMuPDF on at least
   five real PDFs (scanned, multi-column, forms), and Will approves that comparison before the swap PR.
4. **No pricing.**
5. **Plugin auto-update is off.** After an app update, a plugin built for the previous version shows
   **"update available"** (5.4). Nothing downloads until someone asks.
6. **Packs stay working for now.** Plugins are the future user-facing format. The FOUNDATION Part 2
   amendment lands with the plugin core. Nothing needs converting: `~/.prometheus/packs/` on the mini, the one
   running install, exists and is empty (listed 2026-10-13, names only).
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
2. **GPL in the voice path, twice.**
   - The locked `piper-tts` 1.4.2 is **GPL-3.0-or-later** (PyPI metadata; upstream `piper1-gpl`). It is out
     of the plan now that Speech is the OAra Voice stack.
   - That stack's default TTS has the same problem one layer down. `kokoro-onnx` requires `phonemizer`
     (GPL-3.0-or-later) and `espeakng-loader`, which bundles espeak-ng (GPL-3.0 upstream), and imports both
     when it loads.
   - It can be avoided, at a cost; section 1a has the audit and the options.
   - The `say` fallback needs a path in `tts.py`, which today looks only for espeak and piper
     (`tools/builtin/tts.py:86-91`). Whether `say`'s audio needs converting for Telegram voice notes is still
     to be checked.
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
| **Speech plugin** | `speech` (what pip installs for it is open; section 1a) | the OAra Voice stack: the `oara_voice` services, onnxruntime, numpy, Kokoro's G2P, a signed `whisper-server` | not built; section 1a | Kokoro, Whisper and Silero models, pinned (section 1a) | **Plugin, first after Browser** (decided); fallback spoken replies use macOS `say` |
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

## 1a. The Speech plugin: the OAra Voice stack

**What it is.** Will's revision of 2026-10-13: Speech is built from the OAra Voice plugin, not from
Prometheus's `voice` extra. The voice plugin repo is Apache-2.0 in its own `LICENSE` and `pyproject.toml`; it is
not public yet, and Will intends to make it public. As that repo stands at its `main`:

| Part | What it runs | Port today | Model, downloaded on first run |
|---|---|---|---|
| STT | whisper.cpp's `whisper-server` (`POST /inference`) | 8081 | `ggml-base.bin` |
| TTS | `oara_voice.tts_servers.kokoro_server`, through `kokoro-onnx` | 8003 | `kokoro-v1.0.onnx` and `voices-v1.0.bin`, from the `thewh1teagle/kokoro-onnx` GitHub release `model-files-v1.0` |
| VAD | `oara_voice.vad`, Silero VAD v5 on onnxruntime | 8011 (moving to 8012) | `silero_vad.onnx`, from `snakers4/silero-vad` on raw GitHub |
| Bridge | `oara_voice.bridge` (to Prometheus REST 8005 and WS 8010) | 8015 | none |
| Opt-in GPU TTS | `chatterbox_server`; "weights are NOT downloaded automatically" | n/a | Chatterbox (about 1 GB per file) |

Today the stack runs as Docker services: `docker-compose.yml`, CPU and GPU profiles. As a Mac plugin, the
same processes would run as **services** (3.3) under the daemon, bound to 127.0.0.1.

- **The bridge's credential to the daemon** must be minted for it, a scoped device token. It must never be
  the global token. Today it reads `BRIDGE_BACKEND_TOKEN` or `PROMETHEUS_API_TOKEN`.
- **With nothing downloaded**, spoken replies fall back to macOS's `say`.

### Licence audit: everything the stack would ship

**Rules for this audit.**

- Facts come from package metadata (PyPI JSON and the wheels' own `METADATA` and licence files), from
  Hugging Face model cards (`cardData.license`) and from repository licence fields. All were read 2026-10-13.
- Anything those sources do not settle is labelled **UNCLEAR**.
- Sizes are the wheel or file as published, not installed size.
- **This is a reading of licence labels, not legal advice.**

The last column says whether the component ships in the plugin on the recommended path, the espeak-free one
below.

| Component | Licence (source) | Size | Ships |
|---|---|---|---|
| `oara_voice` (the plugin's own code) | Apache-2.0 (`pyproject.toml`, `LICENSE`); the repo is not public yet | about 100 KB of source | **yes**, once the repo is public (decision 1 below) |
| `kokoro-onnx` 0.6.1 | MIT (wheel `LICENSE`; the PyPI licence field is empty) | 0.03 MB | **no, not as packaged**: it imports `phonemizer` and `espeakng_loader` at load (`tokenizer.py`), and `Kokoro()` builds that tokenizer even with `is_phonemes=True` (`__init__.py:78`). A small runner of our own, its ONNX session and vocabulary without espeak, replaces it; its MIT licence allows reuse with attribution |
| `phonemizer` 3.4.0 | **GPL-3.0-or-later** (PyPI classifier, wheel `LICENSE`) | 0.1 MB | **no** |
| `phonemizer-fork` 3.3.2 (misaki's `[en]` extra) | **GPL-3.0-or-later** (PyPI classifier) | 0.1 MB | **no** |
| `espeakng-loader` 0.2.4 | **UNCLEAR**: the wheel declares no licence and carries no licence file. Its repository says MIT, but it bundles `libespeak-ng.dylib` (0.5 MB) and espeak-ng's data, and espeak-ng's repository is **GPL-3.0** | 9.9 MB (macOS arm64) | **no** |
| `misaki` 0.9.4 (Kokoro's own G2P, English) | Apache-2.0 (PyPI classifier, wheel `LICENSE`) | 3.6 MB | **yes**, on the espeak-free path, without its `[en]` extra (that extra pulls `phonemizer-fork` and `espeakng-loader`) |
| `spacy` 3.8.16 (misaki's English G2P imports it) | MIT (PyPI) | 6.5 MB wheel, plus its own dependencies, **not measured** | **yes**, on the espeak-free path |
| `en_core_web_sm` (spaCy English model) | MIT (Hugging Face card `spacy/en_core_web_sm`) | 12.8 MB | **yes**: pinned. Otherwise misaki downloads it at runtime (`misaki/en.py:499-503`) |
| `num2words` 0.5.14 (misaki's English G2P imports it) | **LGPL** (PyPI classifier; version not stated) | 0.2 MB | **yes**, with its licence text, as the app already does for LGPL packages |
| `onnxruntime` (1.26.0 in `uv.lock`) | MIT | 18.0 MB | **yes** |
| `numpy` (2.4.6 in `uv.lock`) | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 (PyPI) | 5.2 MB | **yes** |
| fastapi, httpx, websockets, pydantic, pyyaml, python-multipart | already in the app's base | n/a | no (already present) |
| `uvicorn[standard]` extras: uvloop, httptools, watchfiles | MIT / MIT / MIT (PyPI) | about 2 MB | **no**: the app runs plain uvicorn, and the services can too |
| `whisper-server` (whisper.cpp) | MIT (repository) | **not measured** | **yes**: built from a pinned tag in CI with Metal, signed by our team (today it is a Docker image) |
| Kokoro-82M v1.0, ONNX | Apache-2.0 (Hugging Face cards `hexgrad/Kokoro-82M`, `onnx-community/Kokoro-82M-v1.0-ONNX`). **UNCLEAR** for the file the plugin downloads today: that GitHub release carries no licence statement of its own | fp32 325.5 MB; fp16 177.5 MB; int8 92.4 MB | **yes**: fetched at add time, pinned. Prefer the `onnx-community` copy, whose card states the licence |
| `voices-v1.0.bin` | **UNCLEAR**: no licence statement at its release. The voices come from `hexgrad/Kokoro-82M` (Apache-2.0 card) | 28.2 MB (the server's comment says "~5 MB") | **yes**: pinned, once its source is settled |
| Whisper `ggml-base.bin` | MIT (Hugging Face card `ggerganov/whisper.cpp`) | 148.0 MB | **yes**: fetched at add time, pinned |
| Silero VAD v5, `silero_vad.onnx` | MIT (repository `snakers4/silero-vad`) | 2.3 MB | **yes**: pinned. Today it is fetched from the repository's `master` branch, which is not a pin |
| Chatterbox weights | **UNCLEAR**: the Hugging Face card now says MIT, but the plugin's own source says the weights "carry no explicit license tag" | about 1 GB per weight file | **no**: GPU tier |
| `chatterbox-tts` 0.1.7, plus `torch` 2.6, `torchaudio`, `librosa`, `s3tokenizer`, `resemble-perth` | MIT; torch Apache/BSD family; resemble-perth MIT (PyPI) | torch about 127 MB; resemble-perth 34.4 MB | **no**: GPU tier |
| macOS `say` | part of macOS | 0 | n/a: nothing distributed |

**Does Kokoro pull in espeak-ng? Yes, as the plugin uses it today.** `kokoro-onnx` 0.6.1 requires `phonemizer`
(GPL-3.0-or-later) and `espeakng-loader` (bundles espeak-ng, GPL-3.0 upstream), and loads both whenever it
loads.

**Can it be avoided? Yes:**

1. Phonemize English with misaki's own G2P. Its English class does not import espeak. Its fallback for unknown
   words is off by default (`misaki/en.py:496`, `:505`); the espeak fallback lives in a separate module.
2. Feed the phonemes to the Kokoro ONNX model through a small runner of our own.

**What it costs, to be measured before choosing:**

- **Unknown words go unspoken.** Words missing from misaki's lexicon get no pronunciation. The default
  `unk` is "❓", and that affects names, rare words and code identifiers.
- **spaCy and its English model come along.** They are about 19 MB of wheels before spaCy's own dependencies,
  which are not yet measured, plus `num2words` (LGPL).
- **English only at first.** misaki's other languages bring other dependencies, which this audit did not
  cover.

**The alternative** is to keep espeak and meet the GPL's terms. Will ruled that out for piper, and the same
reasoning would apply here.

### The espeak-free measurement (2026-10-13)

**Setup.**
- **Replies:** 50 real assistant replies from the running install's conversation history, picked for names,
  places and numbers. The texts stayed on Will's Mac; only the counts below are published.
- **Voice and model:** each reply, with markdown stripped, was spoken by Kokoro v1.0 (fp32 ONNX, voice
  `af_heart`) twice:
  - **A**, today's path: `kokoro-onnx` phonemizing with espeak-ng;
  - **B**, espeak-free: misaki 0.9.4's English G2P with no fallback, the phonemes fed to the same model.
- **Scoring:** both audios were transcribed by faster-whisper `small.en`, and each source word was aligned to
  each transcript.
  - A word counts as **mangled by B** when Whisper heard it in A and not in B.
  - Words Whisper missed in both are its own noise and are counted separately.
  - **Dropped** means misaki gave a word no phonemes at all, so it is not spoken.

| 2,740 words | In the text | Dropped by B | Wrong in B only | Wrong in A only | Wrong in both |
|---|---|---|---|---|---|
| Names (capitalised, not sentence-initial) | 503 | see below | 30 (6.0%) | 22 (4.4%) | 75 |
| Numbers | 473 | 0 | 68 (14.4%) | 31 (6.6%) | 73 |
| Other words | 1,764 | see below | 100 (5.7%) | 42 (2.4%) | 111 |
| **All** | **2,740** | **49 (1.8%)** | **198** | **95** | **259** |

**What the numbers mean, from listening and reading the transcripts:**

- **Names fail in the way that matters most.**
  - Misaki has no pronunciation for a name outside its lexicon, and **drops it silently**: the sentence is
    spoken with the name missing. That happened to project, product and company names, and to a few
    ordinary words ("verified", "denied", "backends").
  - espeak at least attempts every word, so a listener hears something name-shaped.
  - 49 words were dropped across 50 replies, about one per reply.
- **Numbers were worse on B,** but most of B's number misses sit in hashes, commit ids and file paths, which
  neither pipeline speaks usefully (259 words were wrong in both).
  - Plain quantities were mostly fine in both. One long byte count came out wrong in B.
  - Whatever path is chosen, replies need a speech-text pass that summarises hashes and paths instead of
    reading them.
- **Overall, B made about twice as many errors of its own as A:** 198 against 95.

**Verdict: names fail on the espeak-free path,** so option B is written up below, as Will asked. Neither
option is built.

### The third option, measured: misaki plus a permissive fallback for unknown words only (2026-10-13)

**Setup.** The same kind of 50 replies, re-selected by the same query: 2,760 words (496 names, 490 numbers).
The set is near-identical to the first run's, not identical, so every pipeline was re-run on this one set.

**Candidates**, licences from package metadata and model cards:
- **gruut 2.4.0.** MIT. A CMUdict-style lexicon plus a CRF model for words outside it; 38 MB of English data.
  **UNCLEAR:** the `gruut-lang-en` wheel carries no licence file and does not say where its lexicon comes from.
- **OpenPhonemizer 0.1.2.** BSD-3-Clause-Clear, model weights included. A neural model trained to produce
  espeak-style IPA. A 175 MB checkpoint, plus PyTorch.
  - The checkpoint is a pickle. It loads only with PyTorch's safe loader and three allow-listed tokenizer classes;
    the alternative runs whatever code the file holds.
- **Ruled out:**
  - `g2p-en`: Apache-2.0 itself, but depends on `distance`, which is GPL;
  - the `cmudict` Python package: GPL-3.0-or-later.

**How it was wired.** A fallback passed to misaki's own `fallback=` hook gets *whole compound tokens* (a path, a
hash, "llama.cpp") whenever any piece of one is unknown. That replaces misaki's own spelling of the known pieces
and made both candidates worse than no fallback at all. So misaki ran with no fallback, and **only the pieces it
left without phonemes** went to the candidate. That is 163 pieces, compound pieces included.

| | espeak (A) | no fallback (B) | misaki + gruut | misaki + OpenPhonemizer |
|---|---|---|---|---|
| Pieces left silent | 0 | **163** | 0 | 0 |
| Words worse than espeak / better than espeak | n/a | 194 / 99 | 154 / 112 | 143 / 107 |
| Names worse / better than espeak (of 496) | n/a | 28 / 22 | **23 / 24** | **21 / 25** |
| Numbers worse / better than espeak (of 490) | n/a | 70 / 35 | 44 / 39 | 38 / 36 |
| Unknown word-like names heard exactly (of 107) | **34** | 3 | 19 | 24 |

**What it means:**

- **Silent drops are gone with either fallback.**
- **Across all names, both are level with espeak.**
- **On the hardest words, espeak is still ahead.** These are names no dictionary knows, such as model,
  project and command names: espeak 34, OpenPhonemizer 24, gruut 19. Whisper's exact-match test is strict
  for words like these, so listen to the samples rather than trust the count alone.
- **gruut and OpenPhonemizer come out close.** gruut needs no PyTorch and no pickle, which is why Will leans
  towards it.
- **The speakable-text pass** (checklist) addresses the remaining number and fragment errors with any option.


Found while measuring: on macOS, espeak-ng exits the process (`Error processing file .../phontab`) when its
data directory's path is long. It failed at about 200 characters, and a short path worked. Under
`~/Library/Application Support/Prometheus/plugins/speech/<version>/site-packages/espeakng_loader/espeak-ng-data`
that limit is close, so option B must install espeak's data at a short, fixed path and test the real one.

### Option B: keep espeak, in a separate process whose source is published

**Shape.**
- Speech output runs as its own program: the voice plugin's TTS service, which already is one (`kokoro_server`
  over HTTP).
- That program is Kokoro plus `kokoro-onnx` plus `phonemizer` (GPL-3.0-or-later) plus espeak-ng (GPL-3.0).
- It is distributed **as a GPL-3.0-or-later work**, with its complete corresponding source published:
  - the voice plugin's own code (Apache-2.0, which GPL-3.0 can include);
  - `kokoro-onnx` (MIT);
  - `phonemizer`;
  - the exact espeak-ng source the bundled library was built from.
- Prometheus talks to it only over HTTP on 127.0.0.1, as it talks to any service, and stays MIT.
- Whether that boundary is enough is a legal question, not answered here. **This is not legal advice.**

**What it needs:**

1. **A separate signed executable.** The TTS service goes into the plugin as a **service** (3.3): its own
   process and its own Python environment inside the plugin, signed by our team, never imported into the
   daemon.
2. **Source, published with every release that ships it:**
   - a `Prometheus-plugin-speech-tts-source-<version>.zip` asset beside the plugin zip, holding the TTS
     service's source, phonemizer's source, and the espeak-ng tag and source the loader wheel was built from;
   - the GPL text in the plugin;
   - a line in the market listing saying the speech service is GPL-3.0 and where its source is.
3. **The espeak data path fix** above, tested on the real install path.
4. **No GPL code in the daemon's process, ever.** A test fails if the daemon's own environment can import
   `phonemizer` or `espeakng_loader`.

**What it costs:**
- A second licence story in the product: "Prometheus is MIT; its speech service is GPL".
- A source asset to build and check per release.
- An extra process for a feature that, with option A, would not need one. The voice stack is already
  separate processes, so the extra process costs little.

**What it keeps:** espeak's handling of any word, including names it has never seen. That is exactly where
the espeak-free path failed.

### Will's decisions on Speech (2026-10-13)

1. **The voice plugin repo becomes public.** Will makes it public himself, after a scan of its full git
   history for tailnet addresses, hostnames, personal names and secrets. The scan is reported to him; no history
   is rewritten for him. Building Speech in this repository's release job then needs no private access.
2. **espeak: measure the espeak-free path first.**
   - The test set is 50 real assistant replies, heavy on names, places and numbers.
   - Count the words dropped or mangled, with audio samples to listen to.
   - If names fail, write up option B: keep espeak, but run speech output in a separate process whose source
     is published.
   - Neither is built yet.
3. **The `[speech]` extra** (terminal installs) gives speech-to-text plus the macOS `say` fallback until the
   voice plugin is on PyPI. `[voice]` keeps working as an alias. Publishing the voice plugin to PyPI is a
   checklist item.

### Voice ready to ship: checklist

| Item | State (2026-10-13) |
|---|---|
| Beacon desktop 0.4.4: VAD on 8012, set in config | **done** (merged 2026-10-13) |
| The Android client: its VAD port 8011 → 8012 | **open**: still 8011, in two copies of the client |
| The voice plugin repo: the VAD port 8011 hard-coded in its config and compose file (its issue 5) | **open** |
| The licence audit above | **done** |
| espeak-free measurement: 50 real replies, words dropped or mangled, audio samples (decision 2) | **done**: names fail (silent drops); see above |
| Option B written up (decision 2) | **done**, not built; Will chooses |
| History scan of the voice plugin repo before it goes public (decision 1) | **done**: git history and CI logs clean; one issue's text to edit before going public (reported to Will) |
| Issue 5's text made generic before the repo goes public | **done**; the original revision must be deleted from the issue's edit history (Will, in the GitHub UI) |
| The voice plugin repo made public | Will |
| The voice plugin published to PyPI, so `[speech]` can carry the full stack (decision 3) | not started |
| `[speech]` = speech-to-text plus `say`, with `[voice]` as an alias (decision 3) | with the plugin core |
| `say` fallback in `tts.py` (and Telegram voice-note format) | not started |
| A **speakable-text pass** before any TTS, needed with every option: skip or summarise hashes, commit ids and file paths; read numbers, units and code identifiers sensibly. The measurement's 259 words missed by *both* pipelines are mostly this. | not started |
| A third option measured: Kokoro's converter plus a permissively licensed fallback for unknown words only (Will, 2026-10-13: his pick if near espeak on names, otherwise option B) | **done**: misaki + gruut is level with espeak on names overall and behind on unknown names (19 against 34); Will decides |
| Models pinned (URLs, SHA-256s) and `whisper-server` built and signed in CI | not started |
| The plugin core (sections 3 to 8) | waits until 0.9.9 is out |

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
| **Services** (`services.json`: first-party processes the daemon starts, watches and stops) | Added for Speech, whose stack runs as separate processes: VAD WebSocket, TTS, `whisper-server` and the bridge. Each listens on 127.0.0.1 only, on a port from config. A crash stays in its own process. Status is the process's own `/health` | **Never**: a service is code |
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
   - **Speech:** pin each model the stack downloads on first run today: the Kokoro ONNX model, the voices
     file, the Whisper ggml model, the Silero VAD model and, on the espeak-free path, spaCy's English model.
     Each is pinned as a URL or revision plus its SHA-256, like `similarity.py`. The URLs and sizes are in
     section 1a.
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
  as "The Speech plugin is for 0.9.8; this app is 0.9.9. Update available (<size>)". Nothing downloads by
  itself.
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

- the Speech plugin, right after the Browser plugin (section 1a's checklist);
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
