# The computer-use candidate-table corpus — harvest spec

A corpus of real candidate tables, each annotated with the answer a human says
is correct, so that a chooser can be *scored* rather than trusted.

This document is the spec. It was written before any harvesting, because the
measurements below change what is worth harvesting.

---

## 1. Why milestone 2's table cannot be the corpus

Milestone 2 executed one real click. It offered **4 candidates**, from a window
containing one label and one button, with a goal that named the button.

A uniformly random chooser scores **25%** on a 4-candidate table. One sample at
25% chance distinguishes nothing at all: every chooser, including a coin flip,
is consistent with the observed result. The table proves the *path* executes. It
cannot score a *chooser*.

## 2. What was measured, and on what

Measured 2026-09-20 on the operator box, reading the live AT-SPI tree directly
and applying this repo's own `_CLICKABLE_ROLES`, `_EDITABLE_ROLES` and
`max_candidates` rules from `candidates.py` — not a model of them.

### 2.1 Real tables are much bigger than four

| app | tree nodes | click | type | **table size** |
|---|---:|---:|---:|---:|
| gnome-shell | 159 | 15 | 1 | 19 |
| gjs (Desktop Icons) | 39 | 0 | 0 | 3 |
| **gnome-calculator** | 93 | 32 | 2 | **37** |
| baobab | 34 | 4 | 0 | 7 |
| gnome-text-editor | 78 | 7 | 0 | 10 |
| gnome-system-monitor | 95 | 15 | 0 | 18 |

**min 3 · median 18 · max 37.**

Two consequences, both load-bearing elsewhere:

- A pocket calculator builds **37** candidates — three short of
  `max_candidates`. ⚠ An earlier version of this line claimed that also
  overflowed a 36-label alphabet. **That was wrong.** SimpleJev's
  `CHOICE_LABELS` is `string.ascii_uppercase + string.ascii_lowercase[:24]` =
  **50** labels, so 37 fits with thirteen to spare and 40 fits with ten. The
  36 came from assuming A–Z + 0–9 instead of reading the constant. The
  abstain-on-overflow guard in `chooser.py` stays as insurance against a future
  cap change; it does not fire today.
- Windows below ~10 candidates exist and are mostly trivial.

### 2.2 Table size is NOT what makes a table discriminating

Self-supervised check: for each candidate in each real table, treat it as the
correct answer and derive the goal from its own description, then ask
`RuleChooser` to recover it.

| app | N | recovered uniquely | tied | missed | discriminating |
|---|---:|---:|---:|---:|---:|
| gnome-shell | 18 | 16 | 1 | 1 | 11% |
| gjs | 3 | 3 | 0 | 0 | 0% |
| gnome-calculator | 35 | 35 | 0 | 0 | **0%** |
| baobab | 7 | 3 | 1 | 3 | **57%** |
| gnome-text-editor | 10 | 10 | 0 | 0 | 0% |
| gnome-system-monitor | 18 | 18 | 0 | 0 | 0% |
| **total** | **91** | **85** | 2 | 4 | **6%** |

**`RuleChooser` recovers its own goal uniquely on 93% of rows.** The
35-candidate calculator is **0%** discriminating; the 7-candidate baobab window
is **57%**.

So size is necessary and nowhere near sufficient. A corpus whose goals are
copied from candidate descriptions is ~94% trivial *regardless of how big the
tables are*, because `RuleChooser` scores by substring overlap and the goal
hands it the substring.

**The goal phrasing is the experiment. The table is just the setting.**

### 2.3 With task-language goals, the deterministic chooser abstains half the time

Nine goals phrased the way a person states a task ("add two numbers together",
"undo what I just did", "find out how much space is left"), against the same
real tables:

| app | N | abstained | picked |
|---|---:|---:|---:|
| gnome-shell | 18 | 4 | 5 |
| gjs | 3 | 8 | 1 |
| gnome-calculator | 35 | 3 | 6 |
| baobab | 7 | 5 | 4 |
| gnome-text-editor | 10 | 3 | 6 |
| gnome-system-monitor | 18 | 4 | 5 |
| **total** | | **27** | **27** |

**`RuleChooser` abstains on 50% of task-language goals.**

This splits the corpus into two populations that answer *different questions*,
and conflating them would produce a single meaningless accuracy number:

- **Abstain rows (~50%)** — `RuleChooser` returns `abstain` with confidence 0.0.
  There is nothing to pair against. The question here is **coverage**: does the
  classifier find the right answer where the rule finds nothing? This is where a
  fast chooser earns its place.
- **Picked rows (~50%)** — both choosers answer. The question here is
  **regression**: do they agree, and when they disagree, who is right?

## 3. Minimum table size

Stated as asked, with the reasoning.

A uniformly random chooser scores `1/N`. For a correct answer to be worth more
than luck, the chance floor has to sit well below the accuracy being claimed.
To detect an accuracy of 0.70 against the chance floor at α=0.05 and 80% power,
by one-sample binomial:

| N | chance floor | tables needed |
|---:|---:|---:|
| 4 | 25% | ~17 |
| 10 | 10% | ~9 |
| 20 | 5% | ~8 |
| 37 | 2.7% | ~7 |

**Minimum: N ≥ 10.** Below it the chance floor eats the signal — at N=4, one row
in four is correct by luck, so a 20-row corpus expects 5 lucky hits and
separating 5 from 8 needs far more rows than the annotation effort justifies.
Above N≈10 the returns flatten: going from 10 to 37 saves two tables.

The measured median real table is **18**, so N ≥ 10 is satisfied by most real
windows without special effort. Windows below it (gjs=3, baobab=7) are recorded
and excluded from headline scoring by a threshold applied **at scoring time**,
derived from the stored `candidate_count` rather than written onto the row —
so raising or lowering the floor never requires rewriting the corpus. They are
kept because baobab is the most *ambiguous* window measured, and ambiguity is
the scarce thing.

**N ≥ 10 is a floor, not a target, and it is the weaker of the two
requirements.** The binding one is §2.2: goals must be written in task language
by a human, not derived from the table.

## 4. Harvest spec

### 4.1 Apps

Installed and confirmed observable on this box:

**Harvest:** `gnome-calculator` (large, dense, repetitive labels),
`gnome-text-editor`, `gnome-system-monitor`, `baobab`, `nautilus`,
`gnome-control-center` (deep, many panes), `evince`, `gnome-disks`.

**Do not harvest:**

- `seahorse` — a key manager. Its accessible tree is key names, fingerprints and
  identities. Never a corpus row.
- `firefox` — `browser_*` is outside this milestone entirely, and browser chrome
  plus page content is where scraped text is most likely to be personal.
- Any window belonging to a running session of the operator's own work — mail,
  chat, terminals, editors with real content open.

Harvest from **clean instances opened for the purpose**, closed afterwards. A
corpus row is a permanent artifact; a window opened on real work is not a safe
thing to make permanent.

### 4.2 Volume — CORRECTED 2026-09-20 to what this box can actually produce

⚠ **An earlier version of this section targeted ~100 tables. That target is not
achievable here, and leaving it in the spec would have left a number nobody
could hit.**

Measured: **five** installed apps clear the N ≥ 10 floor, **eight** if the
content roles are added. At ~5 goals per app that is:

| | apps ≥ N10 | tables |
|---|---:|---:|
| today | 5 | **~25** |
| + chrome roles | 5 | ~25 |
| + content roles | **8** | **~40** |

`gnome-control-center` never reaches the accessibility bus and cannot be
measured. `baobab`, `eog`, `evince` and `clocks` never cross regardless — they
are genuinely sparse, not role-limited.

**~40 is the ceiling a GNOME desktop gives, and installing apps to raise it was
declined: a corpus harvested from a box changed to produce it is a measurement
of the change.**

#### RULED 2026-09-20 — no content roles. Harvest at ~25 tables.

Reasons, in order, and none of them is the power math:

1. **Regression is unanswerable at 40 and stays so.** The 25→40 jump buys
   coverage rows for a question already answerable, not the question that is
   blocked.
2. **Reversible one way only.** Adding roles later costs one re-harvest.
   Un-writing filenames costs the corpus, because replay requires verbatim
   descriptions.
3. **A corpus containing real filenames can never be published or shared.**
   That constraint outlives the power math.
4. Adding roles also invalidates every stored click grant (§8).

#### What ~25 tables powers — sized to 12, not to 20

At the measured abstain rate, ~25 tables is roughly **12 abstain rows**, not 20.
Computed at the measured median table size N = 18, so chance = 1/18 = 5.6% and
0.67 correct is the expectation from guessing:

| | |
|---|---|
| critical value | **3 of 12 correct** rejects "no better than chance", p = 0.026 |
| power at true p = 0.30 | **0.75** |
| power at true p = 0.40 | **0.92** |
| power at true p = 0.50 | **0.98** |
| power at true p = 0.20 | 0.44 — the blind spot |

**12 rows clears the chance floor decisively for any classifier worth having.**
The claim it supports is: *detects a classifier at ≥30% accuracy in the abstain
region with 75% power, and at ≥40% with 92%.* It is NOT "measures accuracy
precisely", and it would miss a classifier scraping 20% more often than not.

For comparison, 20 rows would give 0.89 at p = 0.30 — about fourteen points,
and nothing at all at p ≥ 0.40. That is the size of what the content-role trade
would have bought on this question.

A 20% classifier in a region where the current baseline is **zero** is marginal
by construction, and the go/no-go question is "does it beat doing nothing", not
"what exactly is its accuracy".

#### The regression ceiling is a property of the BOX

Not of the method, and not a harvest parameter. Five apps clear the N ≥ 10
floor here; eight would with content roles. Neither reaches the ~25 discordant
pairs McNemar needs, and more goals cannot fix it because the limit is
**distinct tables**.

**A regression verdict needs a second machine with a different application
set.** That is its own decision, to be taken on its own terms — not something a
harvest can be tuned into providing.

### 4.3 Goals

The single most important rule, and the one the measurement in §2.2 exists to
justify:

- **Goals are written by a human, in task language, before seeing the table.**
- **A goal must not be copy-derived from any candidate's description.** If the
  goal contains the button's label verbatim, the row is trivial by construction.
- Aim for 3–5 goals per table, mixing: one the rule should get, one the rule
  should abstain on, one genuinely ambiguous between two candidates.

An automated check should flag any row whose goal shares more than a threshold
of content words with its annotated-correct candidate's description, and the
harvest tool should refuse to store it silently.

## 5. Three findings that change the harvest — verified 2026-09-20

Found by Track A, reproduced here before spending anyone's time.

### 5.1 `window_id` does not scope the observation

Two Nautilus windows, different folders, different files:

```
window[0] 'probe-a'   127 elements -> 18 candidates
window[1] 'probe-b'   127 elements -> 18 candidates
OVERLAP: 34/38 = 89%   unique to window[1]: 2, both just the title
```

**Two genuinely different states of one app produce near-identical tables.**
The file contents do not even differ, because file rows are `table cell` and
are not offered.

⚠ **This invalidates the "40 states" plan.** Six text-editor states and six
Nautilus states would not have been twelve distinct tables — they would have
been roughly two, plus titles. The harvest must go **wide across apps**, not
deep into states of one app.

### 5.2 A file chooser writes real paths into the candidate table

Measured on a live GTK chooser: `/home/will/Documents`, `/home/will/Videos`,
`/media/will/WD_BLACK` and a full scratchpad path — all as **offered
candidates**, because the sidebar is `list item` and `list item` IS in
`_CLICKABLE_ROLES`.

Not merely `elements_json`. **Candidate descriptions**, which the
replay-surface ruling stores VERBATIM and must: a redacted description is a
different input and scores a different question. So these cannot be scrubbed
without breaking the thing the corpus exists for.

**The only control is not capturing the state.** File chooser, open dialog and
save-as dialog are in `REFUSE_STATE` in the harvester. This supersedes "close
anything with real content first" for this surface — that warning predates
`elements_json` and assumed the risk was document content. A chooser pointed at
an empty directory still lists the user's bookmarks.

### 5.3 The deterministic baseline was stuck, not merely weak

`RuleChooser` never read `request.history`. On a stable table it returned the
same pick forever — four steps, same candidate, while history grew underneath
it.

**Fixed before any measurement**: an action already in `history` is excluded,
matched on DESCRIPTION rather than id (ids are snapshot-bound and change every
observation, so an id-keyed exclusion would exclude nothing in a real loop).
Exhaustion now abstains, which is the honest answer and is exactly the region
an additive classifier is authoritative in.

Raising the baseline first is the same argument as the embeddings baseline:
*"beats RuleChooser" means nothing if RuleChooser was weaker than it needed to
be.* A repeat-refusal is deterministic, free, and needs no model.

⚠ **CORRECTED.** An earlier version of this note claimed the pilot's 40%
abstain rate was invalidated by this change. **It was not.** Harvest rows are
single first steps with EMPTY history, so the exclusion has nothing to exclude
and scores identically. Verified by replaying all 22 stored tables through the
new chooser: **0 of 22 picks differ.**

The pilot is disposable on ONE count — the role-set fingerprint moved
(`9364275ceaeb5877` → `660427fcc1052cbf`) — not two.

### 5.4 Install survey — the app list that clears the floor

Measured, not listed: every installed candidate app launched and its real
candidate count taken. **An app list is not the deliverable; an app list that
clears N ≥ 10 is.**

| app | elements | candidates | |
|---|---:|---:|---|
| gnome-calculator | 98 | **35** | use |
| gnome-logs | 108 | **26** | use |
| gnome-system-monitor | 106 | **22** | use |
| org.gnome.Nautilus | 127 | **18** | use |
| gnome-text-editor | 103 | **17** | use |
| gnome-font-viewer | 70 | 8 | below floor |
| baobab | 66 | 8 | below floor |
| eog | 21 | 7 | below floor |
| org.gnome.Characters | 132 | 7 | below floor |
| gnome-disks | 26 | 4 | below floor |
| evince | 9 | 4 | below floor |
| org.gnome.clocks | 40 | 3 | below floor |

**Five of twelve clear the floor.** `gnome-control-center` never reaches the
accessibility bus at all and could not be measured.

The pattern: apps that clear it have **toolbars and headerbars full of
buttons** — a keypad, a sidebar, column headers. Apps below it are
**content viewers** whose content is `image`, `document text` or `table cell`
and is therefore not offered.

**So the ceiling is role coverage, not app count.** With the pending roles
added:

| | now | if widened |
|---|---:|---:|
| org.gnome.Characters | 7 | **24** |
| gnome-disks | 4 | **17** |
| gnome-font-viewer | 8 | **18** |
| apps clearing N ≥ 10 | **5** | **8** |

Three apps cross the floor purely from role coverage, and the five that already
clear it barely move — so widening buys breadth, not depth.

⚠ That is a measurement, not a recommendation. Widening is still the halted
decision, it now costs a re-prompt of every stored click grant, and each role
is its own consent surface.

### 5.5 Chrome vs content — the role gain, split

The decision rule was: *if chrome roles alone get to 8 apps, take them and
defer content.* Measured per role, marginal gain over the current set:

| role | bucket | carries | gain |
|---|---|---|---|
| `combo box` | chrome | no user data | **+0 on every app** |
| `spin button` | chrome | no user data | **+0 on every app** |
| `page tab` | chrome | no user data | **+0** — already in the set, the dead-string fix landed |
| `tree item` | content | — | **+0 on every app** |
| `menu`, `menu bar`, `popup menu`, `check/radio menu item` | — | — | **+0 on every app** |
| **`table cell`** | **content** | file rows, log lines, sidebar paths | font-viewer **+9**, disks **+12** |
| **`table row`** | **content** | character grid, process rows | Characters **+17** |
| `table`, `tree table`, `page tab list` | — | — | +1 or +2, never decisive |

**Chrome buys nothing. Every app that crosses the floor crosses on content
roles.**

```
apps clearing N>=10   now: 5   chrome only: 5   content: 8   both: 8
```

So the trade is real and unavoidable, exactly as anticipated. `table cell` and
`table row` are simultaneously:

* the only thing that takes the corpus from ~25 to ~40 tables, and
* what puts real filenames, log lines and process names into every table —
  stored **verbatim** by the replay-surface ruling, which cannot scrub them
  without changing the input a chooser is scored on.

Refusing file-chooser states contained the path leak to one surface. Adding
`table cell` un-contains it to every file-manager, log-viewer and
process-list view.

**This comes back to Will. It is not a coverage decision dressed as a role
decision.**

## 6. The correct answer is not the executed answer

The two differ exactly when the run was wrong, which is the case the corpus
exists to capture. A schema that can only record what happened cannot score
anything.

Four annotation states, deliberately distinct:

| state | meaning |
|---|---|
| *(unannotated)* | no human has looked at this row yet. **The default.** Never confuse with "no correct answer". |
| a candidate id | that candidate was the right action |
| `none_correct` | the table was usable, but no candidate in it was the right action — `abstain` was correct |
| `table_unusable` | the observation should not have produced a table at all |

`none_correct` and `table_unusable` are separate because they score differently:
on the first, a chooser that abstains is **right**; on the second, the fault is
upstream of the chooser and the row must be excluded from scoring entirely, not
counted as a loss.

A corpus that collapses unannotated into `none_correct` silently scores every
un-reviewed row as "abstain was correct" — which would make a chooser that
always abstains look perfect on an unannotated corpus.

## 7. Table expressiveness — DIAGNOSED 2026-09-20, deliberately NOT fixed here

Verified independently at `424edc1`, by AST over the source rather than by
reading prose.

| | |
|---|---|
| `ACTION_MODELS` declares | `observe, click, scroll, press_key, type_text, invoke_menu, verify` (7) |
| `build_candidates` emits | `computer_click`, `computer_type_text` (caller-supplied text only), `computer_press_key` |
| `press_key` offers | exactly `return`, `tab`, `escape` |
| `ALLOWED_KEYS` declares | **15** keys — so **12 are unreachable** |
| modifier combinations | **none exist at all** — Ctrl+Z, Ctrl+S have no representation |

**`scroll` and `invoke_menu` can never be selected.** All three `Candidate(`
constructions in the entire `src/` tree are inside `build_candidates`, and it
has exactly one caller (`loop.py:118`). There is no other path into a table.
Both verbs nonetheless carry full models, schemas, gate wiring, and
operator-facing consent phrases — `_VERB_PHRASES` offers to grant "scroll
anything" and "use any menu item" for capabilities that cannot be exercised.

### Was the narrowing deliberate? No — the docstring asserts the opposite

`actions.py` carries a section headed **"WHAT IS IN v1, AND WHAT IS
DELIBERATELY NOT"**, and lists `scroll` and `invoke_menu` under **In**.
`build_candidates`' own docstring says "Every bounded action worth offering for
this snapshot", which reads as completeness.

So this is not an undocumented decision. It is a documented claim that the code
contradicts, under the one heading whose entire purpose is to separate
deliberate omission from oversight. That is §1's orphan shape — a capability
fully built and unreachable — inside the subsystem whose premise is that code
and claim must not diverge.

**Not fixed in this PR, on instruction.** Fixing `build_candidates` is
deterministic, adds no dependency, needs no licence and costs no latency — and
it changes the measurement, so it belongs in its own change with its own
before/after.

### Why this reorders the plan

Part of the measured 50% abstain rate is **not chooser weakness — it is a table
that cannot express the goal**. Those rows are `none_correct` by construction
and no chooser, local or hosted, will ever improve them.

Worked example. Goal: *"undo my last change"* in a text editor. The table offers
"Click the button labelled 'Open'", "Click the text area", "Press escape".
`RuleChooser` scores `{undo, last, change}`, gets zero overlap, abstains. A
classifier reads the same table and finds nothing that undoes anything, because
`invoke_menu(['Edit','Undo'])` was never offered and Ctrl+Z is not a candidate.
**Both abstain correctly. The gap was never the chooser.**

### Revised sequence

1. Harvest with `none_correct_reason` recorded (this PR).
2. Split the `none_correct` rows by reason.
3. If `verb_not_offered` / `key_not_offered` are a large share, **fix
   `build_candidates` first and re-measure.**
4. Only then judge a classifier — against the smaller remaining gap, and
   against an **embeddings baseline** rather than against substring matching.

Judging a classifier before step 3 measures the table's limits and reports them
as the chooser's.

## 8. Role coverage — SURVEYED 2026-09-20, deliberately NOT widened

Measured against live AT-SPI trees. **Diagnosis only.** Widening
`_CLICKABLE_ROLES` is a new **consent surface** — every added role is a class of
thing the system may be asked to click — so each one wants its own decision.

### `tab` is a dead entry

`_CLICKABLE_ROLES` contains `tab`. AT-SPI's own role enum contains **`page tab`
and never `tab`**, so that entry has never matched anything. Tab switching has
been unreachable the whole time, behind an allowlist entry that makes the set
look like it covers it.

Same shape as the a11y-bus defect: a constant in the code that does not match
what the system emits, with nothing to notice the mismatch. A survey of the
emitted roles catches it; reading the set does not.

### Roles present in real windows and never offered

| role | nodes seen | where |
|---|---:|---|
| `panel` | 217 | everywhere — structural, correctly not offered |
| **`table cell`** | **40** | the file chooser's file list |
| `label` | 31 | everywhere — structural |
| `image` | 23 | icons |
| **`table row`** | **9** | the file chooser |
| `scroll pane` / `scroll bar` | 10 | — |
| **`page tab` / `page tab list`** | **4** | the editor's document tabs |
| `list` | 4 | — |
| `tree table` | 1 | the file chooser |

**A file chooser's actual content is `table cell`.** None of it is offered, so
"open my documents" in a file dialog has no candidate that selects a file.

### Roles AT-SPI defines that the set does not contain

`combo box` · `menu` · `menu bar` · `popup menu` · `check menu item` ·
`radio menu item` · `spin button` · `table` · `table cell` · `table row` ·
`tree table` · `page tab` · `page tab list`

Only `menu item` is present, so an **open menu** offers its items but the thing
that opens it, and the menu bar itself, are not candidates.

⚠ **`combo box` could not be measured against a real preferences pane** —
`gnome-control-center` exits immediately on this box and never registers on the
accessibility bus. That it is absent from the set is a fact about the code; how
much of a prefs pane it accounts for is not yet measured and should be, during
the real harvest, before any decision to widen.

### ⚠ HALT — widening the role set retroactively widens every stored grant

**Demonstrated, not argued.** The extent is `target:app:verb:delivery_mode` and
carries **no role-set term**. Grants match by exact whole-value compare. So:

```
BEFORE adding 'table cell'        AFTER adding 'table cell'
  click-0  'Open'                   click-0  'Open'
                                    click-1  'tax-return-2025.pdf'
                                    click-2  'passwords.kdbx'
  extent: mini:…nautilus:click:background   ← IDENTICAL
```

A grant stored for `mini:org.gnome.nautilus:click:background` — "click anything
in Nautilus" — silently starts meaning "select files", **with no new prompt**.
The grant was given when the offered set was buttons; it is honoured after the
set includes file rows.

This is the milestone-1 target-term argument in a new place: *the extent must
name every term that changes what is authorised.* It named `target` because one
grant should not span machines. It does not name the role set, and the role set
is what decides which elements become clickable.

**RULED 2026-09-20 — invalidate stored click grants on a role-set change.**
Not versioning: `describe()` still says "click anything", so a version suffix
improves the machine half of the consent and leaves the human half equally
vague — §17 drift. Done while it was free: `registered: 0`, the loop unwired,
approximately zero stored click grants. Never cheaper than today.

The set is fingerprinted (`role_set_fingerprint()`, covering BOTH role sets
since an element becomes a candidate through either), the fingerprint is
stamped beside the grants on every persist, and a mismatch **drops every
`computer_action` grant and surfaces the count on `/api/status`**. Path,
command and tool grants are untouched — the role set says nothing about them.
An ABSENT fingerprint does not drop: absent is not "changed", and dropping on
absence would punish every existing install once for a set that may never have
moved.

The options considered:

| option | what it costs | what it buys |
|---|---|---|
| **Version the role set into the extent** — `mini:nautilus:click:background:r3` | every role-set change invalidates grants for that version; the value gets longer and less readable at the prompt | the grant states what it covered; old grants keep their original meaning rather than being reinterpreted |
| **Invalidate stored click grants on a set change** | a re-prompt for every app after any change | the extent stays four terms and readable; the blast radius is visible at the moment of change |

### Dead entries — CORRECTED 2026-09-20 under the exemption

`tab` → `page tab` (dead-string correction; the entry always meant tabs).
`button` **deleted** — AT-SPI emits `push button` / `toggle button` /
`radio button`, all three already present, so it was redundant as well as dead
and removing it offers strictly *less*. Fingerprint moved
`9364275ceaeb5877` → `660427fcc1052cbf`, which is itself the invalidation
working. `KNOWN_DEAD` is now empty.

**Nothing else added.** `combo box`, `table cell`, `tree item` and the rest
remain genuine additions that do not inherit the exemption, and each now costs
a re-prompt of every stored click grant — which is the point.

**`page tab` is exempt, and the exemption is written down here so the next
addition cannot inherit it.** `_CLICKABLE_ROLES` contains `tab`, which AT-SPI
never emits; correcting it to `page tab` restores what the entry was always
meant to cover rather than extending the set to a new class of control. That
reasoning applies to a **dead-string correction only**. `combo box`,
`table cell`, `tree item` and every other candidate role are genuine additions
and none of them inherits this exemption.

### Dead entries in the offered set — audited 2026-09-20

| entry | status | fix |
|---|---|---|
| `tab` | **DEAD** — AT-SPI emits `page tab` | correct the string (exempt, above) |
| `button` | **DEAD** — AT-SPI emits `push button` / `toggle button` / `radio button`, all three already present | delete it; redundant as well as dead, and removing it offers strictly less |
| `link`, `list item`, `menu item`, `check box`, `radio button`, `push button`, `toggle button` | live | — |
| all of `_EDITABLE_ROLES` | live | — |

Pinned by `tests/test_clickable_roles_are_live.py`, which interrogates AT-SPI's
own enum rather than a second copy of the names, and holds both dead entries in
a `KNOWN_DEAD` ratchet — new dead entries fail, and fixing one means deleting
its line.

⚠ That test **skips where the AT-SPI bindings are absent**, which includes the
project venv and CI. Skipped is not passed, so it does not belong in CI: it
runs in the **on-box verification ritual** under system python and is named in
its output. A guard that skips in CI is a guard that runs when someone
remembers.

### Apps the loop structurally cannot operate

**`gnome-system-monitor`'s process list never reaches AT-SPI at all.** Zero
row-like nodes (`table row`, `table cell`, `tree table`, `table`, `tree item`,
`list item`) anywhere in its 118-node full tree, showing or not. This is not a
role gap and no role set reaches it — the widget does not expose its rows.

**Do not spend harvest states on its process view.** Its tab switcher is
worth capturing (that one was a real role gap, now fixed); its process list is
not capturable by anything.

### Why this precedes the real harvest

Forty states collected against the current role set would understate
reachability across the whole corpus, and every `role_not_clickable` row would
be recorded against a set nobody had looked at. The survey is cheap; the
harvest is not.

## 9. Label provenance — and the harvest order

`label_source` is required whenever `correct_id` is set. There is **no
default**: `human` would be the comfortable one and the wrong one.

| value | meaning |
|---|---|
| `human` | a person decided, having looked at the table. **The only labels a score may be called accuracy against.** |
| `model` | a model proposed it, nobody checked. A score here is **agreement with a model**. |
| `model_confirmed` | a model proposed it and a person confirmed or corrected it. Weaker than `human` — the person saw a suggestion first, and anchoring is real — so it is tracked separately rather than folded into either neighbour. |

**Every summary and score reports its label mix, in words.** A number over
`model` rows prints `AGREEMENT WITH A MODEL — not accuracy`, and a mixed corpus
refuses to be summarised as either. A reader who skims must not be able to pick
up the wrong word, so the rule is in the output rather than in a footnote, and a
test asserts that the word "accuracy" never appears except inside a denial of it.

The reason is the one `tests/fixtures/divergence_traces.py` already states: *a
calibration round that cannot tell the two apart is calibrating against its own
author.* A chooser graded against labels another model produced can only be
measured on how alike they are.

### Harvest order — do not reorder

1. **Will hand-labels ~20 tables COLD**, before any model proposal exists. This
   is the calibration set and it must not see a suggestion first — that is what
   makes it a control rather than a confirmation.
2. **Claude labels all ~100**, recorded as `model`.
3. **Compare against the 20.** Report agreement.
4. **Will decides** whether the remaining ~80 are usable as-is, need confirming
   (`model_confirmed`), or get hand-labelled.

⚠ Step 1 precedes step 2 for a reason that cannot be recovered afterwards: once
a human has seen a model's answer, the label is `model_confirmed` at best and
the calibration set no longer exists.

## 10. Provenance on every row

Four fields exist so a row cannot quietly mean something other than it appears
to. Each one is stored, not inferred at read time.

| field | why it cannot be left out |
|---|---|
| `driver_kind` | **`fixture` vs `cua`.** Every test in this repo drives the loop with `FixtureDriver`. If those rows entered the corpus indistinguishably from real ones, a chooser would be scored against tables the test suite invented. `tests/fixtures/divergence_traces.py` already states this rule: *"a calibration round that cannot tell the two apart is calibrating against its own author."* It is **derived from the driver**, never supplied by a caller, because the row it would be wrong on is exactly a fixture row that reads as real. |
| `goal_source` | `human` \| `derived` \| `unknown`. §2.2 measures that a description-derived goal makes 93% of rows trivial, so goal provenance *is* the experiment. The default is `unknown`, not `human` — a row that merely forgot the flag must not read as one a person vouched for. |
| `harvest_session` | Groups rows captured under the same discipline. Two sessions with different goal-writing rigour mix irreversibly without it. |
| `code_fingerprint` | A hash of `_CLICKABLE_ROLES` and `_EDITABLE_ROLES` — the constants that *define* which elements become candidates at all. Change them and a row harvested beforehand describes a different universe of options; scoring across the change silently blends two experiments. |

`table_fingerprint` is also stored, over the sorted `(id, description)` pairs, so
the same window captured twice is identifiable rather than double-counted.

## 11. Redaction

Candidate descriptions are built from AT-SPI labels scraped off a live desktop:
window titles, button text, field labels, and sometimes document content. A
corpus is a permanent, re-readable artifact, so what goes in it is a privacy
decision, not a formatting one.

- Never store element tokens (snapshot-bound, meaningless later) or raw
  arguments. The chooser's view — `{id, description}` — is the whole record.
- Never store a typed payload. `text_to_type` is the caller's, not the table's.
- Apply the project's existing redaction before writing, not after.
- The app allowlist in §4.1 is the primary control; redaction is the backstop.

## 12. What this spec does not cover

`hf-server`, SimpleJev, any classifier, and any scoring harness. Those come
after a corpus exists and after the licence question is answered. This document
describes the corpus and nothing downstream of it.
