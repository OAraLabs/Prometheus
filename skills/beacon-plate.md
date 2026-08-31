---
name: beacon-plate
description: The OAra house visual style for any HTML page, artifact, plate or report produced for Will or for Beacon/Prometheus/OAra. Palette (NOX dark, LVX light), type, and components lifted from beacon-desktop tokens.css. Use whenever building an HTML artifact or standalone page for this user, in preference to generic web-design skills.
---

# Beacon Plate — the OAra house style

Any HTML page built for this user renders in this system, so a plate and the
Beacon app read as one family. This supersedes generic frontend-design and
web-design-guidelines advice for OAra work; those still apply to third-party
or client work with its own brand.

## Never convert Markdown to HTML

The spec and the plate are different artefacts with different jobs. A converted
`.md` renders as a README with better fonts, which is the one thing a plate
exists not to be. Author the HTML directly.

## Tokens

Dark-first: Beacon's default is NOX. Bare `:root` is NOX; LVX arrives via
`@media (prefers-color-scheme: light)` guarded `:root:not([data-theme="dark"])`,
and again under `:root[data-theme="light"]` so an explicit toggle wins both ways.
Neutrals swap between plates; **accents never do**.

```css
:root {                                  /* NOX — dark, the default */
  --paper:#0f1219; --ink:#ece3cc; --ink-soft:#d8cfb6;
  --ink2:#ada287;  --ink3:#b3a075; --ink4:#82775a;
  --line:rgba(190,168,110,.26); --line2:rgba(190,168,110,.15); --line3:rgba(190,168,110,.10);
  --geo:#8a7a4e; --geo-strong:#c9b682; --node:#161a24; --crop:#c9b682; --idle:#9a8a5e;
  --glow1:rgba(120,150,200,.06); --glow2:rgba(184,150,82,.09);
  /* shared accents — identical on both plates */
  --sang:#b8473a; --sang-deep:#a23b2e; --sang-soft:rgba(184,71,58,.12);
  --lapis:#3e6b8f; --lapis-soft:rgba(62,107,143,.14);
  --online:#46a98c; --online-soft:rgba(70,169,140,.14); --on-sang:#f5efe2;
  --font-display:'Cormorant Garamond',Georgia,serif;
  --font-body:'EB Garamond',Georgia,serif;
  --font-mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;
  --track-caps:.18em; --measure:68ch;
  color-scheme: dark;
}
/* LVX — light. Repeat this block under BOTH the media query and [data-theme="light"]. */
  --paper:#faf9f7; --ink:#2e2418; --ink-soft:#3a2f20;
  --ink2:#6b5b45;  --ink3:#7a6748; --ink4:#9a8868;
  --line:rgba(70,56,36,.40); --line2:rgba(70,56,36,.25); --line3:rgba(70,56,36,.16);
  --geo:#6b5b45; --geo-strong:#2e2418; --node:#ffffff; --crop:#2e2418; --idle:#8a7656;
  --glow1:rgba(255,255,255,.55); --glow2:rgba(138,138,138,.07);
  color-scheme: light;
```

Fonts load from Google Fonts, the one host the artifact CSP admits:
`https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=EB+Garamond:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap`

## The ground is not --paper

`--paper` is the base, not the background. Two radial washes composite over it.
Drop them and the plate goes flat and reads like any other dark page. Paper,
node, and BOTH glows move together when re-tuning — neutralising paper alone
leaves the wash in place and the light plate still reads yellow.

```css
body{ background:
  radial-gradient(1000px 620px at 8% -5%,   var(--glow1), transparent 72%),
  radial-gradient(900px  700px at 104% 104%, var(--glow2), transparent 72%),
  var(--paper);
  background-attachment: fixed; color: var(--ink-soft);
  font-family: var(--font-body); font-size:17px; line-height:1.62; }
```

## Type

Cormorant Garamond display (weight 300 for the title, 600 for headings), EB
Garamond body, IBM Plex Mono for everything structural — eyebrows, chips,
badges, table headers, code. Document scale runs taller than the app's 11–30px
UI scale: body 17px, display `clamp(40px,7vw,64px)`. A plate is read at arm's
length, not operated at 14px.

Uppercase mono labels carry `letter-spacing: var(--track-caps)`. Prose stays at
`--measure`. Headings take `text-wrap: balance`.

## Components, and when each earns its place

- **Eyebrow** — letterspaced mono over the title. Locates the page in a series.
  Omit it on a page that belongs to no series.
- **Chips** — bordered mono pills in a flex-wrap row, on `--node`. Use for a set
  scanned at a glance. A dozen bullets reads as a chore; chips read as one set.
- **Badges** — carry state, so colour is data, never decoration.
  `--online` shipped/healthy · `--ink3` on `--line` open · `--lapis` phase ·
  `--sang` blocked/danger.
- **Numerals** — display-face numerals in a left gutter, in `--geo`. Only for
  content that is genuinely ranked or sequential. On an unordered list they are
  decoration pretending to be information.
- **Rails** — paired columns with a 2px `--line` left border, mono label above.
  Accent one rail with `--sang` when one side is the live claim.
- **Note** — `--node` ground, 2px `--geo` left border. For evidence and caveats.
- **Crop marks** — 14px corner brackets in `--crop` on the plate container.

## Content rules that outrank the CSS

1. Every claim carries its evidence inline — a commit SHA, a PR number, a
   measured value. A status plate whose claims cannot be checked is worse than
   no plate; it manufactures confidence.
2. Say what was verified and against what. "Verified against the code at
   `main @ 6ff2a37`, not the README" is what makes the rest worth reading.
3. Title the page like a product, not a caption: a short specific noun phrase,
   no explainer after a dash.
