---
tier: core
name: beacon-plate
description: House style for any HTML page, report or PDF you make — NOX/LVX palettes, type, components, print rules. Use it unless the user asks for another style or has their own brand.
---

# Beacon Plate — the default house style

Every HTML page or PDF document you produce uses this system unless the user
asks for something else or gives you their own brand. A page made here and the
Beacon app read as one family: calm, precise, dark-first. It replaces generic
web-design defaults; a user's own brand always wins over it.

## Never convert Markdown to HTML

A spec and a plate are different things. A converted `.md` renders as a README
with better fonts, which is exactly what a plate is meant not to be. Write the
HTML directly.

## Tokens

Dark first: NOX is the default. Bare `:root` is NOX. LVX (light) applies under
`@media (prefers-color-scheme: light)` guarded by `:root:not([data-theme="dark"])`,
and again under `:root[data-theme="light"]`, so an explicit toggle wins both
ways. Neutrals swap between plates; **accents never do**.

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
/* LVX — light. The same block goes in BOTH places. */
@media (prefers-color-scheme: light) { :root:not([data-theme="dark"]) {
  --paper:#faf9f7; --ink:#2e2418; --ink-soft:#3a2f20;
  --ink2:#6b5b45;  --ink3:#7a6748; --ink4:#9a8868;
  --line:rgba(70,56,36,.40); --line2:rgba(70,56,36,.25); --line3:rgba(70,56,36,.16);
  --geo:#6b5b45; --geo-strong:#2e2418; --node:#ffffff; --crop:#2e2418; --idle:#8a7656;
  --glow1:rgba(255,255,255,.55); --glow2:rgba(138,138,138,.07);
  color-scheme: light;
} }
:root[data-theme="light"] { /* repeat the LVX values above */ }
```

Fonts: `https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=EB+Garamond:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap`.
Loading them means the reader's browser requests them from Google Fonts. If the
user wants a page that makes no outside requests, leave the link out; the font
stacks fall back to Georgia and Menlo, and the page still reads well.

## The ground is not --paper

`--paper` is the base, not the background. Two radial washes sit over it; drop
them and the page goes flat. Paper, node and BOTH glows move together when you
re-tune.

```css
body{ background:
  radial-gradient(1000px 620px at 8% -5%,   var(--glow1), transparent 72%),
  radial-gradient(900px  700px at 104% 104%, var(--glow2), transparent 72%),
  var(--paper);
  background-attachment: fixed; color: var(--ink-soft);
  font-family: var(--font-body); font-size:17px; line-height:1.62;
  font-variant-numeric: lining-nums; }
```

## Type

- **Cormorant Garamond** for display: weight 300 for the title, 600 for headings.
- **EB Garamond** for body text.
- **IBM Plex Mono** for everything structural: eyebrows, chips, badges, table
  headers and code.

The document scale is taller than an app's: body 17px, display
`clamp(40px,7vw,64px)`. A page is read at arm's length.

Uppercase mono labels take `letter-spacing: var(--track-caps)`. Prose stays
within `--measure`. Headings take `text-wrap: balance`.

## Components, and when each earns its place

- **Eyebrow** — letterspaced mono above the title. It places the page in a
  series. Leave it out when the page belongs to no series.
- **Chips** — bordered mono pills in a wrapping row, on `--node`. Use them for a
  set the reader scans at a glance; a dozen bullets reads as a chore.
- **Badges** — they carry state, so colour is data, never decoration:
  - `--online`: shipped or healthy;
  - `--ink3` on `--line`: open;
  - `--lapis`: in progress;
  - `--sang`: blocked or danger.
- **Numerals** — display-face numerals in a left gutter, in `--geo`. Only for
  content that is genuinely ranked or sequential; on an unordered list they are
  decoration pretending to be information.
- **Rails** — paired columns with a 2px `--line` left border and a mono label
  above. Accent one rail with `--sang` when that side is the live claim.
- **Note** — `--node` ground, 2px `--geo` left border. For evidence and caveats.
- **Crop marks** — 14px corner brackets in `--crop` on the page container.

## Print and PDF

A PDF is the same page printed, with print rules added. Print in LVX by default:
dark pages waste ink and feel heavy on paper. Use NOX only if the user asks for a
dark PDF.

```css
@page { size: Letter; margin: 18mm 16mm 20mm;          /* A4 for non-US readers */
        background: #faf9f7; }                          /* paper to the sheet edge */
@page { @bottom-right { content: counter(page) " / " counter(pages);
        font: 9pt 'IBM Plex Mono', monospace; color: #7a6748; } }
@media print {
  :root { /* paste the LVX values here, so print is light whatever the screen theme */ }
  body { background: none; margin: 0; font-size: 11.5pt; line-height: 1.5; }
  .crop, .glow { display: none; }                     /* screen-only ornament */
  h1, h2, h3 { break-after: avoid; }
  .note, .rails, figure, table, .chips { break-inside: avoid; }
  thead { display: table-header-group; }              /* repeat table headers */
  p { orphans: 3; widows: 3; }
  .badge, .chip { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  a[href^="http"]::after { content: " (" attr(href) ")"; font: 9pt var(--font-mono); color: var(--ink3); }
}
```

Render with headless Chrome or Chromium:

```
google-chrome --headless=new --no-pdf-header-footer --virtual-time-budget=6000 \
  --print-to-pdf=report.pdf file:///absolute/path/report.html
```

- The paper size comes from `@page`. Its `background` paints the margins too;
  a background on `body` alone leaves a white frame around every page.
- `--virtual-time-budget` lets the web fonts load before printing. If you're
  offline, the PDF uses the fallback fonts: say so.
- Open the PDF and check the page breaks before you hand it over. A heading must
  never sit alone at the bottom of a page, and a table or note must never split.

## Content rules that outrank the CSS

1. Every claim carries its evidence inline: a commit, a PR number, a measured
   value, a source. A status page whose claims can't be checked is worse than
   no page, because it manufactures confidence.
2. Say what was verified and against what. "Verified against the code at
   `main @ 6ff2a37`, not the README" is what makes the rest worth reading.
3. Title the page like a product, not a caption: a short, specific noun phrase,
   with no explainer after a dash.
