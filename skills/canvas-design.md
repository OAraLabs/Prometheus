---
name: canvas-design
description: Create visual art and design as .png and .pdf files using design philosophy methodology. Use when the user asks to create a poster, piece of art, design, or other static visual piece. Create original visual designs.
license: CC-BY-SA-4.0
---

# Canvas Design

Instructions for creating design philosophies - aesthetic movements that are EXPRESSED VISUALLY. Output .md files (philosophy), .pdf files, and .png files using `file_write` and `bash` with Python imaging libraries.

Complete in two steps:
1. Design Philosophy Creation (.md file)
2. Express by creating it on a canvas (.pdf or .png file)

## DESIGN PHILOSOPHY CREATION

Create a VISUAL PHILOSOPHY (not layouts or templates) interpreted through:
- Form, space, color, composition
- Images, graphics, shapes, patterns
- Minimal text as visual accent

### THE CRITICAL UNDERSTANDING
- What is received: Subtle input or instructions from the user as a foundation; it should not constrain creative freedom.
- What is created: A design philosophy/aesthetic movement.
- What happens next: The philosophy is then EXPRESSED VISUALLY - creating file output that is 90% visual design, 10% essential text.

### HOW TO GENERATE A VISUAL PHILOSOPHY

**Name the movement** (1-2 words): "Brutalist Joy" / "Chromatic Silence" / "Metabolist Dreams"

**Articulate the philosophy** (4-6 paragraphs - concise but complete):

Express how the philosophy manifests through:
- Space and form
- Color and material
- Scale and rhythm
- Composition and balance
- Visual hierarchy

**CRITICAL GUIDELINES:**
- **Avoid redundancy**: Each design aspect mentioned once.
- **Emphasize craftsmanship REPEATEDLY**: Stress that the final work should appear meticulously crafted, the product of deep expertise, master-level execution.
- **Leave creative space**: Be specific about aesthetic direction, but concise enough for interpretive choices at extremely high craftsmanship.

### PHILOSOPHY EXAMPLES

**"Concrete Poetry"**
Philosophy: Communication through monumental form and bold geometry.
Visual expression: Massive color blocks, sculptural typography, Brutalist spatial divisions. Ideas expressed through visual weight and spatial tension.

**"Chromatic Language"**
Philosophy: Color as the primary information system.
Visual expression: Geometric precision where color zones create meaning. Typography minimal - small sans-serif labels letting chromatic fields communicate.

**"Analog Meditation"**
Philosophy: Quiet visual contemplation through texture and breathing room.
Visual expression: Paper grain, ink bleeds, vast negative space. Photography and illustration dominate. Japanese photobook aesthetic.

**The design philosophy should be 4-6 paragraphs long.** Output as a .md file using `file_write`.

---

## DEDUCING THE SUBTLE REFERENCE

Before creating the canvas, identify the subtle conceptual thread from the original request. The topic is a subtle, niche reference embedded within the art itself - not always literal, always sophisticated. The design philosophy provides the aesthetic language. The deduced topic provides the soul.

---

## CANVAS CREATION

With both the philosophy and conceptual framework established, express it on a canvas. Use `bash` with Python libraries (Pillow, matplotlib, reportlab, cairo) to generate the visual output.

**IMPORTANT**: Even for movie/game/book content, the approach should be sophisticated. This should be art, not something cartoony or amateur.

To create museum or magazine quality work, use the design philosophy as the foundation. Create one single page, highly visual, design-forward PDF or PNG output (unless asked for more pages). Treat the abstract philosophical design as if it were a scientific bible, borrowing the visual language of systematic observation.

**Text as a contextual element**: Text is always minimal and visual-first, but let context guide whether that means whisper-quiet labels or bold typographic gestures. Nothing falls off the page and nothing overlaps. Every element must be contained within the canvas boundaries with proper margins.

**Use Python for rendering:**
```bash
python3 -c "
from PIL import Image, ImageDraw, ImageFont
# ... generate the design
img.save('output.png')
"
```

Or for PDF output:
```bash
python3 -c "
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
# ... generate the design
"
```

**CRITICAL**: Create work that looks like it took countless hours. Make it appear as though someone at the absolute top of their field labored over every detail.

Output the final result as a downloadable .pdf or .png file using `file_write`, alongside the design philosophy as a .md file.

---

## MULTI-PAGE OPTION

When additional pages are requested, create more creative pages along the same philosophy but distinctly different. Bundle in the same .pdf or many .pngs. Make the pages tell a story in a tasteful way with full creative freedom.
