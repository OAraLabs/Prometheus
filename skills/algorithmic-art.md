---
name: algorithmic-art
description: Creating algorithmic art using p5.js with seeded randomness and interactive parameter exploration. Use when users request creating art using code, generative art, algorithmic art, flow fields, or particle systems. Create original algorithmic art rather than copying existing artists' work.
license: CC-BY-SA-4.0
---

# Algorithmic Art

Algorithmic philosophies are computational aesthetic movements expressed through code. Output .md files (philosophy), .html files (interactive viewer), and .js files (generative algorithms).

This happens in two steps:
1. Algorithmic Philosophy Creation (.md file)
2. Express by creating p5.js generative art (.html + .js files)

## ALGORITHMIC PHILOSOPHY CREATION

Create an ALGORITHMIC PHILOSOPHY (not static images or templates) interpreted through:
- Computational processes, emergent behavior, mathematical beauty
- Seeded randomness, noise fields, organic systems
- Particles, flows, fields, forces
- Parametric variation and controlled chaos

### THE CRITICAL UNDERSTANDING
- What is received: Some subtle input or instructions by the user as a foundation; it should not constrain creative freedom.
- What is created: An algorithmic philosophy/generative aesthetic movement.
- What happens next: The philosophy is then EXPRESSED IN CODE - creating p5.js sketches that are 90% algorithmic generation, 10% essential parameters.

### HOW TO GENERATE AN ALGORITHMIC PHILOSOPHY

**Name the movement** (1-2 words): "Organic Turbulence" / "Quantum Harmonics" / "Emergent Stillness"

**Articulate the philosophy** (4-6 paragraphs - concise but complete):

Express how this philosophy manifests through:
- Computational processes and mathematical relationships
- Noise functions and randomness patterns
- Particle behaviors and field dynamics
- Temporal evolution and system states
- Parametric variation and emergent complexity

**CRITICAL GUIDELINES:**
- **Avoid redundancy**: Each algorithmic aspect mentioned once.
- **Emphasize craftsmanship REPEATEDLY**: Stress that the final algorithm should appear meticulously crafted, refined with care, the product of deep computational expertise.
- **Leave creative space**: Be specific about algorithmic direction, but concise enough to allow room for interpretive implementation choices.

### PHILOSOPHY EXAMPLES

**"Organic Turbulence"**
Philosophy: Chaos constrained by natural law, order emerging from disorder.
Algorithmic expression: Flow fields driven by layered Perlin noise. Thousands of particles following vector forces, their trails accumulating into organic density maps. Color emerges from velocity and density.

**"Quantum Harmonics"**
Philosophy: Discrete entities exhibiting wave-like interference patterns.
Algorithmic expression: Particles on a grid, each carrying a phase value evolving through sine waves. Interference creates bright nodes and voids. Simple harmonic motion generates complex emergent mandalas.

**"Recursive Whispers"**
Philosophy: Self-similarity across scales, infinite depth in finite space.
Algorithmic expression: Branching structures that subdivide recursively. Each branch randomized but constrained by golden ratios. L-systems generate tree-like forms that feel both mathematical and organic.

**The algorithmic philosophy should be 4-6 paragraphs long.** Output as a .md file using `file_write`.

---

## DEDUCING THE CONCEPTUAL SEED

Before implementing, identify the subtle conceptual thread from the original request. The concept is a subtle, niche reference embedded within the algorithm itself - not always literal, always sophisticated. The algorithmic philosophy provides the computational language. The deduced concept provides the soul.

---

## P5.JS IMPLEMENTATION

With the philosophy AND conceptual framework established, express it through code.

### TECHNICAL REQUIREMENTS

**Seeded Randomness (Art Blocks Pattern)**:
```javascript
let seed = 12345;
randomSeed(seed);
noiseSeed(seed);
```

**Parameter Structure - FOLLOW THE PHILOSOPHY**:
```javascript
let params = {
  seed: 12345,
  // Parameters that control YOUR algorithm:
  // Quantities, Scales, Probabilities, Ratios, Angles, Thresholds
};
```

**Core Algorithm - EXPRESS THE PHILOSOPHY**:

The algorithmic philosophy should dictate what to build. Think "how to express this philosophy through code?" not "which pattern should I use?"

**Canvas Setup**: Standard p5.js structure:
```javascript
function setup() {
  createCanvas(1200, 1200);
}

function draw() {
  // Your generative algorithm
}
```

### CRAFTSMANSHIP REQUIREMENTS

- **Balance**: Complexity without visual noise, order without rigidity
- **Color Harmony**: Thoughtful palettes, not random RGB values
- **Composition**: Even in randomness, maintain visual hierarchy and flow
- **Performance**: Smooth execution, optimized for real-time if animated
- **Reproducibility**: Same seed ALWAYS produces identical output

### OUTPUT FORMAT

Output:
1. **Algorithmic Philosophy** - As markdown file
2. **Single HTML file** - Self-contained interactive generative art

The HTML file contains everything: p5.js (from CDN), the algorithm, parameter controls, and UI - all in one file that works in any browser. Use `file_write` to create the output files.

---

## INTERACTIVE FILE CREATION

Create a single, self-contained HTML file with:

### REQUIRED FEATURES

**1. Parameter Controls**
- Sliders for numeric parameters (particle count, noise scale, speed, etc.)
- Color pickers for palette colors
- Real-time updates when parameters change
- Reset button to restore defaults

**2. Seed Navigation**
- Display current seed number
- "Previous" and "Next" buttons to cycle through seeds
- "Random" button for random seed
- Input field to jump to specific seed

**3. Single File Structure**
```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.7.0/p5.min.js"></script>
  <style>/* All styling inline */</style>
</head>
<body>
  <div id="canvas-container"></div>
  <div id="controls"><!-- All parameter controls --></div>
  <script>
    // ALL p5.js code inline here
  </script>
</body>
</html>
```

No external files, no imports (except p5.js CDN). Everything inline.

---

## THE CREATIVE PROCESS

**User request** -> **Algorithmic philosophy** -> **Implementation**

1. **Interpret the user's intent** - What aesthetic is being sought?
2. **Create an algorithmic philosophy** (4-6 paragraphs)
3. **Implement it in code** - Build the algorithm
4. **Design appropriate parameters** - What should be tunable?
5. **Build matching UI controls** - Sliders/inputs for those parameters

Use `file_write` to save the philosophy .md and the self-contained .html file.
