---
name: gif-creator
description: Knowledge and utilities for creating animated GIFs optimized for messaging platforms (Telegram, web). Provides constraints, validation patterns, and animation concepts using PIL/Pillow. Use when users request animated GIFs like "make me a GIF of X doing Y".
---

# GIF Creator

A toolkit for creating animated GIFs optimized for messaging platforms like Telegram.

## Platform Requirements

**Telegram:**
- Sticker GIFs: 512x512 max (one side must be exactly 512px)
- Inline GIFs: 320x320 recommended for quick loading
- Max file size: 256KB for stickers, no strict limit for regular

**General web/messaging:**
- Emoji GIFs: 128x128
- Message GIFs: 480x480

**Parameters:**
- FPS: 10-30 (lower is smaller file size)
- Colors: 48-128 (fewer = smaller file size)
- Duration: Keep under 3 seconds for sticker/emoji GIFs

## Core Workflow

Write a Python script via `file_write` and execute with `bash`:

```python
from PIL import Image, ImageDraw
import imageio
import math

frames = []
width, height = 128, 128
num_frames = 12
fps = 10

for i in range(num_frames):
    frame = Image.new('RGB', (width, height), (240, 248, 255))
    draw = ImageDraw.Draw(frame)

    # Draw your animation using PIL primitives
    t = i / num_frames
    # ... animation logic ...

    frames.append(frame)

# Save as GIF
frames[0].save(
    'output.gif',
    save_all=True,
    append_images=frames[1:],
    duration=1000 // fps,
    loop=0,
    optimize=True
)
```

## Drawing Graphics

### Working with User-Uploaded Images

Load and work with images using PIL:
```python
from PIL import Image
uploaded = Image.open('file.png')
# Use directly, or as reference for colors/style
```

### Drawing from Scratch

Use PIL ImageDraw primitives:

```python
from PIL import ImageDraw

draw = ImageDraw.Draw(frame)

# Circles/ovals
draw.ellipse([x1, y1, x2, y2], fill=(r, g, b), outline=(r, g, b), width=3)

# Polygons (stars, triangles, any shape)
points = [(x1, y1), (x2, y2), (x3, y3)]
draw.polygon(points, fill=(r, g, b), outline=(r, g, b), width=3)

# Lines
draw.line([(x1, y1), (x2, y2)], fill=(r, g, b), width=5)

# Rectangles
draw.rectangle([x1, y1, x2, y2], fill=(r, g, b), outline=(r, g, b), width=3)
```

Do not use emoji fonts (unreliable across platforms).

### Making Graphics Look Good

- **Use thicker lines** - Always `width=2` or higher. Thin lines look choppy.
- **Add visual depth** - Gradients for backgrounds, layered shapes for complexity
- **Make shapes interesting** - Add highlights, rings, patterns. Stars with glows, circles with inner rings.
- **Use vibrant colors** - Complementary colors, good contrast, dark outlines on light shapes
- **For complex shapes** (hearts, snowflakes) - Combine polygons and ellipses, calculate points for symmetry

## Animation Concepts

### Shake/Vibrate
Offset position with `math.sin()` or `math.cos()` oscillation, small random variations for natural feel.

### Pulse/Heartbeat
Scale size with `math.sin(t * freq * 2 * math.pi)`. Scale between 0.8 and 1.2 of base size.

### Bounce
Quadratic or cubic easing for gravity. Dampen each successive bounce.

### Spin/Rotate
`image.rotate(angle, resample=Image.BICUBIC)`. Sine wave for wobble instead of linear.

### Fade In/Out
RGBA image with adjustable alpha, or `Image.blend(img1, img2, alpha)`.

### Slide
Start position outside frame, ease to target. Quadratic ease-out for smooth stop.

### Explode/Particle Burst
Particles with random angles and velocities. Apply gravity (`vy += g`), fade out over time.

## Optimization Strategies

When the file size needs to be smaller:

1. **Fewer frames** - Lower FPS (10 instead of 20) or shorter duration
2. **Fewer colors** - Reduce palette to 48 or fewer
3. **Smaller dimensions** - 128x128 instead of 480x480
4. **Remove duplicate frames** - Skip frames that are identical
5. **Use optimize=True** in PIL's save method

```python
# Maximum optimization
frames[0].save(
    'emoji.gif',
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0,
    optimize=True,
    colors=48
)
```

## Dependencies

```bash
pip install pillow imageio numpy
```

Combine animation concepts creatively (bouncing + rotating, pulsing + sliding) for compelling results.
