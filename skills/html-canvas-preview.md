---
name: html-canvas-preview
description: Generate and serve HTML content locally for previewing games, visualizations, dashboards, and interactive demos. Adapted from OpenClaw canvas patterns for local file-based workflows.
version: 1.0.0
tags: [HTML, Visualization, Dashboard, Preview, Web]
---

# HTML Canvas Preview

Generate HTML content and serve it locally for previewing games, visualizations, dashboards, and interactive demos.

## Overview

Write HTML/CSS/JS files to a local directory and serve them with Python's built-in HTTP server. Useful for:

- Displaying games, visualizations, dashboards
- Previewing generated HTML content
- Interactive demos and prototypes

## Workflow

### 1. Create HTML content

Write self-contained HTML files using `file_write`:

```bash
cat > /tmp/canvas/my-viz.html << 'HTML'
<!DOCTYPE html>
<html>
<head><title>My Visualization</title></head>
<body>
  <h1>Hello Canvas!</h1>
  <canvas id="c" width="600" height="400"></canvas>
  <script>
    const ctx = document.getElementById('c').getContext('2d');
    // Draw something
    ctx.fillStyle = '#3498db';
    ctx.fillRect(50, 50, 200, 150);
  </script>
</body>
</html>
HTML
```

### 2. Serve locally

```bash
mkdir -p /tmp/canvas
cd /tmp/canvas && python3 -m http.server 8080 &
echo "Preview at: http://localhost:8080/my-viz.html"
```

For Tailscale access across devices:

```bash
# Find your Tailscale IP
tailscale ip -4

# Serve on all interfaces
cd /tmp/canvas && python3 -m http.server 8080 --bind 0.0.0.0 &
echo "Preview at: http://$(tailscale ip -4):8080/my-viz.html"
```

### 3. Live development

For iterative development, just overwrite the file and refresh the browser:

```bash
# Update the file
cat > /tmp/canvas/my-viz.html << 'HTML'
<!DOCTYPE html>
<html>
<head><title>Updated Viz</title></head>
<body>
  <h1>Updated content</h1>
</body>
</html>
HTML
# Browser refresh picks up changes automatically
```

For auto-reload, add a simple script to the HTML:

```html
<script>
  // Auto-reload every 2 seconds during development
  setTimeout(() => location.reload(), 2000);
</script>
```

### 4. Stop the server

```bash
# Find and kill the HTTP server
pkill -f "python3 -m http.server 8080"
```

## Tips

- Keep HTML self-contained (inline CSS/JS) for simplest portability
- Use `/tmp/canvas/` as the default working directory
- For complex multi-file projects, organize with subdirectories
- Save screenshots with a browser automation tool or just describe the output
- Access from other machines on the Tailnet by binding to 0.0.0.0
