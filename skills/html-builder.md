---
name: html-builder
description: Build rich, interactive HTML pages and single-page applications using modern frontend web technologies (React via CDN, Tailwind CSS, vanilla JS). Write complete HTML files via file_write. Use for dashboards, data visualizations, interactive tools, landing pages, or any complex frontend output.
---

# HTML Builder

Build rich, interactive HTML pages and write them to disk via `file_write`. This replaces artifact-based workflows with direct file output.

## Core Workflow

1. Plan the page structure and components needed
2. Write a complete HTML file via `file_write`
3. Open in browser or serve locally for testing if needed

## Stack Options

### Vanilla HTML + Tailwind (simplest)

For most pages, use Tailwind CSS via CDN:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 min-h-screen">
    <!-- Content here -->
</body>
</html>
```

### React via CDN (interactive apps)

For stateful, component-based UIs:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>App Title</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        function App() {
            const [state, setState] = React.useState(initialValue);
            return (
                <div className="container mx-auto p-4">
                    {/* Components here */}
                </div>
            );
        }
        ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    </script>
</body>
</html>
```

### With Chart.js (data visualization)

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
```

### With D3.js (complex visualizations)

```html
<script src="https://d3js.org/d3.v7.min.js"></script>
```

## Design Guidelines

Avoid common "AI slop" patterns:
- Do NOT use excessive centered layouts with purple gradients
- Do NOT use uniform rounded corners everywhere
- Do NOT default to Inter font for everything
- DO use proper visual hierarchy and whitespace
- DO use appropriate color schemes (reference theme-factory skill if needed)
- DO make layouts responsive with Tailwind breakpoints

## Writing the File

Use `file_write` to create the HTML file:

```
file_write to ~/project/output.html
```

For larger applications, split into multiple files:
- `index.html` - Main page
- `styles.css` - Custom styles (if Tailwind is insufficient)
- `app.js` - Application logic (if too large for inline script)

## Testing

After writing the file, optionally:

1. Serve locally to verify:
   ```bash
   cd /path/to/output && python -m http.server 8080 &
   ```

2. Use the webapp-testing skill with Playwright for automated verification

3. Take a screenshot for visual inspection:
   ```python
   from playwright.sync_api import sync_playwright
   with sync_playwright() as p:
       browser = p.chromium.launch(headless=True)
       page = browser.new_page()
       page.goto('file:///path/to/output.html')
       page.screenshot(path='/tmp/preview.png', full_page=True)
       browser.close()
   ```

## Common Patterns

### Dashboard Layout
```html
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-6">
    <div class="bg-white rounded-lg shadow p-4">
        <h3 class="text-lg font-semibold mb-2">Metric</h3>
        <p class="text-3xl font-bold">42</p>
    </div>
    <!-- More cards -->
</div>
```

### Navigation
```html
<nav class="bg-white shadow-sm border-b">
    <div class="max-w-7xl mx-auto px-4 flex items-center h-16">
        <a href="#" class="font-bold text-xl">App Name</a>
        <div class="ml-auto flex gap-4">
            <a href="#" class="text-gray-600 hover:text-gray-900">Link</a>
        </div>
    </div>
</nav>
```

### Form
```html
<form class="max-w-md mx-auto space-y-4">
    <div>
        <label class="block text-sm font-medium mb-1">Email</label>
        <input type="email" class="w-full border rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500">
    </div>
    <button type="submit" class="w-full bg-blue-600 text-white py-2 rounded-md hover:bg-blue-700">
        Submit
    </button>
</form>
```

## Self-Contained Principle

Every HTML file should be fully self-contained and openable with `file://` in a browser. All dependencies via CDN. No build step required.
