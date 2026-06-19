---
name: webapp-testing
description: Toolkit for interacting with and testing local web applications using Playwright. Supports verifying frontend functionality, debugging UI behavior, capturing browser screenshots, and viewing browser logs. Use when testing web apps, automating browser interactions, or debugging frontend issues.
---

# Web Application Testing

Test local web applications by writing Python Playwright scripts and running them via `bash`.

## Decision Tree

```
User task -> Is it static HTML?
    Yes -> Use file_read to inspect HTML and identify selectors
           -> Write Playwright script using selectors
    No (dynamic webapp) -> Is the server already running?
        No -> Start the server, then write Playwright script
        Yes -> Reconnaissance-then-action:
            1. Navigate and wait for networkidle
            2. Take screenshot or inspect DOM
            3. Identify selectors from rendered state
            4. Execute actions with discovered selectors
```

## Starting a Server

Start the dev server in the background, then run automation:

```bash
# Start server in background
cd /path/to/project && npm run dev &
SERVER_PID=$!
sleep 3  # Wait for server to be ready

# Run automation
python /tmp/test_script.py

# Clean up
kill $SERVER_PID
```

For multiple servers (backend + frontend):
```bash
cd backend && python server.py &
BACKEND_PID=$!
cd frontend && npm run dev &
FRONTEND_PID=$!
sleep 5

python /tmp/test_script.py

kill $BACKEND_PID $FRONTEND_PID
```

## Writing Playwright Scripts

Write the script via `file_write`, then run via `bash`:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)  # Always headless
    page = browser.new_page()
    page.goto('http://localhost:5173')
    page.wait_for_load_state('networkidle')  # CRITICAL: Wait for JS

    # ... your automation logic

    browser.close()
```

## Reconnaissance-Then-Action Pattern

1. **Inspect rendered DOM**:
   ```python
   page.screenshot(path='/tmp/inspect.png', full_page=True)
   content = page.content()
   page.locator('button').all()
   ```

2. **Identify selectors** from inspection results

3. **Execute actions** using discovered selectors

## Common Pitfall

Do NOT inspect the DOM before waiting for `networkidle` on dynamic apps. Always wait for `page.wait_for_load_state('networkidle')` before inspection.

## Best Practices

- Use `sync_playwright()` for synchronous scripts
- Always close the browser when done
- Use descriptive selectors: `text=`, `role=`, CSS selectors, or IDs
- Add appropriate waits: `page.wait_for_selector()` or `page.wait_for_timeout()`
- Take screenshots to `/tmp/` and use `file_read` to inspect them visually
- Write scripts to `/tmp/` via `file_write` to keep the workspace clean

## Example: Full Test Flow

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Navigate
    page.goto('http://localhost:3000')
    page.wait_for_load_state('networkidle')

    # Screenshot for inspection
    page.screenshot(path='/tmp/initial.png', full_page=True)

    # Interact
    page.fill('input[name="email"]', 'test@example.com')
    page.click('button[type="submit"]')
    page.wait_for_selector('.success-message')

    # Verify
    page.screenshot(path='/tmp/after_submit.png', full_page=True)
    assert page.locator('.success-message').is_visible()

    browser.close()
```

## Dependencies

```bash
pip install playwright
playwright install chromium
```
