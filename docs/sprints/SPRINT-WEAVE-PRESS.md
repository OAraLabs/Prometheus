# SPRINT: Printing Press CLI Auto-Discovery
**Codename:** WEAVE-PRESS
**Date:** 2026-05-09
**Estimated Time:** 1 session (~2 hours)
**Dependencies:** Printing Press factory binary installed, SYMBIOTE approval queue pattern
**Additive Only**

---

## What This Sprint Does

When Prometheus can't handle a request because it lacks a CLI for a specific
service, it checks the Printing Press library for a matching CLI. If one exists,
it offers to install it. If the user approves, it installs the binary, copies
the skill file, and uses it immediately.

Flow:

```
User: "Check my Cal.com calendar for tomorrow"
    ↓
Prometheus: no CLI for Cal.com
    ↓
Check Printing Press library → cal-com-pp-cli exists
    ↓
Telegram: "I found a Cal.com CLI in Printing Press. Want me to install it?"
    ↓
User: "Yes"
    ↓
Install binary + copy skill → use immediately
    ↓
Prometheus: "Here's your calendar for tomorrow: ..."
```

---

## Read These Files First

```
src/prometheus/tools/builtin/bash.py                    # how bash tool works
src/prometheus/symbiote/coordinator.py                  # approval queue pattern
src/prometheus/gateway/telegram.py                      # where /approve flow lives
src/prometheus/skills/loader.py                         # how skills get loaded
src/prometheus/skills/registry.py                       # skill registry internals
src/prometheus/learning/skill_creator.py                # how skills get written to auto/
~/.prometheus/skills/                                   # existing skill files
```

Answer before writing code:
1. How does the approval queue work — exact enqueue + approve pattern?
2. How does SkillRegistry reload skills at runtime — can it hot-load or does daemon need restart?
3. Where is the printing-press-library cloned to on this machine? (find it)
4. What's the directory structure of the cloned library — where are SKILL.md files relative to each CLI?
5. Is the `printing-press` factory binary installed? (`which printing-press`)

---

## Step 1 — PrintingPressRegistry (~100 lines)

**New file:** `src/prometheus/tools/printing_press.py`

```python
class PrintingPressRegistry:
    """Discovers and manages Printing Press CLIs.

    Two sources:
    1. Local library clone (fast, offline) — check first
    2. GitHub API fallback (if clone not found)

    Responsibilities:
    - List available CLIs from the library
    - Check if a CLI is already installed (which <name>)
    - Install a CLI (go install <path>)
    - Copy SKILL.md to ~/.prometheus/skills/
    - Verify installation
    """

    def __init__(self, library_path: Path = None):
        """Find the cloned printing-press-library on disk.

        Search order:
        1. Explicit library_path from config
        2. ~/printing-press-library/
        3. /tmp/printing-press-library/

        If not found anywhere, set self.library_path = None
        and fall back to GitHub API for discovery.
        """

    def list_available(self) -> list[dict]:
        """List all CLIs in the library.

        Returns list of:
        {
            "name": "cal-com-pp-cli",
            "category": "productivity",
            "description": "...",  # from SKILL.md first line
            "install_path": "github.com/mvanhorn/...",
            "skill_path": "/path/to/SKILL.md",
            "installed": bool,  # check with shutil.which()
        }

        Reads from local library clone directory structure.
        Each CLI has a cmd/ directory and a cli-skills/ SKILL.md.
        """

    def is_installed(self, cli_name: str) -> bool:
        """Check if CLI binary is on PATH."""
        return shutil.which(cli_name) is not None

    async def install(self, cli_name: str) -> InstallResult:
        """Install a CLI via go install.

        1. Find the go install path from the library
        2. Run: go install <path>@latest
        3. Verify: which <cli_name>
        4. Copy SKILL.md to ~/.prometheus/skills/<name>.md
        5. Trigger skill registry reload

        Returns InstallResult with success, version, skill_path.
        """

    async def search(self, query: str) -> list[dict]:
        """Fuzzy search available CLIs by name or description.

        Used when the agent knows the service name but not
        the exact CLI name. "calendar" → cal-com-pp-cli.
        """

    def _find_install_path(self, cli_name: str) -> str | None:
        """Find the go install path for a CLI from the library structure."""

    def _find_skill_path(self, cli_name: str) -> Path | None:
        """Find the SKILL.md for a CLI from the library structure."""


@dataclass
class InstallResult:
    success: bool
    cli_name: str
    version: str
    skill_installed: bool
    skill_path: Path | None
    error: str | None
```

---

## Step 2 — Tool-Not-Found Hook (~80 lines)

**Modify:** `src/prometheus/engine/agent_loop.py`

Find where the agent loop handles unknown tool errors or circuit breaker trips.
Add a hook that fires when the model tries to call a tool/CLI that doesn't exist
or when a bash command returns "command not found".

```python
# In the error handling path for tool execution failures:

async def _on_tool_not_found(self, tool_name: str, user_message: str) -> str | None:
    """Check Printing Press when a tool/CLI isn't available.

    Trigger conditions:
    1. Bash tool returns "command not found: <cli-name>"
    2. Model mentions a service name we don't have a tool for

    Does NOT trigger for:
    - Internal tool names (file_read, web_fetch, etc.)
    - Already-installed CLIs that just errored
    - Background/automated tasks (SENTINEL, GEPA)

    Returns a suggestion message to inject into the conversation,
    or None if no matching CLI found.
    """
    if not self.printing_press:
        return None

    # Extract service name from the error or user message
    # Search Printing Press registry
    matches = await self.printing_press.search(tool_name)

    if not matches:
        return None

    best = matches[0]
    if best["installed"]:
        return None  # already installed, error is something else

    # Return suggestion — don't auto-install
    return (
        f"I found a CLI for {best['name']} in the Printing Press library. "
        f"Want me to install it? Say 'yes' to proceed."
    )
```

**Important:** This hook only fires for user-initiated sessions, never for
background tasks. Check `session.origin` before suggesting.

---

## Step 3 — Install Approval Flow (~60 lines)

**Modify:** `src/prometheus/gateway/telegram.py`

When the agent suggests installing a CLI and the user says yes,
route through the approval queue.

```python
# Add to Telegram command handler:

/press list          → show all available Printing Press CLIs
/press search <query> → search for a CLI by name/service
/press install <name> → install a specific CLI (queues approval)
/press installed      → show what's already installed
/press update         → git pull the library clone to get new CLIs
```

The `/press install` command enqueues to the same ApprovalQueue
used by `/gepa run` and `/symbiote`. Follow the exact same pattern.

DO NOT TOUCH existing commands.

---

## Step 4 — Library Auto-Update (~30 lines)

**Add to SENTINEL cron or daemon startup:**

```python
# On daemon startup — pull latest library if clone exists:
async def _update_printing_press_library():
    """git pull the printing-press-library to get new CLIs.

    Only if:
    - Library clone exists on disk
    - Last pull was > 24 hours ago
    - Network is available

    Silent on failure — stale library is fine.
    """
    lib_path = Path.home() / "printing-press-library"
    if not lib_path.exists():
        return

    result = await asyncio.create_subprocess_exec(
        "git", "-C", str(lib_path), "pull", "--ff-only",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await result.wait()
```

---

## Step 5 — Skill Hot-Reload (~40 lines)

After installing a CLI and copying its SKILL.md, the skill registry
needs to pick it up without a daemon restart.

**Modify:** `src/prometheus/skills/loader.py` or `registry.py`

```python
def reload_skills(self):
    """Re-scan ~/.prometheus/skills/ and update the registry.

    Called after a Printing Press CLI install.
    Does NOT remove existing skills — only adds new ones.
    """
```

If hot-reload is already supported, just call it after install.
If not, add it — it's essential for the install-and-use-immediately flow.

---

## Step 6 — Config

**Add to `prometheus.yaml`:**

```yaml
printing_press:
  enabled: true
  library_path: null          # auto-detect from ~/printing-press-library/
  auto_suggest: true          # suggest CLIs when command not found
  auto_update_library: true   # git pull on daemon startup
```

**Add to `prometheus.yaml.default`:**
Same keys, `enabled: false` for public defaults.

---

## Step 7 — Tests

**New file:** `tests/test_printing_press.py`

```python
def test_registry_finds_local_library():
    """Discovers printing-press-library clone on disk."""

def test_registry_lists_available_clis():
    """Lists CLIs with name, category, install status."""

def test_registry_detects_installed():
    """is_installed returns True for CLIs on PATH."""

def test_registry_search_fuzzy():
    """search('calendar') finds cal-com-pp-cli."""

def test_registry_search_no_match():
    """search('xyzzy') returns empty list."""

def test_install_copies_skill():
    """After install, SKILL.md exists in ~/.prometheus/skills/."""

def test_skill_hot_reload():
    """New skill is available without daemon restart."""

def test_tool_not_found_suggests_cli():
    """'command not found: cal-com-pp-cli' triggers suggestion."""

def test_tool_not_found_ignores_internal_tools():
    """Internal tool names don't trigger Printing Press lookup."""

def test_tool_not_found_ignores_background_tasks():
    """SENTINEL/GEPA tasks don't trigger suggestions."""

def test_press_list_command():
    """Telegram /press list returns formatted CLI list."""

def test_press_install_queues_approval():
    """/press install enqueues to ApprovalQueue."""
```

**Add to `tests/test_wiring.py`:**

```python
def test_printing_press_registry_wired():
    """PrintingPressRegistry is instantiated in daemon."""

def test_printing_press_hook_wired():
    """Tool-not-found hook is registered in agent loop."""
```

---

## Commit

```bash
python3 -m pytest tests/ -v --tb=short
git add -A && git commit -m "WEAVE-PRESS: Printing Press CLI auto-discovery and install"
```

---

## After This Sprint

Test the full flow in Telegram:

```
# See what's available:
/press list

# Search for something:
/press search calendar

# Install something new:
/press install cal-com-pp-cli

# Use it immediately:
What's on my calendar tomorrow?
```

Also test the auto-suggest flow:

```
# Ask for something you don't have a CLI for:
Check my Sentry error rates for the last 24 hours

# Prometheus should respond:
# "I found a Sentry CLI in Printing Press. Want me to install it?"
```

---

## Rules

- ADDITIVE ONLY — do not touch existing tools or commands
- Installation goes through ApprovalQueue — never auto-install
- Only suggest for user-initiated tasks, never background
- The library clone stays where it is — don't move or restructure it
- Skill files are copied, not symlinked — survives library deletion
- `go install` requires Go on PATH — check first, error gracefully if missing
- `python3 -m pytest` not `uv run pytest`
- Update PROMETHEUS.md when done

---

*"Need a tool? The press prints one."*
