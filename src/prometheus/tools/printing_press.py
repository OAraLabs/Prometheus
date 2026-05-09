# Source: Original implementation for Prometheus (WEAVE-PRESS sprint)
# License: MIT
# Purpose: Discover and install Printing Press CLIs from a local clone of
#          github.com/mvanhorn/printing-press-library.
#
# When a user request needs a CLI that Prometheus doesn't yet have, the
# agent checks this registry, suggests an install, queues the action via
# ApprovalQueue, and (on approval) runs ``go install`` + copies SKILL.md
# into ~/.prometheus/skills/. The skill registry then hot-reloads so the
# new tool is available within the same conversation.

"""Printing Press CLI registry — discover, search, install."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from prometheus.config.paths import get_config_dir

log = logging.getLogger(__name__)


_DEFAULT_GIT_REMOTE = "https://github.com/mvanhorn/printing-press-library.git"
_DEFAULT_LIBRARY_LOCATIONS: tuple[Path, ...] = (
    Path.home() / "printing-press-library",
    Path("/tmp/printing-press-library"),
    Path.home() / "go" / "pkg" / "mod" / "github.com" / "mvanhorn"
    / "printing-press-library",
)
_GO_BIN_DIR = Path.home() / "go" / "bin"
_INSTALL_TIMEOUT_SECONDS = 180.0
_GIT_PULL_TIMEOUT_SECONDS = 30.0


@dataclass
class CLIRecord:
    """One CLI entry surfaced from the library."""

    name: str             # Registry name, e.g. "slack"
    skill_name: str       # Skill folder name, e.g. "pp-slack"
    category: str         # e.g. "productivity"
    description: str      # Short description (first 200 chars)
    install_module: str   # go install path
    bin_name: str         # Binary name on disk, e.g. "slack-pp-cli"
    skill_path: Path      # Path to SKILL.md in the library clone
    installed: bool       # True if bin is on PATH or in ~/go/bin

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "skill_name": self.skill_name,
            "category": self.category,
            "description": self.description,
            "install_module": self.install_module,
            "bin_name": self.bin_name,
            "skill_path": str(self.skill_path),
            "installed": self.installed,
        }


@dataclass
class InstallResult:
    """Outcome of a ``PrintingPressRegistry.install()`` call."""

    success: bool
    cli_name: str
    bin_name: str = ""
    version: str = ""
    skill_installed: bool = False
    skill_path: Path | None = None
    on_path: bool = False
    error: str | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class PrintingPressRegistry:
    """Discover and install CLIs from a local printing-press-library clone.

    Source order for the library clone (first hit wins):
      1. ``library_path`` constructor arg (from prometheus.yaml)
      2. ``~/printing-press-library/``
      3. ``/tmp/printing-press-library/``
      4. ``~/go/pkg/mod/github.com/mvanhorn/printing-press-library``

    If no clone is found anywhere, ``library_path`` is None and lookup
    methods return empty lists. (A future commit can add GitHub API
    fallback; for this sprint we require a local clone.)
    """

    def __init__(
        self,
        library_path: str | Path | None = None,
        *,
        skills_dest: Path | None = None,
    ) -> None:
        self._library_path: Path | None = self._locate_library(library_path)
        self._skills_dest: Path = skills_dest or (
            get_config_dir() / "skills"
        )
        self._skills_dest.mkdir(parents=True, exist_ok=True)
        # Optional reload hook — set by the daemon so install can refresh
        # the running skill registry.  Signature: () -> None.
        self._reload_callback: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def library_path(self) -> Path | None:
        return self._library_path

    def is_available(self) -> bool:
        """True iff a library clone was found."""
        return self._library_path is not None

    def set_reload_callback(self, callback: Any) -> None:
        """Set a zero-arg callback to fire after a successful install.

        Used by the daemon to call ``SkillRegistry.reload()`` so the
        new SKILL.md is picked up without restarting.
        """
        self._reload_callback = callback

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_available(self) -> list[CLIRecord]:
        """Enumerate CLIs from ``cli-skills/`` in the library clone."""
        if self._library_path is None:
            return []
        skills_root = self._library_path / "cli-skills"
        if not skills_root.is_dir():
            return []
        records: list[CLIRecord] = []
        for skill_dir in sorted(skills_root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                rec = self._record_from_skill(skill_md)
            except Exception:
                log.debug(
                    "PrintingPress: failed to parse %s", skill_md, exc_info=True
                )
                continue
            if rec is not None:
                records.append(rec)
        return records

    def search(self, query: str, *, limit: int = 10) -> list[CLIRecord]:
        """Fuzzy-match available CLIs by name, skill name, or description.

        Ranking (highest first):
          1. Exact match on name / bin_name / skill_name
          2. Substring match on name / skill_name
          3. Substring match on description
        """
        q = (query or "").strip().lower()
        if not q:
            return []
        all_records = self.list_available()
        scored: list[tuple[int, CLIRecord]] = []
        for rec in all_records:
            score = 0
            for field_val, weight_exact, weight_sub in (
                (rec.name.lower(), 100, 50),
                (rec.skill_name.lower(), 90, 45),
                (rec.bin_name.lower(), 80, 40),
                (rec.description.lower(), 0, 10),
            ):
                if not field_val:
                    continue
                if field_val == q:
                    score = max(score, weight_exact)
                elif q in field_val:
                    score = max(score, weight_sub)
            if score > 0:
                scored.append((score, rec))
        scored.sort(key=lambda t: (-t[0], t[1].name))
        return [rec for _, rec in scored[:limit]]

    def is_installed(self, bin_name: str) -> bool:
        """True if the binary is on PATH or present at ~/go/bin/<name>."""
        if not bin_name:
            return False
        if shutil.which(bin_name) is not None:
            return True
        # ~/go/bin is the default Go install destination; it may not be
        # on PATH but the binary is still functional from there.
        return (_GO_BIN_DIR / bin_name).is_file()

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    async def install(self, cli_name: str) -> InstallResult:
        """Install a CLI via ``go install`` and copy its SKILL.md.

        ``cli_name`` may be the registry name (``slack``), the
        ``pp-<name>`` skill folder, or the binary name (``slack-pp-cli``).
        """
        if shutil.which("go") is None:
            return InstallResult(
                success=False,
                cli_name=cli_name,
                error="`go` is not on PATH — install Go before running press install",
            )

        rec = self._resolve_record(cli_name)
        if rec is None:
            return InstallResult(
                success=False,
                cli_name=cli_name,
                error=(
                    f"No CLI matching '{cli_name}' in the Printing Press library "
                    f"({self._library_path or 'library not cloned'})"
                ),
            )

        # Run go install <module>@latest
        module_at_latest = f"{rec.install_module}@latest"
        try:
            proc = await asyncio.create_subprocess_exec(
                "go", "install", module_at_latest,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "GO111MODULE": "on"},
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_INSTALL_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return InstallResult(
                success=False,
                cli_name=rec.name,
                bin_name=rec.bin_name,
                error=f"go install timed out after {_INSTALL_TIMEOUT_SECONDS:.0f}s",
            )
        except FileNotFoundError:
            return InstallResult(
                success=False,
                cli_name=rec.name,
                bin_name=rec.bin_name,
                error="`go` binary disappeared mid-install",
            )

        if proc.returncode != 0:
            err = (stderr or b"").decode("utf-8", errors="replace").strip()
            return InstallResult(
                success=False,
                cli_name=rec.name,
                bin_name=rec.bin_name,
                error=f"go install failed (rc={proc.returncode}): {err[:400]}",
            )

        # Verify the binary lands somewhere we can find it
        bin_on_path = shutil.which(rec.bin_name) is not None
        bin_in_gopath = (_GO_BIN_DIR / rec.bin_name).is_file()
        if not (bin_on_path or bin_in_gopath):
            return InstallResult(
                success=False,
                cli_name=rec.name,
                bin_name=rec.bin_name,
                error=(
                    f"go install reported success but binary {rec.bin_name!r} "
                    f"not found on PATH or in {_GO_BIN_DIR}"
                ),
            )

        # Copy SKILL.md into ~/.prometheus/skills/<skill_name>.md
        skill_target = self._skills_dest / f"{rec.skill_name}.md"
        skill_installed = False
        try:
            shutil.copy2(rec.skill_path, skill_target)
            skill_installed = True
        except OSError as exc:
            log.warning(
                "PrintingPress: install %s succeeded but skill copy failed: %s",
                rec.name, exc,
            )

        # Hot-reload skill registry if the daemon wired a callback
        if self._reload_callback is not None and skill_installed:
            try:
                self._reload_callback()
            except Exception:
                log.exception(
                    "PrintingPress: skill registry reload callback failed"
                )

        return InstallResult(
            success=True,
            cli_name=rec.name,
            bin_name=rec.bin_name,
            version="latest",
            skill_installed=skill_installed,
            skill_path=skill_target if skill_installed else None,
            on_path=bin_on_path,
        )

    async def update_library(self) -> tuple[bool, str]:
        """Run ``git pull --ff-only`` in the library clone.

        Returns ``(ok, message)``.  No-op if the library isn't a git
        clone or no clone was found.
        """
        if self._library_path is None:
            return False, "library not cloned"
        if not (self._library_path / ".git").exists():
            return False, "library path is not a git checkout"
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", str(self._library_path), "pull", "--ff-only",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_GIT_PULL_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            return False, "git pull timed out"
        except FileNotFoundError:
            return False, "`git` not on PATH"
        text = (stdout or b"").decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            err = (stderr or b"").decode("utf-8", errors="replace").strip()
            return False, f"git pull failed: {err[:200]}"
        return True, text or "already up to date"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _locate_library(self, override: str | Path | None) -> Path | None:
        candidates: list[Path] = []
        if override:
            candidates.append(Path(override).expanduser())
        candidates.extend(_DEFAULT_LIBRARY_LOCATIONS)
        for p in candidates:
            try:
                resolved = p.expanduser().resolve()
            except (OSError, RuntimeError):
                continue
            # Library is valid if it contains cli-skills/
            if (resolved / "cli-skills").is_dir():
                return resolved
        return None

    def _record_from_skill(self, skill_md: Path) -> CLIRecord | None:
        """Parse a SKILL.md frontmatter into a ``CLIRecord``."""
        text = skill_md.read_text(encoding="utf-8", errors="replace")
        frontmatter = _extract_frontmatter(text)
        if not frontmatter:
            return None
        skill_name = str(frontmatter.get("name") or skill_md.parent.name)
        description = str(frontmatter.get("description") or "")[:400]
        # Pull install info from metadata.openclaw.install[0]
        meta = frontmatter.get("metadata") or {}
        openclaw = meta.get("openclaw") or {}
        installs = openclaw.get("install") or []
        if not isinstance(installs, list) or not installs:
            return None
        first_install = installs[0]
        if not isinstance(first_install, dict):
            return None
        if first_install.get("kind") != "go":
            # Sprint scope: Go installs only.  npm/etc. could ship later.
            return None
        module = first_install.get("module")
        bins = first_install.get("bins") or []
        if not module or not bins:
            return None
        bin_name = str(bins[0])

        # Registry name is the part after 'pp-' on the skill folder, when
        # it has that prefix; otherwise use the folder name directly.
        registry_name = skill_name[3:] if skill_name.startswith("pp-") else skill_name

        # Category: derive from the install module path
        # github.com/.../library/<category>/<service>/cmd/...
        category = ""
        parts = str(module).split("/")
        if "library" in parts:
            i = parts.index("library")
            if i + 1 < len(parts):
                category = parts[i + 1]

        bin_on_disk = (
            shutil.which(bin_name) is not None
            or (_GO_BIN_DIR / bin_name).is_file()
        )

        return CLIRecord(
            name=registry_name,
            skill_name=skill_name,
            category=category,
            description=description,
            install_module=str(module),
            bin_name=bin_name,
            skill_path=skill_md,
            installed=bin_on_disk,
        )

    def _resolve_record(self, cli_name: str) -> CLIRecord | None:
        """Find a record by registry name, skill folder, or binary name."""
        if not cli_name:
            return None
        normalized = cli_name.strip().lower()
        records = self.list_available()
        # Exact match on any of the three names
        for rec in records:
            if normalized in (
                rec.name.lower(),
                rec.skill_name.lower(),
                rec.bin_name.lower(),
            ):
                return rec
        # Fall back to top fuzzy hit
        matches = self.search(cli_name, limit=1)
        return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Frontmatter parser (PyYAML, which is already a project dep)
# ---------------------------------------------------------------------------


def _extract_frontmatter(text: str) -> dict[str, Any] | None:
    """Return the YAML frontmatter as a dict, or None if absent/malformed."""
    if not text.startswith("---"):
        return None
    # Find the closing ---
    rest = text[3:]
    end = rest.find("\n---")
    if end == -1:
        return None
    block = rest[:end].lstrip("\n")
    try:
        loaded = yaml.safe_load(block)
    except yaml.YAMLError:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def detect_command_not_found(bash_output: str) -> str | None:
    """Extract the missing command from a 'command not found' bash error.

    Bash emits one of:
      bash: line N: <name>: command not found
      <name>: command not found
      command not found: <name>
    Returns the candidate name or None if no pattern matched.
    """
    if not bash_output:
        return None
    # ``command not found`` appears in any of three forms across shells
    for line in bash_output.splitlines():
        line = line.strip()
        lowered = line.lower()
        if "command not found" not in lowered:
            continue
        # Form 1: "<name>: command not found"
        if ": command not found" in lowered:
            head = line.split(": command not found", 1)[0]
            head = head.split(":")[-1].strip()
            if head and " " not in head:
                return head
        # Form 2: "command not found: <name>"
        if "command not found:" in lowered:
            tail = line.split("command not found:", 1)[1].strip()
            tail = tail.split()[0] if tail else ""
            if tail:
                return tail
    return None
