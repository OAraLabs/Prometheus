"""Migration tool for importing data from Hermes Agent or OpenClaw into Prometheus.

Runs pre-agent with zero runtime dependencies — only stdlib + PyYAML.

Usage:
    python -m prometheus migrate --from hermes
    python -m prometheus migrate --from openclaw
    python -m prometheus migrate --from hermes --dry-run
    python -m prometheus migrate --from openclaw --source ~/.clawdbot
    python -m prometheus migrate --from hermes --overwrite
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Memory char limits (must match hermes_memory_tool.py)
_MEMORY_MAX_CHARS = 12_000
_USER_MAX_CHARS = 8_000


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class MigrationItem:
    """Single item to be migrated."""

    category: str            # "identity", "memory", "skills", "config", "secrets"
    source_path: Path
    dest_path: Path
    description: str
    action: str = "copy"     # "copy", "remap", "skip", "manual"
    status: str = "pending"  # "pending", "done", "skipped", "conflict", "error"
    conflict: str | None = None


@dataclass
class MigrationReport:
    """Summary of a migration run."""

    source: str
    source_path: Path
    timestamp: str
    items: list[MigrationItem] = field(default_factory=list)

    @property
    def migrated(self) -> list[MigrationItem]:
        return [i for i in self.items if i.status == "done"]

    @property
    def skipped(self) -> list[MigrationItem]:
        return [i for i in self.items if i.status in ("skipped", "conflict")]

    @property
    def errors(self) -> list[MigrationItem]:
        return [i for i in self.items if i.status == "error"]

    @property
    def manual(self) -> list[MigrationItem]:
        return [i for i in self.items if i.action == "manual"]


@dataclass
class MigrationOptions:
    """User-selected migration options."""

    source: str                    # "hermes" or "openclaw"
    source_path: Path
    dest_path: Path | None = None  # override ~/.prometheus/
    dry_run: bool = False
    overwrite: bool = False
    preset: str = "user-data"      # "full" includes secrets
    skill_conflict: str = "skip"   # "skip", "overwrite", "rename"

    @property
    def migrate_secrets(self) -> bool:
        return self.preset == "full"


# ------------------------------------------------------------------
# Source detection
# ------------------------------------------------------------------

def detect_sources() -> dict[str, Path]:
    """Detect installed agent systems. Returns {name: path} for found sources."""
    sources: dict[str, Path] = {}

    hermes_path = Path.home() / ".hermes"
    if hermes_path.exists() and (hermes_path / "config.yaml").exists():
        sources["hermes"] = hermes_path

    for dirname in (".openclaw", ".clawdbot", ".moldbot"):
        oc_path = Path.home() / dirname
        if not oc_path.exists():
            continue
        for config_name in ("openclaw.json", "clawdbot.json", "moldbot.json"):
            if (oc_path / config_name).exists():
                sources["openclaw"] = oc_path
                break
        if "openclaw" in sources:
            break

    return sources


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _deep_get(d: dict, dotted_key: str) -> Any:
    """Get nested dict value by dotted key path."""
    for key in dotted_key.split("."):
        if isinstance(d, dict):
            d = d.get(key)  # type: ignore[assignment]
        else:
            return None
    return d


def _deep_set(d: dict, dotted_key: str, value: Any) -> None:
    """Set nested dict value by dotted key path, creating intermediates."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


# ------------------------------------------------------------------
# Config key mappings
# ------------------------------------------------------------------

HERMES_TO_PROMETHEUS_KEYS: dict[str, str] = {
    "model.default": "model.model",
    "model.provider": "model.provider",
    "gateway.telegram.token": "gateway.telegram_token",
    "gateway.telegram.enabled": "gateway.telegram_enabled",
    "gateway.telegram.allowed_users": "gateway.allowed_chat_ids",
    "compression.threshold": "context.compression_trigger",
}

HERMES_PROVIDER_MAP: dict[str, str] = {
    "openrouter": "openai",
    "nous": "openai",
    "anthropic": "anthropic",
    "openai": "openai",
    "ollama": "ollama",
    "custom": "llama_cpp",
    "copilot": "openai",
}

OPENCLAW_MODEL_MAP: dict[str, str] = {
    "anthropic/claude-sonnet-4-20250514": "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-haiku-4-5-20251001": "anthropic/claude-haiku-4-5-20251001",
    "ollama/qwen3.5:32b": "llama_cpp/auto",
    "ollama/gemma4:26b": "llama_cpp/auto",
}


# ------------------------------------------------------------------
# Base migrator
# ------------------------------------------------------------------

class _BaseMigrator:
    """Shared copy/report logic for both migrators."""

    def __init__(self, options: MigrationOptions) -> None:
        self.options = options
        self.src = options.source_path
        self.dst = options.dest_path or (Path.home() / ".prometheus")

    def _safe_copy(self, item: MigrationItem, archive_dir: Path) -> None:
        """Copy file/directory, archiving existing destination if overwriting."""
        if item.dest_path.exists() and self.options.overwrite:
            archive_path = archive_dir / "archive" / item.dest_path.name
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            if item.dest_path.is_dir():
                shutil.copytree(item.dest_path, archive_path)
            else:
                shutil.copy2(item.dest_path, archive_path)

        item.dest_path.parent.mkdir(parents=True, exist_ok=True)
        if item.source_path.is_dir():
            shutil.copytree(item.source_path, item.dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(item.source_path, item.dest_path)

    def _copy_with_overflow(
        self, item: MigrationItem, archive_dir: Path, max_chars: int
    ) -> None:
        """Copy a memory file, trimming and archiving overflow if too large."""
        content = item.source_path.read_text(encoding="utf-8")
        if len(content) <= max_chars:
            self._safe_copy(item, archive_dir)
            return

        # Keep most-recent lines (end of file) that fit within limit
        lines = content.strip().splitlines()
        kept: list[str] = []
        total = 0
        for line in reversed(lines):
            if total + len(line) + 1 > max_chars:
                break
            kept.insert(0, line)
            total += len(line) + 1

        overflow_lines = lines[: len(lines) - len(kept)]

        # Archive overflow
        overflow_path = archive_dir / f"{item.dest_path.name}.overflow"
        overflow_path.parent.mkdir(parents=True, exist_ok=True)
        overflow_path.write_text("\n".join(overflow_lines), encoding="utf-8")

        # Write trimmed content
        item.dest_path.parent.mkdir(parents=True, exist_ok=True)
        item.dest_path.write_text("\n".join(kept), encoding="utf-8")
        item.description += f" (trimmed: {len(kept)} lines kept, {len(overflow_lines)} archived)"

    def _load_prometheus_config(self) -> dict:
        """Load existing prometheus.yaml or return empty dict."""
        config_path = Path("config/prometheus.yaml")
        if config_path.exists():
            with config_path.open(encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        return {}

    def _save_report(self, report: MigrationReport, archive_dir: Path) -> None:
        """Save migration report as YAML."""
        report_data = {
            "source": report.source,
            "source_path": str(report.source_path),
            "timestamp": report.timestamp,
            "summary": {
                "migrated": len(report.migrated),
                "skipped": len(report.skipped),
                "errors": len(report.errors),
                "manual": len(report.manual),
            },
            "items": [
                {
                    "category": i.category,
                    "source": str(i.source_path),
                    "destination": str(i.dest_path),
                    "description": i.description,
                    "action": i.action,
                    "status": i.status,
                    "conflict": i.conflict,
                }
                for i in report.items
            ],
        }
        path = archive_dir / "migration_report.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(report_data, fh, default_flow_style=False)

    def _execute_items(self, report: MigrationReport) -> None:
        """Execute all items in the report."""
        archive_dir = self.dst / "migration" / report.source / report.timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)

        for item in report.items:
            if item.action in ("skip", "manual"):
                item.status = "skipped"
                continue
            if item.status == "conflict" and not self.options.overwrite:
                continue
            try:
                if item.action == "copy":
                    if item.category == "memory" and item.dest_path.name == "MEMORY.md":
                        self._copy_with_overflow(item, archive_dir, _MEMORY_MAX_CHARS)
                    elif item.category == "memory" and item.dest_path.name == "USER.md":
                        self._copy_with_overflow(item, archive_dir, _USER_MAX_CHARS)
                    else:
                        self._safe_copy(item, archive_dir)
                elif item.action == "remap":
                    self._remap_config(item)
                item.status = "done"
            except Exception as exc:
                item.status = "error"
                item.conflict = str(exc)

        self._save_report(report, archive_dir)

    def _remap_config(self, item: MigrationItem) -> None:
        """Override in subclass."""
        raise NotImplementedError


# ------------------------------------------------------------------
# Hermes migrator
# ------------------------------------------------------------------

class HermesMigrator(_BaseMigrator):
    """Migrate from Hermes Agent (~/.hermes/) to Prometheus (~/.prometheus/)."""

    def scan(self) -> MigrationReport:
        """Scan source and build migration plan (no writes)."""
        report = MigrationReport(
            source="hermes",
            source_path=self.src,
            timestamp=datetime.now().strftime("%Y%m%dT%H%M%S"),
        )
        self._scan_identity(report)
        self._scan_memory(report)
        self._scan_skills(report)
        self._scan_config(report)
        self._scan_cron(report)
        if self.options.migrate_secrets:
            self._scan_secrets(report)
        return report

    def execute(self) -> MigrationReport:
        """Scan then execute."""
        report = self.scan()
        if self.options.dry_run:
            return report
        self._execute_items(report)
        return report

    # -- scanners -------------------------------------------------------

    def _scan_identity(self, report: MigrationReport) -> None:
        for filename in ("SOUL.md", "AGENTS.md"):
            src = self.src / filename
            if not src.exists():
                continue
            dst = self.dst / filename
            item = MigrationItem(
                category="identity", source_path=src, dest_path=dst,
                description=f"Identity file: {filename}",
            )
            if dst.exists():
                item.status = "conflict"
                item.conflict = f"Prometheus already has {filename}"
            report.items.append(item)

    def _scan_memory(self, report: MigrationReport) -> None:
        for filename in ("MEMORY.md", "USER.md"):
            src = self.src / "memories" / filename
            if not src.exists():
                src = self.src / filename
            if not src.exists():
                continue
            dst = self.dst / filename
            item = MigrationItem(
                category="memory", source_path=src, dest_path=dst,
                description=f"Memory file: {filename}",
            )
            if dst.exists():
                item.status = "conflict"
                item.conflict = f"Prometheus already has {filename}"
            report.items.append(item)

        daily_src = self.src / "memories" / "daily"
        if daily_src.exists() and any(daily_src.iterdir()):
            count = len(list(daily_src.glob("*.md")))
            report.items.append(MigrationItem(
                category="memory", source_path=daily_src,
                dest_path=self.dst / "memory" / "daily",
                description=f"Daily memory notes ({count} files)",
            ))

    def _scan_skills(self, report: MigrationReport) -> None:
        skills_src = self.src / "skills"
        if not skills_src.exists():
            return
        for skill_md in skills_src.rglob("*.md"):
            rel = skill_md.relative_to(skills_src)
            dst = self.dst / "skills" / "imported" / rel
            item = MigrationItem(
                category="skills", source_path=skill_md, dest_path=dst,
                description=f"Skill: {rel}",
            )
            if dst.exists():
                if self.options.skill_conflict == "skip":
                    item.status = "conflict"
                    item.conflict = f"Skill '{rel}' already exists"
                elif self.options.skill_conflict == "rename":
                    item.dest_path = dst.with_stem(dst.stem + "-hermes")
            report.items.append(item)

    def _scan_config(self, report: MigrationReport) -> None:
        src = self.src / "config.yaml"
        if src.exists():
            report.items.append(MigrationItem(
                category="config", source_path=src,
                dest_path=Path("config/prometheus.yaml"),
                description="Config remapping (Hermes config.yaml)",
                action="remap",
            ))

    def _scan_cron(self, report: MigrationReport) -> None:
        cron_src = self.src / "cron"
        if cron_src.exists() and any(cron_src.iterdir()):
            count = len(list(cron_src.glob("*")))
            report.items.append(MigrationItem(
                category="config", source_path=cron_src,
                dest_path=self.dst / "cron",
                description=f"Cron jobs ({count} files)",
            ))

    def _scan_secrets(self, report: MigrationReport) -> None:
        env_src = self.src / ".env"
        if env_src.exists():
            report.items.append(MigrationItem(
                category="secrets", source_path=env_src,
                dest_path=Path("(env var guidance)"),
                description="API keys from .env",
                action="manual",
            ))

    # -- config remapping -----------------------------------------------

    def _remap_config(self, item: MigrationItem) -> None:
        with item.source_path.open(encoding="utf-8") as fh:
            hermes_cfg = yaml.safe_load(fh) or {}

        prom_cfg = self._load_prometheus_config()

        for hkey, pkey in HERMES_TO_PROMETHEUS_KEYS.items():
            val = _deep_get(hermes_cfg, hkey)
            if val is not None:
                _deep_set(prom_cfg, pkey, val)

        provider = _deep_get(hermes_cfg, "model.provider")
        if provider:
            _deep_set(prom_cfg, "model.provider",
                       HERMES_PROVIDER_MAP.get(provider, provider))

        with item.dest_path.open("w", encoding="utf-8") as fh:
            yaml.dump(prom_cfg, fh, default_flow_style=False, sort_keys=False)


# ------------------------------------------------------------------
# OpenClaw migrator
# ------------------------------------------------------------------

class OpenClawMigrator(_BaseMigrator):
    """Migrate from OpenClaw (~/.openclaw/) to Prometheus (~/.prometheus/)."""

    def __init__(self, options: MigrationOptions) -> None:
        super().__init__(options)
        self.oc_config = self._load_openclaw_config()
        self.workspace = self._find_workspace()

    def _load_openclaw_config(self) -> dict:
        for name in ("openclaw.json", "clawdbot.json", "moldbot.json"):
            path = self.src / name
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    return json.load(fh)
        return {}

    def _find_workspace(self) -> Path | None:
        agents = self.oc_config.get("agents", {})
        for agent_conf in agents.values():
            ws = agent_conf.get("workspace")
            if ws:
                ws_path = Path(ws).expanduser()
                if ws_path.exists():
                    return ws_path
        for candidate in (Path.home() / "clawd", self.src / "workspace"):
            if candidate.exists():
                return candidate
        return None

    def scan(self) -> MigrationReport:
        report = MigrationReport(
            source="openclaw",
            source_path=self.src,
            timestamp=datetime.now().strftime("%Y%m%dT%H%M%S"),
        )
        if self.workspace:
            self._scan_workspace(report)
            self._scan_skills(report)
            self._scan_memory_notes(report)
        self._scan_config(report)
        if self.options.migrate_secrets:
            self._scan_secrets(report)
        return report

    def execute(self) -> MigrationReport:
        report = self.scan()
        if self.options.dry_run:
            return report
        self._execute_items(report)
        return report

    # -- scanners -------------------------------------------------------

    def _scan_workspace(self, report: MigrationReport) -> None:
        assert self.workspace is not None
        for filename in ("SOUL.md", "AGENTS.md", "MEMORY.md", "USER.md"):
            src = self.workspace / filename
            if not src.exists():
                continue
            cat = "identity" if filename in ("SOUL.md", "AGENTS.md") else "memory"
            dst = self.dst / filename
            item = MigrationItem(
                category=cat, source_path=src, dest_path=dst,
                description=f"Workspace file: {filename}",
            )
            if dst.exists():
                item.status = "conflict"
                item.conflict = f"Prometheus already has {filename}"
            report.items.append(item)

    def _scan_skills(self, report: MigrationReport) -> None:
        assert self.workspace is not None
        skills_src = self.workspace / "skills"
        if not skills_src.exists():
            return
        for skill_md in skills_src.rglob("*.md"):
            rel = skill_md.relative_to(skills_src)
            dst = self.dst / "skills" / "imported" / rel
            item = MigrationItem(
                category="skills", source_path=skill_md, dest_path=dst,
                description=f"Skill: {rel}",
            )
            if dst.exists():
                item.status = "conflict"
                item.conflict = f"Skill '{rel}' already exists"
            report.items.append(item)

    def _scan_memory_notes(self, report: MigrationReport) -> None:
        assert self.workspace is not None
        memory_src = self.workspace / "memory"
        if memory_src.exists() and any(memory_src.iterdir()):
            count = len(list(memory_src.rglob("*.md")))
            report.items.append(MigrationItem(
                category="memory", source_path=memory_src,
                dest_path=self.dst / "memory",
                description=f"Memory notes ({count} files)",
            ))

    def _scan_config(self, report: MigrationReport) -> None:
        for name in ("openclaw.json", "clawdbot.json", "moldbot.json"):
            src = self.src / name
            if src.exists():
                report.items.append(MigrationItem(
                    category="config", source_path=src,
                    dest_path=Path("config/prometheus.yaml"),
                    description=f"Config remapping ({name})",
                    action="remap",
                ))
                break

    def _scan_secrets(self, report: MigrationReport) -> None:
        env_src = self.src / ".env"
        if env_src.exists():
            report.items.append(MigrationItem(
                category="secrets", source_path=env_src,
                dest_path=Path("(env var guidance)"),
                description="API keys from .env",
                action="manual",
            ))

    # -- config remapping -----------------------------------------------

    def _remap_config(self, item: MigrationItem) -> None:
        prom_cfg = self._load_prometheus_config()

        agents = self.oc_config.get("agents", {})
        main_agent = agents.get("main", {})
        model = main_agent.get("model")
        if isinstance(model, str):
            mapped = OPENCLAW_MODEL_MAP.get(model, model)
            if "/" in mapped:
                provider, model_name = mapped.split("/", 1)
                _deep_set(prom_cfg, "model.provider", provider)
                _deep_set(prom_cfg, "model.model", model_name)
        elif isinstance(model, dict):
            primary = model.get("primary", model.get("default", ""))
            if primary and "/" in primary:
                provider, model_name = primary.split("/", 1)
                _deep_set(prom_cfg, "model.provider", provider)
                _deep_set(prom_cfg, "model.model", model_name)

        channels = self.oc_config.get("channels", {})
        tg = channels.get("telegram", {})
        if tg.get("enabled"):
            _deep_set(prom_cfg, "gateway.telegram_enabled", True)

        with item.dest_path.open("w", encoding="utf-8") as fh:
            yaml.dump(prom_cfg, fh, default_flow_style=False, sort_keys=False)


# ------------------------------------------------------------------
# CLI printing
# ------------------------------------------------------------------

def _print_plan(report: MigrationReport, options: MigrationOptions) -> None:
    print(f"\n  Prometheus Migration: {report.source}")
    print(f"  Source:  {report.source_path}")
    print(f"  Target:  ~/.prometheus/")
    print(f"  Preset:  {options.preset}")
    print(f"  Mode:    {'DRY RUN' if options.dry_run else 'LIVE'}\n")

    categories: dict[str, list[MigrationItem]] = {}
    for item in report.items:
        categories.setdefault(item.category, []).append(item)

    for cat, items in categories.items():
        print(f"  {cat.upper()}:")
        for item in items:
            tag = ""
            if item.status == "conflict":
                tag = f" [CONFLICT: {item.conflict}]"
            elif item.action == "manual":
                tag = " [MANUAL]"
            arrow = "->" if item.action != "skip" else "xx"
            print(f"    {arrow} {item.description}{tag}")
        print()


def _print_results(report: MigrationReport) -> None:
    print(f"\n  Migration complete: {len(report.migrated)} migrated, "
          f"{len(report.skipped)} skipped, {len(report.errors)} errors")

    if report.errors:
        print("\n  ERRORS:")
        for item in report.errors:
            print(f"    x {item.description}: {item.conflict}")

    if report.manual:
        print("\n  MANUAL STEPS:")
        for item in report.manual:
            if item.category == "secrets":
                print(f"    Copy API keys from {item.source_path} to your shell profile:")
                print(f"      cat {item.source_path}")
                print(f"      export OPENAI_API_KEY=sk-...")

    print()


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def run_migration(args: Any) -> bool:
    """Main entry point, called from __main__.py. Returns True on success."""
    source = args.source_type
    source_path: Path

    if hasattr(args, "source_path") and args.source_path:
        source_path = Path(args.source_path).expanduser()
    else:
        detected = detect_sources()
        found = detected.get(source)
        if not found:
            print(f"\n  Could not find {source} installation.")
            if source == "hermes":
                print("  Expected: ~/.hermes/config.yaml")
            else:
                print("  Expected: ~/.openclaw/openclaw.json")
            print("  Use --source /path/to/directory to specify manually.\n")
            return False
        source_path = found

    options = MigrationOptions(
        source=source,
        source_path=source_path,
        dry_run=getattr(args, "dry_run", False),
        overwrite=getattr(args, "overwrite", False),
        preset=getattr(args, "preset", "user-data"),
        skill_conflict=getattr(args, "skill_conflict", "skip"),
    )

    migrator: _BaseMigrator
    if source == "hermes":
        migrator = HermesMigrator(options)
    elif source == "openclaw":
        migrator = OpenClawMigrator(options)
    else:
        print(f"  Unknown source: {source}")
        return False

    report = migrator.scan()
    _print_plan(report, options)

    if options.dry_run:
        print("  Dry run — no files were changed.\n")
        return True

    if not getattr(args, "yes", False):
        try:
            resp = input("  Proceed with migration? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Migration cancelled.\n")
            return False
        if resp and resp != "y":
            print("  Migration cancelled.\n")
            return False

    report = migrator.execute()
    _print_results(report)
    return len(report.errors) == 0
