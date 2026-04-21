"""Project configuration system — named presets for different workloads."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from prometheus.config.paths import get_config_dir

log = logging.getLogger(__name__)

_PROJECTS_DIR = "anatomy/projects"


@dataclass
class ModelSlot:
    """A model slot within a project configuration."""

    name: str
    role: str = "primary"
    engine: str = "llama_cpp"
    machine: str = ""
    vram_estimate_gb: float = 0.0
    port: int = 8080
    gguf_file: str | None = None
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class ProjectConfig:
    """A named infrastructure configuration preset."""

    name: str
    description: str = ""
    models: list[ModelSlot] = field(default_factory=list)
    services: list[str] = field(default_factory=list)
    notes: str = ""
    last_used: str | None = None
    active: bool = False


class ProjectConfigStore:
    """Load and save project configs from ``~/.prometheus/anatomy/projects/``."""

    def __init__(self, projects_dir: Path | None = None) -> None:
        self._dir = projects_dir or (get_config_dir() / _PROJECTS_DIR)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def projects_dir(self) -> Path:
        return self._dir

    def list_projects(self) -> list[ProjectConfig]:
        """Return all project configs sorted by name."""
        configs: list[ProjectConfig] = []
        for path in sorted(self._dir.glob("*.yaml")):
            try:
                configs.append(self._load(path))
            except Exception:
                log.warning("Failed to load project config: %s", path)
        return configs

    def get(self, name: str) -> ProjectConfig | None:
        """Get a project config by name."""
        path = self._dir / f"{name}.yaml"
        if not path.exists():
            return None
        return self._load(path)

    def get_active(self) -> ProjectConfig | None:
        """Return the currently active project config (if any)."""
        for cfg in self.list_projects():
            if cfg.active:
                return cfg
        return None

    def save(self, config: ProjectConfig) -> None:
        """Save a project config to YAML."""
        path = self._dir / f"{config.name}.yaml"
        data = {
            "name": config.name,
            "description": config.description,
            "models": [
                {
                    "name": m.name,
                    "role": m.role,
                    "engine": m.engine,
                    "machine": m.machine,
                    "vram_estimate_gb": m.vram_estimate_gb,
                    "port": m.port,
                    "gguf_file": m.gguf_file,
                    "extra_flags": m.extra_flags,
                }
                for m in config.models
            ],
            "services": config.services,
            "notes": config.notes,
            "last_used": config.last_used,
            "active": config.active,
        }
        path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")

    def activate(self, name: str) -> bool:
        """Set *name* as the active config, deactivating all others."""
        found = False
        for cfg in self.list_projects():
            was_active = cfg.active
            cfg.active = cfg.name == name
            if cfg.active:
                found = True
            if cfg.active != was_active:
                self.save(cfg)
        return found

    def summaries(self) -> list[dict]:
        """Return lightweight summary dicts for AnatomyWriter."""
        result: list[dict] = []
        for cfg in self.list_projects():
            d: dict = {"name": cfg.name, "description": cfg.description}
            if cfg.models:
                d["models"] = ", ".join(m.name for m in cfg.models)
            if cfg.active:
                d["status"] = "active"
            elif cfg.last_used:
                d["last_used"] = cfg.last_used
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> ProjectConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        models = [
            ModelSlot(
                name=m.get("name", ""),
                role=m.get("role", "primary"),
                engine=m.get("engine", "llama_cpp"),
                machine=m.get("machine", ""),
                vram_estimate_gb=float(m.get("vram_estimate_gb", 0)),
                port=int(m.get("port", 8080)),
                gguf_file=m.get("gguf_file"),
                extra_flags=m.get("extra_flags", []),
            )
            for m in data.get("models", [])
        ]
        return ProjectConfig(
            name=data["name"],
            description=data.get("description", ""),
            models=models,
            services=data.get("services", []),
            notes=data.get("notes", ""),
            last_used=data.get("last_used"),
            active=data.get("active", False),
        )
