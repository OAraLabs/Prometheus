"""LSP server definitions and language mapping.

Maps file extensions to language server commands. Provides project-root
detection by walking up the directory tree looking for root markers.

Modeled after OpenCode's ``server.ts`` / ``language.ts`` pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LSPServerDef:
    """Definition for a language server."""

    language_id: str
    extensions: list[str]
    command: list[str]
    root_markers: list[str] = field(default_factory=list)
    install_command: list[str] | None = None
    initialization_options: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Built-in server definitions
# ------------------------------------------------------------------

BUILTIN_SERVERS: dict[str, LSPServerDef] = {
    "python": LSPServerDef(
        language_id="python",
        extensions=[".py", ".pyi"],
        command=["pyright-langserver", "--stdio"],
        root_markers=["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", ".git"],
        install_command=["pip", "install", "pyright"],
    ),
    "typescript": LSPServerDef(
        language_id="typescript",
        extensions=[".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"],
        command=["typescript-language-server", "--stdio"],
        root_markers=["tsconfig.json", "package.json", ".git"],
        install_command=["npm", "install", "-g", "typescript-language-server", "typescript"],
    ),
    "go": LSPServerDef(
        language_id="go",
        extensions=[".go"],
        command=["gopls", "serve"],
        root_markers=["go.mod", "go.sum", ".git"],
    ),
    "rust": LSPServerDef(
        language_id="rust",
        extensions=[".rs"],
        command=["rust-analyzer"],
        root_markers=["Cargo.toml", ".git"],
    ),
    "c": LSPServerDef(
        language_id="c",
        extensions=[".c", ".h", ".cpp", ".hpp", ".cc", ".cxx"],
        command=["clangd"],
        root_markers=["compile_commands.json", "CMakeLists.txt", "Makefile", ".git"],
    ),
}

# Extension → language ID for textDocument/didOpen
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".ts": "typescript", ".tsx": "typescriptreact",
    ".js": "javascript", ".jsx": "javascriptreact",
    ".mjs": "javascript", ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".cc": "cpp", ".cxx": "cpp",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".md": "markdown", ".html": "html", ".css": "css",
}


def find_project_root(filepath: Path, root_markers: list[str]) -> Path:
    """Walk up from *filepath* to find a project root via marker files.

    Returns the first directory containing any marker, or the filesystem
    root as a last resort.
    """
    current = filepath.parent if filepath.is_file() else filepath
    while True:
        for marker in root_markers:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            return filepath.parent  # fallback: directory of the file
        current = parent


def get_server_for_file(
    filepath: str | Path,
    custom_servers: dict[str, dict] | None = None,
) -> LSPServerDef | None:
    """Return the server definition for a file, or ``None`` if unsupported.

    Custom server definitions from config override builtins with the same key
    and can add new languages.
    """
    ext = Path(filepath).suffix.lower()
    if not ext:
        return None

    # Merge custom servers over builtins
    servers = dict(BUILTIN_SERVERS)
    for name, cfg in (custom_servers or {}).items():
        if name in servers:
            base = servers[name]
            servers[name] = LSPServerDef(
                language_id=cfg.get("language_id", base.language_id),
                extensions=cfg.get("extensions", base.extensions),
                command=cfg.get("command", base.command),
                root_markers=cfg.get("root_markers", base.root_markers),
                install_command=cfg.get("install_command", base.install_command),
                initialization_options=cfg.get("initialization_options", base.initialization_options),
            )
        else:
            servers[name] = LSPServerDef(
                language_id=cfg.get("language_id", name),
                extensions=cfg.get("extensions", []),
                command=cfg.get("command", []),
                root_markers=cfg.get("root_markers", [".git"]),
                install_command=cfg.get("install_command"),
                initialization_options=cfg.get("initialization_options", {}),
            )

    for server in servers.values():
        if ext in server.extensions:
            return server
    return None
