"""LSP integration — compiler-grade code intelligence for Prometheus."""

from prometheus.lsp.languages import LSPServerDef, find_project_root, get_server_for_file
from prometheus.lsp.client import LSPClient, Location, Diagnostic, HoverInfo, DocumentSymbol
from prometheus.lsp.orchestrator import LSPOrchestrator

__all__ = [
    "LSPServerDef",
    "find_project_root",
    "get_server_for_file",
    "LSPClient",
    "LSPOrchestrator",
    "Location",
    "Diagnostic",
    "HoverInfo",
    "DocumentSymbol",
]
