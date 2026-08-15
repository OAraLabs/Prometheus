"""Coding mode (SPRINT-coding-mode v2) — sandboxed iterate-to-green coding runs.

A coding run drives the existing agent loop with coding-specific tools
(`prometheus.coding.tools`) inside an execution sandbox
(`prometheus.coding.sandbox`), under a policy layer that refuses
"done" without test evidence (`prometheus.coding.policy`). The terminal
artifact is a feature branch in a dedicated clone of the target repo —
never a merge, never a push.
"""

from prometheus.coding.sandbox import (
    SANDBOX_BACKENDS,
    BwrapSandbox,
    DockerSandbox,
    ProcessSandbox,
    Sandbox,
    SandboxBackendUnavailable,
    SandboxConstructionError,
    SandboxResult,
    SandboxViolation,
    cleanup_stale_docker_containers,
    create_sandbox,
    docker_available,
    list_stale_docker_containers,
)
from prometheus.coding.tools import build_coding_registry

__all__ = [
    "SANDBOX_BACKENDS",
    "BwrapSandbox",
    "DockerSandbox",
    "ProcessSandbox",
    "Sandbox",
    "SandboxBackendUnavailable",
    "SandboxConstructionError",
    "SandboxResult",
    "SandboxViolation",
    "build_coding_registry",
    "cleanup_stale_docker_containers",
    "create_sandbox",
    "docker_available",
    "list_stale_docker_containers",
]
