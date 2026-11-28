"""Minimal out-of-process Extension hook registry for Prometheus.

Extensions (such as OAra Gym or OAra Voice) run as separately installed,
separately permissioned companion services — never in-process inside the
Prometheus daemon. This package provides config parsing, loopback-by-default
validation, manifest discovery, and strictly separated read vs. control
proxying so Beacon and operator clients can reach configured Extensions
through Prometheus's existing paired-device auth without exposing backend
addresses, bearer tokens, or env var names.
"""

from prometheus.extensions.registry import (
    ExtensionConfigError,
    ExtensionEntry,
    ExtensionRegistry,
    ExtensionUnavailable,
    load_extension_registry,
)

__all__ = [
    "ExtensionConfigError",
    "ExtensionEntry",
    "ExtensionRegistry",
    "ExtensionUnavailable",
    "load_extension_registry",
]
