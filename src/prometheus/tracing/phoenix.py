"""Phoenix/OpenTelemetry tracing integration — env-gated.

Enable with PROMETHEUS_TRACING=1 environment variable.
When disabled (default), all functions are zero-cost no-ops.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

_tracer_provider: Any = None
_tracer: Any = None
_initialized = False


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled via env var."""
    return os.environ.get("PROMETHEUS_TRACING", "").lower() in ("1", "true", "yes")


def init_tracing(config: dict[str, Any] | None = None) -> Any | None:
    """Initialize OpenTelemetry + Phoenix tracing.

    Returns the TracerProvider if enabled, None otherwise.
    Gated by PROMETHEUS_TRACING=1 env var or config tracing.enabled.
    """
    global _tracer_provider, _tracer, _initialized

    if _initialized:
        return _tracer_provider

    cfg = (config or {}).get("tracing", {})
    env_enabled = is_tracing_enabled()
    cfg_enabled = cfg.get("enabled", False)

    if not env_enabled and not cfg_enabled:
        log.debug("Tracing disabled (set PROMETHEUS_TRACING=1 to enable)")
        _initialized = True
        return None

    try:
        from phoenix.otel import register
        from opentelemetry import trace

        endpoint = cfg.get("phoenix_endpoint", "http://127.0.0.1:6006")
        service_name = cfg.get("service_name", "prometheus")

        tracer_provider = register(
            project_name=service_name,
            endpoint=f"{endpoint.rstrip('/')}/v1/traces",
        )
        _tracer_provider = tracer_provider
        _tracer = trace.get_tracer("prometheus")
        _initialized = True
        log.info("Phoenix tracing enabled -> %s", endpoint)
        return _tracer_provider

    except ImportError:
        log.warning(
            "PROMETHEUS_TRACING=1 but phoenix/opentelemetry not installed. "
            "Install with: pip install 'prometheus[evals]'"
        )
        _initialized = True
        return None
    except Exception as exc:
        log.warning("Failed to initialize tracing: %s", exc)
        _initialized = True
        return None


def get_tracer():
    """Return the OTel tracer, or a _NoOpTracer if tracing is disabled."""
    if _tracer is not None:
        return _tracer
    return _NOOP_TRACER


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider."""
    global _tracer_provider, _tracer, _initialized
    if _tracer_provider is not None:
        try:
            if hasattr(_tracer_provider, "force_flush"):
                _tracer_provider.force_flush()
            if hasattr(_tracer_provider, "shutdown"):
                _tracer_provider.shutdown()
        except Exception as exc:
            log.warning("Error shutting down tracing: %s", exc)
    _tracer_provider = None
    _tracer = None
    _initialized = False


class _NoOpSpan:
    """Stub span that discards all operations."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoOpTracer:
    """Stub tracer that returns no-op spans."""

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs):
        yield _NoOpSpan()


_NOOP_TRACER = _NoOpTracer()
