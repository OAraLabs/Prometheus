"""Span decorators and context managers for Prometheus tracing."""

from __future__ import annotations

import asyncio
import functools
from contextlib import contextmanager
from typing import Any, Callable

from prometheus.tracing.phoenix import get_tracer


@contextmanager
def span_context(name: str, attributes: dict[str, Any] | None = None):
    """Context manager for creating ad-hoc trace spans.

    No-op when tracing is disabled.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes and hasattr(span, "set_attribute"):
            for k, v in attributes.items():
                span.set_attribute(
                    k, str(v) if not isinstance(v, (int, float, bool)) else v
                )
        yield span


def traced(name: str | None = None, attributes: dict[str, str] | None = None):
    """Decorator that wraps a function in an OTel span.

    Handles both sync and async functions. When tracing is disabled,
    the function executes with minimal overhead.
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with span_context(span_name, attributes):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with span_context(span_name, attributes):
                    return func(*args, **kwargs)

            return sync_wrapper

    return decorator
