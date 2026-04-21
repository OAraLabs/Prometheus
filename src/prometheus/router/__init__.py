"""Model routing — provider + adapter selection per request.

Re-exports the public router API so callers can `from prometheus.router import ...`
without reaching into module internals.
"""

from prometheus.router.model_router import (
    ModelRouter,
    RouteDecision,
    RouteReason,
    RouterConfig,
    TaskClassification,
    TaskClassifier,
    TaskType,
    load_router_config,
)

__all__ = [
    "ModelRouter",
    "RouteDecision",
    "RouteReason",
    "RouterConfig",
    "TaskClassification",
    "TaskClassifier",
    "TaskType",
    "load_router_config",
]
