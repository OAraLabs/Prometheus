"""SENTINEL — proactive daemon layer for Prometheus.

Source: Novel code for Prometheus Sprint 9.
Transforms Prometheus from reactive to proactive via signal-driven
Activity Observer and idle-time AutoDream Engine.
"""

from prometheus.sentinel.signals import ActivitySignal, SignalBus

__all__ = [
    "ActivitySignal",
    "SignalBus",
]
