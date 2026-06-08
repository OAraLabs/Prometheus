"""Tasks package — background task lifecycle management."""

from prometheus.tasks.completion_handler import TaskCompletionHandler
from prometheus.tasks.manager import BackgroundTaskManager, get_task_manager
from prometheus.tasks.store import TaskStore
from prometheus.tasks.types import (
    OnComplete,
    TaskRecord,
    TaskStatus,
    TaskType,
    TERMINAL_STATUSES,
)

__all__ = [
    "BackgroundTaskManager",
    "OnComplete",
    "TaskCompletionHandler",
    "TaskRecord",
    "TaskStatus",
    "TaskStore",
    "TaskType",
    "TERMINAL_STATUSES",
    "get_task_manager",
]
