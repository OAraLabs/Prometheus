"""Gateway package — Telegram, Slack adapters, cron, heartbeat, archive."""

from prometheus.gateway.archive_writer import ArchiveWriter
from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.cron_scheduler import run_scheduler_loop, scheduler_status
from prometheus.gateway.cron_service import (
    delete_cron_job,
    load_cron_jobs,
    upsert_cron_job,
    validate_cron_expression,
)
from prometheus.gateway.heartbeat import Heartbeat
from prometheus.gateway.platform_base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from prometheus.gateway.telegram import TelegramAdapter

# Slack adapter is optional — requires slack-bolt
try:
    from prometheus.gateway.slack import SlackAdapter
except ImportError:
    SlackAdapter = None  # type: ignore[assignment,misc]

__all__ = [
    "ArchiveWriter",
    "BasePlatformAdapter",
    "Heartbeat",
    "MessageEvent",
    "MessageType",
    "Platform",
    "PlatformConfig",
    "SendResult",
    "SlackAdapter",
    "TelegramAdapter",
    "delete_cron_job",
    "load_cron_jobs",
    "run_scheduler_loop",
    "scheduler_status",
    "upsert_cron_job",
    "validate_cron_expression",
]
