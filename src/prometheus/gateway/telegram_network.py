"""Telegram network fallback configuration.

Source: Novel code for Prometheus Sprint 6 (inspired by Hermes telegram_network).
Handles optional proxy/network configuration for regions where Telegram
direct connections may be unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TelegramNetworkConfig:
    """Optional network overrides for the Telegram adapter."""

    proxy_url: str | None = None
    base_url: str | None = None  # override api.telegram.org
    base_file_url: str | None = None
    connect_timeout: float = 30.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    pool_timeout: float = 10.0

    def to_bot_kwargs(self) -> dict:
        """Return kwargs suitable for telegram.Bot / ApplicationBuilder."""
        kwargs: dict = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.base_file_url:
            kwargs["base_file_url"] = self.base_file_url
        return kwargs

    def to_request_kwargs(self) -> dict:
        """Return kwargs for httpx-based request configuration."""
        kwargs: dict = {
            "connect_timeout": self.connect_timeout,
            "read_timeout": self.read_timeout,
            "write_timeout": self.write_timeout,
            "pool_timeout": self.pool_timeout,
        }
        if self.proxy_url:
            kwargs["proxy_url"] = self.proxy_url
        return kwargs
