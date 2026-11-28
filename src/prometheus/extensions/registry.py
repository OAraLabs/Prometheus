"""Extension registry, manifest discovery, and separated read/control proxy.

Security invariants:
- Extensions run out-of-process; no third-party or extension code is ever
  imported into Prometheus.
- Loopback URLs (``127.0.0.1``, ``localhost``, ``::1``) are allowed by default;
  non-loopback URLs require ``allow_remote: true`` explicitly in config.
- Read calls (``manifest``, ``health``, ``status``) use ``url`` + ``token_env``
  ONLY. Write calls (``control/intent``, ``control/confirm``) use
  ``control_url`` + ``control_token_env`` ONLY — a read misconfiguration can
  never reach a write route.
- Public responses NEVER include backend URLs, tokens, or env var names.
"""

from __future__ import annotations

import ipaddress
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
MANIFEST_SAFE_KEYS = frozenset({"schema", "name", "version", "kind", "capabilities", "scopes", "api_version"})


class ExtensionConfigError(ValueError):
    """Invalid or unsafe extension configuration."""


class ExtensionUnavailable(RuntimeError):
    """Extension service could not be reached or answered an error."""

    def __init__(self, message: str, *, status_code: int = 503, code: str = "extension_unavailable") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    h = host.strip().lower().strip("[]")
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _validate_url(url: str, *, allow_remote: bool, label: str) -> str:
    s = (url or "").strip().rstrip("/")
    parsed = urlparse(s)
    if parsed.scheme not in ("http", "https"):
        raise ExtensionConfigError(f"{label}: scheme must be http or https")
    if not parsed.hostname:
        raise ExtensionConfigError(f"{label}: missing host")
    if parsed.username or parsed.password:
        raise ExtensionConfigError(f"{label}: embedded URL credentials are forbidden")
    if not allow_remote and not _is_loopback_host(parsed.hostname):
        raise ExtensionConfigError(
            f"{label}: non-loopback host requires allow_remote: true in config"
        )
    return s


@dataclass(frozen=True)
class ExtensionEntry:
    name: str
    url: str
    enabled: bool = True
    kind: str = "extension"
    token_env: str | None = None
    control_url: str | None = None
    control_token_env: str | None = None
    status_path: str = "/v1/status"
    manifest_path: str = "/integration/manifest"
    health_path: str = "/health"
    timeout_s: float = 5.0
    allow_remote: bool = False

    def read_headers(self) -> dict[str, str]:
        if not self.token_env:
            return {}
        tok = os.environ.get(self.token_env, "").strip()
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def control_headers(self) -> dict[str, str]:
        if not self.control_token_env:
            return {}
        tok = os.environ.get(self.control_token_env, "").strip()
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def public_summary(self, *, healthy: bool | None = None, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        """Sanitized view safe to return over REST — never leaks URLs, keys, or env names."""
        out: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "enabled": self.enabled,
            "has_control": bool(self.control_url),
        }
        if healthy is not None:
            out["healthy"] = healthy
        if manifest is not None:
            out["manifest"] = sanitize_manifest(manifest)
        return out


def sanitize_manifest(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {k: raw[k] for k in sorted(raw) if k in MANIFEST_SAFE_KEYS}


class ExtensionRegistry:
    """Read-only registry of configured out-of-process Extensions."""

    def __init__(self, entries: dict[str, ExtensionEntry] | None = None) -> None:
        self._entries: dict[str, ExtensionEntry] = dict(entries or {})

    def names(self) -> list[str]:
        return sorted(self._entries)

    def get(self, name: str) -> ExtensionEntry | None:
        if not NAME_RE.match(name or ""):
            return None
        return self._entries.get(name)

    def enabled_entries(self) -> list[ExtensionEntry]:
        return [self._entries[k] for k in sorted(self._entries) if self._entries[k].enabled]

    async def probe(
        self,
        entry: ExtensionEntry,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        """Probe ``health`` and ``/integration/manifest`` on *entry* without leaking addresses."""
        owns = client is None
        cli = client or httpx.AsyncClient(timeout=entry.timeout_s)
        healthy = False
        manifest: dict[str, Any] | None = None
        try:
            headers = entry.read_headers()
            try:
                hr = await cli.get(f"{entry.url}{entry.health_path}", headers=headers)
                healthy = hr.status_code == 200
            except Exception:
                healthy = False

            if healthy:
                try:
                    mr = await cli.get(f"{entry.url}{entry.manifest_path}", headers=headers)
                    if mr.status_code == 200:
                        manifest = mr.json()
                except Exception:
                    manifest = None
        finally:
            if owns:
                await cli.aclose()
        return entry.public_summary(healthy=healthy, manifest=manifest)

    async def fetch_status(
        self,
        entry: ExtensionEntry,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        """Fetch read-only status from *entry* using its read URL + read token only."""
        owns = client is None
        cli = client or httpx.AsyncClient(timeout=entry.timeout_s)
        try:
            try:
                r = await cli.get(f"{entry.url}{entry.status_path}", headers=entry.read_headers())
            except Exception as exc:
                raise ExtensionUnavailable("extension status unreachable", status_code=503) from exc
            if r.status_code >= 400:
                raise ExtensionUnavailable(
                    f"extension answered {r.status_code}",
                    status_code=502 if r.status_code >= 500 else r.status_code,
                    code="extension_error",
                )
            data = r.json()
            if not isinstance(data, dict):
                raise ExtensionUnavailable("extension status must be a JSON object", status_code=502)
            return data
        finally:
            if owns:
                await cli.aclose()

    async def post_control(
        self,
        entry: ExtensionEntry,
        subpath: str,
        payload: dict[str, Any],
        *,
        client: httpx.AsyncClient | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Proxy a two-step control call (``intent`` or ``confirm``) to ``control_url`` only."""
        if subpath not in ("intent", "confirm"):
            raise ExtensionUnavailable("unsupported control verb", status_code=400, code="bad_control_verb")
        if not entry.control_url:
            raise ExtensionUnavailable("control service not configured", status_code=403, code="control_not_configured")

        owns = client is None
        cli = client or httpx.AsyncClient(timeout=entry.timeout_s)
        try:
            try:
                r = await cli.post(
                    f"{entry.control_url}/v1/control/{subpath}",
                    json=payload,
                    headers=entry.control_headers(),
                )
            except Exception as exc:
                raise ExtensionUnavailable("extension control unreachable", status_code=503) from exc
            try:
                body = r.json()
                if not isinstance(body, dict):
                    body = {"detail": "non-object response"}
            except Exception:
                body = {"detail": "invalid json response"}
            return r.status_code, body
        finally:
            if owns:
                await cli.aclose()


def load_extension_registry(config: dict[str, Any] | None) -> ExtensionRegistry:
    """Build an :class:`ExtensionRegistry` from ``config.get('extensions', {})``."""
    raw = (config or {}).get("extensions")
    if not isinstance(raw, dict):
        return ExtensionRegistry()

    entries: dict[str, ExtensionEntry] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise ExtensionConfigError(f"invalid extension name {name!r}")
        if not isinstance(spec, dict):
            raise ExtensionConfigError(f"extensions.{name} must be a mapping")
        enabled = bool(spec.get("enabled", True))
        if not enabled:
            entries[name] = ExtensionEntry(name=name, url="", enabled=False, kind=str(spec.get("kind") or "extension"))
            continue
        allow_remote = bool(spec.get("allow_remote", False))
        url = _validate_url(str(spec.get("url") or ""), allow_remote=allow_remote, label=f"extensions.{name}.url")
        control_url_raw = spec.get("control_url")
        control_url = (
            _validate_url(str(control_url_raw), allow_remote=allow_remote, label=f"extensions.{name}.control_url")
            if control_url_raw else None
        )
        timeout_s = max(0.5, min(30.0, float(spec.get("timeout_s", 5.0))))
        entries[name] = ExtensionEntry(
            name=name,
            url=url,
            enabled=True,
            kind=str(spec.get("kind") or "extension"),
            token_env=str(spec["token_env"]) if spec.get("token_env") else None,
            control_url=control_url,
            control_token_env=str(spec["control_token_env"]) if spec.get("control_token_env") else None,
            status_path=str(spec.get("status_path") or "/v1/status"),
            manifest_path=str(spec.get("manifest_path") or "/integration/manifest"),
            health_path=str(spec.get("health_path") or "/health"),
            timeout_s=timeout_s,
            allow_remote=allow_remote,
        )
    return ExtensionRegistry(entries)
