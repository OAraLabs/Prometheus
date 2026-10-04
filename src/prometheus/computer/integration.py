"""The Cua desktop driver as an Integration — never a builtin (decision 4).

WHAT DECISION 4 ASKS FOR, AND WHERE EACH PART LIVES
----------------------------------------------------
* **Health known before dispatch.** :meth:`ComputerIntegration.probe` runs
  ordered checks and RECORDS each answer; it never raises. A task (the door,
  PR 5) forces a probe before it starts; ``/api/status`` reads the cache and
  never probes. Same shape as ``providers/backends.BackendRegistry``.
* **Off means nothing.** ``computer_use.enabled`` defaults to false and only a
  literal ``true`` turns it on. While off, no driver is constructed, no probe
  touches the box, and no target is declared.
* **Lifecycle.** The runtime starts on the first probe that gets that far and
  is shut down by :meth:`close` (daemon shutdown) or by a probe that finds it
  unusable. There is NO timer-driven restart: a failed start is retried only
  by the next probe somebody asks for — the LSP precedent of never quietly
  retrying a broken server.

THE TELEMETRY FLOOR (design Q1)
--------------------------------
cua-driver reports usage to PostHog by default, and the pinned 0.28.2 binary
ignores ``DO_NOT_TRACK``. Its own two opt-outs are forced to ``"0"`` BEFORE
``import cua_driver`` (``cua._require_sdk`` calls :func:`apply_telemetry_floor`
first) and re-asserted by every probe. A floor, not a config key: decision 3
is "nothing leaves the machine".

SUPPORTED VERSIONS ARE A CODE CONSTANT
---------------------------------------
The adapter's input builders are written against one SDK's API; config cannot
widen that. A driver outside :data:`SUPPORTED_DRIVER_VERSIONS` is
``version-mismatch`` and the integration is down. It moves with the exact pin
in ``pyproject.toml`` and the on-box check, in one change.

HOSTING
-------
In-process (``embedded``) — the only mode the adapter implements. Cua's
supervised private worker gives native crash containment and is the design's
recommendation (Appendix A), but it is chosen only after an on-box check
shows it works on X11 and that the telemetry opt-out reaches the worker. That
check needs a display, so it is not made here.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping

from prometheus.computer.driver import (
    STATE_UNKNOWN,
    Driver,
    DriverUnavailable,
    PreconditionResult,
    check_preconditions,
)
from prometheus.computer.targets import KIND_LOCAL, Target, TargetRegistry

logger = logging.getLogger(__name__)

#: The cua-driver versions this adapter was validated against. A FLOOR: config
#: cannot widen it (see the module docstring).
SUPPORTED_DRIVER_VERSIONS: frozenset[str] = frozenset({"0.28.2"})

#: cua-driver's own telemetry opt-outs, forced off. ``DO_NOT_TRACK`` is not
#: here because 0.28.2 does not read it.
TELEMETRY_FLOOR: dict[str, str] = {
    "CUA_DRIVER_RS_TELEMETRY_ENABLED": "0",
    "CUA_TELEMETRY_ENABLED": "0",
}

#: How the runtime is hosted. See "HOSTING" above.
EXECUTION_MODE = "embedded"

DEFAULT_TTL_S = 60
DEFAULT_TIMEOUT_S = 5.0

# Check states. ``unknown`` is a third answer and never reads as ``ok``.
OK = "ok"
DEGRADED = "degraded"
DOWN = "down"
UNKNOWN = "unknown"

# Rollup states.
STATE_DISABLED = "disabled"
STATE_READY = "ready"
STATE_DEGRADED = "degraded"
STATE_DOWN = "down"
STATE_NOT_PROBED = "unknown"

#: Words that name the driver in an MCP server's command line.
_CUA_DRIVER_NAMES = ("cua-driver", "cua_driver")


def apply_telemetry_floor(env: MutableMapping[str, str] | None = None) -> None:
    """Force cua-driver's telemetry opt-outs off in *env* (default: os.environ)."""
    target = os.environ if env is None else env
    for key, value in TELEMETRY_FLOOR.items():
        target[key] = value


def telemetry_floor_holds(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return all(source.get(k) == v for k, v in TELEMETRY_FLOOR.items())


def installed_driver_version() -> str | None:
    """The installed cua-driver's version, or None when it is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("cua-driver")
    except PackageNotFoundError:
        return None


def mcp_servers_running_cua_driver(servers: Mapping[str, Any]) -> list[str]:
    """Names of MCP servers whose command line runs cua-driver.

    Such a server exposes the driver's RAW tools as ``mcp__*``: the gate still
    prompts for every call, but none of them goes through the candidate table
    or the extent, so the operator is told.
    """
    hits = []
    for name, definition in (servers or {}).items():
        if not isinstance(definition, Mapping):
            continue
        words = [str(definition.get("command") or "")]
        words += [str(a) for a in (definition.get("args") or [])]
        if any(os.path.basename(w).lower().startswith(_CUA_DRIVER_NAMES)
               for w in words):
            hits.append(str(name))
    return sorted(hits)


@dataclass(frozen=True)
class Check:
    name: str
    state: str
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "state": self.state, "detail": self.detail}


def _iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _number(raw: Any, default: float, key: str, errors: list[str]) -> float:
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = -1.0
    if value <= 0:
        errors.append(f"computer_use.probe.{key} must be a positive number; "
                      f"using {default:g}")
        return default
    return value


class ComputerIntegration:
    """The driver's health, lifecycle and target binding. See the module doc."""

    def __init__(
        self,
        *,
        enabled: bool,
        targets: TargetRegistry | None = None,
        config_errors: list[str] | None = None,
        ttl_s: float = DEFAULT_TTL_S,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        mcp_servers: Mapping[str, Any] | None = None,
        adapter_factory: Callable[[str], Any] | None = None,
        preconditions: Callable[[], PreconditionResult] = check_preconditions,
        version_reader: Callable[[], str | None] = installed_driver_version,
        clock: Callable[[], float] = time.monotonic,
        env: MutableMapping[str, str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.targets = targets if enabled else None
        self.config_errors = list(config_errors or [])
        self.ttl_s = ttl_s
        self.timeout_s = timeout_s
        self._mcp_servers = dict(mcp_servers or {})
        self._adapter_factory = adapter_factory or _default_adapter
        self._preconditions = preconditions
        self._version_reader = version_reader
        self._clock = clock
        self._env = env
        self._lock = asyncio.Lock()
        #: Serialises the check BODY. A probe that times out stops WAITING,
        #: but its thread runs on; the next run must not overlap it.
        self._run_lock = threading.Lock()
        self._adapter: Any = None
        self._bound: str | None = None
        self._checks: list[Check] = []
        self._version: str | None = None
        self._probed_at: float | None = None
        self._probed_at_iso: str | None = None
        if enabled:
            apply_telemetry_floor(env)

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None, **kwargs: Any
    ) -> ComputerIntegration:
        """Build from the ``computer_use`` block. Never raises on bad config:
        an error is recorded and reported by the probe as ``down``."""
        block = (config or {}).get("computer_use") or {}
        if not isinstance(block, Mapping):
            block = {}
        # ONLY A LITERAL TRUE. `bool(value)` would let a quoted "false" — or
        # any non-empty string — switch on desktop control.
        enabled = block.get("enabled", False) is True
        errors: list[str] = []
        probe = block.get("probe") or {}
        if not isinstance(probe, Mapping):
            errors.append("computer_use.probe must be a mapping")
            probe = {}
        ttl_s = _number(probe.get("ttl_s", 60), DEFAULT_TTL_S, "ttl_s", errors)
        timeout_s = _number(probe.get("timeout_s", 5.0), DEFAULT_TIMEOUT_S,
                            "timeout_s", errors)
        targets = _local_targets(block.get("targets"), errors) if enabled else None
        return cls(enabled=enabled, targets=targets, config_errors=errors,
                   ttl_s=ttl_s, timeout_s=timeout_s, **kwargs)

    # ── reads (no I/O) ──────────────────────────────────────────────────

    @property
    def state(self) -> str:
        if not self.enabled:
            return STATE_DISABLED
        if self._probed_at is None:
            return STATE_NOT_PROBED
        states = {c.state for c in self._checks}
        if DOWN in states:
            return STATE_DOWN
        if states & {DEGRADED, UNKNOWN}:
            return STATE_DEGRADED
        return STATE_READY

    def is_stale(self) -> bool:
        return (self._probed_at is None
                or (self._clock() - self._probed_at) > self.ttl_s)

    def driver(self) -> Driver | None:
        """The bound driver when nothing is DOWN; otherwise None. A task must
        not dispatch through an integration whose health is not known."""
        if self.state in (STATE_READY, STATE_DEGRADED) and self._bound:
            return self._adapter
        return None

    def snapshot(self) -> dict[str, Any]:
        """JSON-ready view from the CACHE — no I/O, safe for /api/status."""
        return {
            "enabled": self.enabled,
            "state": self.state,
            "version": self._version,
            "supported_versions": sorted(SUPPORTED_DRIVER_VERSIONS),
            "execution_mode": EXECUTION_MODE,
            "telemetry": ("forced_off" if telemetry_floor_holds(self._env)
                          else "not_forced"),
            "checks": [c.as_dict() for c in self._checks],
            "config_errors": list(self.config_errors),
            "probed": self._probed_at is not None,
            "checked_at": self._probed_at_iso,
            "stale": self.is_stale() if self.enabled else False,
            "ttl_s": self.ttl_s,
        }

    def status_view(self) -> dict[str, Any]:
        """The subset ``/api/status``'s computer block carries."""
        snap = self.snapshot()
        return {k: snap[k] for k in ("state", "version", "execution_mode",
                                     "telemetry", "checked_at")}

    # ── probe ───────────────────────────────────────────────────────────

    async def probe(self, *, force: bool = False) -> dict[str, Any]:
        """Run the checks (TTL-cached unless *force*). Never raises."""
        if not self.enabled:
            return self.snapshot()
        async with self._lock:
            if force or self.is_stale():
                try:
                    checks = await asyncio.wait_for(
                        asyncio.to_thread(self._run_checks),
                        timeout=self.timeout_s)
                except asyncio.TimeoutError:
                    checks = [Check("probe", DOWN,
                                    f"probe timed out after {self.timeout_s:g}s")]
                except Exception as exc:  # noqa: BLE001 — a recorded state
                    checks = [Check("probe", DOWN,
                                    f"{exc.__class__.__name__}: {exc}")]
                self._checks = checks
                self._probed_at = self._clock()
                self._probed_at_iso = _iso_now()
                logger.info("computer-use integration: %s", self.state)
        return self.snapshot()

    def _run_checks(self) -> list[Check]:
        """The ordered checks. Stops at the first DOWN in the chain — there is
        no point starting a runtime on a dead display — but the MCP check is
        independent and always runs."""
        with self._run_lock:
            checks = self._chain()
            checks.append(self._mcp_check())
            if any(c.state == DOWN for c in checks):
                # A runtime that failed its probe is not kept, and nothing is
                # left bound to the target. Released HERE, in the run that
                # found it — never by a caller that merely stopped waiting.
                self._release()
            return checks

    def _chain(self) -> list[Check]:
        out: list[Check] = []

        apply_telemetry_floor(self._env)
        if not telemetry_floor_holds(self._env):  # pragma: no cover - defensive
            return [Check("telemetry", DOWN, "the telemetry opt-outs could "
                          "not be forced off")]
        out.append(Check("telemetry", OK, "cua-driver telemetry forced off"))

        if self.config_errors:
            out.append(Check("config", DOWN, "; ".join(self.config_errors)))
            return out
        out.append(Check("config", OK))

        self._version = self._version_reader()
        if self._version is None:
            out.append(Check("driver", DOWN, (
                "cua-driver is not installed — `uv sync --extra computer`")))
            return out
        if self._version not in SUPPORTED_DRIVER_VERSIONS:
            out.append(Check("driver", DOWN, (
                f"version-mismatch: cua-driver {self._version} is installed; "
                f"this adapter was validated against "
                f"{', '.join(sorted(SUPPORTED_DRIVER_VERSIONS))}")))
            return out
        out.append(Check("driver", OK, f"cua-driver {self._version}"))

        local = self._local_target_name()
        if local is None:
            out.append(Check("target", DOWN, (
                "no local target is declared in computer_use.targets — "
                "declare one (e.g. `local: {kind: local}`); none is implied")))
            return out
        out.append(Check("target", OK, local))

        pre = self._preconditions()
        if not pre:
            state = UNKNOWN if pre.state == STATE_UNKNOWN else DOWN
            out.append(Check("substrate", state, pre.reason or pre.state))
            return out
        out.append(Check("substrate", OK, pre.state))

        adapter = self._adapter
        if adapter is None or self._bound != local:
            self._release()
            adapter = self._adapter_factory(local)
            self._adapter = adapter
        try:
            adapter.start()
        except DriverUnavailable as exc:
            out.append(Check("runtime", DOWN, str(exc)))
            return out
        out.append(Check("runtime", OK, "the runtime started and reports "
                         "itself available"))

        list_apps = getattr(adapter, "list_apps", None)
        if list_apps is None:
            out.append(Check("observe", UNKNOWN, (
                "this build has no discovery (list_apps), so the observe half "
                "is not established")))
        else:
            try:
                apps = list_apps()
            except DriverUnavailable as exc:
                out.append(Check("observe", DOWN, str(exc)))
                return out
            if not apps:
                out.append(Check("observe", DOWN, (
                    "list_apps returned no applications — an empty desktop is "
                    "shaped exactly like a working one, so it is refused")))
                return out
            out.append(Check("observe", OK,
                             f"{len(apps)} application(s) visible"))

        if self.targets is not None:
            self.targets.bind(local, adapter)
            self._bound = local
        return out

    def _mcp_check(self) -> Check:
        hits = mcp_servers_running_cua_driver(self._mcp_servers)
        if hits:
            return Check("mcp", DEGRADED, (
                f"cua-driver-also-configured-as-mcp: {', '.join(hits)} — its "
                f"raw tools bypass the candidate table (they still prompt)"))
        return Check("mcp", OK)

    def _local_target_name(self) -> str | None:
        if self.targets is None:
            return None
        for name in self.targets.names():
            if self.targets.get(name).kind == KIND_LOCAL:
                return name
        return None

    # ── lifecycle ───────────────────────────────────────────────────────

    def _release(self) -> None:
        adapter, self._adapter = self._adapter, None
        if self._bound and self.targets is not None:
            self.targets.unbind(self._bound)
        self._bound = None
        if adapter is not None:
            try:
                adapter.shutdown()
            except Exception:  # noqa: BLE001
                logger.warning("computer-use: driver shutdown failed",
                               exc_info=True)

    def close(self) -> None:
        """Daemon shutdown: stop the runtime and unbind the target."""
        self._release()


def _default_adapter(target: str) -> Any:
    from prometheus.computer.cua import CuaDriverAdapter

    return CuaDriverAdapter(target=target)


def _local_targets(raw: Any, errors: list[str]) -> TargetRegistry:
    """Declared targets, LOCAL ONLY (decision 3). Errors are recorded, never
    raised; a refused entry is simply not declared."""
    registry = TargetRegistry()
    if raw is None:
        return registry
    if not isinstance(raw, Mapping):
        errors.append("computer_use.targets must be a mapping of name -> target")
        return registry
    for name, spec in raw.items():
        spec = spec or {}
        if not isinstance(spec, Mapping):
            errors.append(f"computer_use.targets.{name} must be a mapping")
            continue
        kind = str(spec.get("kind", KIND_LOCAL))
        if kind != KIND_LOCAL:
            errors.append(
                f"computer_use.targets.{name}: kind {kind!r} is refused — "
                f"local only (nothing leaves the machine, decision 3)")
            continue
        registry.declare(Target(
            name=str(name), kind=kind,
            description=str(spec.get("description", "")),
        ))
    return registry
