"""Advertising ``_prometheus._tcp`` on the home network, and ONLY there.

A phone on the same Wi-Fi has no address to type. This module lets the daemon announce itself over mDNS /
DNS-SD so the phone can find it, say what it is called (the six ``GET /api/hello`` fields, as the TXT
record), and ask to pair. It is deliberately timid:

* **It advertises only when it should.** Home-network mode (``web/network.py``), ``discovery.mdns`` on, a
  usable library and at least one usable address. Never on a loopback bind, and (a deliberate narrowing of the
  contract, documented there) not in ``open`` mode either: an existing install that listens on every
  interface would otherwise start announcing its name on the LAN the day it upgraded.
* **The address set is conservative.** A private IPv4 (RFC 1918) or link-local address on an interface the
  daemon actually listens on. Never a public address, never Tailscale's CGNAT range, never a tunnel, bridge or
  container interface, however private its address looks. IPv6 is not advertised.
* **Nothing it announces is a secret.** The TXT record is ``hello_txt`` (``web/hello.py``) of the very
  dictionary ``GET /api/hello`` answers with, so the two cannot drift; a field hello does not have is a
  ``ValueError`` here, not a broadcast.
* **It cannot hurt the daemon.** ``zeroconf`` is the optional ``discovery`` extra (LGPL-2.1-or-later). Absent,
  the daemon starts, serves and pairs exactly as before, the advertiser says so once, loudly, and reports the
  reason. A registration that fails is a status, not an exception, and is tried again. ``start()`` waits a
  couple of seconds for the first registration and then lets it finish in the background, so a stuck
  multicast socket cannot hold up the web server.
* **It follows the network.** Wi-Fi roams. The address set is re-evaluated on a timer; a new set is a new
  backend (zeroconf fixes its interfaces when it is built), while a change in what hello says is an update,
  not a re-registration.

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from prometheus.web.bind import is_all_interfaces
from prometheus.web.hello import hello_txt
from prometheus.web.loopback import is_loopback_address
from prometheus.web.network import HOME, THIS_MAC, NetworkSettings, NetworkState

logger = logging.getLogger("prometheus.discovery")

SERVICE_TYPE = "_prometheus._tcp.local."

#: A DNS label is at most 63 bytes; a TXT string (``key=value``) at most 255.
_LABEL_BYTES = 63
_TXT_STRING_BYTES = 255

DEFAULT_INSTANCE_NAME = "Prometheus"

#: How long ``start()`` waits for the first registration before letting it finish in the background.
START_WAIT_SECONDS = 2.0
#: How often the address set is looked at again.
REFRESH_SECONDS = 30.0

_INSTALL_HINT = "pip install 'oara-prometheus[discovery]'"

# Interfaces that are tunnels, bridges, containers, VMs or Apple's private links. Their addresses are private
# by number and wrong by nature: a phone on the Wi-Fi cannot reach them, and an address that reaches somewhere
# else is not one to announce. Matched case-insensitively on the name AND on Windows' friendly name
# (``vEthernet (WSL)`` starts with ``veth``).
_SKIPPED_INTERFACES = re.compile(
    r"^(utun|tun|tap|wg|tailscale|docker|br-|veth|virbr|awdl|llw|lo\d*$|zt|bridge|vmnet|vboxnet|gif|stf|ppp|"
    r"ipsec|anpi|ap\d|cni|flannel|cali|kube)",
    re.IGNORECASE,
)

# Private (RFC 1918) and link-local IPv4. Deliberately NOT ``ipaddress``'s ``is_private``, which also covers
# loopback, the benchmarking range and more; and not 100.64.0.0/10, which is Tailscale's.
_ADVERTISABLE = tuple(ipaddress.ip_network(net) for net in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16"))


# ── which addresses ──────────────────────────────────────────────────────────

def eligible_addresses(adapters: Iterable[Any], *, bind: str) -> list[str]:
    """The IPv4 addresses worth announcing, given the interfaces (``ifaddr.get_adapters()`` shape) and the bind.

    A loopback bind announces nothing. A wildcard bind (``0.0.0.0``, ``::``) announces every eligible address.
    A specific bind announces only that address, and only if the rules above accept it: a bind they refuse is
    not rescued by being specific.
    """
    if is_loopback_address(bind):
        return []
    wildcard = is_all_interfaces(bind)
    specific: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    if not wildcard:
        try:
            specific = ipaddress.ip_address(bind)
        except ValueError:
            return []
    found: list[str] = []
    for adapter in adapters:
        names = (getattr(adapter, "name", ""), getattr(adapter, "nice_name", ""))
        if any(isinstance(name, str) and _SKIPPED_INTERFACES.match(name) for name in names):
            continue
        for entry in getattr(adapter, "ips", ()):
            if not getattr(entry, "is_IPv4", False) or not isinstance(entry.ip, str):
                continue
            try:
                address = ipaddress.IPv4Address(entry.ip)
            except ValueError:
                continue
            if not any(address in net for net in _ADVERTISABLE):
                continue
            if specific is not None and address != specific:
                continue
            if str(address) not in found:
                found.append(str(address))
    return found


# ── what it is called and what it says ───────────────────────────────────────

def _cut(text: str, limit: int) -> str:
    """*text* cut to at most *limit* UTF-8 bytes, never through a character."""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def instance_name(display_name: str) -> str:
    """The DNS-SD instance name for a display name: one label (at most 63 bytes), never empty."""
    name = _cut((display_name or "").strip(), _LABEL_BYTES).strip()
    return name or DEFAULT_INSTANCE_NAME


def txt_properties(hello: Mapping[str, Any]) -> dict[str, bytes]:
    """The TXT record for a hello answer: its six fields as bytes, each ``key=value`` within 255 bytes.

    Raises :class:`ValueError` for a field hello does not have, or a missing one (``hello_txt``).
    """
    return {
        key: _cut(value, _TXT_STRING_BYTES - len(key.encode("utf-8")) - 1).encode("utf-8")
        for key, value in hello_txt(hello).items()
    }


def _server_name(properties: Mapping[str, bytes]) -> str:
    """The host name the service points at, kept apart from the machine's own ``<name>.local``.

    A second responder claiming the machine's real mDNS name would fight the operating system's for it. This
    one is derived from the instance fingerprint, which the TXT record already publishes.
    """
    fp = properties.get("fp", b"").decode("ascii", errors="ignore")
    suffix = fp[:8] if re.fullmatch(r"[0-9a-f]{8,}", fp) else ""
    return f"prometheus-{suffix}.local." if suffix else "prometheus.local."


def service_info(*, name: str, port: int, properties: Mapping[str, bytes], addresses: Sequence[str],
                 server: str | None = None) -> Any:
    """The ``zeroconf.ServiceInfo`` for this daemon. Needs the library; raises ``ImportError`` without it."""
    from zeroconf import ServiceInfo

    return ServiceInfo(
        type_=SERVICE_TYPE,
        name=f"{instance_name(name)}.{SERVICE_TYPE}",
        port=port,
        properties=dict(properties),
        parsed_addresses=list(addresses),
        server=server or _server_name(properties),
    )


# ── the backend: zeroconf, or a stand-in ─────────────────────────────────────

class Backend(Protocol):
    """The four things the advertiser needs from an mDNS library."""

    async def register(self, *, name: str, port: int, properties: Mapping[str, bytes],
                       addresses: Sequence[str]) -> None: ...

    async def update(self, *, name: str, port: int, properties: Mapping[str, bytes],
                     addresses: Sequence[str]) -> None: ...

    async def unregister(self) -> None: ...

    async def close(self) -> None: ...


class ZeroconfBackend:
    """``zeroconf``'s ``AsyncZeroconf`` bound to exactly the given addresses. Built inside a running loop."""

    def __init__(self, addresses: Sequence[str]) -> None:
        from zeroconf import IPVersion
        from zeroconf.asyncio import AsyncZeroconf

        self._zc = AsyncZeroconf(interfaces=list(addresses), ip_version=IPVersion.V4Only)
        self._info: Any = None

    async def register(self, *, name: str, port: int, properties: Mapping[str, bytes],
                       addresses: Sequence[str]) -> None:
        info = service_info(name=name, port=port, properties=properties, addresses=addresses)
        # A name already on the network is renamed ("Kitchen Mac (2)"), never refused.
        announced = await self._zc.async_register_service(info, allow_name_change=True)
        await announced
        self._info = info

    async def update(self, *, name: str, port: int, properties: Mapping[str, bytes],
                     addresses: Sequence[str]) -> None:
        from zeroconf import ServiceInfo

        current = self._info
        if current is None:
            return await self.register(name=name, port=port, properties=properties, addresses=addresses)
        info = ServiceInfo(type_=SERVICE_TYPE, name=current.name, port=port, properties=dict(properties),
                           parsed_addresses=list(addresses), server=current.server)
        await (await self._zc.async_update_service(info))
        self._info = info

    async def unregister(self) -> None:
        await self._zc.async_unregister_all_services()
        self._info = None

    async def close(self) -> None:
        await self._zc.async_close()


def _default_adapters() -> list[Any]:
    import ifaddr          # zeroconf's own dependency: absent exactly when zeroconf is

    return list(ifaddr.get_adapters())


# ── the advertiser ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AdvertiserStatus:
    advertising: bool
    #: Why not, when not. ``None`` while advertising.
    reason: str | None


class Advertiser:
    """Announces the daemon while the conditions hold, follows the network, and withdraws on stop.

    *hello* and *display_name* are callables so a change in either is picked up on the next evaluation.
    *backend_factory* takes the address list and returns a :class:`Backend` (or raises ``ImportError`` when
    the library is missing); *adapters* returns the interfaces. Both default to the real thing and exist so a
    test never touches the network.
    """

    def __init__(
        self,
        *,
        state: NetworkState,
        settings: NetworkSettings,
        port: int,
        hello: Callable[[], Mapping[str, Any]],
        display_name: Callable[[], str],
        backend_factory: Callable[[Sequence[str]], Backend] | None = None,
        adapters: Callable[[], Iterable[Any]] | None = None,
        refresh_seconds: float = REFRESH_SECONDS,
    ) -> None:
        self._state = state
        self._settings = settings
        self._port = port
        self._hello = hello
        self._display_name = display_name
        self._factory: Callable[[Sequence[str]], Backend] = backend_factory or ZeroconfBackend
        self._adapters = adapters or _default_adapters
        self._refresh_seconds = refresh_seconds
        self._status = AdvertiserStatus(False, "the advertiser has not started")
        self._lock = asyncio.Lock()
        self._backend: Backend | None = None
        self._addresses: tuple[str, ...] = ()
        self._name = ""
        self._properties: dict[str, bytes] = {}
        self._loop_task: asyncio.Task[None] | None = None
        self._first: asyncio.Task[None] | None = None
        self._stopped = False
        self._library_missing = False
        self._logged_reason: str | None = None

    # ── public ───────────────────────────────────────────────────────────

    @property
    def status(self) -> AdvertiserStatus:
        return self._status

    async def start(self) -> None:
        """Evaluate once (waiting at most ``START_WAIT_SECONDS``) and keep following the network."""
        if self._stopped or self._first is not None:
            return
        self._first = asyncio.ensure_future(self.refresh())
        await asyncio.wait({self._first}, timeout=START_WAIT_SECONDS)
        if self._wants_to_advertise() and self._loop_task is None:
            self._loop_task = asyncio.ensure_future(self._follow_the_network())

    async def refresh(self) -> None:
        """Re-evaluate: register, re-register on a new address set, update on a new hello, or withdraw."""
        async with self._lock:
            if self._stopped or self._library_missing:
                return
            try:
                await self._reconcile()
            except asyncio.CancelledError:
                await self._withdraw()
                raise
            except ImportError as exc:
                await self._withdraw()
                self._library_missing = True
                self._set(False, f"zeroconf is not installed ({_INSTALL_HINT})", warn=True, detail=str(exc))
            except Exception as exc:                   # a status, never a crash; tried again next time
                await self._withdraw()
                self._set(False, f"{type(exc).__name__}: {exc}", warn=True)

    async def stop(self) -> None:
        """Withdraw the service, close the library and stop following the network. Idempotent."""
        self._stopped = True
        tasks = [task for task in (self._loop_task, self._first) if task is not None]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        async with self._lock:
            await self._withdraw()
            if self._status.advertising:
                self._status = AdvertiserStatus(False, "the advertiser was stopped")

    # ── internals ────────────────────────────────────────────────────────

    def _wants_to_advertise(self) -> bool:
        return self._settings.mdns and self._state.mode == HOME

    def _why_not(self) -> str | None:
        if not self._settings.mdns:
            return "discovery.mdns is off"
        if self._state.mode == THIS_MAC:
            return "not advertising: the daemon listens on this machine only"
        if self._state.mode != HOME:
            return ("not advertising: the daemon is not in home network mode (it listens beyond this machine "
                    "without having been set to home network)")
        return None

    async def _follow_the_network(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_seconds)
            await self.refresh()

    async def _reconcile(self) -> None:
        reason = self._why_not()
        if reason is not None:
            await self._withdraw()
            self._set(False, reason)
            return
        addresses = tuple(sorted(eligible_addresses(self._adapters(), bind=self._state.bind)))
        if not addresses:
            await self._withdraw()
            self._set(False, "no usable address: no private IPv4 address on an interface this daemon listens on")
            return
        name = instance_name(self._display_name())
        properties = txt_properties(self._hello())
        offer: dict[str, Any] = dict(name=name, port=self._port, properties=properties,
                                     addresses=list(addresses))

        backend = self._backend
        if backend is not None and addresses == self._addresses:
            if name != self._name:                       # the name IS the identity: a new one is a new service
                await backend.unregister()
                await backend.register(**offer)
            elif properties != self._properties:
                await backend.update(**offer)
            self._name, self._properties = name, properties
            self._set(True, None, announce=f"advertising {SERVICE_TYPE} as {name!r} on {', '.join(addresses)}")
            return

        await self._withdraw()                           # a new address set: zeroconf's interfaces are fixed
        backend = self._factory(addresses)               # ImportError here means the library is missing
        self._backend, self._addresses = backend, addresses
        await backend.register(**offer)
        self._name, self._properties = name, properties
        self._set(True, None, announce=f"advertising {SERVICE_TYPE} as {name!r} on {', '.join(addresses)}")

    async def _withdraw(self) -> None:
        backend, self._backend = self._backend, None
        self._addresses, self._name, self._properties = (), "", {}
        if backend is None:
            return
        for step in (backend.unregister, backend.close):
            try:
                await step()
            except Exception:                            # best effort: the service is going away anyway
                logger.debug("mDNS: %s failed while withdrawing", step.__name__, exc_info=True)

    def _set(self, advertising: bool, reason: str | None, *, announce: str | None = None,
             warn: bool = False, detail: str | None = None) -> None:
        """Record the status, and say what changed once rather than on every evaluation."""
        self._status = AdvertiserStatus(advertising, reason)
        key = reason if not advertising else announce
        if key == self._logged_reason:
            return
        self._logged_reason = key
        if advertising:
            logger.info("mDNS: %s", announce)
        elif warn:
            logger.warning("mDNS: not advertising: %s%s", reason, f" ({detail})" if detail else "")
        else:
            logger.info("mDNS: %s", reason)
