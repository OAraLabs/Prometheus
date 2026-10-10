"""Real-socket helpers for the bind tests: what is ACTUALLY listening, and what
a client with a chosen ``Host`` header actually gets back.

Nothing in here mocks a bind. The point of the web.bind tests is that the
address the operator asked for is the address the kernel was given, and the only
honest evidence of that is the kernel's own socket table (``lsof`` on macOS,
``/proc/net/tcp*`` on Linux) plus ``getsockname()`` on the live servers.

Everything blocking (raw clients) is synchronous on purpose: call it through
``asyncio.to_thread`` when the server under test shares the caller's event loop.
"""

from __future__ import annotations

import gc
import ipaddress
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class ListenersUnavailable(RuntimeError):
    """No way to read this process's listening sockets on this machine."""


@dataclass(frozen=True)
class Listener:
    host: str  # an IP literal, or "*" (lsof's spelling of a wildcard bind)
    port: int

    @property
    def is_wildcard(self) -> bool:
        if self.host == "*":
            return True
        try:
            return ipaddress.ip_address(self.host).is_unspecified
        except ValueError:
            return False

    @property
    def is_loopback(self) -> bool:
        try:
            ip = ipaddress.ip_address(self.host)
        except ValueError:
            return False
        mapped = getattr(ip, "ipv4_mapped", None)
        return ip.is_loopback or bool(mapped is not None and mapped.is_loopback)


# ---------------------------------------------------------------------------
# The kernel's view: which TCP sockets in LISTEN state does a PID own?
# ---------------------------------------------------------------------------


def parse_lsof_listeners(text: str) -> list[Listener]:
    """Parse ``lsof -F n`` output (``n<host>:<port>`` lines)."""
    out: list[Listener] = []
    for line in text.splitlines():
        if not line.startswith("n"):
            continue
        name = line[1:]
        host, _, port = name.rpartition(":")
        if not port.isdigit():
            continue
        out.append(Listener(host.strip("[]"), int(port)))
    return out


def _proc_hex_ip(hex_addr: str) -> str:
    raw = bytes.fromhex(hex_addr)
    if len(raw) == 4:
        return socket.inet_ntoa(raw[::-1])
    words = b"".join(raw[i:i + 4][::-1] for i in range(0, 16, 4))
    return str(ipaddress.IPv6Address(words))


def parse_proc_net_tcp(text: str, inodes: set[str]) -> list[Listener]:
    """Parse ``/proc/net/tcp`` or ``tcp6``: LISTEN rows (state 0A) whose inode
    is one of *inodes* (the socket inodes of the process under test)."""
    out: list[Listener] = []
    for line in text.splitlines()[1:]:
        cols = line.split()
        if len(cols) < 10 or cols[3] != "0A" or cols[9] not in inodes:
            continue
        addr, _, port_hex = cols[1].rpartition(":")
        out.append(Listener(_proc_hex_ip(addr), int(port_hex, 16)))
    return out


def _proc_socket_inodes(pid: int) -> set[str]:
    inodes: set[str] = set()
    fd_dir = Path(f"/proc/{pid}/fd")
    for fd in fd_dir.iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[len("socket:["):-1])
    return inodes


def process_listeners(pid: int | None = None) -> list[Listener]:
    """Every TCP socket in LISTEN state owned by *pid* (default: this process).

    Raises :class:`ListenersUnavailable` when the machine offers no way to ask.
    """
    pid = os.getpid() if pid is None else pid
    proc_problem = ""
    if Path("/proc/net/tcp").exists() and Path(f"/proc/{pid}/fd").exists():
        try:
            inodes = _proc_socket_inodes(pid)
            found: list[Listener] = []
            for table in ("/proc/net/tcp", "/proc/net/tcp6"):
                p = Path(table)
                if p.exists():
                    found += parse_proc_net_tcp(p.read_text(), inodes)
            return found
        except (OSError, ValueError) as exc:
            proc_problem = f"/proc was not usable ({type(exc).__name__}: {exc}); "
    try:
        proc = subprocess.run(
            ["lsof", "-nP", "-a", "-p", str(pid), "-iTCP", "-sTCP:LISTEN", "-Fn"],
            capture_output=True, text=True, timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ListenersUnavailable(f"{proc_problem}lsof is not usable: {exc}") from exc
    if proc.returncode not in (0, 1):  # 1 == "nothing matched"
        raise ListenersUnavailable(
            f"{proc_problem}lsof exited {proc.returncode}: {proc.stderr[:200]}")
    return parse_lsof_listeners(proc.stdout)


def new_listeners(before: list[Listener], pid: int | None = None) -> list[Listener]:
    known = set(before)
    return [lst for lst in process_listeners(pid) if lst not in known]


# ---------------------------------------------------------------------------
# The live servers, found by type (no production seam needed)
# ---------------------------------------------------------------------------


def live_instances(cls: type[T]) -> list[T]:
    return [o for o in gc.get_objects() if isinstance(o, cls)]


def sockname(sock: Any) -> tuple[str, int]:
    name = sock.getsockname()
    return str(name[0]), int(name[1])


# ---------------------------------------------------------------------------
# Raw clients: the Host header is whatever the test says it is
# ---------------------------------------------------------------------------


def _raw_exchange(
    ip: str, port: int, request: bytes, *, timeout: float = 5.0, whole: bool = False,
) -> bytes:
    """Send *request*, read the reply. By default stop at the end of the headers
    (a WebSocket upgrade never closes); with *whole* read until the peer closes
    (the request carries ``Connection: close``)."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect((ip, port))
        s.sendall(request)
        chunks: list[bytes] = []
        try:
            while True:
                data = s.recv(65536)
                if not data:
                    break
                chunks.append(data)
                if not whole and b"\r\n\r\n" in b"".join(chunks):
                    break
        except socket.timeout:
            pass
        return b"".join(chunks)


def _status_of(response: bytes) -> int:
    first = response.split(b"\r\n", 1)[0].split()
    return int(first[1]) if len(first) >= 2 and first[1].isdigit() else 0


def http_status(
    port: int,
    host_headers: str | list[str] | None,
    *,
    path: str = "/api/status",
    ip: str = "127.0.0.1",
) -> int:
    """GET *path* with exactly the Host header(s) given; return the status code."""
    if host_headers is None:
        hosts: list[str] = []
    elif isinstance(host_headers, str):
        hosts = [host_headers]
    else:
        hosts = host_headers
    lines = [f"GET {path} HTTP/1.1"] + [f"Host: {h}" for h in hosts]
    lines += ["Connection: close", "", ""]
    return _status_of(_raw_exchange(ip, port, "\r\n".join(lines).encode()))


def http_response(
    port: int, host_header: str, *, path: str = "/api/status", ip: str = "127.0.0.1",
) -> bytes:
    req = f"GET {path} HTTP/1.1\r\nHost: {host_header}\r\nConnection: close\r\n\r\n"
    return _raw_exchange(ip, port, req.encode(), whole=True)


def ws_handshake_status(
    port: int,
    host_headers: str | list[str],
    *,
    ip: str = "127.0.0.1",
) -> int:
    """Open a WebSocket handshake with exactly the Host header(s) given and
    return the status code (101 = upgraded, 403 = refused)."""
    hosts = [host_headers] if isinstance(host_headers, str) else host_headers
    lines = ["GET / HTTP/1.1"] + [f"Host: {h}" for h in hosts]
    lines += [
        "Upgrade: websocket",
        "Connection: Upgrade",
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==",
        "Sec-WebSocket-Version: 13",
        "", "",
    ]
    return _status_of(_raw_exchange(ip, port, "\r\n".join(lines).encode()))


# ---------------------------------------------------------------------------
# Reachability from a non-loopback address of this machine
# ---------------------------------------------------------------------------


def local_non_loopback_address() -> str | None:
    """An IPv4 address of this machine that is not loopback, or None.

    A UDP ``connect`` sends nothing; it only asks the routing table which local
    address would be used to reach a documentation-range address.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            ip = str(s.getsockname()[0])
    except OSError:
        return None
    try:
        return None if ipaddress.ip_address(ip).is_loopback else ip
    except ValueError:
        return None


def tcp_connects(ip: str, port: int, *, timeout: float = 2.0) -> bool:
    """True when a TCP connection to ip:port is accepted. A refusal, a timeout
    (a stealth-mode firewall drops instead of resetting) and an unreachable
    network all count as "no"."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((ip, port))
        return True
    except OSError:
        return False


def tcp_refused(ip: str, port: int, *, timeout: float = 2.0) -> bool:
    """True when the connection is actively REFUSED (RST)."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((ip, port))
    except ConnectionRefusedError:
        return True
    except OSError:
        return False
    return False


def ipv6_loopback_usable() -> bool:
    """Can this machine bind and reach ::1? (Skip reason for the IPv6 tests.)"""
    if not socket.has_ipv6:
        return False
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
        return True
    except OSError:
        return False


def package_src_root() -> Path:
    """The directory the importing process loaded ``prometheus`` from — what a
    child process must be pointed at so it runs THIS tree, not another checkout
    that happens to be installed in the venv."""
    import prometheus

    return Path(prometheus.__file__).resolve().parents[1]


def repo_config_of_loaded_package() -> Path:
    """Where a CHILD process would look for a checkout-local live config
    (conftest only patches the parent's copy of this constant)."""
    return package_src_root().parent / "config" / "prometheus.yaml"


def free_port() -> int:
    """A currently-free loopback TCP port (never one of the daemon's 8005/8010)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
