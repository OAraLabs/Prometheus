"""What the model is told about the machines: what they can do, not where they are.

The anatomy text (the prompt's Infrastructure section, ``ANATOMY.md``, the ``anatomy``
tool) names hosts, addresses and URLs: this machine's hostname, the inference server's
address and port, every tailnet peer by name and address, the directory a model file
sits in. A model needs none of that to know it runs on "local llama.cpp, 32k, vision",
and a reply that quotes it puts the network's layout wherever the reply goes, a
Telegram chat included. So by default the model gets the capabilities only.

:func:`without_addresses` is the one transform. It removes, from any text meant for
the model:

- URLs (``http://…``, ``ws://…``), with the brackets around them;
- IPv4 and IPv6 addresses, with a port if one follows;
- tailnet DNS names (``*.ts.net``);
- this machine's own hostname (as "this machine") and every host the config names
  (as "another machine");
- the tailnet peer list: the ``- Tailscale:`` line keeps the peer count and loses the
  address and the per-peer lines;
- the directory of a ``.gguf`` model path (another machine's username is in it).

Loopback (``localhost``, ``127.x``, ``::1``) and ``0.0.0.0`` are left alone: they name
no machine on the network.

``anatomy.include_addresses: true`` (only a literal ``true``) turns the transform off.
The owner's own ``/anatomy`` command is a separate renderer and always shows everything.
"""

from __future__ import annotations

import ipaddress
import platform
import re
from collections.abc import Iterable, Mapping
from urllib.parse import urlsplit

WITHHELD = "[address withheld]"
THIS_MACHINE = "this machine"
ANOTHER_MACHINE = "another machine"

_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"})

_URL_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s()<>\[\]{}\"'`|,;]+", re.IGNORECASE)
# A URL in brackets goes with its brackets: "llama_cpp (http://h:8080)" -> "llama_cpp".
_BRACKETED_URL_RE = re.compile(r"[ \t]*\((?P<url>[a-z][a-z0-9+.-]*://[^\s()]+)\)", re.IGNORECASE)
_TS_NET_RE = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+ts\.net\.?", re.IGNORECASE)
_IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?(?![\w.])")
# Candidates only; ipaddress decides. "23:25:33" is a time, not an address.
_IPV6_CANDIDATE_RE = re.compile(r"(?<![\w:.])\[?[0-9a-f]*:[0-9a-f:]*:[0-9a-f:.]*\]?(?::\d{1,5})?(?![\w:])",
                                re.IGNORECASE)
_GGUF_PATH_RE = re.compile(r"(?:~|/)[^\s`'\"()|]*/(?P<file>[^\s/`'\"()|]+)\.gguf\b")
_TAILSCALE_LINE_RE = re.compile(r"^(?P<indent>[ \t]*)- Tailscale: .*?\((?P<count>\d+) peers?\)[ \t]*$")


def include_addresses(config: Mapping | None) -> bool:
    """``anatomy.include_addresses``: only a literal ``true`` lets addresses through."""
    anatomy = (config or {}).get("anatomy") or {}
    return isinstance(anatomy, Mapping) and anatomy.get("include_addresses", False) is True


def _is_loopback(host: str) -> bool:
    host = host.strip("[]").lower()
    if host in _LOOPBACK_NAMES or host == "0.0.0.0":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def local_names() -> frozenset[str]:
    """This machine's own names: the node name, and its first label when it has dots."""
    node = (platform.node() or "").strip().lower()
    names = {node, node.split(".", 1)[0]} if node else set()
    return frozenset(n for n in names if len(n) >= 3 and not _is_loopback(n))


def network_names(config: Mapping | None) -> frozenset[str]:
    """Every host the config names: the host of each URL in it, and each ``*_host`` value.

    Walks the whole config rather than a list of keys, so a backend or a gateway added
    later is covered without anyone remembering to add it here.
    """
    found: set[str] = set()

    def walk(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                walk(v, str(k))
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v, key)
        elif isinstance(value, str) and value.strip():
            text = value.strip()
            if "://" in text:
                try:
                    host = urlsplit(text).hostname or ""
                except ValueError:
                    host = ""
                if host:
                    found.add(host.lower())
            elif key == "host" or key.endswith("_host"):
                found.add(text.split(":", 1)[0].lower())

    walk(config or {})
    return frozenset(h for h in found if len(h) >= 3 and not _is_loopback(h))


def _drop_tailscale_peers(text: str) -> tuple[str, set[str]]:
    """*text* without the per-peer lines, and the peer names those lines carried.

    The names are returned so the caller withholds them wherever else they appear
    (the inference host is often one of them).
    """
    out: list[str] = []
    peers: set[str] = set()
    peer_indent: str | None = None
    for line in text.splitlines(keepends=True):
        if peer_indent is not None:
            stripped = line.lstrip(" \t")
            indent = line[: len(line) - len(stripped)]
            if stripped.startswith("- ") and len(indent) > len(peer_indent):
                # One peer, written "- <name>: <address> (<online|offline>)".
                name = stripped[2:].split(":", 1)[0].strip()
                if name:
                    peers.add(name.lower())
                continue
            peer_indent = None
        m = _TAILSCALE_LINE_RE.match(line.rstrip("\r\n"))
        if m:
            n = int(m.group("count"))
            ending = line[len(line.rstrip("\r\n")):]
            out.append(f"{m.group('indent')}- Tailscale: connected ({n} peer{'' if n == 1 else 's'})" + ending)
            peer_indent = m.group("indent")
            continue
        out.append(line)
    return "".join(out), peers


def _replace_url(match: re.Match) -> str:
    url = match.group(0)
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        host = ""
    return url if host and _is_loopback(host) else WITHHELD


def _replace_bracketed_url(match: re.Match) -> str:
    url = match.group("url")
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        host = ""
    return match.group(0) if host and _is_loopback(host) else ""


def _replace_ipv4(match: re.Match) -> str:
    candidate = match.group(0).split(":", 1)[0]
    try:
        addr = ipaddress.IPv4Address(candidate)
    except ValueError:
        return match.group(0)  # 999.1.2.3 and the like: not an address
    return match.group(0) if addr.is_loopback or addr.is_unspecified else WITHHELD


def _replace_ipv6(match: re.Match) -> str:
    raw = match.group(0)
    core = raw
    if core.startswith("["):
        core = core[1:].split("]", 1)[0]
    try:
        addr = ipaddress.IPv6Address(core)
    except ValueError:
        try:  # an address followed by :port without brackets is ambiguous; try without the last group
            addr = ipaddress.IPv6Address(core.rsplit(":", 1)[0])
        except ValueError:
            return raw
    return raw if addr.is_loopback or addr.is_unspecified else WITHHELD


def _replace_name(text: str, name: str, replacement: str) -> str:
    pattern = re.compile(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])", re.IGNORECASE)
    return pattern.sub(replacement, text)


def without_addresses(
    text: str,
    names: Iterable[str] = (),
    *,
    here: Iterable[str] | None = None,
) -> str:
    """*text* with every address, URL and machine name taken out (see the module doc).

    *names* are other hosts to withhold (usually :func:`network_names`); *here* are this
    machine's own names, default :func:`local_names`.
    """
    if not text:
        return text
    own = {n.lower() for n in (here if here is not None else local_names()) if n}
    out, peers = _drop_tailscale_peers(text)
    # A phone on the tailnet can be named "localhost"; loopback names stay readable.
    others = {n for n in ({n.lower() for n in names if n} | peers) if len(n) >= 3 and not _is_loopback(n)} - own
    out = _BRACKETED_URL_RE.sub(_replace_bracketed_url, out)
    out = _URL_RE.sub(_replace_url, out)
    out = _TS_NET_RE.sub(WITHHELD, out)
    out = _GGUF_PATH_RE.sub(lambda m: m.group("file"), out)
    out = _IPV6_CANDIDATE_RE.sub(_replace_ipv6, out)
    out = _IPV4_RE.sub(_replace_ipv4, out)
    # Longest first, so "gpu-box" is not half-replaced by a shorter "gpu".
    for name in sorted(own, key=len, reverse=True):
        out = _replace_name(out, name, THIS_MACHINE)
    for name in sorted(others, key=len, reverse=True):
        out = _replace_name(out, name, ANOTHER_MACHINE)
    return out


def for_model(text: str, config: Mapping | None, *, extra_names: Iterable[str] = ()) -> str:
    """*text* as the model may see it under *config*: unchanged when addresses are on."""
    if include_addresses(config):
        return text
    return without_addresses(text, set(network_names(config)) | {n.lower() for n in extra_names if n})
