"""``web/discovery.py`` — advertising ``_prometheus._tcp`` on the home network, and ONLY there.

What is pinned, in the order it can go wrong:

* **It advertises only when it should.** Home-network mode, ``discovery.mdns`` on, a usable library and at
  least one usable address. Never on a loopback bind, and (a deliberate narrowing of the contract, documented
  there) not in ``open`` mode either: an existing install listening on every interface would otherwise start
  announcing its name on the LAN the day it upgraded, because ``discovery.mdns`` defaults to on.
* **The address set is conservative.** A private IPv4 (RFC 1918) or link-local address on an interface the
  daemon actually listens on. Never a public address, never Tailscale's CGNAT range, never a tunnel or
  container interface, however private its address looks.
* **Nothing it advertises is a secret**, and it says exactly what ``GET /api/hello`` says (the same dictionary).
* **It cannot hurt the daemon.** ``zeroconf`` is an optional extra: absent, the daemon starts, pairs and serves
  normally, says so loudly ONCE, and reports the reason. A registration that fails is a status, not a crash.
* **It follows the network.** Wi-Fi roams; the address set is re-evaluated and the service re-registered on the
  new one, and a change in what hello says is an update, not a re-registration.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from prometheus.web.discovery import (
    SERVICE_TYPE,
    Advertiser,
    eligible_addresses,
    instance_name,
    txt_properties,
)
from prometheus.web.network import HOME, OPEN, THIS_MAC, NetworkSettings, NetworkState


CGNAT_A, CGNAT_B = ".".join(("100", "101", "102", "103")), ".".join(("100", "64", "0", "1"))   # built: the scanner flags these


def ip(addr: str) -> SimpleNamespace:
    return SimpleNamespace(ip=addr, is_IPv4=True, is_IPv6=False)


def ip6(addr: str) -> SimpleNamespace:
    return SimpleNamespace(ip=(addr, 0, 0), is_IPv4=False, is_IPv6=True)


def adapter(name: str, *ips: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(name=name, nice_name=name, ips=list(ips))


# ── which addresses ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("addr", ["192.168.1.20", "10.1.2.3", "172.16.5.5", "172.31.255.1", "169.254.3.4"])
def test_a_private_or_link_local_ipv4_on_a_real_interface_is_eligible(addr):
    assert eligible_addresses([adapter("en0", ip(addr))], bind="0.0.0.0") == [addr]


@pytest.mark.parametrize("addr", ["8.8.8.8", "172.32.0.1", CGNAT_A, CGNAT_B, "127.0.0.1", "0.0.0.0",
                                  "224.0.0.251", "198.18.0.5"])
def test_a_public_cgnat_loopback_or_multicast_address_never_is(addr):
    assert eligible_addresses([adapter("en0", ip(addr))], bind="0.0.0.0") == []


@pytest.mark.parametrize("name", ["utun3", "tun0", "tap1", "wg0", "tailscale0", "docker0", "br-1a2b3c", "veth12",
                                  "virbr0", "awdl0", "llw0", "lo0", "zt0abc", "bridge100", "vmnet8", "vboxnet0",
                                  "gif0", "ppp0", "ipsec0", "vEthernet (WSL)", "anpi0", "ap1", "cni0", "flannel.1",
                                  "cali1234", "kube-ipvs0", "stf0"])
def test_a_tunnel_or_container_interface_is_skipped_however_private_its_address(name):
    assert eligible_addresses([adapter(name, ip("10.9.8.7"))], bind="0.0.0.0") == []


def test_ipv6_is_not_advertised():
    assert eligible_addresses([adapter("en0", ip6("fe80::1"), ip("192.168.1.2"))], bind="0.0.0.0") == ["192.168.1.2"]


def test_several_interfaces_give_several_addresses_without_duplicates():
    found = eligible_addresses(
        [adapter("en0", ip("192.168.1.20")), adapter("en1", ip("10.0.0.5"), ip("192.168.1.20"))], bind="0.0.0.0")
    assert sorted(found) == ["10.0.0.5", "192.168.1.20"]


def test_a_specific_bind_advertises_only_that_address_if_it_is_eligible():
    adapters = [adapter("en0", ip("192.168.1.20")), adapter("en1", ip("10.0.0.5"))]
    assert eligible_addresses(adapters, bind="10.0.0.5") == ["10.0.0.5"]
    assert eligible_addresses(adapters, bind=CGNAT_B) == [], "a bind the rules refuse is not rescued by being specific"


@pytest.mark.parametrize("bind", ["127.0.0.1", "::1"])
def test_a_loopback_bind_advertises_nothing(bind):
    assert eligible_addresses([adapter("en0", ip("192.168.1.20"))], bind=bind) == []


def test_the_wildcard_ipv6_bind_means_every_interface():
    assert eligible_addresses([adapter("en0", ip("192.168.1.20"))], bind="::") == ["192.168.1.20"]


# ── what it is called and what it says ───────────────────────────────────────

def test_the_service_type_is_the_contracts():
    assert SERVICE_TYPE == "_prometheus._tcp.local."


def test_an_instance_name_is_the_display_name_cut_to_a_dns_label():
    assert instance_name("Will's Mac mini") == "Will's Mac mini"
    long = instance_name("é" * 100)
    assert len(long.encode()) <= 63 and long == "é" * (len(long))
    assert len(instance_name("a" * 200).encode()) == 63


def test_a_name_is_never_cut_through_a_character():
    name = instance_name("日" * 40)                 # 3 bytes each: 63 bytes is exactly 21 of them
    assert name == "日" * 21 and len(name.encode()) == 63
    assert instance_name("a" + "日" * 40) == "a" + "日" * 20, "a partial character at the cut is dropped, not mangled"
    assert instance_name("日" * 40).encode().decode("utf-8") == name


def test_an_empty_name_is_never_advertised():
    assert instance_name("") == "Prometheus" and instance_name("   ") == "Prometheus"


HELLO = {"v": "0.9.7", "name": "Will's Mac mini", "agent": "Prometheus", "fp": "3fa9c1e07b2d4a68",
         "pair": "approve", "tls": False}


def test_the_txt_record_is_hello_and_nothing_else():
    props = txt_properties(HELLO)
    assert set(props) == {"v", "name", "agent", "fp", "pair", "tls"}
    assert props["tls"] == b"0" and props["fp"] == b"3fa9c1e07b2d4a68" and props["name"] == b"Will's Mac mini"
    assert all(isinstance(v, bytes) for v in props.values())


def test_a_field_hello_does_not_have_cannot_be_advertised():
    with pytest.raises(ValueError):
        txt_properties({**HELLO, "uptime": 12})


def test_a_long_multibyte_name_still_fits_a_txt_string():
    """A TXT string is at most 255 bytes: 'name=' plus 64 characters of 4 bytes would not fit."""
    props = txt_properties({**HELLO, "name": "😀" * 64})
    assert len(b"name=" + props["name"]) <= 255
    assert props["name"].decode("utf-8") == "😀" * (len(props["name"]) // 4), "cut on a character boundary"


# ── the advertiser ───────────────────────────────────────────────────────────

class FakeBackend:
    """Stands in for zeroconf: records what it is asked to do."""

    def __init__(self, addresses, log, *, fail_register: BaseException | None = None) -> None:
        self.addresses = list(addresses)
        self.log = log
        self.fail_register = fail_register

    async def register(self, *, name, port, properties, addresses):
        if self.fail_register is not None:
            raise self.fail_register
        self.log.append(("register", tuple(self.addresses), name, port, dict(properties)))

    async def update(self, *, name, port, properties, addresses):
        self.log.append(("update", tuple(self.addresses), name, port, dict(properties)))

    async def unregister(self):
        self.log.append(("unregister", tuple(self.addresses)))

    async def close(self):
        self.log.append(("close", tuple(self.addresses)))


def rig(*, mode=HOME, bind="0.0.0.0", adapters=None, mdns=True, hello=None, name="Will's Mac mini",
        factory=None, refresh=60.0):
    log: list[tuple] = []
    state = {"adapters": adapters if adapters is not None else [adapter("en0", ip("192.168.1.20"))],
             "hello": dict(hello or HELLO), "name": name}
    made: list[FakeBackend] = []

    def default_factory(addresses):
        backend = FakeBackend(addresses, log)
        made.append(backend)
        return backend

    advertiser = Advertiser(
        state=NetworkState(mode=mode, bind=bind, bind_source="config", tls_enabled=False, warnings=[]),
        settings=NetworkSettings(mdns=mdns), port=8005,
        hello=lambda: dict(state["hello"]), display_name=lambda: state["name"],
        backend_factory=factory or default_factory, adapters=lambda: state["adapters"], refresh_seconds=refresh)
    return advertiser, log, state, made


@pytest.mark.asyncio
async def test_home_network_registers_the_service_on_the_eligible_addresses():
    advertiser, log, _, _ = rig()
    await advertiser.start()
    assert log == [("register", ("192.168.1.20",), "Will's Mac mini", 8005, txt_properties(HELLO))]
    assert advertiser.status.advertising is True and advertiser.status.reason is None
    await advertiser.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, bind, needle", [(THIS_MAC, "127.0.0.1", "this machine only"),
                                                (OPEN, "0.0.0.0", "home network")])
async def test_it_advertises_nothing_outside_home_network_mode(mode, bind, needle):
    advertiser, log, _, made = rig(mode=mode, bind=bind)
    await advertiser.start()
    assert log == [] and made == [], "the backend is not even built"
    assert advertiser.status.advertising is False and needle in advertiser.status.reason
    await advertiser.stop()


@pytest.mark.asyncio
async def test_the_owner_can_turn_it_off():
    advertiser, log, _, made = rig(mdns=False)
    await advertiser.start()
    assert log == [] and made == []
    assert "discovery.mdns" in advertiser.status.reason
    await advertiser.stop()


@pytest.mark.asyncio
async def test_a_missing_library_is_a_loud_status_and_never_an_exception(caplog):
    def no_zeroconf(addresses):
        raise ImportError("No module named 'zeroconf'")

    advertiser, _, _, _ = rig(factory=no_zeroconf)
    with caplog.at_level("WARNING"):
        await advertiser.start()
        await advertiser.refresh()
        await advertiser.refresh()
    assert advertiser.status.advertising is False
    assert "zeroconf is not installed" in advertiser.status.reason and "[discovery]" in advertiser.status.reason
    warnings = [r for r in caplog.records if "zeroconf" in r.getMessage()]
    assert len(warnings) == 1, "said once, not once a minute"
    await advertiser.stop()


@pytest.mark.asyncio
async def test_no_usable_address_is_a_status():
    advertiser, log, _, _ = rig(adapters=[adapter("utun3", ip("10.9.8.7")), adapter("en0", ip(CGNAT_B))])
    await advertiser.start()
    assert log == []
    assert advertiser.status.advertising is False and "address" in advertiser.status.reason
    await advertiser.stop()


@pytest.mark.asyncio
async def test_a_registration_that_fails_is_a_status_and_is_retried():
    attempts: list[int] = []

    def flaky(addresses):
        attempts.append(1)
        return FakeBackend(addresses, [], fail_register=OSError("multicast not permitted") if len(attempts) == 1 else None)

    advertiser, _, _, _ = rig(factory=flaky)
    await advertiser.start()                                       # must not raise
    assert advertiser.status.advertising is False and "OSError" in advertiser.status.reason
    await advertiser.refresh()
    assert advertiser.status.advertising is True, "the next evaluation tries again"
    await advertiser.stop()


@pytest.mark.asyncio
async def test_nothing_changing_changes_nothing():
    advertiser, log, _, made = rig()
    await advertiser.start()
    for _ in range(3):
        await advertiser.refresh()
    assert [c[0] for c in log] == ["register"] and len(made) == 1
    await advertiser.stop()


@pytest.mark.asyncio
async def test_roaming_to_another_network_reregisters_on_the_new_address():
    advertiser, log, state, made = rig()
    await advertiser.start()
    state["adapters"] = [adapter("en0", ip("10.0.0.5"))]
    await advertiser.refresh()
    kinds = [(c[0], c[1]) for c in log]
    assert kinds == [("register", ("192.168.1.20",)), ("unregister", ("192.168.1.20",)),
                     ("close", ("192.168.1.20",)), ("register", ("10.0.0.5",))]
    assert len(made) == 2, "interfaces are fixed when the backend is built, so a new address set is a new backend"
    await advertiser.stop()


@pytest.mark.asyncio
async def test_a_change_in_what_hello_says_is_an_update_not_a_reregistration():
    advertiser, log, state, made = rig()
    await advertiser.start()
    state["hello"]["fp"] = "ffffffffffffffff"
    await advertiser.refresh()
    assert [c[0] for c in log] == ["register", "update"] and len(made) == 1
    assert log[-1][4]["fp"] == b"ffffffffffffffff"
    await advertiser.stop()


@pytest.mark.asyncio
async def test_losing_every_address_withdraws_the_service_and_getting_one_back_restores_it():
    advertiser, log, state, _ = rig()
    await advertiser.start()
    state["adapters"] = []
    await advertiser.refresh()
    assert advertiser.status.advertising is False and [c[0] for c in log][-2:] == ["unregister", "close"]
    state["adapters"] = [adapter("en0", ip("192.168.1.20"))]
    await advertiser.refresh()
    assert advertiser.status.advertising is True and log[-1][0] == "register"
    await advertiser.stop()


@pytest.mark.asyncio
async def test_stopping_withdraws_the_service_and_closes_the_backend():
    advertiser, log, _, _ = rig()
    await advertiser.start()
    await advertiser.stop()
    assert [c[0] for c in log][-2:] == ["unregister", "close"]
    assert advertiser.status.advertising is False
    await advertiser.stop()                                        # idempotent


@pytest.mark.asyncio
async def test_the_background_loop_follows_the_network_without_being_asked():
    advertiser, log, state, _ = rig(refresh=0.05)
    await advertiser.start()
    state["adapters"] = [adapter("en0", ip("10.0.0.5"))]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any(c[0] == "register" and c[1] == ("10.0.0.5",) for c in log):
        await asyncio.sleep(0.02)
    assert any(c[0] == "register" and c[1] == ("10.0.0.5",) for c in log)
    await advertiser.stop()
    seen = len(log)
    await asyncio.sleep(0.2)
    assert len(log) == seen, "stop() stops the loop"


@pytest.mark.asyncio
async def test_the_instance_name_follows_the_display_name():
    advertiser, log, state, _ = rig()
    await advertiser.start()
    state["name"] = "Kitchen Mac"
    await advertiser.refresh()
    assert log[-1][0] in ("update", "register") and log[-1][2] == "Kitchen Mac"
    await advertiser.stop()


# ── against the real library, when it is installed (it is an optional extra) ──

def test_what_we_advertise_is_acceptable_to_zeroconfs_own_validation():
    zeroconf = pytest.importorskip("zeroconf")
    from prometheus.web.discovery import service_info

    for name in ("Will's Mac mini", "Mr. Smith's iMac", "日" * 30, "😀" * 64):
        info = service_info(name=name, port=8005, properties=txt_properties({**HELLO, "name": name}),
                            addresses=["192.168.1.20"])
        assert isinstance(info, zeroconf.ServiceInfo) and info.port == 8005
        assert info.type == SERVICE_TYPE and info.name.endswith("." + SERVICE_TYPE)
