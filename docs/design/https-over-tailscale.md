# HTTPS over Tailscale: design note

**Status:** proposal v0.2, 2026-11-12. Nothing is built. For the release after 0.9.21.
**Written from:** the code at v0.9.20 and `main` `2ce3ae4`, Tailscale's documentation, the CLI help of Tailscale 1.102, and
Tailscale's source for Serve (`ipn/ipnlocal/serve.go`, read 2026-11-11).
**Marked throughout:** *measured* (run against the real code), *documented* (Tailscale says so), *source* (read in Tailscale's
code, not yet observed on a machine), *UNVERIFIED* (none of these).
**v0.2:** the owner chose option A (section 5). v0.1 recommended B because Serve's behaviour was unknown; Tailscale's source
answers most of it (section 2a). B stays written down as the fallback.

## The question

A phone on the owner's tailnet must reach the daemon over HTTPS at the machine's full `<name>.<tailnet>.ts.net` name, with
**no ATS exception in the iOS app** and **no weaker security** than today. The phone-onboarding plan proposes
`tailscale serve` terminating TLS "with Prometheus keeping to localhost behind it". This note tests that against the code
and sets it beside the other way to get a certificate onto the daemon.

## 1. What exists today

- **Plain HTTP over the tailnet.** REST on `:8005`, WebSocket on `:8010`; the tailnet leg is WireGuard-encrypted, the HTTP
  inside it is not (`docs/guide/install.md`, "Remote access over Tailscale"). iOS refuses plain HTTP to a name like that.
- **Home-network TLS** (`docs/design/lan-tls-pr5.md`, `web/tls.py`). With `network.home_network: true` and a bind beyond
  loopback, the daemon serves TLS itself on `web.tls_api_port` (8006) and `web.tls_ws_port` (8011), and plain HTTP stays on
  loopback only. The certificate is the **instance certificate**, self-signed over the instance key. A client pins that key
  (`tls.spki_sha256` in the pairing answers). No public CA signs it, so iOS's default trust refuses it.
- **Loopback trust.** Some routes are the person-at-the-machine's alone and test that the request came from this machine
  (`web/loopback.py`: `is_same_machine`, `is_this_host`, `is_loopback_host_header`). A server bound to loopback also refuses
  any request whose `Host` is not a loopback name (`LoopbackHostGuard`), as DNS-rebinding protection.
- **`web.trusted_proxies`** (`web/serving.py`): addresses whose forwarded headers uvicorn believes when it works out the client.

## 2. Option A: `tailscale serve` terminates TLS and forwards to `127.0.0.1`

Tailscale's documentation shows Serve proxying to a local `http://127.0.0.1` service. For Prometheus that means the daemon stays
on loopback and every remote client arrives from the Serve process. The code was written to expect a relay like this
(`web/loopback.py` names `tailscale serve`), and what it expects is that it be **noticed and distrusted**, not supported.

| # | Finding | Status |
|---|---|---|
| A1 | The loopback `Host` guard refuses the forwarded public name. A loopback-bound server answers `403 host_not_allowed` to `Host: my-server.tail1234.ts.net` (and `…:443`); `localhost` and `127.0.0.1` pass. The module's own rule: a proxy "must send `Host: localhost`; forwarding the public name is refused". | *measured*. Whether Serve rewrites or keeps `Host`: **UNVERIFIED** (Tailscale does not say). If it keeps it, A fails at the first request. |
| A2 | Every remote client has the same peer, `127.0.0.1`, so every per-source limit becomes global. Two phones asking to join: the second gets `429 per_source_pending`. The owner's prompt shows `source_ip: 127.0.0.1` for any device, so the owner loses "where is this from". | *measured* |
| A3 | The owner-code limit is per source. Five wrong tries by anyone behind the proxy shut out **everyone** behind it: the owner's own phone, with the right code, got `401 invalid_code`. Any device on the tailnet can deny the owner an owner-code redemption. | *measured* |
| A4 | The same-machine tests look for relay headers (`X-Forwarded-For`, `Via`, ...). Tailscale documents that Serve adds **identity headers** (`Tailscale-User-Login`, `Tailscale-User-Name`, ...), and that **requests from tagged devices and from Funnel carry none**. It does not document `X-Forwarded-For`. So a Serve-relayed request can look exactly like a local one. Every route that trusts loopback also needs a secret (the pairing file secret, the global token, a worker token), so this is a loss of one factor, not an open door, but it changes the trust model and has to be a decision. | identity headers *documented*; forwarded headers **UNVERIFIED** |
| A5 | `web.trusted_proxies = [127.0.0.1]` would restore the real client address, but only if Serve sends a forwarded header, and `is_same_machine` warns that a range that is too wide lets a forged header read as `127.0.0.1`. | **UNVERIFIED** |
| A6 | Beacon uses REST **and** a separate WebSocket port. Serve publishes one HTTPS port per target, so there would be two. The answers that tell a client where to connect (`endpoints.rest`, `endpoints.ws`) are built from the `Host` it used and `web.ws_port`/`web.tls_ws_port`; behind Serve the daemon cannot know the ports Serve exposes. WebSocket proxying by Serve is not stated in the docs. | **UNVERIFIED** |

What A would need before it is supportable: an allowed-host setting for the guard (A1), a public-endpoints setting (A6),
limiter keys that read the real client behind a *named* proxy (A2, A3), and proof of what Serve sends (A4, A5). That is four
changes in security-sensitive code, resting on three behaviours nobody has observed. (A variant, `--tls-terminated-tcp` with
`--proxy-protocol`, hands the daemon the real client address, but the daemon does not speak the PROXY protocol, and the `Host` problem remains.)

## 3. Option B: the daemon terminates TLS itself, with a certificate Tailscale issued

`tailscale cert <name>.<tailnet>.ts.net` (flags `--cert-file`, `--key-file`, `--min-validity`, per `tailscale cert --help`)
writes a Let's Encrypt certificate that any phone trusts with the default trust store. The daemon already has the whole
listener (`web/tls.py`: ports, cipher policy, WSS, the plan); the **only** hardwired part is the certificate's source, the
instance certificate.

**What changes**

1. *Certificate source.* Two settings, `web.tls_cert_file` and `web.tls_key_file`, replace the instance certificate in the
   `LanTlsPlan` when both are set. Checked at boot, loudly: both files load, the key matches the certificate, the certificate
   is not expired, and its names include the name clients use. A failure falls back to the instance certificate and says why,
   never to plain HTTP on the LAN (the existing `decide_lan` rule).
2. *`oara doctor` row:* the name, the issuer and the days left; a warning under 14.
3. *The pairing answers' `tls` object.* Today it is the SPKI pin of the instance key. With a public-CA certificate the leaf key
   is different, so a client that pins would fail. The object needs a trust mode: `{"trust": "ca"}` (no pin) for an issued
   certificate, `{"trust": "pin", "spki_sha256": ...}` for the instance one. Clients take the pin only when `trust` says `pin`.
   The reference `docs/reference/pairing-join-requests.md` documents the field and would change with it.
4. *Key custody.* The private key must not be readable by a model-written shell. `~/.prometheus/node/` is already denied
   to it by the kernel floor (AppArmor and Seatbelt profiles), so the key goes there, mode 0600.
5. *Renewal.* Let's Encrypt certificates last 90 days, and Tailscale's documentation says that for certificates delivered as
   files the owner is responsible for renewal (`tailscaled` does not do it). A documented timer runs `tailscale cert --min-validity <n>h ...` and restarts the
   daemon (or the daemon re-reads the files on a signal). The doctor row is the net under it.

**What does not change.** No proxy, so no new trust decision: the peer is the real peer, `Host` is the real name, the limiters
key on real addresses, every loopback test means what it means today, and WebSockets use the existing `:8011`. Plain HTTP stays
on loopback. The URL carries a port (`https://<name>.<tailnet>.ts.net:8006`), which is fine for iOS.

**Costs and risks**

- **Public disclosure.** Tailscale's documentation: every machine name a certificate is issued for goes into the public
  Certificate Transparency log, permanently, and it advises not to enable HTTPS if a machine name is sensitive. Rename the
  machine *before* the first `tailscale cert`; a machine's name cannot be taken back out of the log afterwards.
- *HTTPS Certificates* and MagicDNS must be enabled in the tailnet's admin console (an owner action, not Prometheus's).
- Let's Encrypt rate limits (Tailscale warns of waits of a day or more if you request too often), so renewal asks for a certificate only when one is needed (`--min-validity`).
- One more thing for the owner to set up. A's "one command" is easier for a person; B is easier to get right.

## 4. The others

- **Plain HTTP and an ATS exception in the app:** rejected by the plan's own rule.
- **The instance certificate with client-side pinning** (what exists): it works for the desktop Beacon. Whether iOS code can accept
  a pinned self-signed certificate without it counting as an exception is the iOS session's call, not this note's.

## 2a. What Tailscale's source says Serve sends (v0.2)

| Question | Source says | Effect |
|---|---|---|
| `Host` | kept: `r.Out.Host = r.In.Host` | the daemon sees the `.ts.net` name; A1 stands until the guard allows it (D2) |
| `X-Forwarded-For` | set from the source address, overwriting a client's value | the real client address is available, and the request is not "this machine": `is_same_machine` already treats the header as a relay (A4, A5 resolved, pending E1) |
| `X-Forwarded-Host`, `X-Forwarded-Proto` | Host always; Proto `https` when the leg was TLS | both already relay headers; a second check on the name |
| `Tailscale-User-*` | cleared, then set only for non-tagged, non-Funnel callers | not used for any decision |
| WebSocket upgrade | Go's `httputil.ReverseProxy` | **UNVERIFIED** (E2) |
| Certificate renewal for Serve | provisioned automatically; renewal not stated | **UNVERIFIED** (E4); the doctor row is the net |

## 5. Decision: A, `tailscale serve` in front of a loopback daemon

```
phone ──HTTPS 443──▶ tailscaled (Serve, Tailscale's certificate) ──HTTP──▶ 127.0.0.1:8005   REST
      ──HTTPS 8443─▶ tailscaled (Serve)                          ──HTTP──▶ 127.0.0.1:8010   WebSocket
```

The daemon stays on `127.0.0.1`; plain HTTP never leaves the machine. Tailscale holds the certificate's private key, so
Prometheus never does. Never `tailscale funnel`: that is the public internet.

**Trust.** Clients use the normal certificate chain for the name they dialled, and **never pin** the certificate, its key or
an intermediate (the leaf changes at every renewal, about every 90 days). Before sending the token, they prove the daemon is
the one they paired with by the node-key identity challenge, checked against the `node_pub` stored at pairing. TLS says "this
is the machine with that name"; the challenge says "that machine is the Prometheus I paired with".

**Daemon changes**, each with a red test first:

- **D1, one settings block**, written only by `oara tailscale setup`: `web.tailscale: {enabled, name, rest_port: 443,
  ws_port: 8443}`, with `web.bind` required to stay loopback. A bad block is dropped out loud.
- **D2, the Host guard accepts the name only when relayed**: both guards (REST and the WebSocket bridge) accept
  `Host: <name>` only from a loopback peer whose `X-Forwarded-Host` is the same name. A DNS-rebinding page cannot produce that.
- **D3, the client address, believed narrowly.** Not `web.trusted_proxies = [127.0.0.1]`, which would let any local process
  choose its own source with one header. A small middleware takes `X-Forwarded-For` as the client only when: loopback peer,
  the feature on, exactly one value, a tailnet address (100.64.0.0/10 or fd7a:115c:a1e0::/48), and `X-Forwarded-Host` is the
  name. `is_same_machine` is unchanged and stays false for a relayed request. Of the unauthenticated limits, join requests and
  owner-code redemption already have a global cap; hello and the identity challenge (60 a minute per source) do not, so this
  change adds a global ceiling there, so that a local process rotating forged tailnet addresses gains no budget (A2, A3).
- **D4, where to connect.** For a relayed request at the name, the pairing answers give `https://<name>` and
  `wss://<name>:8443`, and `tls: {"trust": "ca", "host": "<name>"}` with no `spki_sha256`. Additive: `trust: "pin"` keeps
  meaning the instance certificate. `docs/reference/pairing-join-requests.md` and its contract test change with it.
- **D5, `oara tailscale check | setup | off`.** `setup` prints the plan (the two Serve commands and the settings block) and
  changes nothing without `--apply`. It never runs `sudo`, `tailscale cert`, `funnel`, `up`, `login` or `set`; when Serve
  needs an operator it prints the one line for the owner to run.
- **D6, `oara doctor`, "Tailscale HTTPS"** (no rows when off): CLI found and running, the MagicDNS name matches; Serve maps
  443 and 8443 to the loopback ports; **Funnel off** (error); `web.bind` loopback (error otherwise); a live handshake to
  `<name>:443` that verifies against the default trust store, with days left (warning under 14); and an end-to-end identity
  challenge through `https://<name>` that verifies with the node key. A check that could not run says "not checked".
- **D7, docs** (install guide, pairing reference) and **D8**, Mission Control's CSP lists `wss://<name>:8443`.

**The owner's switch.** In the tailnet's admin console, **DNS** page: MagicDNS on, then **HTTPS Certificates > Enable HTTPS**.
**Every machine name a certificate is issued for goes into the public Certificate Transparency logs, permanently.** Rename a
machine before its first certificate if its name says too much.

## 6. Experiments (need the owner's say-so; none run)

- **E1** (plain HTTP, nothing in the public log): an echo server on `127.0.0.1:18090`, `tailscale serve --bg --http=18088
  http://127.0.0.1:18090`, and from another tailnet device `curl -i http://<name>:18088/ -H 'X-Forwarded-For: 1.2.3.4'
  -H 'Host: x.example'`. Record the peer, `Host`, every `X-Forwarded-*`, `Forwarded`, `Via`, `Tailscale-*`, and whether the
  forged header survives. Repeat from a tagged node. Then `--http=18088 off`.
- **E2**, WebSockets through the same entry.
- **E3** (puts the name in the public log): the two real entries to an echo server; check the chain a phone sees.
- **E4**, renewal: record `notAfter`, look again at day 60 and 90.

## 7. Build plan

Three PRs, red tests first, an independent review each: (1) daemon D1 to D4 and D8; (2) `oara tailscale` and the doctor
rows, against a fake `tailscale` on `PATH`; (3) docs and the pairing reference.

## Sources

Tailscale Serve: <https://tailscale.com/docs/features/tailscale-serve> (identity headers, none for tagged devices or Funnel),
<https://tailscale.com/kb/1242/tailscale-serve> (flags, `serve --http`, the `127.0.0.1` target). Enabling HTTPS:
<https://tailscale.com/kb/1153/enabling-https> (Let's Encrypt, 90 days, renewal for file certificates, Certificate Transparency
and machine names, rate limits). `tailscale cert --help` and `tailscale serve --help` (1.102.2).
