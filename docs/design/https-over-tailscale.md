# HTTPS over Tailscale: design note

**Status:** proposal v0.1, 2026-11-10. Nothing is built. For the release after 0.9.21.
**Written from:** the code at v0.9.20 (unchanged on `main`), Tailscale's documentation, and the CLI help of Tailscale 1.102.
**Marked throughout:** *measured* (run against the real code), *documented* (Tailscale says so), *UNVERIFIED* (not shown by either).

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

## 5. Recommendation

**B, in three small steps**, each its own PR: (1) certificate source, boot validation, the doctor row and the renewal
documentation; (2) the `tls.trust` field in the pairing answers and the reference; (3) a renewal helper (a script and a
systemd timer) if the documentation alone proves too fiddly. **A stays possible later** as a documented recipe, once
experiments E1 to E3 below have shown what Serve really sends and the four prerequisites in section 2 exist.

## 6. Experiments (run before anyone relies on A; they need the owner's say-so)

**E1: what the backend sees through Serve** (plain HTTP, so no certificate and nothing in the public log). On a machine with
Tailscale, run a 20-line echo server on `127.0.0.1:18090` that prints the TCP peer and every header; then
`tailscale serve --bg --http=18088 http://127.0.0.1:18090`. From *another* tailnet device:
`curl -i http://<name>:18088/ -H 'X-Forwarded-For: 1.2.3.4' -H 'Host: x.example'`. Record: the peer address, `Host`, every
`X-Forwarded-*`, `Forwarded`, `Via`, `Tailscale-*`, and whether the forged `X-Forwarded-For` survives. Repeat from a *tagged*
node. Finish with `tailscale serve --http=18088 off`. This answers A1, A4 and A5.
**E2: WebSockets** through the same entry (`websockets` client against `ws://<name>:18088/`). This answers A6.
**E3: the certificate path on a throwaway machine name** you do not mind seeing in the public log: `tailscale cert --cert-file
c.crt --key-file k.key <name>.<tailnet>.ts.net`, then the daemon's TLS listener with `web.tls_cert_file`/`web.tls_key_file`
(after step 1 exists), checked with `curl` and from a phone.

## 7. For the owner to decide

1. Is B the supported path, with A parked behind E1 to E3?
2. Machine names: the certificate publishes them. Rename before enabling HTTPS if the current ones say more than you want public.
3. May Prometheus change the `tls` object in the pairing answers (a documented, additive field), given that the iOS and desktop
   sessions build to `docs/reference/pairing-join-requests.md`?

## Sources

Tailscale Serve: <https://tailscale.com/docs/features/tailscale-serve> (identity headers, none for tagged devices or Funnel),
<https://tailscale.com/kb/1242/tailscale-serve> (flags, `serve --http`, the `127.0.0.1` target). Enabling HTTPS:
<https://tailscale.com/kb/1153/enabling-https> (Let's Encrypt, 90 days, renewal for file certificates, Certificate Transparency
and machine names, rate limits). `tailscale cert --help` and `tailscale serve --help` (1.102.2).
