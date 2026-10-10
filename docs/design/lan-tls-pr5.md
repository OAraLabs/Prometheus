# TLS on the home network (pairing PR 5)

Pairing PR 5 of 5. The contract is `docs/PAIRING-APPROVAL-API.md` §7.3 (approved, D5). Beacon's v0.4 multi-computer spec
builds against it and states what it assumes from this PR. This note says what PR 5 builds, what it leaves to #700, and
where it departs from the letter of either.

## What Beacon checks, and so what must be true

- **The pin is the SHA-256 of the certificate's SubjectPublicKeyInfo (SPKI DER), all 32 bytes.** Beacon reads it from the
  TLS handshake. Names, dates and the issuer are ignored.
- **Hello's `fp` is the first 16 hex characters of that same hash.** It's already `instance_key.instance_fingerprint()`:
  16 hex of SHA-256 over the instance key's SPKI DER. So the certificate must be made **over the instance key**, and
  then the handshake's SPKI *is* the instance key's SPKI. Beacon also checks the handshake key against
  `instance_public_key` in the `POST /api/pair/requests` answer (#698). That's the same key, so it holds without
  change.
- **Hello's `tls` is true when the port that was asked speaks TLS.** It's per listener, not per daemon: hello on the
  loopback port of a home-network daemon says `false`, and hello on its TLS port says `true`.
- **Ports:** REST 8006 and WebSocket 8011 by default (`web.tls_api_port`, `web.tls_ws_port`). Loopback keeps plain HTTP
  on 8005 and 8010, so Beacon on the same Mac, the app's launcher, `oara doctor` and curl are unchanged.

## What PR 5 builds

1. **The certificate** (`config/instance_cert.py`).
   - Self-signed X.509 over the instance key (P-256, ECDSA-SHA256).
   - SANs: `localhost`, `<hostname>.local`, and `prometheus-<first 8 hex of fp>.local` (the name #700's mDNS
     advertises).
   - Valid for 5 years, starting a day back for clock skew.
   - Written to `instance.crt` beside `instance.key` in the node directory, mode 0644 (it's public), atomically.
   - Made at daemon boot (first run), right after the key. Re-issued **over the same key** when it's missing,
     unreadable, made for another key, or within 30 days of expiry. The pin (the key) never changes. The key file is
     never written here.
   - Setup mode makes neither (it creates no `~/.prometheus` state); the TLS listeners only exist in a configured
     daemon.
2. **The decision** (`web/tls.py`, `lan_tls_plan`). TLS runs when `network.home_network: true` **and** the resolved bind
   reaches beyond loopback. The plan names the LAN address (the bind), the two TLS ports and the two files. Otherwise
   there's no plan, and nothing changes for any install that predates this.
3. **The listeners** (`web/launcher.py`, `web/serving.py`, `web/ws_server.py`).
   - With a plan, the plain REST and WebSocket listeners bind **127.0.0.1**, which is "loopback stays plain". A second
     uvicorn server serves the same app with the certificate on `<bind>:tls_api_port`, and the bridge serves a second,
     TLS WebSocket on `<bind>:tls_ws_port`.
   - Without a plan, there is exactly the listener set of today.
   - Separate ports, so the daemon never has to tell interfaces apart to run two kinds of listener on one port
     (§7.3).
4. **Hello** sets `tls` from the request's own scheme. Everything else in hello is unchanged.
5. **Configuration:** `web.tls_api_port: 8006` and `web.tls_ws_port: 8011` in the template and the generated reference.
   An invalid port falls back to the default and says so.

## What it leaves to #700, and why

`network.home_network` is #700's setting: its `NetworkSettings` is the one reader, `GET`/`PUT /api/network` report and
change it, and its mDNS advertiser announces it. PR 5 lands on main first, by Will's order, so it reads that one key
itself, with #700's exact rule: `true` or `false`, anything else is off and said out loud. When #700 merges main (the
next step), it does three things:
- it replaces that read with `NetworkSettings`;
- it passes the real TLS state into `describe()` (its `tls` parameter already exists) and `_tls_block()`
  (`enabled`, and the full `spki_sha256` hex Beacon reads);
- it advertises the TLS REST port with `tls=1`.

Until then, `network.home_network: true` on main turns TLS on and nothing else.

## Departures, stated

- **"Generated at first run"** means the daemon's first boot after this lands. An install that already has an instance key
  gets its certificate then, over that key. The pin doesn't move, because the key doesn't.
- **The bundled dashboard over the LAN** shows a browser warning (a self-signed certificate). Beacon and the iOS app pin
  and are unaffected. That's the contract's stated cost (§7.2 A).
- **Not built:** a commit-reveal for first contact (§7.5, reserved), and live switching (the restart route is its own PR).

## Tests (red first)

- **The certificate:**
  - its SPKI equals the instance key's, computed independently;
  - SANs, validity and file mode are as above;
  - idempotent;
  - re-issued over the same key near expiry, and when made for another key;
  - the key file untouched.
- **The plan:** none on loopback, none with `home_network` off or not a boolean, the default and custom ports, and an
  invalid port.
- **Real listeners on 127.0.0.1** (a LAN bind in a test would raise the macOS firewall prompt):
  - the TLS REST port's certificate SPKI SHA-256 equals the instance key's;
  - hello there says `tls: true`, and its `fp` is the 16-hex prefix;
  - the plain port says `tls: false`;
  - a WSS handshake on the TLS WebSocket port presents the same key;
  - with a plan, the plain listeners are on loopback.
- **The daemon** makes the certificate at boot, after the key.
