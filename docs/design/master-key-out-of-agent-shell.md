# Keeping the master key out of the agent's shell on macOS

Release gate (b) of Prometheus.app (`docs/design/macos-app-installer.md`). Will, 2026-10-10: the master key must
not be readable by the agent's shell on macOS. Two designs were asked for and compared; one is recommended
and built.

## The exposure today

`~/.config/prometheus/env` holds the daemon's API token (the master key: every REST route, including answering
the agent's own approval prompts). It also holds the provider keys (`ANTHROPIC_API_KEY`, …) and the gateway
bot tokens (Telegram, Slack, Discord). The Telegram bot is the approval channel itself.

The path-declaring tools (`read_file`, `write_file`, `edit_file`) cannot open it: `security.denied_paths` ships
`/*/.config/*/*env`. Shell commands are not covered by that list, because the gate is handed a command string
and never sees a path. On Linux the **read floor** closes that gap below the tool layer: AppArmor
(`deploy/apparmor/prometheus-bash`) denies `~/.ssh`, `~/.gnupg` and `~/.config/*/*env` to every shell a model
writes (the bash tool, task shells, model-made cron jobs, coding runs; `security/shell_floor.py`). On macOS
there is no AppArmor, and `security.bash_confinement: auto` reports `unsupported`. So today the agent's shell
on a Mac can `cat ~/.config/prometheus/env`.

## Option 1: keep only a hash of the token on disk

The daemon stores `sha256(token)`, and any local client that needs to call the API holds a device credential
instead.

- **For:** the plaintext master key exists nowhere at rest, so nothing on disk can be read for it, whatever
  path the reader takes. It doesn't depend on any OS mechanism.
- **Against: it protects one secret of several.** The provider keys and bot tokens stay in the same file,
  readable by the shell. So do `~/.ssh` and `~/.gnupg`. A leaked Telegram token is a leaked approval channel.
- **Against: it moves the problem.** Every local client that presents the token today needs another
  credential: `oara pair`, `oara token`, `oara doctor`'s live check, operator cron jobs and scripts that call
  the API, and the documented `curl -H "Authorization: Bearer …"`. That credential (an owner device token is
  operator-equivalent) has to be stored somewhere the same user can read. That's the same file problem again,
  unless it goes behind a Keychain item that demands the person's presence. That would be a new design of its
  own.
- **Against: cost.** It's a migration for every existing install's plaintext token, plus changes to rotation
  and to setup mode's `issue_setup_credential`. It's the larger build, and it breaks things that work.

## Option 2: a macOS sandbox profile on the agent's shell (Seatbelt)

Every shell a model writes is exec'd through `sandbox-exec -p <profile>`. The profile allows everything except
reading or writing `~/.ssh`, `~/.gnupg` and `~/.config/*/*env`: the same three families as the AppArmor
profile. Since 2026-10-13 it also denies the same-Mac pairing directory and renaming the directories above
what it denies (below).

- **For: it's the macOS half of a floor that already exists.** It uses the same config key
  (`security.bash_confinement`), the same modes (`auto`/`required`/`off`) and the same doctrine (verify by
  outcome at boot, fail loud, never silently unconfined). It reports the same way (`/api/status`,
  `oara doctor`, each call's `read_floor` metadata), and it covers the same scope (`shell_floor.py`).
- **For: it covers every secret in the file,** and `~/.ssh` and `~/.gnupg`, not just the master key.
- **For: no client changes, no migration, and no root.** `sandbox-exec` is unprivileged.
- **For: measured on this Mac (macOS 15.6.1), unprivileged.** With a profile denying a scratch "home":
  - `cat …/.config/prometheus/env`: `Operation not permitted`;
  - `cd …/.ssh && cat id*`: refused;
  - a Python child opening the env file: `PermissionError` (children inherit the sandbox);
  - an ordinary read and write: unaffected.
- **Against: `sandbox-exec` is deprecated by Apple.** It still ships (macOS 15), and it's what Chrome and
  several coding agents sandbox with. The boot probe verifies it by outcome, so if a future macOS drops it,
  `auto` says so loudly and `required` refuses to run the shell. The floor never quietly disappears.
- **Against, shared with the Linux floor:** a floor confines the process tree it starts. A command that asks a
  separate, unconfined service to act for it isn't inside that tree. On macOS that's launchd
  (`launchctl submit`), AppleEvents and `open`; on Linux, `systemd-run --user` over D-Bus, which the AppArmor
  profile allows. This isn't measured here. Neither option closes it for the other secrets. Option 1 would
  close it for the master key only.

## Recommendation: option 2, now

It isn't close. Option 2 makes the gate's sentence true for every secret in the file, matches the floor Linux
already has, and changes nothing for any client. Option 1 protects one secret, breaks the clients that use it,
and re-creates the problem wherever their replacement credential lives.

What option 1 uniquely adds, no plaintext master key at rest, is worth keeping as a later complement. That
means a hashed token together with a presence-gated client credential, designed as its own change once the
local clients have somewhere safe to keep a credential.

**The narrow opt-out (Will, 2026-10-10):** `security.bash_allow_ssh: true` lifts only the `~/.ssh` rule, for `git` over SSH. `~/.gnupg` and `~/.config/*/*env` stay denied, so the master key stays out. Without it, `bash_confinement: off` was the only way, and it opened everything. It works on macOS (Seatbelt). On Linux the AppArmor profile is a file root loaded, so the daemon warns at boot that the key has no effect there.

**The same-Mac pairing secret (2026-10-13).** The app keeps `pair.secret` in
`~/Library/Application Support/Prometheus/pairing/` (`config/local_pairing.py`). `POST /api/pair/local`, and
`POST /api/setup/pair` in setup mode, trade it for an owner device token when the request comes from loopback
with a loopback Host and no Origin. Every model-written shell on the same Mac meets all three conditions, and a
`python3 -c` with urllib doesn't match the curl/wget approval patterns. So while the secret exists (setup mode
before Beacon pairs, or after `Prometheus --pair` until it is used), a shell that could read the file could mint
itself operator-equivalent credentials. Writing to the file is as bad as reading it: a secret the shell planted
would be accepted just the same. Measured before the fix: under the profile, `cat .../pairing/pair.secret`
printed the secret, and `read_file` was ALLOW at AUTO trust. The local-pairing module's note that the file
"widens nothing" holds for a person's processes, not for a shell the floor keeps out of `~/.config/prometheus/env`.

Now:

- The Seatbelt profile denies reading and writing the pairing directory, and also the one
  `PROMETHEUS_LOCAL_PAIRING_DIR` names when it is set. `security.bash_allow_ssh` doesn't change that.
- The gate's credential floor (`checker._ALWAYS_DENIED_PATHS`, unoverridable) and the shipped `denied_paths`
  name `/*/Library/Application Support/Prometheus/pairing` and `/*/.local/share/prometheus/pairing`. The gate
  also denies the directory this process would actually mint into, as it does its own config file.
- A config holding, exactly and in order, a `denied_paths` list an earlier version shipped (setup copies the
  template, so every install has one) is given the new list, with one INFO line at boot. Any other list is the
  operator's, is used as written and still gets the override WARNING (`DENIED_PATHS_DEFAULTS_SHIPPED`).
- grep and glob prune the floor out of results under any root, not only the configured list. Before this,
  `denied_paths: []`, or a config written before an entry joined the floor, let `grep --root ~` print a floored
  file.

**Renaming a parent escaped the floor (found in the same measurement).** A path-matched rule is escaped by
moving a directory above it: `mv ~/.config ~/cfg` was allowed, because `~/.config` itself isn't
`~/.config/*/*env`, and `cat ~/cfg/prometheus/env` then printed the master key. Moving
`.../Application Support/Prometheus` did the same to the pairing secret. The profile now refuses writes
(`literal`) to each directory above a denied root, up to but not including the home: `~/.config`, `~/Library`,
`~/Library/Application Support` and `.../Prometheus`. Creating, writing and deleting things inside those
directories still works (measured). What's refused is renaming, replacing or re-moding the directory itself, so
`touch ~/.config` now fails. Hard links and symlinks into a denied path are refused too, and so is moving a
prepared directory into place. The boot probe checks the pairing secret and the `~/.config` rename by outcome.

**Linux, measured afterwards (#766):** the AppArmor profile (`deploy/apparmor/prometheus-bash`) had the same hole. With the shipped profile loaded on Linux 7.0, `mv ~/.config ~/cfg && cat ~/cfg/prometheus/env` printed a dummy env file. `deny @{HOME}/.config/ w,` refuses that rename and the move outside the home, and nothing else measured changed (#776). The boot probe renames `~/.config` onto itself, which is refused under the rule and a no-op without it. A host still on the old profile keeps being confined by it, and the boot logs an ERROR with the reload. The profile is a root-loaded file, so each Linux host runs `sudo apparmor_parser -r -W /etc/apparmor.d/prometheus-bash` after upgrading. The pairing directory only exists for the Mac app (`PROMETHEUS_INSTALL_KIND=app`) unless someone names one by hand.

## What gets built

- `permissions/confinement.py`: a Seatbelt read floor beside the AppArmor one.
  - It applies on darwin when `/usr/bin/sandbox-exec` exists.
  - The profile is generated from the real home directory, with paths resolved, because Seatbelt matches real
    paths.
  - The boot preflight verifies by outcome against a scratch home: a denied family is refused, an ordinary file
    is read.
  - The wrapper is `sandbox-exec -p <profile> --`.
  - `auto` on a Mac becomes the floor instead of `unsupported`. `unsupported` remains only where neither
    mechanism exists.
- The same call sites as the AppArmor floor (`shell_floor.py`). The template, the docs, the refusal message and
  `oara doctor` name the mechanism that's actually in use.
- Tests: the profile text, wrapping, modes and reporting run everywhere. On macOS (CI's `test-macos`) real
  `sandbox-exec` tests run against a scratch home and check refusal by outcome.
