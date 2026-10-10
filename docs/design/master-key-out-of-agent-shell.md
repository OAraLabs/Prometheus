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
profile.

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
