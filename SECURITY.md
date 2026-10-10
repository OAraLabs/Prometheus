# Security policy

## Reporting a vulnerability

Please report security problems privately, through GitHub's private
vulnerability reporting:

**[Report a vulnerability](https://github.com/OAraLabs/Prometheus/security/advisories/new)**
(or the repository's **Security** tab → **Report a vulnerability**).

Do not open a public issue, discussion or pull request for a vulnerability.
Issues and GitHub handles are public, and a report there tells everyone
before a fix exists.

If you cannot use GitHub's form, email support@oara.ai with the subject
"Security" and we will move the report into a private advisory.

A useful report says:

- the Prometheus version (`oara --version`) and how it was installed
  (PyPI, Homebrew, `uv tool` / `pipx`, or a checkout)
- what an attacker can do, and from where (a chat message, a file the agent
  reads, the network, a local user)
- the steps to reproduce, with any tokens, keys, host names and IP addresses
  removed

We will acknowledge the report in the advisory, keep you updated there, and
credit you in the advisory when the fix ships unless you ask us not to.

## Supported versions

Prometheus is pre-1.0. Fixes land on `main` and ship in the next release;
only the latest release on [PyPI](https://pypi.org/project/oara-prometheus/)
is supported.

## Scope

In scope: the code in this repository — the daemon, the CLI, the gateways,
the REST and WebSocket control plane, and the security gate.

Some limits are documented and are not vulnerabilities by themselves — for
example, the workspace boundary on `write_file` / `edit_file` is a speed
bump, not confinement, and the default coding sandbox is process-level, not a
container. They are listed under
[Security & permissions](docs/guide/features.md#security--permissions) and
[known limits](https://oara.ai/docs/limits/). A way to get past a control
that the docs say *does* hold is in scope.

Beacon, Beacon for iOS and OAra Voice are closed source and are not in this
repository. Report problems with them to support@oara.ai.
