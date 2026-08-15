# BwrapSandbox — built, tested, and blocked on this host by one sysctl

Follow-up to `20260813T053000Z-kernel-sandbox-scoping.md`, which recommended
"a `BwrapSandbox` behind the existing interface" with the acceptance test
"the shell redirect must fail." That class is now written
(`src/prometheus/coding/sandbox.py`) and tested
(`tests/test_bwrap_sandbox.py`, 15 passed unconditionally). **The acceptance
test itself cannot pass on the deployment host as currently configured — not
the implementation is wrong, but because the host blocks unprivileged bwrap
from completing at all, in any network configuration.** This is a real,
reproducible, well-diagnosed host constraint, not a flake, and not an
artifact of the agent's own tool-execution context (ruled out explicitly,
below).

## The finding, precisely

```
kernel.apparmor_restrict_unprivileged_userns = 1     # Ubuntu 24.04 default
```

Bisected every individual `--unshare-*` flag bwrap accepts, one at a time:

| Flags | Result |
|---|---|
| `--unshare-user` (alone, or with pid/ipc/uts/cgroup, no net) | `bwrap: setting up uid map: Permission denied` |
| same set **+ `--unshare-net`** | gets past uid-map; then `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted` |
| `--unshare-all` (bwrap's own shorthand) | identical to the row above — it always includes net |
| `--unshare-all --share-net` | back to `Permission denied` at uid-map |

Two independent failures, and the flag set determines which one you hit —
never neither.

**Failure 1 — uid-map, gated by network's presence.** Matches Ubuntu's own
`unprivileged_userns` AppArmor profile, confirmed present at
`/etc/apparmor.d/unprivileged_userns` and readable — "Special profile
transitioned to by unconfined when creating an unprivileged user namespace."
The transition does not reliably trigger for a userns-only request; adding
`--unshare-net` to the same call does. (Root cause not fully pinned beyond
this — plausibly the kernel-side heuristic gating the transition was tuned
against the flag combination real container tooling actually sends, which in
practice always includes net.)

**Failure 2 — loopback bring-up, unconditional once net IS unshared.**
`strace` shows the netlink exchange directly:

```
clone(..., CLONE_NEWNS|CLONE_NEWUSER|CLONE_NEWNET|SIGCHLD) = <pid>   # succeeds
socket(AF_NETLINK, ...) = 4                                          # succeeds
sendto(4, [RTM_NEWADDR ... 127.0.0.1/8 on lo], ...) = 40             # sent
recvfrom(4, [{error=-EPERM, ...}]) = 60                               # kernel says no
```

Namespace creation succeeds. The specific netlink call to bring up loopback
is refused by the kernel. `--info-fd` confirms all six namespaces (user
implied, mnt, pid, ipc, uts, cgroup, net) exist before this point.

**Ruled out, not assumed:**

- *Dropped capability at an outer layer* — `/proc/self/status` `CapBnd`
  includes `CAP_NET_ADMIN` (bit 12 set: `000001ffffffffff`). The bounding
  set is full; this is not a stripped-capability case.
- *Nested sandbox / not the real host* — `systemd-detect-virt` → `none`;
  `/proc/1/comm` → `systemd`; `/proc/1/cgroup` → `0::/init.scope` (root
  cgroup, not a container's); no `/.dockerenv`; `systemctl --user
  is-active prometheus.service` → `active`; `hostname` returns the host's own
  name, not a container id. This
  session's shell runs directly on the deployment host, not inside a nested container
  the agent harness applies over Bash calls. The daemon will hit the exact
  same wall.
- *Flakiness* — the uid-map failure reproduced identically across three
  consecutive runs.

Net effect: **as configured today, no bwrap invocation — isolated or
networked — completes for an unprivileged user on this host.**

## What's built and verified anyway

`BwrapSandbox(ProcessSandbox)` in `coding/sandbox.py`. Inherits `resolve()`,
env scrub, and the dedicated-clone jail root from `ProcessSandbox` unchanged
— those were never the gap. Overrides only `run()`, wrapping the command in
a `bwrap` invocation: mount+pid+ipc+uts(+net) namespace, jail root the only
writable bind, OS/toolchain dirs and everything on `$PATH` read-only,
`denied_paths` entries inside root additionally re-bound read-only (closing
a gap `ProcessSandbox.run()` never covered — previously `denied_paths` was
`resolve()`-only, so a shell command could write there even though a file
tool couldn't).

A sentinel is printed before the real command so `run()` can tell "bwrap
itself failed to start" from "the command ran and exited nonzero" — the
former raises `SandboxConstructionError` naming bwrap's own diagnostic,
never misreported as the command's exit code.

`BwrapSandbox.self_check()` runs both network policies with a throwaway
namespace and reports which (if either) actually works — it reproduces this
entire finding in one call, and is what the test suite skips on, loudly,
carrying the full detail string in the skip reason rather than a bare
"skipped."

**Verified live, unconditionally (15 tests, host-independent):**
argv construction (network flag present/absent as configured, root bound
read-write, `denied_paths` RO-shadow ordered *after* the RW root bind so it
actually shadows), the sentinel wrapper, `resolve()` unchanged,
`self_check()`'s own shape, and — concretely — that a forced bwrap-setup
failure raises `SandboxConstructionError` naming bwrap rather than silently
returning a result that misattributes bwrap's exit code to the command.

**Not verified, and cannot be from inside this session:** the acceptance
test itself (`sb.run("echo x > <outside>")` failing to create the file) and
the 8 other behavioral parity tests (env scrub under real execution, timeout
tree-kill, `denied_paths` refusing a live shell write, output truncation).
All 9 are written, gated on `self_check().ok`, and will run the moment the
host allows it — nothing further to build.

Full repo suite: 4548 passed, 13 skipped (9 of those are this file; the rest
pre-exist and are unrelated).

## The fix, and why it stops here

The minimal, understood fix is a **system-level security-setting change** —
loosen or disable `kernel.apparmor_restrict_unprivileged_userns`, or add a
narrower AppArmor allowance scoped to `/usr/bin/bwrap` specifically. Either
requires root and trades away a real defense-in-depth measure (this sysctl
exists to blunt a documented unprivileged-userns CVE class) for the
capability this class needs to do its job. That trade is Will's to weigh and
apply, not something to change unilaterally mid-implementation — "modifying
system or security settings" is a bright line, not a judgment call.

Two candidate commands, for reference, **not run**:

```bash
# Broadest — every unprivileged-userns caller on the box, not just bwrap:
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
# persist: echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/60-bwrap.conf

# Narrower, more work to get right, smaller blast radius:
# an AppArmor profile addition scoped to /usr/bin/bwrap allowing `userns,`
```

Neither has been tested to confirm it clears **both** failures — the
sysctl is documented as gating failure 1; whether it also resolves failure 2
(the loopback EPERM, which read as a kernel-level netlink refusal rather
than an AppArmor block in the `strace` trace) is unconfirmed and should be
checked live once toggled, before assuming the class is unblocked.

## Recommendation

1. **Ship the code and tests as-is.** They are correct per bwrap's
   documented semantics, cost nothing while dormant, and the class is not
   wired as any call site's active backend — there is no regression surface.
2. **Will decides on the sysctl** (or the narrower AppArmor alternative),
   on his own machine, in his own time.
3. **The moment either is applied**, re-run `BwrapSandbox.self_check()` —
   if `ok=True`, `pytest tests/test_bwrap_sandbox.py` immediately exercises
   the acceptance test for real, no code changes needed.
