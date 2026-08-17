# (a′) Kernel-sandbox scoping — how far the existing foundation reaches

Survey only. The question: `permissions/sandbox.py` exists and coding runs
use a sandboxed clone — is that a foundation for real containment, or a name?

**Answer: one of the two is a foundation, the other is a name — and the real
one has the SAME bash gap as the main path.** Which makes (a′) the answer in
two places, not one.

## Two things called "sandbox", and they are not alike

### `permissions/sandbox.py` — `SandboxedExecution`

Not a sandbox in any kernel sense: `asyncio.create_subprocess_exec` with a
scrubbed environment, a cwd set to `workspace_root`, a timeout and an output
cap. No namespace, no seccomp, no chroot, **no path enforcement at all**.

And — **it is not wired.** Every reference outside its own module is the
`__init__.py` export and its own tests. Nothing in `src/` constructs it.
That is the fourth instance of Standing-Principles §2e found in one day, and
it was found by applying §2e minutes after writing it.

So the "foundation" this option was hoped to build on is a class nothing
calls, whose name promises more than its body delivers.

### `coding/sandbox.py` — `ProcessSandbox`

A real jail, and **honestly labelled** — its own docstring says *"This is
process-level hygiene, not a container boundary — the honest limit of
ProcessSandbox, stated here so nobody mistakes it for one."*

What it genuinely does: a **dedicated full `git clone`** as the jail root
(spec chose hard isolation over disk savings), an allowlisted environment so
daemon secrets never reach the child, and **path containment with
`resolve()` called BEFORE the check**, so a symlink inside the clone pointing
outside it is refused rather than followed.

Verified by outcome, not by reading:

```
sb.resolve("<outside the jail>")        -> refused (SandboxViolation)   ✓
sb.run("echo -n escaped > <outside>")   -> rc=0, file created outside    ✗
```

**The containment guards the FILE TOOLS. The shell is `create_subprocess_shell(cmd, cwd=root)` — cwd only.** A redirect to an absolute path escapes the jail exactly as it escapes the workspace boundary on the main path.

This is not a criticism of the design: its docstring says it is not a
container. But it means the coding sandbox is *the same shape* as everything
else — tool-layer containment, shell-shaped hole — and the bash-boundary
survey's conclusion applies to it unchanged.

## What (a′) would actually be

Replace the *process* backend with a *kernel* one, in the place the interface
already anticipates: `coding/sandbox.py` names `DockerSandbox` as
"interface-shaped future work; this module is the interface it will
implement." That is the single best thing found in this survey — the seam
exists and was designed for exactly this.

Mechanism options, cheapest first:

| Approach | Gets you | Cost |
|---|---|---|
| `bwrap` (bubblewrap) | read-only binds outside the jail, tmpfs elsewhere; unprivileged; ~1 binary | least invasive; needs per-run policy and a fallback when absent |
| `unshare` + mount ns | same idea, no dependency | more plumbing, more failure modes |
| container (`DockerSandbox`) | strongest, image/network control too | daemon dependency, image lifecycle, slowest start |

All three make the shell question moot: the kernel decides, so it no longer
matters which program writes or how the path was constructed.

## Scope, honestly

**Two consumers, not one.** Coding runs (via the existing interface) and the
main agent loop's `bash`. The first is a backend swap behind a designed seam.
The second is not — the main loop's bash has no jail root to speak of, and
giving it one is a decision about what the agent is *for*: `~/projects`,
`~/.prometheus` and `/tmp` are legitimate targets today, and a namespace that
permits all three plus the system read-only is close to no confinement at all
for anything except the two Prometheus checkouts.

That is the real finding for scoping: **for coding runs a kernel sandbox is a
well-shaped, bounded piece of work. For the main loop it is a product
question first** — what should an agent with a shell be allowed to touch —
and only then an implementation.

## Recommendation

1. **Fix the labelling now, separately and cheaply**: `permissions/sandbox.py`
   is unwired and misnamed. Either delete it (it is dead) or rename it to what
   it is (`ScrubbedSubprocess`). Leaving a class called `SandboxedExecution`
   in `permissions/` invites exactly the assumption this survey started from.
2. **Coding runs**: a `BwrapSandbox` behind the existing interface is the
   tractable, bounded version of (a′). Its acceptance test is the two lines
   above — the shell redirect must fail.
3. **Main-loop bash**: do not scope an implementation yet. The prior question
   is which directories the agent is permitted to touch at all, and the
   current answer (three broad roots) does not leave a namespace much to
   enforce.
