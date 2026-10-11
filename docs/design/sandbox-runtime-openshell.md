# Running Prometheus inside an OpenShell sandbox

**Status: Stage 1 shipped (2026-10-11).** Will moved it ahead of v0.4. The sandbox image, the policy, the
smoke test and the guide are in `packaging/openshell/` and docs/guide/openshell.md, tested with OpenShell 0.1.3
on Linux amd64; the guide lists exactly what was and was not tested. Stages 2 and 3 below are still open. The
rest of this note is the research as of 2026-10-10.

## For Will (one page)

**Can a third-party agent be onboarded?** Not through the managed onboarding layer (NemoClaw). It has no
extension SDK, its `onboard --agent` picks only from agent manifests it ships, and its community repo
limits contributed sandboxes to four named harnesses. A new agent starts as an upstream Discussion. The
route that works is plain OpenShell: our own image, started with `openshell sandbox create --from
<image> -- oara daemon`, plus a network policy we write.

**Does OpenShell need Docker?** It needs a runtime driver: Docker Engine/Desktop 28+, Podman 5, Kubernetes
1.29+ (with its Agent Sandbox controller), or a libkrun MicroVM. Linux amd64 and arm64 are supported,
macOS through Docker Desktop, WSL 2 experimentally. OpenShell 0.1.x requires Landlock ABI 3 (Linux 6.2+).

**Licence and maturity.** OpenShell and NemoClaw are both Apache-2.0, which grants no trademark rights.
NemoClaw calls itself alpha and an early preview, with production use unsupported. OpenShell releases are
dev, pre-release and stable, weekly, and experimental interfaces can change in a patch release. NemoClaw
pins OpenShell 0.0.116 while OpenShell's own docs are at 0.1.3, and the two differ in how model calls
are routed (below).

**Feasibility.** Feasible as a bring-your-own image. The blockers are inbound access for Beacon, our
shell floors, two egress details, and the version split. Effort and stages are at the end.

## How a sandboxed agent is set up

- **Image.** Any standard Linux OCI image, not distroless or `FROM scratch`. It must declare a non-root
  `USER`, have `/sandbox` writable by that user, and carry `iproute2`. The image's `ENTRYPOINT` and `CMD`
  are replaced; the start command goes after `--`. On Docker and Podman the image's `WORKDIR` becomes the
  workspace (default `/sandbox`). Root is rejected.
- **Blueprint YAML** is a NemoClaw construct (image digest, agent manifest, network policy, inference
  profile, version bounds). Plain OpenShell has no blueprint; a sandbox is an image plus a policy.
- **Credentials.** The workload never sees real secrets: its environment holds placeholders, and the
  egress proxy substitutes the real value after the policy and endpoint binding allow the request.
  Substitution works in headers, Basic auth, query strings and URL path segments. Request bodies and
  WebSocket text are opt-in. An unresolved placeholder fails closed.
- **Model routing.** In 0.0.116, `https://inference.local/v1` is a sandbox-local endpoint the gateway
  routes to one configured provider (OpenAI-compatible paths, Anthropic `/v1/messages`). **0.1.0 removed
  it**: the workload calls the provider's own endpoint with a placeholder key. Servers on the host are
  reached at `host.openshell.internal`, not `localhost`.
- **Network policy.** Egress is denied by default. A rule pairs endpoints (host with wildcards, ports,
  allowed IPs, a protocol of rest/websocket/graphql/mcp/json-rpc/tcp, method/path/query rules) with the
  binaries allowed to use them, matched on the real executable path. Network rules reload live; filesystem
  and process settings need a new sandbox.
- **Inbound.** Every workload socket is bound to loopback. The ways in are `openshell forward start
  <port>` (an SSH tunnel) and `openshell service expose` (a gateway URL). Plain HTTP works from loopback
  clients only; LAN access needs the gateway's HTTPS listener with a configured SAN.
- **Process and filesystem.** Landlock for the filesystem; seccomp plus a privilege drop for processes,
  with `no_new_privs`, no capabilities, and `mount`, `pivot_root`, `setns`, `ptrace` and user-namespace
  creation (`unshare`, `clone` with `CLONE_NEWUSER`) refused.

**The reference agents.** Hermes ships a custom image (a slim Node base with `sandbox` and `gateway`
users), a manifest naming its gateway command, health URL, forwarded ports and config directory under
`/sandbox`, a policy addition for its model and package egress, and per-agent Telegram, Slack and Discord
policy files. Deep Agents Code is terminal-only: a pip-installed CLI, strict Landlock, egress to the
model endpoint, GitHub and PyPI, and a note that its traffic is attributed to the Python interpreter.

## What Prometheus needs

| Need | Fit | Work |
|---|---|---|
| Telegram | Long polling to `api.telegram.org`; the bot token sits in the URL path, which placeholders cover. | Adapt Hermes's Telegram policy. |
| Slack | Socket Mode: `slack.com` API, then WSS URLs Slack hands back (`wss-primary`/`wss-backup.slack.com`). Tokens in headers. | Adapt Hermes's Slack policy, including its WebSocket credential rewrite. |
| Discord | `discord.com` REST and the `gateway.discord.gg` WebSocket. The bot token is sent inside the IDENTIFY frame, which is WebSocket text, opt-in for substitution. | Verify the opt-in works for IDENTIFY. |
| Beacon (REST 8005, WS 8010) | Loopback only inside the sandbox. Same machine: a forward per port. Another device: a gateway URL over HTTPS. Beacon pairs to `host:8005`/`8010` directly today. | **The main blocker.** Beacon-side support for a gateway URL, or a documented forward. |
| Model providers | Every provider takes a base URL. 0.1.x: the real endpoint with a placeholder key. Ollama or llama.cpp on the host: `host.openshell.internal`. | Policy entries per provider. `model.base_url`, or `PROMETHEUS_OLLAMA_URL` / `PROMETHEUS_LLAMA_CPP_URL` for the main model. |
| The API token | It authenticates **inbound** requests, so it cannot be a placeholder: it must be real inside the sandbox, minted into `$HOME/.config/prometheus/env` or mounted with `PROMETHEUS_API_TOKEN_FILE`. The agent's own shell can read it, as in Docker. | None beyond the image. |
| AppArmor read floor | A profile cannot be loaded from inside (no capabilities). | Decide the posture (below). |
| bubblewrap write floor | Needs user namespaces and `mount`, both refused by the seccomp filter. | Decide the posture (below). |
| `web_fetch` | Resolves DNS itself and connects to the IP (the SSRF guard), so a host-based allowlist refuses it. | Connect by name through the proxy inside a sandbox, or list `allowed_ips`. |
| Background `git fetch` | `deployment.origin_fetch` runs every 300 s. An installed image is not a checkout, so it should find nothing to fetch (not yet checked). | Confirm it stays quiet in Stage 1. |
| Per-binary rules | Almost all our egress is the Python interpreter (the rest is `git` and `curl` in the agent's shell), so a rule cannot tell a gateway from model-written Python. | Accept; scope by host instead. |

**The floors.** Under the shipped `auto` they degrade with an ERROR at boot, and `/api/status` reports
them `dark`; no conflict, just noise. Under `required` every shell command would be refused. They do
not add to the sandbox: Landlock's read-only root with writable `/sandbox` and `/tmp` already does what
the write floor does, and denying `~/.ssh` and friends is a Landlock rule. The honest Stage 3 change is a
floor state that says "provided by the outer sandbox" when Prometheus can see it is inside one, instead
of `dark`.

## Stages, effort, and what we can say

| Stage | Work | Effort | What we can honestly claim after it |
|---|---|---|---|
| 0 (#714) | The container image: non-root, `ENV`-only configuration, `HOME`-relative state, a token from a secret file, a health probe. | done, pending merge | Before the first release that publishes it: "a Dockerfile in the repo". After: "an official container image for Linux amd64 and arm64". Nothing about OpenShell. |
| 1 (after v0.4) | A thin image on #714's (`iproute2`, `/sandbox`, `HOME=/sandbox`, `WORKDIR /sandbox`), a policy file for one provider and one gateway, a guide section; run on the 4090 box with OpenShell 0.1.x. | 2–3 days | "Runs in an OpenShell sandbox (tested with OpenShell 0.1.x on Linux amd64; example policy included)". Version-specific, factual, no logos. |
| 2 | Beacon through a forward on the same machine, then through a gateway URL. | 3–5 days, plus Beacon work not yet scoped | Add "Beacon connects", with the method named. |
| 3 | The floor state for an outer sandbox, `web_fetch` by name, Discord IDENTIFY. | 2–3 days | Drop the caveats those cover. |
| Managed onboarding | An upstream Discussion; acceptance is theirs. | not ours to estimate | Nothing until accepted. |

**Never**, unless upstream accepts it: "supported by", "certified", "official integration", "partner",
or any of the vendor's logos or marks. Apache-2.0 §6 grants no trademark rights.

## Sources (read 2026-10-10)

- OpenShell docs: `docs.nvidia.com/openshell/` (latest, 0.1.x) and `docs.nvidia.com/openshell/v0.0.116/`:
  support matrix, sandboxes (overview, runtimes), providers, inference and inference routing, policies
  (schema, network rules, managing policies), security best practices, architecture, gateways, the 0.1.0
  upgrade notes, licence.
- NemoClaw docs: `docs.nvidia.com/nemoclaw/`: how it works, platform support, prerequisites, commands,
  the extension and SDK readiness page, community contributions, licence.
- Repositories, at the commits read: NVIDIA/NemoClaw `2bbb4d1e` (`nemoclaw-blueprint/blueprint.yaml`,
  `agents/hermes/`, `agents/langchain-deepagents-code/`, `src/lib/messaging/channels/*/policy/`),
  NVIDIA/OpenShell `eeba0e79` (`examples/bring-your-own-container`), NVIDIA/OpenShell-Community
  `edb65583` (archived).
- Not verified: separate licence terms for the vendor's published images, and AppArmor's behaviour
  inside a sandbox.
