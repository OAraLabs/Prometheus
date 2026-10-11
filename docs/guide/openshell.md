# OpenShell sandbox

Prometheus can run inside an OpenShell sandbox: its own container, default-deny network egress with per-host rules, and a filesystem confined to its workspace. This page covers the sandbox image, the policy, how to start and reach it, and exactly what has been tested.

[← README](../../README.md) · [Docker](docker.md)

**Tested with OpenShell 0.1.3 on Linux amd64** (Ubuntu, kernel 7.0, Docker Engine 29.2.1, the Docker driver), on 2026-10-11, with `packaging/openshell/smoke_openshell.py --openshell`. See [What was tested](#what-was-tested) and [Not tested](#not-tested).

## What you need

- OpenShell 0.1.x with a gateway using the Docker driver, installed the way its own documentation says. It needs Docker Engine 28 or later and a Linux kernel with Landlock.
- The sandbox image, built from this repository (below). It is a thin layer on the [Docker image](docker.md).

## Build the sandbox image

[`packaging/openshell/Dockerfile`](../../packaging/openshell/Dockerfile) adds what a sandbox needs to the official image:
- `/sandbox` as the workspace, owned by the image's user (10001) and set as `HOME`, so the config, memory, logs and the API token all live in the one writable place;
- `iproute2`.

```bash
docker build -f packaging/openshell/Dockerfile -t prometheus-sandbox .
```

It builds on `ghcr.io/oaralabs/prometheus:latest`. Until that image is published, build the official image first and pass it in: `docker build -t prometheus . && docker build -f packaging/openshell/Dockerfile --build-arg BASE_IMAGE=prometheus -t prometheus-sandbox .`

## The policy

[`packaging/openshell/policy.yaml`](../../packaging/openshell/policy.yaml) denies everything it does not list. It allows:

| Rule | Hosts |
|---|---|
| `model_server_on_host` | `host.openshell.internal:11434` (Ollama) and `:8080` (llama.cpp) on the machine running OpenShell |
| `model_providers` | the cloud providers Prometheus ships presets for: Anthropic, OpenAI, Gemini, xAI, DeepSeek, Kimi, GLM, MiMo, Qwen |
| `telegram` | `api.telegram.org` |
| `slack` | `slack.com`, and the Socket Mode WebSocket hosts `wss-primary.slack.com` and `wss-backup.slack.com` |
| `discord` | `discord.com`, the gateway WebSocket (`gateway.discord.gg`, `*.discord.gg`) and `cdn.discordapp.com` |

Delete the rules you don't use. Every rule names one binary: the image's interpreter, `/usr/local/bin/python3.12`, by its real path, because OpenShell matches the executable the kernel reports and `/opt/prometheus/bin/python` is a symlink to it. Other binaries in the sandbox, such as `curl`, reach nothing. Network rules reload live with `openshell policy set`; the filesystem section (writable: `/sandbox`, `/tmp`) needs a new sandbox.

Inside the sandbox a model server on the host is `host.openshell.internal`, never `localhost`.

## Start it

```bash
openshell sandbox create --name prometheus --from prometheus-sandbox \
  --policy packaging/openshell/policy.yaml --restart-policy on-failure -- /bin/sh -c \
  'test -f "$HOME/.prometheus/prometheus.yaml" || oara setup --noninteractive --probe-url http://host.openshell.internal:11434; exec oara daemon'
```

OpenShell replaces the image's start command with the one after `--`. This one sets Prometheus up against the model server on the host the first time, and only then (a restart keeps the config), and runs the daemon. Change the `--probe-url` to your server, or set `PROMETHEUS_OLLAMA_URL` / `PROMETHEUS_LLAMA_CPP_URL` with `--env` on an existing config. `/sandbox` lives in the sandbox's container: it survives `openshell sandbox stop` and `start`, and is deleted with the sandbox.

## Reach it

Everything inside a sandbox listens on its loopback only. Forward the API (8005) and the WebSocket bridge Beacon streams from (8010) to this machine's loopback:

```bash
openshell forward start 8005 prometheus -d
openshell forward start 8010 prometheus -d
openshell sandbox exec -n prometheus -- oara token show
```

Then connect Beacon on this machine to `127.0.0.1:8005` with that token, or call the API with `Authorization: Bearer <token>`. Stop the forwards with `openshell forward stop 8005 prometheus` (and 8010).

## Secrets

Gateway tokens and provider keys can be passed with `--env NAME=value` at create time; they are then ordinary variables inside the sandbox. OpenShell can instead keep them outside the sandbox and substitute them on the way out (its providers); that path has not been tested with Prometheus. The API token authenticates requests coming *in*, so it cannot be a placeholder: it is minted into `/sandbox/.config/prometheus/env`, or supplied with `--env PROMETHEUS_API_TOKEN=…`.

## Inside the sandbox

- **The bash floors are off by circumstance.** The AppArmor read floor and the bubblewrap write floor cannot run in a sandbox: no AppArmor profile can be loaded from inside, and bubblewrap is not in the image (OpenShell's seccomp filter refuses the user namespaces it needs). The daemon says so at boot, and `/api/status` shows both as `dark`. OpenShell's own Landlock filesystem rules and network policy are the boundary instead. Don't set `security.bash_confinement: required`: it would refuse every shell command.
- **`web_fetch`** resolves a name itself and connects to the address (its SSRF guard), so a host rule does not match it and the policy refuses it. `web_search` is not in the policy either.

## What was tested

`packaging/openshell/smoke_openshell.py --openshell` against OpenShell 0.1.3 (CLI and gateway from the release tarballs, checksums verified; a plaintext loopback gateway, the Docker driver), Linux amd64, 2026-10-11:

1. A stub model server ran on the host; the sandbox was created from the sandbox image with `policy.yaml` plus one rule for the stub's port, and the start command above.
2. `oara setup` inside the sandbox reached the stub as `host.openshell.internal`, and the daemon came up healthy.
3. `openshell forward` brought 8005 and 8010 to the host's loopback. Through them, `/health` answered, `/api/status` refused a request without the token and accepted the minted one, and both bash floors reported `dark`.
4. A REST agent turn through the forward completed a model→tool→model round trip against the model on the host.
5. Egress from the daemon's interpreter: Telegram's Bot API, Slack's `api.test` and Discord's REST API answered over HTTPS, through the sandbox's TLS inspection, and Discord's gateway WebSocket sent its HELLO. A host not in the policy (`example.com`) was refused, and so was `curl` to an allowed host.
6. The filesystem: `/sandbox` writable; `/opt/prometheus`, `/home/prometheus` and `/etc` not.
7. `openshell sandbox stop` and `start`: the start command found the config and did not run setup again, and the API token was unchanged.

CI builds the sandbox image on amd64 and arm64 and runs the same script with `--docker`: the image under plain Docker, with the daemon's state under `/sandbox`.

## Not tested

- arm64 inside OpenShell (CI runs the image on arm64 under plain Docker only), macOS, WSL, and the Podman, Kubernetes and MicroVM drivers.
- Live Telegram, Slack and Discord bots: only that their hosts answer the daemon's interpreter. Slack's Socket Mode WebSocket needs an app token and was not opened.
- Calls to the cloud model providers: their hosts are in the policy, but none was called.
- OpenShell's credential substitution (providers and placeholders) with Prometheus.
- Beacon on another device: forwards bind this machine's loopback; OpenShell's gateway URLs (`openshell service expose`) were not tried.
- OpenShell 0.0.x, including the `inference.local` endpoint it had. This page is for 0.1.x.
- Discovery on the local network (mDNS). It is not expected to work: the sandbox's sockets are bound to its loopback.
