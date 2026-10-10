# Docker

The official container image runs the Prometheus daemon, its web API and the WebSocket bridge Beacon streams from, on Linux. It is built for `linux/amd64` and `linux/arm64` and published to the GitHub Container Registry as `ghcr.io/oaralabs/prometheus`, from the first release whose workflow includes it. Until that release is out, [build it yourself](#build-it-yourself) from a checkout; every command below works the same with a local tag.

[← README](../../README.md) · [Install & first flight](install.md)

What is in the image: the daemon with the Anthropic provider, MCP, the Slack, Discord and Telegram gateways and APNs push, plus `git` and `curl` for the agent's shell. Voice, the Playwright browser tool and computer use are not in it ([add them when you build](#build-it-yourself)).

## Quick start

```bash
docker run -d --name prometheus --restart unless-stopped \
  -p 127.0.0.1:8005:8005 -p 127.0.0.1:8010:8010 \
  -v prometheus-home:/home/prometheus \
  -v prometheus-data:/home/prometheus/.prometheus/data \
  ghcr.io/oaralabs/prometheus:latest
```

A new container has no configuration, so it starts in **setup mode** and prints a six-digit pairing code:

```bash
docker logs prometheus
```

Pair Beacon with this machine's address, port 8005, and that code, then follow its wizard ([the guided install](install.md#the-guided-install-couch-mode)); when it finishes, the same container carries on as the configured daemon. Two things differ in a container:

- The banner names the container's own hostname (a container ID), which nothing outside can reach. Use the host's address: `127.0.0.1:8005` on this machine, or its LAN or Tailscale address once you [publish further](#reaching-it-from-another-device).
- In the Model step, `localhost` is the container itself, so the auto-detected servers will be empty. Enter your inference server's address in the custom base URL field (`http://ollama:11434` in the compose setup below).

Or set it up from the command line. Point `oara setup` at your inference server (`--probe-url` takes any URL the container can reach), restart, and print the token Beacon or `curl` will need:

```bash
docker exec prometheus oara setup --noninteractive --probe-url http://192.168.1.20:11434
docker restart prometheus
docker exec prometheus oara token show
```

## With Ollama (compose)

[`packaging/docker/compose.yaml`](../../packaging/docker/compose.yaml) runs Prometheus beside Ollama. Ollama gets no published port; only Prometheus talks to it, at `http://ollama:11434`, on the project's own network.

```bash
docker compose -f compose.yaml up -d ollama
docker compose -f compose.yaml exec ollama ollama pull qwen3:8b
docker compose -f compose.yaml run --rm prometheus oara setup --noninteractive --probe-url http://ollama:11434
docker compose -f compose.yaml up -d
docker compose -f compose.yaml exec prometheus oara token show
```

Any Ollama model with tool calling works; `qwen3:8b` is just a common one. Setup probes the server, takes the first model it lists, and writes `~/.prometheus/prometheus.yaml` onto the home volume. For GPU acceleration, follow Ollama's Docker instructions for your GPU and add the device lines they give to the `ollama` service. Pin a Prometheus release with `PROMETHEUS_VERSION=<version>` (in your shell or a `.env` file beside `compose.yaml`) instead of `latest`.

## Where your data lives

The container runs as user **10001** with `HOME=/home/prometheus`, and everything Prometheus keeps is under that home:

| Volume | Mounted at | Holds |
|---|---|---|
| home | `/home/prometheus` | `~/.prometheus/prometheus.yaml`, identity (`SOUL.md`, `AGENTS.md`), memory and telemetry databases, the wiki, logs, the agent's workspace, and `~/.config/prometheus/env`: the API token and any gateway or provider keys |
| data | `/home/prometheus/.prometheus/data` | conversations (`lcm.db`), sessions, background tasks, paired devices |

Back both up. A named volume can be copied out with a throwaway container:

```bash
docker run --rm -v prometheus-home:/v --entrypoint tar \
  ghcr.io/oaralabs/prometheus:latest czf - -C /v . > prometheus-home.tgz
```

To use host directories instead of named volumes, create them owned by uid 10001 first, including the data volume's mount point inside the home directory (Docker would otherwise create that path as root, and the daemon could not write to `~/.prometheus`):

```bash
sudo install -d -o 10001 -g 10001 -m 700 \
  prometheus-home prometheus-home/.prometheus prometheus-home/.prometheus/data prometheus-data
```

## The API token

Every API call except pairing, `/api/hello` and `/health` needs `Authorization: Bearer <token>`. The token is never in the image; it comes from one of these, in this order (a `web.api_token` in `prometheus.yaml` would win over all three, so leave that unset):

1. **`PROMETHEUS_API_TOKEN_FILE`**: the path of a file holding the token, which is how a Docker or compose secret arrives (`/run/secrets/<name>`). The file must be readable by uid 10001 and must not be a symlink. If the variable is set and the file is missing or empty, the daemon **refuses to start** and says why in `docker logs`; it never falls back to another token.
2. **`PROMETHEUS_API_TOKEN`** in the container's environment (`-e` or compose `environment:`). Simple, but `docker inspect` shows it to anyone who can run that.
3. **Minted on first start** when neither is set, saved to `~/.config/prometheus/env` on the home volume, and printed once in the log. `oara token show` prints it again. A secret or a variable added later replaces it at the next start.

For a secret in compose, uncomment the three marked blocks in `compose.yaml`, then:

```bash
mkdir -m 700 secrets
openssl rand -base64 32 | tr -d '\n' > secrets/prometheus_api_token
chmod 644 secrets/prometheus_api_token   # readable by uid 10001; the 700 directory keeps others out
```

To rotate a minted token: `docker exec prometheus oara token rotate`, then `docker restart prometheus`. To rotate a token from the environment or a secret, change it there and restart; `oara token rotate` refuses a token that lives in a secret file, because the file would win at the next start.

## Reaching it from another device

The examples publish on `127.0.0.1`, so only this machine can connect. Beacon on another computer or a phone needs both ports published on an address it can reach (`-p 8005:8005 -p 8010:8010` for every interface). The API is plain HTTP with no TLS, and the token is the only lock on an agent that runs shell commands. Prefer a private network such as Tailscale or WireGuard, or a reverse proxy that terminates TLS, over a port open to a LAN you don't control.

Inside the container the daemon listens on all of the container's interfaces (`PROMETHEUS_WEB_BIND=0.0.0.0` is set in the image), so `-p` alone decides who can reach it. Telegram, Slack and Discord connect outbound; they need no published port.

## Health

The image's `HEALTHCHECK` runs `python -m prometheus.web.healthcheck`. It reports healthy while setup mode waits to be paired and once the daemon is configured (with `web.enabled: true`, which setup writes), and it needs no token:

```bash
docker inspect --format '{{.State.Health.Status}}' prometheus
```

## Security inside the container

- The daemon runs as uid 10001, never root. Its code sits in `/opt/prometheus`, owned by root, so the agent cannot rewrite what it runs. 10001 rather than 1000 because, on most Linux hosts, uid 1000 is the first human user, and a process that escaped the container as 1000 would own that person's files.
- **The bash floors do not apply inside a default container.** On a host install, the AppArmor read floor keeps model-written shells out of `~/.ssh`, `~/.gnupg` and credential env files, and the bubblewrap write floor keeps their writes inside the workspace. A container cannot load an AppArmor profile from inside, and bubblewrap is not in the image: Docker's default seccomp profile blocks the unprivileged user namespaces it needs. The daemon logs at boot that the read floor is not in force, and `/api/status` shows neither floor as active (`security.bash_read_floor` and `security.bash_write_floor`). Don't set `security.bash_confinement: required` here: it would refuse every shell command.
- So the container is the boundary. A model-written shell can read and change anything uid 10001 can (the two volumes, including the env file with the API token and your keys), read the rest of the image, and use the network. Don't mount host directories you wouldn't hand to the agent, and never mount the Docker socket.
- Coding runs with `coding.sandbox_type: docker` need a Docker engine the daemon can reach, which this image does not have. Use `process` inside the container.

## Upgrading

```bash
docker compose -f compose.yaml pull && docker compose -f compose.yaml up -d
```

Everything Prometheus keeps is on the volumes, so a new image starts with your config, memory and token. Pin `PROMETHEUS_VERSION` and change it on purpose if you would rather not follow `latest`.

## Build it yourself

From a checkout:

```bash
docker build -t ghcr.io/oaralabs/prometheus:local .
PROMETHEUS_VERSION=local docker compose -f packaging/docker/compose.yaml up -d
```

Other extras: `--build-arg PROMETHEUS_EXTRAS="anthropic mcp slack discord push voice"` (every locked wheel exists for both architectures, but voice also needs audio devices, which a container rarely has). Both architectures at once: `docker buildx build --platform linux/amd64,linux/arm64 .`. Every dependency installs from `uv.lock` by hash, as prebuilt wheels, so an arm64 build compiles nothing.

`packaging/docker/smoke_image.py <image>` is the check CI runs on both architectures: it pairs a fresh container, sets it up against a stub model, takes one agent turn through it, and tries every way the token can arrive.

## Building an image on this one

Everything the daemon needs is in `ENV` or derived from `HOME`, and nothing depends on the `ENTRYPOINT`, so a runtime that replaces the start command still gets the same daemon from `oara daemon`. Set `HOME` to move all state; the health probe is `python -m prometheus.web.healthcheck`; the listen address is `PROMETHEUS_WEB_BIND`.
