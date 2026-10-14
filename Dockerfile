# The official Prometheus image: the daemon, its web API and the Beacon bridge, on Linux amd64 and arm64.
#
#   docker run -d --name prometheus \
#     -p 127.0.0.1:8005:8005 -p 127.0.0.1:8010:8010 \
#     -v prometheus-home:/home/prometheus \
#     -v prometheus-data:/home/prometheus/.prometheus/data \
#     ghcr.io/oaralabs/prometheus:<version>
#
# Guide: docs/guide/docker.md. Compose example with Ollama: packaging/docker/compose.yaml.
# The container smoke test CI and the release run against this image: packaging/docker/smoke_image.py.
#
# ── WHAT IS NEVER IN THE IMAGE ──────────────────────────────────────────────
#
# A token, a key, an env file or a prometheus.yaml. The web API token arrives at run time
# (PROMETHEUS_API_TOKEN, or PROMETHEUS_API_TOKEN_FILE pointing at a secret), or the daemon mints one on
# first start into ~/.config/prometheus/env, which lives on the home volume. .dockerignore is an
# ALLOWLIST, so a developer's own config/prometheus.yaml or .env cannot reach the build context at all.
#
# ── NOTHING RESOLVES FRESH ──────────────────────────────────────────────────
#
# The base image is pinned by digest. uv comes from the same hashed pins as the macOS release
# (packaging/macos/release-tools.txt), the wheel's build backend from packaging/macos/build-constraints.txt,
# and every dependency from uv.lock, installed with --require-hashes --only-binary :all: (nothing compiles
# here, which is also what makes the arm64 build identical in kind to the amd64 one). The one thing taken
# as current is Debian's package archive for the four OS packages below, so their security fixes land.
# Dependabot refreshes the base digest (.github/dependabot.yml), which is why both FROM lines spell it out:
# it does not follow a build ARG. Keep the two lines identical.
#
# ── CONFIGURATION LIVES IN ENV, NOT IN AN ENTRYPOINT SCRIPT ─────────────────
#
# Some sandbox runtimes replace a container's ENTRYPOINT and CMD with their own start command. Everything
# the daemon needs to run correctly here is therefore an ENV line or the code itself, so an image built
# FROM this one gets the same daemon however it is started. tini is only PID 1's job: reap the orphans of
# model-written shells and pass SIGTERM on.

FROM python:3.12-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS build

# The gateway and provider extras a server wants. Voice (onnxruntime), the browser (Playwright and its
# browsers) and computer use (a desktop driver) are left out: build with --build-arg to add them. So is
# discovery (mDNS): its announcements do not cross Docker's default bridge network.
ARG PROMETHEUS_EXTRAS="mcp slack discord push"

ENV UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY packaging/macos/release-tools.txt packaging/macos/build-constraints.txt /build/pins/
RUN python -m pip install --no-cache-dir --require-hashes --only-binary :all: -r /build/pins/release-tools.txt

WORKDIR /build/src
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY config/prometheus.yaml.default config/model_registry.yaml ./config/
COPY templates/ ./templates/
COPY src/ ./src/

# --no-config on `uv pip`: it would read [tool.uv] from pyproject.toml, whose override-dependencies
# --require-hashes refuses as unpinned. The exported file is the whole install (as in build_app.py).
RUN set -eu; \
    extras=""; for e in $PROMETHEUS_EXTRAS; do extras="$extras --extra $e"; done; \
    uv export --frozen --no-dev --no-emit-project $extras -o /build/requirements.txt; \
    uv venv --python /usr/local/bin/python3.12 /opt/prometheus; \
    uv pip install --no-config --python /opt/prometheus/bin/python --no-deps --require-hashes --only-binary :all: --compile-bytecode -r /build/requirements.txt; \
    uv build --wheel --out-dir /build/dist --build-constraints /build/pins/build-constraints.txt --require-hashes; \
    uv pip install --no-config --python /opt/prometheus/bin/python --no-deps --compile-bytecode /build/dist/*.whl; \
    /opt/prometheus/bin/oara --version


FROM python:3.12-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

LABEL org.opencontainers.image.title="Prometheus" \
      org.opencontainers.image.description="An AI agent daemon that runs on hardware you own." \
      org.opencontainers.image.url="https://oara.ai/prometheus" \
      org.opencontainers.image.source="https://github.com/OAraLabs/Prometheus" \
      org.opencontainers.image.licenses="MIT"

# tini: PID 1. git and curl: what an agent's shell reaches for first, and coding runs need git.
#
# The user is 10001, not 1000: a container escape as uid 1000 owns the files of most Linux hosts' first
# user. The two volume mount points are created here, owned by that user, so named volumes start out
# writable; a bind mount must be owned by 10001 (docs/guide/docker.md).
RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends tini git curl ca-certificates; \
    rm -rf /var/lib/apt/lists/*; \
    groupadd --gid 10001 prometheus; \
    useradd --uid 10001 --gid 10001 --home-dir /home/prometheus --create-home --shell /bin/bash prometheus; \
    install -d -o 10001 -g 10001 -m 0700 \
        /home/prometheus/.prometheus /home/prometheus/.prometheus/data \
        /home/prometheus/.config /home/prometheus/.config/prometheus

# Root-owned: the agent runs as 10001 and cannot rewrite the code it runs.
COPY --from=build /opt/prometheus /opt/prometheus

# HOME decides every state path: ~/.prometheus (config, identity, memory, logs, workspace), its data/
# directory (conversations, sessions, devices) and ~/.config/prometheus/env (the token and any gateway or
# provider keys, where the file tools' hard floor refuses to read). An image built on this one moves all
# of it by setting HOME.
#
# PROMETHEUS_WEB_BIND=0.0.0.0: listen on the container's own interfaces and let `-p` decide who can reach
# them. Without it a config written by setup would say 127.0.0.1 and every published port would be dead.
ENV HOME=/home/prometheus \
    PATH=/opt/prometheus/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PROMETHEUS_WEB_BIND=0.0.0.0

USER 10001:10001
WORKDIR /home/prometheus
VOLUME ["/home/prometheus", "/home/prometheus/.prometheus/data"]

# 8005: REST API (pairing, setup, /health). 8010: the WebSocket bridge Beacon streams from.
EXPOSE 8005 8010

# Healthy in setup mode (waiting to be paired) and once configured; no token needed for either.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --start-interval=2s --retries=3 \
    CMD ["/opt/prometheus/bin/python", "-m", "prometheus.web.healthcheck"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["oara", "daemon"]
