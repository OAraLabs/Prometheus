# The Prometheus image, shaped for an OpenShell sandbox. Guide: docs/guide/openshell.md.
#
#   docker build -f packaging/openshell/Dockerfile -t prometheus-sandbox .
#   openshell sandbox create --name prometheus --from prometheus-sandbox \
#     --policy packaging/openshell/policy.yaml -- oara daemon
#
# A thin layer, because the official image (the root Dockerfile) already keeps every setting in ENV and
# every path under HOME, and never relies on its ENTRYPOINT: the sandbox replaces ENTRYPOINT and CMD with
# its own supervisor and runs the command given after `--`. What a sandbox needs on top:
#
#   /sandbox    the workspace, writable by the image's user. It becomes HOME too, so the config,
#               memory, logs and ~/.config/prometheus/env (the API token) live in the one place the
#               filesystem policy makes writable.
#   iproute2    for the sandbox's network namespace isolation.
#
# The user stays the official image's 10001:10001, numeric and non-root, which the sandbox requires.
# BASE_IMAGE is a released tag; to build on an image you built yourself, pass --build-arg BASE_IMAGE=<tag>.

ARG BASE_IMAGE=ghcr.io/oaralabs/prometheus:latest
FROM ${BASE_IMAGE}

USER root
RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends iproute2; \
    rm -rf /var/lib/apt/lists/*; \
    install -d -o 10001 -g 10001 -m 0700 /sandbox

ENV HOME=/sandbox
WORKDIR /sandbox
USER 10001:10001
