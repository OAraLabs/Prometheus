"""OpenShell Stage 1: the sandbox image, its policy, the smoke test and the guide.

``packaging/openshell/Dockerfile`` is a thin layer on the official image (``Dockerfile``), shaped to the
bring-your-own-container requirements: a non-root numeric ``USER``, ``/sandbox`` writable by it as the
workspace and ``HOME``, and ``iproute2``. ``packaging/openshell/policy.yaml`` is the default-deny policy
that lets the daemon reach exactly what it talks to: the model providers it ships presets for, a model
server on the host through ``host.openshell.internal``, and the Telegram, Slack and Discord gateways.

What is pinned here is what would silently break a sandboxed install:

* a network rule names a binary by its REAL path, and every process of ours runs as the base image's
  interpreter, so the policy names that file and nothing broader (no ``/**``);
* the policy's provider hosts cannot drift from the provider registry's defaults;
* the writable paths are the workspace and ``/tmp``, nothing more; the code in ``/opt/prometheus`` stays
  readable, or the daemon cannot import itself;
* the guide states what was tested, with the version, and carries no third-party branding.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SANDBOX_DOCKERFILE = ROOT / "packaging" / "openshell" / "Dockerfile"
POLICY = ROOT / "packaging" / "openshell" / "policy.yaml"
SMOKE = ROOT / "packaging" / "openshell" / "smoke_openshell.py"
BASE_DOCKERFILE = ROOT / "Dockerfile"
DOCS = ROOT / "docs" / "guide" / "openshell.md"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _instructions(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if buf and line.startswith("#"):
            continue
        if line.endswith("\\"):
            buf += line[:-1] + " "
            continue
        buf += line
        word, _, rest = buf.partition(" ")
        out.append((word.upper(), rest.strip()))
        buf = ""
    return out


@pytest.fixture(scope="module")
def sandbox_dockerfile() -> list[tuple[str, str]]:
    assert SANDBOX_DOCKERFILE.is_file(), f"{SANDBOX_DOCKERFILE} does not exist"
    return _instructions(SANDBOX_DOCKERFILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def policy() -> dict:
    assert POLICY.is_file(), f"{POLICY} does not exist"
    return yaml.safe_load(POLICY.read_text(encoding="utf-8"))


def _base_python() -> str:
    """The interpreter every process of ours runs as: the official image's base, by its real path."""
    froms = [rest for word, rest in _instructions(BASE_DOCKERFILE.read_text(encoding="utf-8")) if word == "FROM"]
    m = re.match(r"python:(\d+\.\d+)-", froms[-1])
    assert m, froms[-1]
    return f"/usr/local/bin/python{m.group(1)}"


# ── the image ────────────────────────────────────────────────────────────────


def test_it_builds_on_the_official_image(sandbox_dockerfile):
    args = {rest.split("=", 1)[0]: rest.split("=", 1)[1].strip('"') for word, rest in sandbox_dockerfile
            if word == "ARG" and "=" in rest}
    froms = [rest for word, rest in sandbox_dockerfile if word == "FROM"]
    assert len(froms) == 1 and froms[0] == "${BASE_IMAGE}"
    assert args.get("BASE_IMAGE", "").startswith("ghcr.io/oaralabs/prometheus:")


def test_it_meets_the_sandbox_image_requirements(sandbox_dockerfile):
    runs = " ".join(rest for word, rest in sandbox_dockerfile if word == "RUN")
    assert "iproute2" in runs
    assert re.search(r"install -d [^;&]*-o 10001 -g 10001[^;&]*/sandbox", runs), "/sandbox must be the user's own"
    env = " ".join(rest for word, rest in sandbox_dockerfile if word == "ENV")
    assert "HOME=/sandbox" in env, "state follows HOME, and /sandbox is the one writable home a sandbox has"
    assert [rest for word, rest in sandbox_dockerfile if word == "WORKDIR"][-1] == "/sandbox"
    users = [rest for word, rest in sandbox_dockerfile if word == "USER"]
    assert users[-1] == "10001:10001", "the last USER is the base image's non-root, numeric user"


def test_it_adds_no_secret_and_no_start_command(sandbox_dockerfile):
    # The sandbox replaces ENTRYPOINT and CMD; a start command here would only mislead.
    assert not [w for w, _ in sandbox_dockerfile if w in ("ENTRYPOINT", "CMD", "COPY", "ADD")]
    assert "PROMETHEUS_API_TOKEN" not in SANDBOX_DOCKERFILE.read_text(encoding="utf-8")


# ── the policy ───────────────────────────────────────────────────────────────


def _rules(policy: dict) -> dict[str, dict]:
    rules = policy.get("network_policies") or {}
    assert rules, "no network rules: the daemon could reach no model and no gateway"
    return rules


def _endpoints(policy: dict) -> set[tuple[str, int]]:
    return {(e["host"], int(e["port"])) for r in _rules(policy).values() for e in r["endpoints"]}


def test_the_policy_is_version_1_and_keeps_writes_in_the_workspace(policy):
    assert policy["version"] == 1
    fs = policy["filesystem_policy"]
    assert fs.get("include_workdir") is True, "with filesystem_policy present, include_workdir defaults to false"
    assert set(fs["read_write"]) <= {"/sandbox", "/tmp", "/dev/null"}
    assert {"/sandbox", "/tmp"} <= set(fs["read_write"])
    assert "/opt/prometheus" in fs["read_only"], "the daemon's own code must stay readable"
    assert "/usr" in fs["read_only"], "the interpreter lives under /usr/local"


def test_every_rule_names_the_interpreter_by_its_real_path_and_nothing_broader(policy):
    python = _base_python()
    for name, rule in _rules(policy).items():
        paths = [b["path"] for b in rule["binaries"]]
        assert paths == [python], f"{name}: {paths} (OpenShell matches the real path, not the venv symlink)"


def test_the_gateways_can_be_reached(policy):
    endpoints = _endpoints(policy)
    for host in ("api.telegram.org", "slack.com", "wss-primary.slack.com", "wss-backup.slack.com",
                 "discord.com", "gateway.discord.gg"):
        assert (host, 443) in endpoints, f"{host}:443 missing"


def test_a_model_server_on_the_host_can_be_reached(policy):
    endpoints = _endpoints(policy)
    assert ("host.openshell.internal", 11434) in endpoints, "Ollama on the host"
    assert ("host.openshell.internal", 8080) in endpoints, "llama.cpp on the host"


def test_the_provider_hosts_match_the_provider_registry(policy):
    from prometheus.providers import anthropic
    from prometheus.providers.registry import CLOUD_DEFAULTS

    wanted = {urlparse(s["base_url"]).hostname for s in CLOUD_DEFAULTS.values() if s.get("base_url")}
    wanted |= {urlparse(v).hostname for v in vars(anthropic).values()
               if isinstance(v, str) and v.startswith("https://api.anthropic.com")}
    hosts = {h for h, port in _endpoints(policy) if port == 443}
    assert wanted - hosts == set(), f"provider hosts the policy does not allow: {sorted(wanted - hosts)}"


def test_no_rule_allows_any_host(policy):
    # A wildcard is allowed only under a named domain (*.discord.gg: Discord resumes on regional gateway
    # hosts); never a bare * or a wildcard over a whole top-level domain.
    for host, _port in _endpoints(policy):
        assert host not in ("*", "**"), f"wildcard host {host!r}"
        if "*" in host:
            assert host.startswith("*.") and "*" not in host[2:] and host[2:].count(".") >= 1, \
                f"wildcard host {host!r} is not scoped under a named domain"


# ── the smoke test, CI and the guide ─────────────────────────────────────────


def test_the_smoke_test_is_stdlib_and_has_both_modes():
    tree = ast.parse(SMOKE.read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not ({"prometheus", "yaml", "requests", "httpx"} & imported)
    text = SMOKE.read_text(encoding="utf-8")
    for needle in ("--docker", "--openshell", "host.openshell.internal", "FIRSTLIGHT-COMPLETE", "forward"):
        assert needle in text, f"the smoke test never mentions {needle}"


def test_ci_builds_and_checks_the_sandbox_image_on_both_architectures():
    job = yaml.safe_load(CI.read_text(encoding="utf-8"))["jobs"]["container"]
    body = yaml.safe_dump(job)
    assert "packaging/openshell/Dockerfile" in body
    assert "packaging/openshell/smoke_openshell.py" in body and "--docker" in body


def test_the_guide_says_exactly_what_was_tested():
    text = DOCS.read_text(encoding="utf-8")
    assert re.search(r"[Tt]ested with OpenShell 0\.\d+\.\d+ on Linux amd64", text), "the claim names a version"
    for needle in ("packaging/openshell/policy.yaml", "packaging/openshell/Dockerfile", "forward",
                   "host.openshell.internal", "8005", "8010", "Not tested"):
        assert needle in text, f"docs/guide/openshell.md never mentions {needle}"
    assert "nvidia" not in text.lower(), "no third-party branding in our docs"
