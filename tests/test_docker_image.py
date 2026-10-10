"""The official container image: ``Dockerfile``, ``.dockerignore``, the compose example and the jobs
that build, verify and publish it.

The image runs an agent that executes model-written shell commands, so what is asserted here is the
shape that keeps it honest, not style:

* nothing secret is ever in the image: no token, key or env file in any layer, and the build context is
  an ALLOWLIST, so a developer's own ``config/prometheus.yaml`` or ``.env`` cannot reach it;
* it runs as a fixed non-root user, declares where its state lives (``VOLUME``), says how to probe it
  (``HEALTHCHECK``), and listens on its own interfaces so ``-p`` alone decides exposure;
* nothing resolves fresh: the base image is pinned by digest, the build tools and the build backend by
  hash, and every dependency comes from ``uv.lock`` with ``--require-hashes --only-binary :all:``;
* configuration lives in ``ENV``, never in an entrypoint script: a sandbox runtime that replaces the
  entrypoint (some sandbox runtimes do) must still get the same daemon;
* CI builds the image on amd64 AND arm64 runners and runs the container smoke test on each;
* the release builds both architectures, smokes them, and publishes to GHCR only from a job that runs
  no repository code, holds nothing but ``packages: write``, and waits for the approved PyPI publish.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
COMPOSE = ROOT / "packaging" / "docker" / "compose.yaml"
SMOKE = ROOT / "packaging" / "docker" / "smoke_image.py"
CI = ROOT / ".github" / "workflows" / "ci.yml"
RELEASE = ROOT / ".github" / "workflows" / "release.yml"
DOCS = ROOT / "docs" / "guide" / "docker.md"
SHA = re.compile(r"^[^@\s]+@[0-9a-f]{40}(\s|$)")
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}\b")

HOME = "/home/prometheus"
DATA = f"{HOME}/.prometheus/data"


# ── Dockerfile parsing ───────────────────────────────────────────────────────


def _instructions(text: str) -> list[tuple[str, str]]:
    """``(INSTRUCTION, arguments)`` pairs, with ``\\`` continuations joined and comments dropped."""
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
def dockerfile() -> list[tuple[str, str]]:
    assert DOCKERFILE.is_file(), "there is no Dockerfile at the repository root"
    return _instructions(DOCKERFILE.read_text(encoding="utf-8"))


def _stages(instrs):
    stages: list[list[tuple[str, str]]] = []
    for word, rest in instrs:
        if word == "FROM":
            stages.append([])
        if stages:
            stages[-1].append((word, rest))
    return stages


def _final(instrs):
    return _stages(instrs)[-1]


def _resolve_args(instrs, value: str) -> str:
    """Expand ``${NAME}`` from the global ``ARG NAME=default`` lines (the base image is one)."""
    defaults = {}
    for word, rest in instrs:
        if word == "FROM":
            break
        if word == "ARG" and "=" in rest:
            name, _, default = rest.partition("=")
            defaults[name.strip()] = default.strip().strip('"')
    return re.sub(r"\$\{?(\w+)\}?", lambda m: defaults.get(m.group(1), m.group(0)), value)


def _exec_form(rest: str) -> list[str]:
    value = ast.literal_eval(rest)
    assert isinstance(value, list), f"use the exec (JSON) form: {rest}"
    return value


def _env_pairs(instrs) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for word, rest in instrs:
        if word != "ENV":
            continue
        for m in re.finditer(r'(\w+)=("[^"]*"|\S*)', rest):
            pairs[m.group(1)] = m.group(2).strip('"')
    return pairs


# ── the image ────────────────────────────────────────────────────────────────


def test_every_base_image_is_pinned_by_digest(dockerfile):
    froms = [_resolve_args(dockerfile, rest).split()[0] for word, rest in dockerfile if word == "FROM"]
    assert froms
    for ref in froms:
        if re.fullmatch(r"\w+", ref):     # an earlier stage, by name
            continue
        assert DIGEST.search(ref), f"base image not pinned by digest: {ref}"


def test_it_runs_as_a_fixed_non_root_user(dockerfile):
    users = [rest for word, rest in _final(dockerfile) if word == "USER"]
    assert users, "the final stage never sets USER, so the agent would run as root"
    uid = users[-1].split(":")[0]
    # Numeric, so a runtime can verify non-root without reading /etc/passwd (Kubernetes runAsNonRoot,
    # and a sandbox runtime that rejects root).
    assert uid.isdigit(), f"USER must be numeric: {users[-1]}"
    assert int(uid) >= 10000, "a uid that collides with the host's first user owns that user's files on escape"


def test_it_declares_where_its_state_lives(dockerfile):
    volumes = [v for word, rest in dockerfile if word == "VOLUME" for v in _exec_form(rest)]
    assert HOME in volumes, "the home volume holds the config, identity, memory and the env file with the token"
    assert DATA in volumes, "the data volume holds the conversation store, sessions and devices"


def test_it_exposes_the_api_and_the_websocket_bridge(dockerfile):
    ports = {p for word, rest in dockerfile if word == "EXPOSE" for p in rest.split()}
    assert {"8005", "8010"} <= {p.split("/")[0] for p in ports}


def test_it_listens_on_its_own_interfaces_and_lets_publishing_decide(dockerfile):
    # A new config is written with web.bind 127.0.0.1 unless an address was CHOSEN; inside a container
    # that would make every published port dead. The variable is a choice, so setup carries it over.
    assert _env_pairs(dockerfile).get("PROMETHEUS_WEB_BIND") == "0.0.0.0"


def test_the_healthcheck_is_the_daemon_probe(dockerfile):
    checks = [rest for word, rest in _final(dockerfile) if word == "HEALTHCHECK"]
    assert len(checks) == 1
    assert "prometheus.web.healthcheck" in checks[0]
    assert "--start-period" in checks[0], "the first boot builds databases; give it time before counting failures"


def test_no_secret_is_baked_into_the_image(dockerfile):
    from prometheus.security.env_scrub import is_secret_name

    for word, rest in dockerfile:
        if word in ("ENV", "ARG"):
            for m in re.finditer(r"(\w+)(?:=|\s|$)", rest):
                name = m.group(1)
                assert not is_secret_name(name), f"{word} {name}: a secret-shaped name in the image"
    # Instructions only: the header comment explains where the token comes from instead.
    instructions = " ".join(f"{word} {rest}" for word, rest in dockerfile)
    assert "PROMETHEUS_API_TOKEN" not in instructions
    for word, rest in dockerfile:
        if word in ("COPY", "ADD"):
            assert not re.search(r"(^|[\s/])\.?env\b|prometheus\.yaml\s", rest), f"{word} {rest}"


def test_configuration_lives_in_env_not_in_an_entrypoint_script(dockerfile):
    entry = [_exec_form(rest) for word, rest in _final(dockerfile) if word == "ENTRYPOINT"]
    cmd = [_exec_form(rest) for word, rest in _final(dockerfile) if word == "CMD"]
    # tini as PID 1: it reaps the orphans of model-written shells and forwards SIGTERM. Nothing else.
    assert entry == [["/usr/bin/tini", "--"]]
    assert cmd == [["oara", "daemon"]]


def test_the_runtime_needs_no_env_file_override(dockerfile):
    # The env file (minted token, gateway tokens, provider keys) stays at ~/.config/prometheus/env, where
    # the file tools' hard floor (/*/.config/*/*env) refuses to read it. The home volume persists it.
    assert "PROMETHEUS_ENV_FILE" not in _env_pairs(dockerfile)
    assert _env_pairs(dockerfile).get("HOME") in (None, HOME)


def test_nothing_resolves_fresh(dockerfile):
    runs = " ".join(rest for word, rest in dockerfile if word == "RUN")
    assert "--require-hashes" in runs and "release-tools.txt" in runs, "uv must come from the hashed pins"
    assert "uv export --frozen" in runs
    assert re.search(r"uv pip install[^&;]*--require-hashes[^&;]*--only-binary :all:", runs), \
        "dependencies must install from the lock's hashes, as wheels only"
    assert re.search(r"uv build[^&;]*--build-constraints[^&;]*build-constraints\.txt[^&;]*--require-hashes", runs)
    assert "UV_PYTHON_DOWNLOADS" in _env_pairs(dockerfile), "uv must never fetch its own interpreter"


def test_the_package_installs_into_a_venv_the_agent_cannot_rewrite(dockerfile):
    final = _final(dockerfile)
    user_at = max(i for i, (word, _) in enumerate(final) if word == "USER")
    copies = [i for i, (word, rest) in enumerate(final) if word == "COPY" and "/opt/prometheus" in rest]
    assert copies and all(i < user_at for i in copies)
    assert not any("--chown" in rest for word, rest in final if word == "COPY" and "/opt/prometheus" in rest)


# ── the build context ────────────────────────────────────────────────────────


def _ignore_rules() -> list[str]:
    assert DOCKERIGNORE.is_file(), "no .dockerignore: `docker build .` would send the whole checkout"
    return [ln.strip() for ln in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def test_the_build_context_is_an_allowlist():
    rules = _ignore_rules()
    assert rules[0] == "*", "start from nothing and add back what the build needs"
    allowed = [r[1:] for r in rules if r.startswith("!")]
    for forbidden in ("config/prometheus.yaml", ".env", ".git", ".claude", "tests"):
        assert forbidden not in allowed, f".dockerignore lets {forbidden} into the build context"
    assert not any(r in ("!config", "!config/", "!config/*", "!*") for r in rules)


def test_everything_the_dockerfile_copies_is_in_the_context(dockerfile):
    allowed = [r[1:].rstrip("/") for r in _ignore_rules() if r.startswith("!")]
    for word, rest in dockerfile:
        if word != "COPY" or "--from=" in rest:
            continue
        *sources, _dest = [p for p in rest.split() if not p.startswith("--")]
        for src in sources:
            src = src.rstrip("/")
            assert any(src == a or src.startswith(a + "/") for a in allowed), \
                f"COPY {src}: not in the .dockerignore allowlist"


# ── the compose example ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def compose() -> dict:
    assert COMPOSE.is_file(), f"{COMPOSE} does not exist"
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def test_compose_runs_prometheus_beside_ollama(compose):
    services = compose["services"]
    assert {"prometheus", "ollama"} <= set(services)
    assert services["prometheus"]["image"].startswith("ghcr.io/oaralabs/prometheus:")
    ollama = services["ollama"]["image"]
    assert ollama.startswith("ollama/ollama:") and not ollama.endswith(":latest"), "pin the Ollama version"


def test_compose_publishes_on_loopback_only(compose):
    for spec in compose["services"]["prometheus"].get("ports", []):
        assert str(spec).startswith("127.0.0.1:"), f"{spec}: widen it deliberately, not by default"
    assert not compose["services"]["ollama"].get("ports"), "only Prometheus talks to Ollama"


def test_compose_mounts_the_image_volumes(compose):
    mounts = compose["services"]["prometheus"]["volumes"]
    targets = {str(m).split(":")[1] for m in mounts}
    assert {HOME, DATA} <= targets
    assert set(compose["volumes"]) >= {str(m).split(":")[0] for m in mounts}


def test_compose_carries_no_token(compose):
    text = COMPOSE.read_text(encoding="utf-8")
    env = compose["services"]["prometheus"].get("environment") or {}
    if isinstance(env, list):
        env = dict(e.split("=", 1) for e in env)
    assert not env.get("PROMETHEUS_API_TOKEN"), "a token in the compose file is a token in git"
    assert not re.search(r"PROMETHEUS_API_TOKEN\s*[:=]\s*[A-Za-z0-9_-]{8,}", text)


# ── CI: build and run the image on both architectures ───────────────────────


def _jobs(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]


def _text(obj) -> str:
    return yaml.safe_dump(obj)


def test_ci_builds_and_smokes_the_image_natively_on_amd64_and_arm64():
    job = _jobs(CI)["container"]
    runners = set(job["strategy"]["matrix"]["runner"])
    assert runners == {"ubuntu-24.04", "ubuntu-24.04-arm"}, "native runners, no emulation"
    body = _text(job)
    assert "docker build" in body or "docker buildx build" in body
    assert "packaging/docker/smoke_image.py" in body
    assert "continue-on-error" not in body


def test_every_new_action_is_pinned_to_a_commit():
    for path in (CI, RELEASE):
        for name, job in _jobs(path).items():
            for step in job.get("steps", []):
                if "uses" in step and name in ("container", "image", "publish-image"):
                    assert SHA.match(step["uses"] + " "), f"{path.name}:{name}: unpinned {step['uses']}"


# ── release: build both, smoke both, publish after PyPI ─────────────────────


def test_the_release_builds_both_architectures_and_smokes_them():
    jobs = _jobs(RELEASE)
    image = jobs["image"]
    assert image["needs"] == "build" or image["needs"] == ["build"], "the tag/version check runs first"
    assert image["permissions"] == {"contents": "read"}, "the build runs repository code: it publishes nothing"
    body = _text(image)
    assert "linux/amd64,linux/arm64" in body
    assert "packaging/docker/smoke_image.py" in body
    assert "type=oci" in body, "the publish job pushes the bytes that were built and smoked"


def test_the_image_is_published_by_a_job_that_runs_no_repository_code():
    job = _jobs(RELEASE)["publish-image"]
    assert job["if"].replace(" ", "") == "startsWith(github.ref,'refs/tags/v')"
    assert set(job["needs"]) == {"image", "publish"}, "an image is published only after the approved PyPI publish"
    assert job["permissions"] == {"packages": "write"}
    body = _text(job)
    assert "actions/checkout" not in body
    assert "ghcr.io" in body
    assert "--all" in body, "copy the whole multi-arch index, not one platform"
    assert "continue-on-error" not in body


def _tags_script() -> str:
    steps = [s for s in _jobs(RELEASE)["image"]["steps"] if "tags.txt" in s.get("run", "")]
    assert len(steps) == 1, "exactly one step decides the tags (publish-image only reads them)"
    script = steps[0]["run"]
    lines = script.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("if [["))
    end = next(i for i, ln in enumerate(lines) if ln.strip() == "fi")
    return "\n".join(lines[start:end + 1])


@pytest.mark.parametrize(("version", "tags"), [
    ("0.9.7", ["0.9.7", "0.9", "latest"]),
    ("1.2.0", ["1.2.0", "1.2", "latest"]),
    ("1.0.0rc1", ["1.0.0rc1"]),
    ("1.0.0a2", ["1.0.0a2"]),
    ("1.0.0b3", ["1.0.0b3"]),
    ("1.0.0.dev4", ["1.0.0.dev4"]),
])
def test_a_pre_release_never_moves_latest(version, tags):
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    out = subprocess.run([bash, "-c", f'version="{version}"\n{_tags_script()}\necho "$tags"'],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == tags


def test_the_publish_job_pushes_exactly_the_tags_the_image_job_decided():
    scripts = " ".join(s.get("run", "") for s in _jobs(RELEASE)["publish-image"]["steps"])
    assert "tags.txt" in scripts
    assert not re.search(r":latest\b|\blatest\b", scripts), "the publish job must not add tags of its own"


# ── the smoke test and the docs ──────────────────────────────────────────────


def test_the_smoke_test_is_stdlib_and_covers_every_token_path():
    text = SMOKE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not ({"prometheus", "yaml", "requests", "httpx"} & imported), "runs on a bare runner python"
    for needle in ("PROMETHEUS_API_TOKEN_FILE", "PROMETHEUS_API_TOKEN", "token", "show",
                   "/api/setup/status", "/health", "--probe-url", "FIRSTLIGHT-COMPLETE", "Health"):
        assert needle in text, f"the smoke test never exercises {needle}"


def test_the_docs_page_covers_what_an_operator_needs():
    text = DOCS.read_text(encoding="utf-8")
    for needle in ("ghcr.io/oaralabs/prometheus", "PROMETHEUS_API_TOKEN_FILE", "PROMETHEUS_API_TOKEN",
                   "127.0.0.1:8005", HOME, DATA, "10001", "compose.yaml", "oara token show",
                   "bubblewrap", "AppArmor", "linux/arm64"):
        assert needle in text, f"docs/guide/docker.md never mentions {needle}"
    assert "nvidia" not in text.lower(), "no third-party branding in our docs"


def test_the_install_guide_points_at_the_docker_page():
    assert "docker.md" in (ROOT / "docs" / "guide" / "install.md").read_text(encoding="utf-8")
