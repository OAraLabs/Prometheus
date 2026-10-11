"""`.github/workflows/release-macos.yml` — the job that builds, signs, notarizes and attaches Prometheus.app.

This job holds a Developer ID certificate and an Apple notarization password, and it publishes the file a
person's Mac will later run with login-item privileges. So what is asserted here is not style, it is the
shape that keeps those secrets where they belong and keeps a quiet failure from looking like a release:

* it runs on a tag push or a manual dispatch only, never on a pull request;
* every action is pinned to a commit SHA (the repo's rule, and a signing job is the last place to relax it);
* the certificate secrets reach ONE step (the keychain import), the notarization secrets reach ONE step (the
  one that stores them in a keychain profile, from stdin, never on an argv), and the mode decision sees
  booleans, not values;
* the job runs in the ``release`` environment (a required reviewer approves every run before a secret is
  released to it), and the checkout keeps no git credential;
* nothing resolves fresh: the tools come from hashed pins, the wheel's build backend too, and the wheel is
  built BEFORE the certificate is on the runner, so the only fetched code that runs never runs beside it;
* the temporary keychain and the decoded .p12 are deleted even when the build fails;
* nothing is published unless the run was signed AND notarized AND verified, and `continue-on-error` is
  not used anywhere (release.yml's header explains what a green job that did nothing costs);
* the tag must equal the built version, because an asset published under the wrong tag is a release.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release-macos.yml"
RELEASE_TOOLS = ROOT / "packaging" / "macos" / "release-tools.txt"
SECRETS_SCRIPT = ROOT / "packaging" / "macos" / "set_release_secrets.sh"
SHA = re.compile(r"^[^@\s]+@[0-9a-f]{40}(\s|$)")


@pytest.fixture(scope="module")
def wf():
    assert WORKFLOW.is_file(), f"{WORKFLOW} does not exist"
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _steps(wf):
    return [s for job in wf["jobs"].values() for s in job["steps"]]


def _text(step) -> str:
    return yaml.safe_dump(step)


def test_it_runs_on_tags_and_manual_dispatch_only(wf):
    on = wf.get(True) or wf["on"]       # YAML 1.1 reads a bare `on` as True
    assert set(on) == {"push", "workflow_dispatch"}
    assert on["push"] == {"tags": ["v*"]}


def test_every_action_is_pinned_to_a_commit(wf):
    for step in _steps(wf):
        if "uses" in step:
            assert SHA.match(step["uses"] + " "), f"unpinned action: {step['uses']}"


def test_permissions_are_declared_per_job_and_never_include_an_id_token(wf):
    assert wf.get("permissions") in (None, {}), "no workflow-level grant: each job declares its own"
    for name, job in wf["jobs"].items():
        assert job["permissions"] == {"contents": "write"}, f"{name}: only the release upload needs write"


def test_the_job_is_bounded_and_runs_on_apple_silicon(wf):
    for job in wf["jobs"].values():
        assert 0 < job["timeout-minutes"] <= 90
        assert re.fullmatch(r"macos-1[4-9]|macos-latest", job["runs-on"]), job["runs-on"]


def test_the_certificate_secrets_reach_exactly_one_step(wf):
    holders = [s for s in _steps(wf)
               if s.get("id") != "mode" and ("secrets.CSC_LINK" in _text(s) or "secrets.CSC_KEY_PASSWORD" in _text(s))]
    assert len(holders) == 1, [h.get("name") for h in holders]
    assert "secrets.CSC_LINK" in _text(holders[0]) and "secrets.CSC_KEY_PASSWORD" in _text(holders[0])
    assert "keychain" in holders[0]["name"].lower()
    # The mode step decides from booleans: it never receives a secret's value.
    mode = next(s for s in _steps(wf) if s.get("id") == "mode")
    for value in (mode.get("env") or {}).values():
        assert "!= ''" in str(value) or "== ''" in str(value), f"the mode step must not hold a secret: {value}"


def test_the_notarization_password_goes_into_a_keychain_profile_from_stdin_and_never_onto_an_argv(wf):
    """`ps` shows every process's arguments; `notarytool submit --wait` runs for up to 40 minutes."""
    holders = [s for s in _steps(wf) if "secrets.APPLE_" in _text(s) and s.get("id") != "mode"]
    assert len(holders) == 1, [h.get("name") for h in holders]
    store = holders[0]
    run = store["run"].replace("\\\n", " ")
    assert re.search(r"printf '%s(\\n)?' \"\$APPLE_APP_SPECIFIC_PASSWORD\"\s*\|\s*xcrun notarytool store-credentials", run), run
    assert "--keychain" in run
    build = next(s for s in _steps(wf) if "--notarize" in s.get("run", ""))
    assert "secrets." not in _text(build), "the build step holds no secret at all"
    assert "--notary-profile" in build["run"] and "--notary-keychain" in build["run"]
    for step in _steps(wf):
        assert "--password" not in step.get("run", ""), step.get("name")


def test_the_job_runs_in_the_release_environment(wf):
    for name, job in wf["jobs"].items():
        environment = job.get("environment")
        assert environment == "release" or (isinstance(environment, dict) and environment.get("name") == "release"), name


def test_the_checkout_leaves_no_git_credential_behind(wf):
    checkout = next(s for s in _steps(wf) if "actions/checkout@" in s.get("uses", ""))
    assert (checkout.get("with") or {}).get("persist-credentials") is False


def test_the_tools_are_installed_from_hashed_pins_and_nothing_resolves_fresh(wf):
    install = [s for s in _steps(wf) if "pip install" in s.get("run", "")]
    assert len(install) == 1, [s.get("name") for s in install]
    run = install[0]["run"]
    assert "--require-hashes" in run and "--only-binary :all:" in run
    assert "-r packaging/macos/release-tools.txt" in run and "--upgrade" not in run
    for step in _steps(wf):
        text = step.get("run", "")
        assert "uvx" not in text and "--with " not in text, f"{step.get('name')} resolves a package fresh"


def test_the_release_tools_are_pinned_and_hashed():
    import importlib.util

    spec = importlib.util.spec_from_file_location("macos_build_app", ROOT / "packaging" / "macos" / "build_app.py")
    build_app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_app)
    text = RELEASE_TOOLS.read_text(encoding="utf-8")
    build_app.require_hashes(text)
    pins = re.findall(r"^([a-z0-9][a-z0-9._-]*)==([^\s;]+)", text, re.M)
    assert {name for name, _ in pins} >= {"uv", "packaging"}, pins
    assert all(not line.strip() or line.lstrip().startswith(("#", "--hash")) or "==" in line
               for line in text.splitlines()), "every requirement is pinned with =="


def test_the_wheel_is_built_before_the_certificate_is_on_the_runner(wf):
    steps = _steps(wf)
    wheel = next(i for i, s in enumerate(steps) if "--wheel-only" in s.get("run", ""))
    keychain = next(i for i, s in enumerate(steps) if "security import" in s.get("run", ""))
    assert wheel < keychain, [s.get("name") for s in steps]
    assert "secrets." not in _text(steps[wheel])
    build = next(s for s in steps if "--notarize" in s.get("run", ""))
    assert "--wheel " in build["run"], "the signing build installs the wheel built before the import"


def test_the_decoded_p12_is_deleted_even_when_a_step_fails(wf):
    cleanup = [s for s in _steps(wf) if "rm -f" in s.get("run", "") and "prometheus-signing.p12" in s.get("run", "")
               and "always()" in str(s.get("if"))]
    assert len(cleanup) == 1


def test_the_secrets_script_sets_them_on_the_release_environment():
    text = SECRETS_SCRIPT.read_text(encoding="utf-8")
    sets = [line for line in text.splitlines() if "gh secret set" in line and not line.lstrip().startswith("#")]
    assert sets and all("--env" in line for line in sets), sets
    assert re.search(r'^ENVIRONMENT="release"', text, re.M)
    assert "environments/" in text, "it checks the environment exists before setting anything"


def test_the_build_signs_by_the_fingerprint_it_found_in_its_own_keychain(wf):
    build = next(s for s in _steps(wf) if "--notarize" in s.get("run", ""))
    assert "--identity" in build["run"]
    assert "build_app.py" in build["run"] and "PYTHONPATH" in _text(build)
    keychain = next(s for s in _steps(wf) if "security import" in s.get("run", ""))
    assert "Developer ID Application" in keychain["run"] and "53JM8W47RL" in keychain["run"]
    assert "GITHUB_OUTPUT" in keychain["run"] or "GITHUB_ENV" in keychain["run"], "the fingerprint is handed to the build"


def test_the_temporary_keychain_is_deleted_even_when_the_build_fails(wf):
    cleanup = [s for s in _steps(wf) if "delete-keychain" in s.get("run", "")]
    assert len(cleanup) == 1 and "always()" in str(cleanup[0].get("if"))


def test_the_artifact_is_verified_as_notarized_before_anything_is_attached(wf):
    steps = _steps(wf)
    names = [s.get("name", "") for s in steps]
    verify = next(i for i, s in enumerate(steps) if "verify_app.py" in s.get("run", ""))
    assert "--notarized" in steps[verify]["run"]
    attach = next(i for i, s in enumerate(steps) if "gh release" in s.get("run", ""))
    assert verify < attach, names


def test_a_release_is_attached_only_for_a_tag_and_only_when_notarized(wf):
    attach = next(s for s in _steps(wf) if "gh release" in s.get("run", ""))
    condition = str(attach["if"])
    assert "refs/tags/v" in condition and "notarized" in condition
    assert "--draft" in attach["run"], "a human publishes; the workflow only drafts"
    assert "--clobber" in attach["run"]


def test_without_the_secrets_nothing_is_built_and_the_run_says_so(wf):
    mode = next(s for s in _steps(wf) if s.get("id") == "mode")
    assert "GITHUB_STEP_SUMMARY" in mode["run"] and "::warning" in mode["run"]
    # A TAG without the secrets is a release without its Mac app: that is a red run, not a quiet green one.
    assert "::error" in mode["run"] and "refs/tags/" in mode["run"] and "exit 1" in mode["run"]
    gated = [s for s in _steps(wf) if s is not mode and s.get("name") not in ("Check out", "Set up Python 3.12")
             and "uses" not in s]
    for step in gated:
        condition = str(step.get("if", ""))
        # The keychain cleanup is gated by the import step's own outcome: no import, nothing to delete.
        assert "notarized" in condition or "steps.keychain.outcome" in condition, \
            f"{step.get('name')} must be skipped without the secrets"


def test_a_tag_that_is_not_the_version_fails_the_run(wf):
    check = next(s for s in _steps(wf) if "__version__" in s.get("run", ""))
    assert "GITHUB_REF_NAME" in check["run"] and ("exit 1" in check["run"] or "sys.exit" in check["run"])


def test_no_step_swallows_its_own_failure(wf):
    for step in _steps(wf):
        assert step.get("continue-on-error") in (None, False), step.get("name")
        for line in step.get("run", "").splitlines():
            if "|| true" in line:
                # Tolerated, each for a stated reason: `gh release create` fails when release.yml already made
                # the draft; `grep` exits 1 on no match under `set -e` and the count is checked right after;
                # a keychain that is already gone needs no deleting.
                assert any(ok in line for ok in ("gh release create", "already exists", "grep", "delete-keychain")), \
                    f"{step.get('name')}: {line.strip()}"


def test_a_published_release_is_never_modified(wf):
    attach = next(s for s in _steps(wf) if "gh release" in s.get("run", ""))
    assert "isDraft" in attach["run"], "check the release is still a draft before replacing assets"


def test_the_release_build_cannot_include_pymupdf(wf):
    """PyMuPDF is AGPL-3.0; the app ships without it by default. CI must not opt back in."""
    build = next(s for s in _steps(wf) if "--notarize" in s.get("run", ""))
    assert "--include-pymupdf" not in build["run"]
    assert "--without" not in build["run"] or "pymupdf" in build["run"]
