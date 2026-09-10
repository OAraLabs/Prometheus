"""The "Honest status notes" must agree with the code — audit item 10.

WHY THIS FILE EXISTS
--------------------
``docs/guide/features.md`` carries a section that declares itself the tie
breaker:

    These caveats are part of the reference, not fine print. If a claim
    elsewhere in the docs conflicts with this list, this list wins.

Three of its nine bullets were wrong when audited, and one of those three
(the model router) was wrong **from the day it was written** — the override
commands it declared off had shipped on-by-default two and a half months
earlier. A section whose whole authority is that it tells the truth about
what does not work is the worst place in the repository for silent drift,
and nothing checked it.

⚠ THE ASSERTION DIRECTION MATTERS. Each test below reads the DOCUMENT and
compares it against the code. A test that only asserted the code was right
would pass while the sentence describing it rotted — which is exactly the
state this file replaces. Where a bullet states a NUMBER or a NAME, the
number or name is parsed out of the prose, so changing the code without
changing the sentence fails.

Claims not covered here, and why:

* "the fine-tuning flywheel is data-collection only" — the absence of a
  training loop is covered by the no-``peft``/``transformers`` reality; a
  test asserting an absence would pass forever and prove nothing until
  someone adds one, and then it fails for the right reason. Kept as the
  trajectory-export half, which is checkable.
* "Model IDs ... are forward-looking defaults" — a hedge, not a claim. It
  cannot be falsified and so cannot be gated.
* GEPA "has never promoted a skill" on the reference deployment — a fact
  about ONE machine's ``~/.prometheus``, not about this repository. Its
  falsifiable halves (the ``Skill``-tool filter, the minimum) are covered.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "guide" / "features.md"
HEADING = "## Honest status notes"


def _section() -> str:
    text = DOC.read_text(encoding="utf-8")
    assert HEADING in text, (
        f"{DOC.relative_to(REPO)} no longer has a {HEADING!r} section. It is "
        f"the documented tie-breaker for every other claim in the docs — if "
        f"it was renamed, point this file at the new heading; if it was "
        f"deleted, say so somewhere a reader will find."
    )
    body = text.split(HEADING, 1)[1]
    # Up to the next H2, or end of file.
    nxt = re.search(r"^## ", body, re.M)
    return body[: nxt.start()] if nxt else body


def _bullets() -> list[str]:
    return [ln.strip() for ln in _section().splitlines() if ln.startswith("- ")]


def _template() -> dict:
    return yaml.safe_load(
        (REPO / "config" / "prometheus.yaml.default").read_text(encoding="utf-8")
    )


def _get(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return "<ABSENT>"
        cur = cur[part]
    return cur


def test_the_section_still_claims_authority():
    """If it stops declaring itself the tie-breaker, this whole file is
    guarding something that no longer makes the promise."""
    assert "this list wins" in _section()


# ── bullet 1: the skill count ───────────────────────────────────────

def test_the_skill_count_in_the_prose_is_the_real_count():
    """The number was 102 while the directory held 103 — and the README
    said 103 in two places, so the docs contradicted each other."""
    actual = len(list((REPO / "skills").glob("*.md")))
    bullet = next(b for b in _bullets() if "skills` directory holds" in b
                  or "skills/` directory holds" in b)
    m = re.search(r"holds (\d+) authored skill files", bullet)
    assert m, f"could not find the count in: {bullet[:120]}"
    assert int(m.group(1)) == actual, (
        f"the note says {m.group(1)} authored skill files, the repo has "
        f"{actual}. Update the sentence, not just the directory."
    )


#: Every way the docs state the size of the repo's skill library. Written as
#: patterns rather than one, because the count is spelled three different
#: ways in three files and only ONE of them was found by the first sweep —
#: the other two were stale in the same direction, and one of those sat in
#: the same file as the bullet this test guards.
_SKILL_COUNT_PATTERNS = (
    r"(\d+)-file skill library",
    r"(\d+)-file `skills/` library",
    r"holds (\d+) authored skill files",
)


def test_every_doc_states_the_same_skill_count():
    """Not just the note and the README — EVERY statement of the number.

    The audit found the note saying 102. Fixing only that would have left
    README:180 and features.md:252 saying 102 as well, in a repo whose
    README elsewhere said 103. A count restated in five places needs a
    sweep, not a spot-check.
    """
    actual = len(list((REPO / "skills").glob("*.md")))
    stated: dict[str, set[int]] = {}
    docs = [REPO / "README.md", *sorted((REPO / "docs" / "guide").glob("*.md"))]
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        found = {int(n) for pat in _SKILL_COUNT_PATTERNS
                 for n in re.findall(pat, text)}
        if found:
            stated[str(doc.relative_to(REPO))] = found
    assert stated, "no document states the skill-library size any more"
    wrong = {k: sorted(v) for k, v in stated.items() if v != {actual}}
    assert not wrong, (
        f"the repo has {actual} skill files; these documents disagree: "
        f"{wrong}. A number restated in several places goes stale in some of "
        f"them first, and the disagreement is what hides it."
    )


def test_only_three_skills_are_package_bundled():
    """The bullet names them: commit, debug, plan."""
    bundled = sorted(
        p.stem for p in (REPO / "src" / "prometheus" / "skills" / "builtin").glob("*.md")
    )
    bullet = next(b for b in _bullets() if "package-bundled builtins" in b)
    for name in bundled:
        assert f"`{name}`" in bullet, (
            f"{name} is package-bundled but the note does not name it: {bullet[:160]}"
        )
    m = re.search(r"the \*\*(\d+)\*\* package-bundled builtins", bullet)
    assert m and int(m.group(1)) == len(bundled), (
        f"the note says {m.group(1) if m else '?'} bundled skills, there are "
        f"{len(bundled)}: {bundled}"
    )


# ── bullet 2: trajectory export ─────────────────────────────────────

def test_trajectory_export_is_off_in_the_shipped_template():
    bullet = next(b for b in _bullets() if "fine-tuning flywheel" in b)
    assert "off by default" in bullet
    assert _get(_template(), "trajectory_export.enabled") is False


# ── bullet 3: what actually ships off ───────────────────────────────

#: Every subsystem the "ship off by default" bullet names, mapped to the
#: template key that decides it. A name in the prose with no key here fails
#: below, so the list cannot grow a claim nothing checks.
OFF_BY_DEFAULT_KEYS = {
    "SENTINEL": "sentinel.enabled",
    "divergence detection": "divergence.enabled",
    "LSP": "lsp.enabled",
    "Symbiote": "symbiote.enabled",
    "GEPA": "learning.gepa_enabled",
    "escalation-to-teacher": "router.escalation.enabled",
    "skill refinement": "learning.skill_refinement_enabled",
    "the approval queue": "security.approval_queue.enabled",
    "Printing Press": "printing_press.enabled",
    "tracing": "tracing.enabled",
    "Whisper voice input": "whisper.enabled",
}


@pytest.mark.parametrize("name,key", sorted(OFF_BY_DEFAULT_KEYS.items()))
def test_each_named_subsystem_is_actually_off_in_the_template(name, key):
    bullet = next(b for b in _bullets() if "ship off by default" in b)
    assert name in bullet, (
        f"the note no longer names {name!r}. If it is now on by default, this "
        f"entry should go; if the wording changed, update it here too."
    )
    assert _get(_template(), key) is False, (
        f"the note says {name} ships off by default, but {key} is "
        f"{_get(_template(), key)!r} in config/prometheus.yaml.default"
    )


def test_all_three_gateways_are_off_in_the_template():
    bullet = next(b for b in _bullets() if "ship off by default" in b)
    assert "all three chat gateways" in bullet
    t = _template()
    for key in ("gateway.telegram_enabled", "gateway.slack.enabled",
                "gateway.discord.enabled"):
        assert _get(t, key) is False, f"{key} = {_get(t, key)!r}"


def test_the_model_router_is_not_claimed_off():
    """The bullet that was wrong from the day it was written.

    `router.overrides.enabled` defaults to True and the router is built
    unconditionally, which is what makes /claude and friends work on a
    fresh install. If that default ever flips, this note must change with
    it — so the assertion runs in both directions.
    """
    off = next(b for b in _bullets() if "ship off by default" in b)
    assert "model router" not in off, (
        "the model router is back in the off-by-default list. It is "
        "constructed unconditionally and its override commands default to "
        "enabled — see test_router_overrides_default_to_enabled."
    )
    assert any("model router is NOT off by default" in b for b in _bullets()), (
        "the correction bullet is gone; without it a reader has no way to "
        "learn that /claude works out of the box"
    )


def test_router_overrides_default_to_enabled():
    """The code half of the claim above, asserted from the DEFAULT — the
    template omits the key, so the dataclass default is what a fresh
    install actually gets."""
    from prometheus.router.model_router import RouterConfig

    assert RouterConfig().overrides_enabled is True
    assert _get(_template(), "router.overrides.enabled") is True


def test_the_router_is_built_unconditionally():
    """`create_model_router` has no enable check — it returns a router for
    every config. Asserted by calling it with an empty config."""
    from prometheus.__main__ import create_model_router

    router = create_model_router({}, object(), object(), "some-model")
    assert router is not None


def test_the_inert_router_halves_are_still_inert():
    t = _template()
    assert _get(t, "router.rules") == []
    assert _get(t, "router.fallback") == []
    assert _get(t, "router.smart_routing.enabled") is False
    assert _get(t, "router.escalation.enabled") is False


# ── bullet 4: paid media backends ───────────────────────────────────

def test_auto_never_selects_the_paid_image_backend():
    from prometheus.tools.builtin import image_generate as ig

    bullet = next(b for b in _bullets() if "Paid media backends" in b)
    assert "never picks a paid one" in bullet
    src = Path(ig.__file__).read_text(encoding="utf-8")
    assert "auto NEVER picks the paid backend" in src or \
           "NEVER auto-selected" in src, (
        "the tool no longer documents that `auto` avoids the paid backend — "
        "check _resolve_backend before trusting the note"
    )


# ── bullet 5: the sandbox backends ──────────────────────────────────

def test_the_note_names_every_sandbox_backend_that_exists():
    """The bullet said DockerSandbox was unimplemented while
    `coding/sandbox.py` carried a 315-line DockerSandbox AND a
    BwrapSandbox the note never mentioned."""
    from prometheus.coding.sandbox import SANDBOX_BACKENDS

    bullet = next(b for b in _bullets() if "sandbox" in b.lower())
    for backend in SANDBOX_BACKENDS:
        assert f"`{backend}`" in bullet, (
            f"sandbox backend {backend!r} exists in SANDBOX_BACKENDS but the "
            f"note does not name it: {bullet[:200]}"
        )
    assert "unimplemented" not in bullet.lower(), (
        "the note calls a sandbox backend unimplemented; all of "
        f"{sorted(SANDBOX_BACKENDS)} are implemented"
    )


def test_every_named_backend_has_a_real_class():
    from prometheus.coding import sandbox as sb

    for cls in ("ProcessSandbox", "BwrapSandbox", "DockerSandbox"):
        assert hasattr(sb, cls), f"{cls} is gone — the note names it"


def test_the_default_backend_is_the_one_the_note_calls_the_default():
    bullet = next(b for b in _bullets() if "sandbox" in b.lower())
    m = re.search(r"`(\w+)` \(the default\)", bullet)
    assert m, f"the note no longer marks a default backend: {bullet[:200]}"
    assert _get(_template(), "coding.sandbox_type") == m.group(1)


# ── bullet 6: web chat command parity ───────────────────────────────

def test_the_commands_named_as_web_unreachable_really_are():
    from prometheus.web.slash_router import WEB_NATIVE_ONLY

    bullet = next(b for b in _bullets() if "Web chat lacks parity" in b)
    named = re.findall(r"`/(\w+)`", bullet)
    assert named, f"no commands named in: {bullet[:160]}"
    for cmd in named:
        assert cmd in WEB_NATIVE_ONLY, (
            f"the note says /{cmd} is not available on web chat, but it is "
            f"not in WEB_NATIVE_ONLY — it may have been wired since"
        )


def test_those_commands_do_work_on_all_three_gateways():
    """The other half of the same sentence, and the half most likely to
    rot: it names Telegram, Slack AND Discord."""
    gw = REPO / "src" / "prometheus" / "gateway"
    tg = set(re.findall(r'CommandHandler\("(\w+)"',
                        (gw / "telegram.py").read_text(encoding="utf-8")))
    slack = set(re.findall(r'command\("/prometheus-([a-z-]+)"\)',
                           (gw / "slack.py").read_text(encoding="utf-8")))
    discord = set(re.findall(r'_register\(\w+, "(\w+)"',
                             (gw / "discord.py").read_text(encoding="utf-8")))
    bullet = next(b for b in _bullets() if "Web chat lacks parity" in b)
    named = re.findall(r"`/(\w+)`", bullet)
    for cmd in named:
        for label, registered in (("Telegram", tg), ("Slack", slack),
                                  ("Discord", discord)):
            assert cmd in registered, (
                f"the note says /{cmd} works on {label}, but {label} "
                f"registers no such command"
            )


# ── bullet 7: GEPA ──────────────────────────────────────────────────

def test_gepa_only_counts_skill_tool_rows():
    src = (REPO / "src" / "prometheus" / "learning" / "gepa.py").read_text(
        encoding="utf-8")
    assert 'meta.get("tool_name") != "Skill"' in src, (
        "the note says only Skill-tool rows count toward GEPA's input; the "
        "filter that made that true is gone"
    )


def test_gepas_minimum_is_the_number_the_note_states():
    bullet = next(b for b in _bullets() if "GEPA has not been shown" in b)
    m = re.search(r"below a minimum of (\d+)", bullet)
    assert m, f"the minimum is no longer stated: {bullet[:160]}"
    assert _get(_template(), "learning.gepa_min_traces_required") == int(m.group(1))


def test_golden_traces_are_cloud_only():
    from prometheus.telemetry.tracker import _CLOUD_PROVIDERS, _LOCAL_PROVIDERS

    assert _CLOUD_PROVIDERS and not (_CLOUD_PROVIDERS & _LOCAL_PROVIDERS)
    schema = (REPO / "src" / "prometheus" / "telemetry" / "tracker.py").read_text(
        encoding="utf-8")
    assert "1 = cloud + success + zero retries" in schema, (
        "is_golden no longer documents itself as cloud-only — the note's "
        "'golden traces come from cloud providers only' may have gone stale"
    )


# ── bullet 8: the escalation cost cap ───────────────────────────────

def test_budget_usd_still_has_no_enforcing_reader():
    """The note says `budget_usd` does not stop spending. It is read into
    one dataclass field that nothing consumes. When someone wires it, this
    fails and the note gets rewritten — which is the point."""
    hits = []
    for path in (REPO / "src").rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "escalation_budget_usd" in line or "budget_usd" in line:
                hits.append(f"{path.relative_to(REPO)}:{i}: {line.strip()}")
    # Exactly two, both in the router's config plumbing: the dataclass field
    # and the line that fills it from config. Neither compares it to a spend.
    assert len(hits) == 2 and all("router/model_router.py" in h for h in hits), (
        "the `budget_usd` call sites changed — the note claims the cap is "
        "declared but not enforced. Re-read these and update the note if it "
        "now stops spending:\n  " + "\n  ".join(hits)
    )
