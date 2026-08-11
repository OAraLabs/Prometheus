"""Every registered tool is either ADVERTISED or has a TESTED discovery path.

THE HOLE THIS CLOSES. Deferred loading is ``enabled: auto``, which resolves to
ON for every local provider, so the shipped default advertises 8 of 51 tools.
The other 43 are invisible unless the model thinks to call ``tool_search``.
Nothing forced a decision about which side of that line a tool belonged on, and
nothing noticed when a new one landed on the invisible side — so a new tool was
invisible BY DEFAULT and silently so.

``vault_search`` shipped that way: registered, verified registered by a test,
logged as registered at daemon startup, and unusable. Asked to read the brain
vault, Prometheus correctly said it had no such capability.

So this file makes the classification mandatory and exhaustive: every tool is in
``always_loaded`` or in :data:`DEFERRED_BY_DESIGN` with a reason AND a query
proving ``tool_search`` actually surfaces it. A new tool in neither fails the
build, which is the point — the default becomes loud.

WHAT THIS PROVES, AND WHAT IT DOES NOT. It proves a tool is OFFERED, or findable
when asked for. It does NOT prove the model uses it. Tonight's failure was at
the offer step, so this closes the real hole — but "tool_search existed and the
model never reached for it" is a behavioural gap no structural test can catch.
Stated here so a green run is not read as more than it is.
"""

from __future__ import annotations

import pytest

from tests.support.advertisement import (
    advertised_names,
    assert_tool_discoverable,
    live_always_loaded,
    registered_names,
    template_always_loaded,
    tool_search_hits,
)

# Tool -> (why it is not advertised, a query a user might plausibly prompt that
# must surface it via tool_search).
#
# SEEDED WITH THE CURRENT STATE, DELIBERATELY. Every entry below describes what
# ships today; nothing is promoted in this PR, so the guard is behaviour-neutral
# and reviewable on its own. The reasons are written to be argued with — several
# are marked ⚠, meaning "this is defensible today and probably wrong", and those
# are the promotion candidates for the follow-up review.
DEFERRED_BY_DESIGN: dict[str, tuple[str, str]] = {
    # ── Agent / session control: invoked by name once the model knows the
    # pattern; a user rarely asks for them obliquely.
    "agent": ("Subagent spawn — the model reaches for it deliberately, not from an oblique prompt.", "launch a subagent"),
    "sessions_spawn": ("Background session creation; paired with sessions_list/send.", "spawn a background agent session"),
    "sessions_list": ("Enumerates background sessions; only useful once one exists.", "list agent sessions"),
    "sessions_send": ("Sends into a running session; presupposes a session id.", "send a message to a running session"),
    "ask_user": ("Clarification prompt — the loop surfaces it, not the user.", "ask the user a question"),

    # ── Task management: a family the model discovers together once
    # task_create (which IS advertised) has been used.
    "task_get": ("Reached after task_create, which is always_loaded.", "get background task status"),
    "task_list": ("Same family as task_create.", "list background tasks"),
    "task_output": ("Same family; needs a task id.", "read background task output log"),
    "task_stop": ("Same family; needs a task id.", "stop a background task"),
    "task_update": ("Same family; needs a task id.", "update a task's progress"),

    # ── Cron: explicit scheduling requests name the concept, which scores well.
    "cron_create": ("Scheduling is always asked for explicitly ('every morning at 8').", "create a cron job schedule"),
    "cron_list": ("Only meaningful once a job exists, and the user who created one asks about it by name.", "list cron jobs"),
    "cron_delete": ("Destructive and needs an exact job name, so it follows a cron_list rather than a cold prompt.", "delete a cron job"),

    # ── SYMBIOTE: a deliberate multi-step capability workflow, never incidental.
    "symbiote_scout": ("Capability-gap workflow, entered explicitly.", "search github for a capability gap"),
    "symbiote_harvest": ("Step 2 of the SYMBIOTE workflow.", "harvest modules from a repository"),
    "symbiote_graft": ("Step 3 of the SYMBIOTE workflow.", "graft adapted files with provenance"),
    "symbiote_status": ("Status read for an in-flight SYMBIOTE session.", "symbiote session state"),

    # ── LCM history: the model's own conversation archive. Recall already rides
    # the prompt, so these are for explicit archaeology.
    "lcm_grep": ("Explicit history search; passive recall covers the ambient case.", "search conversation history"),
    "lcm_expand": ("Needs a node id, which only an lcm_grep result supplies — it cannot be the first call in a chain.", "expand a summary node"),
    "lcm_expand_query": ("Overlaps lcm_grep for the entry point; its value is expanding what grep already located.", "search compressed history for context"),
    "lcm_describe": ("Metadata for a summary node; needs a node id.", "summary node metadata"),

    # ── Media generation: expensive and/or keyed, and always requested by name.
    "image_generate": ("Paid/keyed; users ask for 'an image' explicitly.", "generate an image from a prompt"),
    "video_generate": ("PAID (Kling). Explicit request only — must not be ambient.", "generate a video from a prompt"),
    "tts": ("Voice output is a mode, not usually a tool request.", "convert text to speech"),

    # ── Infra / niche.
    "browser": ("Heavy (playwright); bash+curl covers most fetching.", "headless browser automation"),
    "dashboard": ("Serves ad-hoc HTML; rare and explicit.", "serve html on a local url"),
    "notebook_edit": ("Jupyter-specific; edit_file covers the general case.", "edit a jupyter notebook cell"),
    "github_search": ("Overlaps symbiote_scout; explicit.", "search github repositories"),
    "message": ("Outbound send to Discord/Slack/Telegram — explicit by nature.", "send a message to slack or discord"),
    "skill": ("Skill bodies are injected by the skill system, not usually fetched.", "read a skill by name"),
    "todo_write": ("Project checklist append; niche.", "append a todo item"),
    "sentinel_status": ("Subsystem introspection; diagnostic.", "sentinel subsystem status"),
    "wiki_lint": ("Maintenance operation, run deliberately.", "lint the wiki for broken links"),
    "wiki_compile": ("Maintenance operation; the compiler also runs automatically.", "compile the wiki from memory facts"),

    # ── ⚠ PROMOTION CANDIDATES. Defensible today, probably wrong. Each is a tool
    # a user reaches for with phrasing that gives the model NO signal to search
    # its own tool list — which is precisely how vault_search failed.
    "web_search": ("⚠ 'look that up' / 'what's the latest on X' gives no signal to search the tool list.", "search the web"),
    "web_fetch": ("⚠ A bare URL in a prompt gives no signal to search the tool list.", "fetch a url and return its text"),
    "wiki_query": ("⚠ 'what do you know about X' gives no signal to search the tool list.", "search the wiki for knowledge"),
    "vault_search": ("⚠ Same failure mode, observed live 2026-08-10. Promoted in the operator's live config; the shipped default keeps it deferred because a fresh install has no vault.", "search the brain vault"),
    "vault_read": ("⚠ Paired with vault_search; same reasoning.", "read a brain vault page"),
    "memory": ("⚠ 'remember that I...' is a durable-fact signal the model may not connect to a tool.", "manage persistent memory entries"),
    "anatomy": ("Infrastructure introspection; /anatomy and /status cover the human path.", "query infrastructure hardware and gpu state"),
    "download_file": ("bash+curl covers it; explicit by nature.", "download a file from a url"),
    "youtube_transcript": ("A YouTube URL is a strong explicit signal.", "fetch a youtube transcript"),
}


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------

def test_every_registered_tool_is_classified():
    """No tool may be silently invisible. THE point of this file.

    A new tool that is neither advertised nor deliberately deferred fails here,
    so the author has to decide — instead of discovering months later that the
    feature never worked.
    """
    registered = registered_names()
    always = template_always_loaded()
    classified = always | set(DEFERRED_BY_DESIGN)

    unclassified = sorted(registered - classified)
    assert not unclassified, (
        f"{len(unclassified)} tool(s) are registered but neither advertised nor "
        f"classified: {unclassified}\n"
        f"  They are INVISIBLE to the model: registered, never offered, and with "
        f"no tested discovery path. Add each to "
        f"tools.deferred_loading.always_loaded (if a user would expect it to "
        f"just work) or to DEFERRED_BY_DESIGN with a reason and a discovery "
        f"query."
    )


def test_the_classification_is_honest():
    """A stale entry masks real drift — same discipline as KNOWN_UNVERIFIED_DRIFT."""
    registered = registered_names()
    ghosts = sorted(set(DEFERRED_BY_DESIGN) - registered)
    assert not ghosts, (
        f"{ghosts} are in DEFERRED_BY_DESIGN but no longer registered — remove them"
    )
    both = sorted(set(DEFERRED_BY_DESIGN) & template_always_loaded())
    assert not both, (
        f"{both} are BOTH advertised and listed as deferred-by-design. The "
        f"classification must be exclusive or it stops meaning anything."
    )


@pytest.mark.parametrize(
    "name,query",
    sorted((n, q) for n, (_reason, q) in DEFERRED_BY_DESIGN.items()),
)
def test_every_deferred_tool_is_actually_discoverable(name, query):
    """The discovery path is a CLAIM until it is executed.

    tool_search shows the model its top 5 of 51. A deferred tool that never
    places for any plausible query is registered, unadvertised and unfindable —
    dead code with a docstring. This is the admission half: the guard above
    proves nothing is unclassified, this proves the classification is not a
    polite fiction.
    """
    assert_tool_discoverable(name, query)


def test_every_deferred_tool_has_a_real_reason():
    """An empty or placeholder reason is an unmade decision."""
    thin = sorted(
        n for n, (reason, _q) in DEFERRED_BY_DESIGN.items()
        if len(reason.strip()) < 25 or reason.strip().lower() in {"todo", "n/a", "niche"}
    )
    assert not thin, f"these entries need a real reason, not a placeholder: {thin}"


# ---------------------------------------------------------------------------
# The advertised side — admission, per §2c
# ---------------------------------------------------------------------------

def test_the_advertised_set_is_not_empty_or_everything():
    """Both degenerate ends would make every other test here vacuous: an empty
    set fails all advertisement assertions for the wrong reason, and a full set
    means deferred loading is off and the guard is asleep."""
    advertised = advertised_names()
    registered = registered_names()
    assert advertised, "nothing is advertised — the loader is not honouring always_loaded"
    assert advertised < registered, (
        "every tool is advertised — deferred loading is disabled, so this guard "
        "is passing vacuously and would not catch an invisible tool"
    )


def test_the_advertised_set_matches_the_configured_one():
    """Read through the loader, not the config key, so a loader that stops
    honouring always_loaded is caught rather than papered over."""
    assert advertised_names() == template_always_loaded()


def test_the_core_file_tools_are_advertised():
    """The floor: a model with no file tools cannot do the job at all."""
    advertised = advertised_names()
    for name in ("bash", "read_file", "write_file", "edit_file", "grep", "glob"):
        assert name in advertised, f"{name} is not advertised"


def test_tool_search_itself_is_advertised():
    """Load-bearing: it is the ONLY route to the 40-odd deferred tools. If it is
    ever deferred, every DEFERRED_BY_DESIGN entry becomes unreachable at once
    and this file's discovery claims all quietly become false."""
    assert "tool_search" in advertised_names(), (
        "tool_search is not advertised — every deferred tool just became "
        "unreachable, and nothing else in this file would have noticed"
    )


# ---------------------------------------------------------------------------
# The operator's live config, when present
# ---------------------------------------------------------------------------

def test_live_always_loaded_names_are_real_tools():
    """A typo in the operator's always_loaded silently advertises nothing.

    Does not skip when the live config is absent — it asserts that fact
    explicitly instead, because a silently-skipping test is how a suite goes
    green while covering nothing (cf. test_api_code.py skipping for want of an
    unrelated extra).
    """
    live = live_always_loaded()
    if live is None:
        assert not (
            __import__("tests.support.advertisement", fromlist=["LIVE_CONFIG"]).LIVE_CONFIG
        ).exists(), "live config exists but was not read"
        return
    unknown = sorted(live - registered_names())
    assert not unknown, (
        f"the live config's always_loaded names {unknown}, which are not "
        f"registered tools — those entries advertise nothing at all"
    )


def test_tool_search_would_find_the_vault_tools():
    """The specific regression, pinned. The tools were unreachable for a user
    asking about the brain vault; tool_search must at least be able to surface
    them for someone who asks in those words."""
    for query in ("brain vault", "second brain knowledge"):
        hits = tool_search_hits(query)
        assert "vault_search" in hits, f"tool_search({query!r}) -> {hits}"
