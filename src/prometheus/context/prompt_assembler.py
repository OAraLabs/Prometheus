# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/prompts/context.py
# License: MIT
# Modified: renamed imports to prometheus.*; restructured as a standalone
#           build_runtime_system_prompt() function with static/dynamic boundary;
#           added Layer 1 bootstrap loading (SOUL.md, AGENTS.md, MEMORY.md, USER.md)

"""Runtime prompt assembler for Prometheus.

Combines the static system prompt (identity + environment + tool schemas) with
dynamic per-turn content (PROMETHEUS.md, memory pointers, user model, task
state, loaded skills) separated by the ``SYSTEM_PROMPT_DYNAMIC_BOUNDARY``
marker.

Layer 1 bootstrap files (loaded from ``~/.prometheus/``):
  - **SOUL.md** — identity, personality, behavioral rules (static, every prompt)
  - **AGENTS.md** — agent registry, capabilities, spawn rules (static, every prompt)
  - **MEMORY.md** — persistent facts (dynamic, updated by agent)
  - **USER.md** — user model (dynamic, updated by agent)
"""

from __future__ import annotations

import logging
from pathlib import Path

from prometheus.config.paths import get_config_dir
from prometheus.context.environment import get_environment_info
from prometheus.context.prometheusmd import load_project_files_prompt
from prometheus.context.system_prompt import (
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    build_system_prompt,
)

log = logging.getLogger(__name__)


# Etiquette prepended to the injected memory section. Memory is background
# knowledge, not a data source to cite — the model should apply it naturally
# without narrating retrieval (modeled on the consumer prompt's memory rules).
_MEMORY_ETIQUETTE = (
    "Treat the facts below as things you already know about the user and the "
    "work — apply them naturally when relevant. Do not narrate retrieval or "
    'point at this section: avoid "based on my memory", "I can see", "your '
    'profile/data", and similar. If a fact is relevant, just use it; if it is '
    "not, ignore it."
)


def _load_bootstrap_file(filename: str) -> str | None:
    """Load a bootstrap file from ``~/.prometheus/``.

    Returns the file content stripped of leading/trailing whitespace,
    or ``None`` if the file does not exist or is empty.
    """
    path = get_config_dir() / filename
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
        return content or None
    except OSError:
        log.warning("Failed to read bootstrap file: %s", path)
        return None


def _load_memory_and_user() -> str:
    """Load MEMORY.md + USER.md content formatted for the system prompt.

    Uses :func:`prometheus.memory.hermes_memory_tool.format_memory_for_prompt`
    which reads both files, formats them as markdown sections, and applies
    security scanning.
    """
    try:
        from prometheus.memory.hermes_memory_tool import format_memory_for_prompt
        return format_memory_for_prompt()
    except Exception:
        log.debug("Could not load memory/user files for prompt", exc_info=True)
        return ""


def _load_anatomy_summary() -> str | None:
    """Load compact infrastructure summary from ANATOMY.md for the system prompt.

    Extracts only the Active Configuration section (~200-300 tokens),
    skipping project configs, history, and full Mermaid diagrams.
    The agent can use the anatomy tool for full details on demand.
    """
    import re

    path = get_config_dir() / "ANATOMY.md"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    # Extract just the Active Configuration section
    match = re.search(
        r"## Active Configuration\n(.*?)(?=\n## |\Z)",
        text,
        re.DOTALL,
    )
    if not match:
        return None

    active = match.group(1).strip()
    if not active:
        return None

    # The clarifier is injected HERE (not only baked by anatomy_writer) so it
    # also covers ANATOMY.md files on disk written by older scanners: any
    # "Model/Loaded" entries below describe the LOCAL GPU node, and a cloud
    # model serving an overridden session must not adopt them as its identity.
    return (
        "## Infrastructure\n"
        "_Auto-detected local hardware/software inventory. Any model named "
        "below is the LOCAL backend, not necessarily the model serving this "
        "conversation — your identity is the Environment section's "
        "`- Model:` line._\n"
        f"{active}"
    )


def build_runtime_system_prompt(
    *,
    cwd: str,
    config: dict | None = None,
    memory_content: str = "",
    skills: list | None = None,
    task_state: str = "",
    profile: object | None = None,
) -> str:
    """Assemble the full runtime system prompt.

    The result is structured as::

        STATIC (cached / stable in KV cache):
        ├── 1. Bootstrap files (controlled by profile or config)
        ├── 2. Base system prompt (tool usage, coding rules, environment)
        ├── 3. Environment info (OS, shell, git, model)
        └── ─── SYSTEM_PROMPT_DYNAMIC_BOUNDARY ───
        DYNAMIC (changes per turn):
        ├── 4. Reasoning settings / session mode
        ├── 5. Available skills
        ├── 6. PROMETHEUS.md project instructions
        ├── 7. MEMORY.md + USER.md content
        ├── 8. Current task / plan state
        └── 9. Loaded skill content

    Parameters
    ----------
    cwd:
        Current working directory — used for environment detection and
        PROMETHEUS.md discovery.
    config:
        Optional configuration dict.  Recognised keys:

        - ``"system_prompt"`` — custom base prompt (replaces default identity).
        - ``"fast_mode"``     — if truthy, adds a fast-mode hint.
        - ``"effort"``        — reasoning effort level label.
        - ``"passes"``        — reasoning pass count label.
        - ``"bootstrap"``     — bootstrap config (load_soul, load_agents).
    memory_content:
        Pre-formatted memory content to inject (e.g. from MemoryPointer).
        If empty, MEMORY.md + USER.md are loaded automatically.
    skills:
        List of skill dicts with ``"name"`` and ``"description"`` keys.
    task_state:
        Serialised task-tracking state (e.g. todo list) to inject.

        ⚠ There is deliberately no ``loaded_skill_content``. It existed from
        the initial commit, rendered a ``# Loaded Skill`` section, and **no
        caller ever passed it** — so that section had never appeared in the
        product. A skill's body reaches the model as a TOOL RESULT instead
        (``tools/builtin/skill.py`` returns ``ToolResult(output=skill.content)``),
        which is the wired path. Keeping the parameter advertised a
        capability the assembler did not have. If prompt-RESIDENT skill
        loading is wanted — it is a real difference, a tool result can be
        compacted away while a prompt section persists — it should arrive
        with a caller and a test rather than sit here as an unwired hook.
    profile:
        An :class:`~prometheus.config.profiles.AgentProfile` controlling which
        bootstrap files to load.  If ``None``, falls back to config toggles.
    """
    if skills is None:
        skills = []
    if config is None:
        config = {}

    bootstrap_cfg = config.get("bootstrap", {})

    # Resolve which bootstrap files to load
    profile_bootstrap: list[str] | None = None
    if profile is not None:
        from prometheus.config.profiles import AgentProfile
        if isinstance(profile, AgentProfile):
            profile_bootstrap = profile.bootstrap_files

    # ------------------------------------------------------------------
    # Static section
    # ------------------------------------------------------------------
    static_sections: list[str] = []

    if profile_bootstrap is not None:
        # Profile controls which bootstrap files load
        for filename in profile_bootstrap:
            content = _load_bootstrap_file(filename)
            if content:
                static_sections.append(
                    f"<!-- Bootstrap: ~/.prometheus/{filename} -->\n{content}"
                )
    else:
        # Legacy path: individual config toggles
        if bootstrap_cfg.get("load_soul", True):
            soul = _load_bootstrap_file("SOUL.md")
            if soul:
                static_sections.append(
                    f"<!-- Bootstrap: ~/.prometheus/SOUL.md -->\n{soul}"
                )

        if bootstrap_cfg.get("load_agents", True):
            agents = _load_bootstrap_file("AGENTS.md")
            if agents:
                static_sections.append(
                    f"<!-- Bootstrap: ~/.prometheus/AGENTS.md -->\n{agents}"
                )

        # ANATOMY.md — compact infrastructure summary (Layer 1.5)
        anatomy_cfg = config.get("anatomy", {})
        if anatomy_cfg.get("include_in_system_prompt", True):
            anatomy_summary = _load_anatomy_summary()
            if anatomy_summary:
                static_sections.append(
                    f"<!-- Bootstrap: ~/.prometheus/ANATOMY.md -->\n{anatomy_summary}"
                )

    # 3-4. Base system prompt + environment info
    custom_prompt = config.get("system_prompt")
    env = get_environment_info(cwd=cwd)
    model_cfg = config.get("model", {})
    env.model_name = model_cfg.get("model", "")
    env.model_provider = model_cfg.get("provider", "")
    static_prompt = build_system_prompt(custom_prompt=custom_prompt, env=env)
    static_sections.append(static_prompt)

    # ------------------------------------------------------------------
    # Dynamic section
    # ------------------------------------------------------------------
    dynamic_sections: list[str] = []

    # Session mode hints
    if config.get("fast_mode"):
        dynamic_sections.append(
            "# Session Mode\n"
            "Fast mode is enabled. Prefer concise replies, minimal tool use, "
            "and quicker progress over exhaustive exploration."
        )

    effort = config.get("effort", "standard")
    passes = config.get("passes", 1)
    dynamic_sections.append(
        "# Reasoning Settings\n"
        f"- Effort: {effort}\n"
        f"- Passes: {passes}\n"
        "Adjust depth and iteration count to match these settings while "
        "still completing the task."
    )

    # Skills — two tiers. Core skills (frontmatter `tier: core` /
    # `autoload: true`) get name+description rendered inline so the model
    # reaches for them reflexively; the rest stay tool_search-only, keeping
    # the standing prompt cost at ~core-count entries instead of the full
    # catalog (~8.5k tokens at 146 skills). Everything remains loadable via
    # the skill tool; nothing is lost by being un-rendered.
    if skills:
        core = [s for s in skills if isinstance(s, dict) and s.get("core")]
        tail_count = len(skills) - len(core)
        lines = []
        if core:
            lines.append("## Core skills (always available)")
            for s in core:
                name = str(s.get("name", ""))
                desc = str(s.get("description", ""))
                lines.append(f"- **{name}**: {desc}" if desc else f"- **{name}**")
        if tail_count > 0:
            lines.append(
                f"## {tail_count} additional skill{'s' if tail_count != 1 else ''} "
                "available on demand. "
                "Use tool_search to find skills for any task you're unsure how "
                "to approach, then use the skill tool to load the skill's instructions."
            )
        elif not core:
            # skills list present but empty-ish / no dicts — keep old hint
            lines.append(
                "You have additional skills available beyond your loaded tools. "
                "Use tool_search to find skills for any task you're unsure how "
                "to approach, then use the skill tool to load the skill's instructions."
            )
        if lines:
            dynamic_sections.append("# Available Skills\n\n" + "\n".join(lines))

    # Project instruction files (PROMETHEUS.md, HERMES.md, CLAUDE.md, etc.)
    # Stacked loading: collects files from all directory levels walking upward
    context_cfg = config.get("context", {})
    stack_project_files = context_cfg.get("stack_project_files", True)
    max_chars = context_cfg.get("project_file_max_chars", 12000)
    # AGENTS.md is already gated by bootstrap.load_agents above. Discovery must
    # honor that gate too, or turning the registry off would silently be undone
    # the moment an AGENTS.md sits anywhere at or above cwd.
    excluded_conventions = (
        () if bootstrap_cfg.get("load_agents", True) else ("AGENTS.md",)
    )
    project_prompt = load_project_files_prompt(
        cwd,
        max_chars_per_file=max_chars,
        stack=stack_project_files,
        exclude=excluded_conventions,
    )
    if project_prompt:
        dynamic_sections.append(project_prompt)

    # MEMORY.md + USER.md — auto-load if caller didn't provide memory_content
    if not memory_content:
        memory_content = _load_memory_and_user()
    if memory_content:
        dynamic_sections.append(f"# Memory\n\n{_MEMORY_ETIQUETTE}\n\n{memory_content}")

    # User's saved files — so the agent knows what files exist without searching
    try:
        from prometheus.utils.user_files import files_context_block
        files_block = files_context_block()
        if files_block:
            dynamic_sections.append(f"# Saved Files\n\n{files_block}")
    except Exception:
        pass  # Non-critical — skip if module not available

    # Task state
    if task_state:
        dynamic_sections.append(f"# Current Task State\n\n{task_state}")

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    static_part = "\n\n".join(s for s in static_sections if s.strip())
    dynamic_part = "\n\n".join(s for s in dynamic_sections if s.strip())

    return f"{static_part}\n\n{SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n\n{dynamic_part}"
