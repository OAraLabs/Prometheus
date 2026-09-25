"""WP-X.14 — the primary's identity line names the model and its provider.

Every primary turn was told::

    - Model: /models-root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf (provider: unknown)

(every golden trace under tests/fixtures/parity/ showed it). The boot line was
right — ``(provider: llama_cpp)`` — and the routing step rewrote it, because
``ModelRouter._route_primary`` built its RouteDecision with no provider name
and the rewrite turned "no name" into a placeholder. The model name was the
GGUF's path as the server reports it.

Pinned here:

* the primary decision carries the primary's provider name, from the same
  source telemetry uses, so the prompt and the telemetry rows agree;
* one display name for the line — a ``.gguf`` path shows as the file name —
  used by the boot line and the rewrite alike, and by nothing else;
* the rewrite never downgrades a known provider to "unknown" (fails on the
  code before this change);
* the primary line is the same at boot and after routing; fallback and
  override lines are what they were.
"""

from __future__ import annotations

from prometheus.context.environment import EnvironmentInfo
from prometheus.context.system_prompt import (
    _format_environment_section,
    model_display_name,
    rewrite_model_identity,
)
from prometheus.engine.agent_loop import LoopContext, _provider_name_for_telemetry
from prometheus.engine.messages import ConversationMessage
from prometheus.engine.stages.routing import route_turn
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.providers.openai_compat import OpenAICompatProvider
from prometheus.router import ModelRouter, RouteReason, RouterConfig

# What the daemon detects at boot from a llama.cpp server: the GGUF's path.
SERVED_ID = "/models-root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"
PRIMARY_LINE = "- Model: Qwen3.8-27B-UD-Q4_K_XL (provider: llama_cpp)"
NOT_LOCAL = (
    " — the ACTIVE model serving this conversation; any model in the"
    " Infrastructure section is a separate local backend, not you"
)


def _env(model_name: str, provider: str) -> EnvironmentInfo:
    return EnvironmentInfo(
        os_name="Linux", os_version="6.8.0", platform_machine="x86_64", shell="bash",
        cwd="/tmp/prometheus-parity/cwd", home_dir="/tmp/prometheus-parity/home",
        date="2026-09-25", python_version="3.11.15", is_git_repo=False,
        model_name=model_name, model_provider=provider,
    )


def _model_line(prompt: str) -> str:
    lines = [ln for ln in prompt.splitlines() if ln.startswith("- Model:")]
    assert len(lines) == 1, prompt
    return lines[0]


# ── the display name ────────────────────────────────────────────────────────


def test_a_gguf_path_shows_as_the_file_name():
    assert model_display_name(SERVED_ID) == "Qwen3.8-27B-UD-Q4_K_XL"
    assert model_display_name("Qwen3.8-27B.gguf") == "Qwen3.8-27B"
    assert model_display_name("/models/gemma-4-26B.GGUF") == "gemma-4-26B"


def test_every_other_name_is_shown_as_it_is():
    for name in ("qwen3.8-27b", "qwen2.5:7b-instruct", "claude-haiku-4-5-20251001",
                 "gpt-4o", "/models/not-a-gguf"):
        assert model_display_name(name) == name


# ── the boot line ───────────────────────────────────────────────────────────


def test_the_boot_line_names_the_model_and_its_provider():
    section = _format_environment_section(_env(SERVED_ID, "llama_cpp"))
    assert _model_line(section) == PRIMARY_LINE


def test_the_boot_line_keeps_a_registry_name_as_it_is():
    section = _format_environment_section(_env("qwen3.8-27b", "llama_cpp"))
    assert _model_line(section) == "- Model: qwen3.8-27b (provider: llama_cpp)"


# ── the primary decision ────────────────────────────────────────────────────


def _router(primary) -> ModelRouter:
    return ModelRouter(config=RouterConfig(), primary_provider=primary,
                       primary_adapter=None, primary_model=SERVED_ID)


def test_the_primary_decision_names_the_primary_provider():
    primary = LlamaCppProvider(base_url="http://127.0.0.1:1")
    decision = _router(primary).route("hello", context={"session_id": "desktop:x"})
    assert decision.reason is RouteReason.PRIMARY
    assert decision.provider_name == "llama_cpp"
    # The raw served id, untouched: registry matching and telemetry read this.
    assert decision.model_name == SERVED_ID


def test_the_primary_provider_name_is_the_one_telemetry_records():
    """Same source, so the identity line and the tool_calls rows cannot disagree."""
    for primary in (
        LlamaCppProvider(base_url="http://127.0.0.1:1"),
        OpenAICompatProvider(base_url="http://127.0.0.1:1", api_key="x",
                             model="qwen3.8-max", provider_name="qwen"),
    ):
        decision = _router(primary).route("hello", context={"session_id": "desktop:x"})
        assert decision.provider_name == _provider_name_for_telemetry(primary)
    assert _router(OpenAICompatProvider(
        base_url="http://127.0.0.1:1", api_key="x", provider_name="qwen",
    )).route("hello").provider_name == "qwen"


# ── the line after routing ──────────────────────────────────────────────────


def test_the_primary_line_is_the_same_at_boot_and_after_routing():
    primary = LlamaCppProvider(base_url="http://127.0.0.1:1")
    boot_prompt = "You are Prometheus.\n\n" + _format_environment_section(_env(SERVED_ID, "llama_cpp"))
    ctx = LoopContext(provider=primary, model=SERVED_ID, system_prompt=boot_prompt,
                      max_tokens=64, model_router=_router(primary))

    route_turn(ctx, [ConversationMessage.from_user_text("hi")], session_id="desktop:x")

    assert _model_line(ctx.system_prompt) == PRIMARY_LINE
    assert ctx.system_prompt == boot_prompt, "a primary route leaves the identity line alone"
    assert "unknown" not in ctx.system_prompt
    assert ctx.model == SERVED_ID, "only the prompt line shows the display name"


# ── the rewrite ─────────────────────────────────────────────────────────────


def test_the_rewrite_never_downgrades_a_known_provider():
    """Fails before this change: a caller with no name to give produced "(provider: unknown)"."""
    boot = "- Model: qwen3.8-27b (provider: llama_cpp)"
    for no_name in ("unknown", ""):
        out = rewrite_model_identity(boot, model_name="qwen3.8-27b", provider_name=no_name,
                                     serving_is_local_backend=True)
        assert out == boot, (no_name, out)


def test_a_line_that_never_named_a_provider_still_says_so():
    out = rewrite_model_identity("- Model: qwen3.8-27b", model_name="qwen3.8-27b",
                                 provider_name="", serving_is_local_backend=True)
    assert out == "- Model: qwen3.8-27b (provider: unknown)"


def test_the_lines_provider_stands_only_for_the_same_model():
    """The prompt carries the line from turn to turn. A later decision built without a
    provider must not pair its model with the previous turn's provider: a local model
    shown as "(provider: anthropic)" is a false statement, worse than "unknown"."""
    cloud = "- Model: claude-haiku-4-5-20251001 (provider: anthropic)" + NOT_LOCAL
    for no_name in ("", "unknown"):
        out = rewrite_model_identity(cloud, model_name="qwen3.5:9b", provider_name=no_name,
                                     serving_is_local_backend=True)
        assert out == "- Model: qwen3.5:9b (provider: unknown)", (no_name, out)
    # The same model with an empty provider keeps it — also when the caller names the
    # model by its raw served id and the line by its display name (the primary's own case).
    for name in (SERVED_ID, "Qwen3.8-27B-UD-Q4_K_XL"):
        out = rewrite_model_identity(PRIMARY_LINE, model_name=name, provider_name="",
                                     serving_is_local_backend=True)
        assert out == PRIMARY_LINE, (name, out)


def test_fallback_and_override_lines_are_unchanged():
    boot = "# Environment\n" + PRIMARY_LINE + "\n- Git: yes"
    # A fallback to a second local backend (the agent loop's recovery caller).
    assert _model_line(rewrite_model_identity(
        boot, model_name="qwen3.5:9b", provider_name="ollama", serving_is_local_backend=True,
    )) == "- Model: qwen3.5:9b (provider: ollama)"
    # A /claude override: the cloud model is not the local backend, and is told so.
    assert _model_line(rewrite_model_identity(
        boot, model_name="claude-haiku-4-5-20251001", provider_name="anthropic",
        serving_is_local_backend=False,
    )) == "- Model: claude-haiku-4-5-20251001 (provider: anthropic)" + NOT_LOCAL
    # A named-backend override (/4090) serving another GGUF: the same display rule.
    assert _model_line(rewrite_model_identity(
        boot, model_name="/models/Qwen3.6-27B.gguf", provider_name="llama_cpp",
        serving_is_local_backend=True,
    )) == "- Model: Qwen3.6-27B (provider: llama_cpp)"
