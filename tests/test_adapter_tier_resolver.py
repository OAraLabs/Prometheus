"""The adapter tier is decided from what the server says, and says why (WP-X.28, PR 2).

THE DEFECT
----------
The tier was chosen by matching the model's FILE NAME against
``config/model_registry.yaml``. A native tool-calling model the registry did
not know — Bonsai 2 27B, Ornith 1.5 9B, both Qwen-based — landed at ``full``:
tools out of the native field, into the prompt, replies read by a JSON reader.
The Bonsai tier sweep (611 runs) scored ``full`` 25 points under ``light``.
And nothing said which tier had been chosen, or why: ``full`` was silent.

THE EVIDENCE is committed. The production server's ``/props`` (recorded in the
``tool_calls`` golden) carries ``chat_template_caps`` and the template; the
mini's Ollama ``/api/show`` (recorded in every golden as the backend probe)
carries ``capabilities`` and the Go template. The fixtures here are those
payloads, not invented ones.

THE RULES (decided 2026-09-25): override > provider class > registry > chat
template > fallback. The registry precedes the template, so every listed
model keeps exactly the tier it has today; the template decides only for
models the registry does not list; the fallback stays ``full``, loudly.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml

from prometheus.adapter.formatter import GemmaFormatter, PassthroughFormatter, QwenFormatter
from prometheus.adapter.tier import (
    TIER_SOURCE_TEXT,
    TIER_SOURCES,
    TIERS,
    ToolTemplate,
    classify_template,
    resolve_tier,
)
from prometheus.adapter.validator import Strictness
from prometheus.providers.base import ModelProvider
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.providers.ollama import OllamaProvider

REPO = Path(__file__).resolve().parents[1]
PARITY = REPO / "tests" / "fixtures" / "parity"
REGISTRY = REPO / "config" / "model_registry.yaml"

BONSAI = "Ternary-Bonsai-2-27B-PQ2_0.gguf"          # the ladder's exact file name
ORNITH = "Ornith-1.5-9B-Q4_K_M.gguf"
PRODUCTION = "/models-root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"   # listed: qwen-3


def recorded_props() -> dict:
    trace = json.loads((PARITY / "tool_calls.trace.json").read_text())
    for exchange in trace["exchanges"]:
        if exchange.get("path") == "/props":
            return json.loads(exchange["body"])
    raise AssertionError("the tool_calls golden records no /props exchange")


def recorded_show() -> dict:
    trace = json.loads((PARITY / "repaired_tool_call.trace.json").read_text())
    for exchange in trace["exchanges"]:
        if exchange.get("path") == "/api/show":
            return json.loads(exchange["body"])
    raise AssertionError("the repaired_tool_call golden records no /api/show exchange")


@pytest.fixture(scope="module")
def props() -> dict:
    return recorded_props()


@pytest.fixture(scope="module")
def show() -> dict:
    return recorded_show()


@pytest.fixture(scope="module")
def qwen_xml(props) -> ToolTemplate:
    return classify_template(chat_template=props["chat_template"], caps=props["chat_template_caps"])


NOT_NATIVE = ToolTemplate(native=False, call_format=None, evidence="a template with no tool block")
UNREADABLE = ToolTemplate(native=None, call_format=None, evidence="nothing was readable")


# ---------------------------------------------------------------------------
# Reading a template
# ---------------------------------------------------------------------------

class TestClassifyTemplate:
    def test_the_recorded_4090_props_say_native_qwen_xml(self, props):
        verdict = classify_template(
            chat_template=props["chat_template"], caps=props["chat_template_caps"],
        )
        assert (verdict.native, verdict.call_format) == (True, "qwen-xml")
        assert "chat_template_caps" in verdict.evidence

    def test_the_recorded_ollama_show_says_native_qwen_json(self, show):
        verdict = classify_template(
            ollama_capabilities=show["capabilities"], ollama_template=show["template"],
        )
        assert (verdict.native, verdict.call_format) == (True, "qwen-json")
        assert "capabilities" in verdict.evidence

    def test_caps_that_deny_tools_are_not_native(self):
        verdict = classify_template(caps={"supports_tools": False, "supports_tool_calls": False})
        assert verdict.native is False and verdict.call_format is None

    def test_caps_need_both_halves(self):
        # Rendering a tools list without rendering calls back (or the reverse)
        # is not native tool calling.
        assert classify_template(caps={"supports_tools": True, "supports_tool_calls": False}).native is False

    def test_the_template_text_alone_decides_on_an_older_server(self, props):
        # A build that publishes chat_template but no chat_template_caps.
        verdict = classify_template(chat_template=props["chat_template"])
        assert (verdict.native, verdict.call_format) == (True, "qwen-xml")
        assert "chat_template" in verdict.evidence

    def test_a_template_with_no_tool_block_is_not_native(self):
        chatml = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        assert classify_template(chat_template=chatml).native is False

    def test_half_a_caps_dict_falls_to_the_template_text(self, props):
        verdict = classify_template(chat_template=props["chat_template"], caps={"supports_tools": True})
        assert verdict.native is True and "chat_template " in verdict.evidence

    def test_ollama_capabilities_without_tools(self):
        assert classify_template(ollama_capabilities=["completion"]).native is False

    def test_the_ollama_go_template_alone(self, show):
        verdict = classify_template(ollama_template=show["template"])
        assert (verdict.native, verdict.call_format) == (True, "qwen-json")

    def test_nothing_readable_is_unknown_not_false(self):
        verdict = classify_template()
        assert verdict.native is None and verdict.call_format is None
        assert classify_template(caps={}, chat_template="", ollama_template=" ").native is None

    def test_as_dict_round_trips(self, qwen_xml):
        assert qwen_xml.as_dict() == {
            "native": True, "call_format": "qwen-xml", "evidence": qwen_xml.evidence,
        }


# ---------------------------------------------------------------------------
# Deciding the tier: the fixed order
# ---------------------------------------------------------------------------

LISTED_NATIVE = {"key": "qwen-3", "display_name": "Qwen 3",
                 "capabilities": {"function_calling": {"supported": True, "requires": None}}}
LISTED_NOT_NATIVE = {"key": "plain", "display_name": "Plain",
                     "capabilities": {"function_calling": {"supported": False}}}


class TestResolveTier:
    def test_the_vocabulary(self):
        assert TIERS == ("off", "light", "full")
        assert set(TIER_SOURCE_TEXT) == set(TIER_SOURCES)

    def test_override_beats_everything(self, qwen_xml):
        d = resolve_tier(provider_name="llama_cpp", model_name=PRODUCTION, template=qwen_xml,
                         registry_entry=LISTED_NATIVE, override="full")
        assert (d.tier, d.source) == ("full", "override")
        assert "adapter.model_tiers" in d.detail

    def test_auto_is_absence_and_a_bad_value_is_refused(self):
        d = resolve_tier(provider_name="llama_cpp", model_name=BONSAI, override="auto")
        assert d.source == "fallback"
        with pytest.raises(ValueError, match="model_tiers"):
            resolve_tier(provider_name="llama_cpp", model_name=BONSAI, override="medium")

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini", "xai", "deepseek", "qwen"])
    def test_a_cloud_provider_is_off_whatever_else_says(self, provider, qwen_xml):
        d = resolve_tier(provider_name=provider, model_name="anything", template=qwen_xml,
                         registry_entry=LISTED_NATIVE)
        assert (d.tier, d.source) == ("off", "provider_class")

    def test_a_listed_native_model_is_light_by_the_registry(self, qwen_xml):
        d = resolve_tier(provider_name="llama_cpp", model_name=PRODUCTION, template=qwen_xml,
                         registry_entry=LISTED_NATIVE)
        assert (d.tier, d.source, d.disagreement) == ("light", "registry", None)
        assert "Qwen 3" in d.detail and d.call_format == "qwen-xml"

    def test_the_registry_precedes_a_disagreeing_template(self):
        # A listed model on a server whose template reports no tools (say,
        # started without --jinja): the registry decides, and the record says
        # they disagree so the boot line can warn.
        d = resolve_tier(provider_name="llama_cpp", model_name=PRODUCTION, template=NOT_NATIVE,
                         registry_entry=LISTED_NATIVE)
        assert (d.tier, d.source) == ("light", "registry")
        assert d.disagreement and "disagrees" in d.disagreement

    def test_a_listed_model_without_native_calling_is_full(self, qwen_xml):
        d = resolve_tier(provider_name="llama_cpp", model_name="plain-7b", template=qwen_xml,
                         registry_entry=LISTED_NOT_NATIVE)
        assert (d.tier, d.source) == ("full", "registry")
        assert d.disagreement  # the template says native, the registry says no

    def test_an_unlisted_model_follows_its_template(self, qwen_xml):
        d = resolve_tier(provider_name="llama_cpp", model_name=BONSAI, template=qwen_xml)
        assert (d.tier, d.source, d.call_format) == ("light", "template", "qwen-xml")
        assert "calls as qwen-xml" in d.detail
        d = resolve_tier(provider_name="ollama", model_name="ornith:9b", template=NOT_NATIVE)
        assert (d.tier, d.source) == ("full", "template")

    def test_an_unlisted_model_with_no_readable_template_falls_back_to_full(self):
        d = resolve_tier(provider_name="llama_cpp", model_name=BONSAI, template=UNREADABLE)
        assert (d.tier, d.source) == ("full", "fallback")
        assert "no registry entry" in d.detail and UNREADABLE.evidence in d.detail
        d = resolve_tier(provider_name="llama_cpp", model_name=BONSAI, template=None,
                         registry_note="the registry file is missing")
        assert (d.tier, d.source) == ("full", "fallback")
        assert "the registry file is missing" in d.detail and "not read" in d.detail

    def test_a_gguf_path_is_shown_by_its_file_name(self, qwen_xml):
        d = resolve_tier(provider_name="llama_cpp", model_name=PRODUCTION, registry_entry=LISTED_NATIVE)
        assert "Qwen3.8-27B-UD-Q4_K_XL.gguf" in d.detail and "/models-root" not in d.detail
        assert d.decided_for == PRODUCTION
