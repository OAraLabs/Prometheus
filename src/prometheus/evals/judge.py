"""PrometheusJudge — LLM-as-judge using the local llama.cpp endpoint.

Evaluates agent outputs against expected behavior descriptions.
Uses the same OpenAI-compatible /v1/chat/completions API as the main model.

Key reliability feature (Sprint 14): **Constrained decoding** via llama.cpp's
``response_format`` with ``json_schema`` type. The server converts the schema
to a GBNF grammar and masks invalid tokens at decode time — the model
physically cannot produce invalid JSON. This eliminates parse failures that
plagued G-Eval and raw JSON prompting with local models.

That makes a reply with no verdict rare, not impossible: a server can ignore
``response_format``, and a reply can come back empty or cut short. Every
reply is read by ONE strict parser, ``parse_judge_reply``. A reply that holds
no verdict comes back ``unparseable`` with no score. It used to become 0.0
(a model failure) or the first number in its text (a stray "1." read as a
pass).

Supports two evaluation modes:
- evaluate(): JSON with constrained decoding (primary, used by metrics)
- evaluate_geval(): G-Eval chain-of-thought (kept for manual/debug use)
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

# JSON Schema for judge scoring responses.
# Passed to llama.cpp's response_format — converted to GBNF grammar
# under the hood, constraining token generation at decode time.
JUDGE_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}

_JUDGE_SYSTEM_PROMPT = """\
/no_think
You are a strict evaluation judge. Do NOT use internal reasoning. Respond directly.

Rate the agent's task completion from 0.0 to 1.0:
- 1.0 = Task fully completed as expected
- 0.7 = Task mostly completed with minor issues
- 0.5 = Task partially completed
- 0.3 = Task attempted but largely failed
- 0.0 = Task not attempted or completely wrong

Respond with a JSON object: {"score": <float>, "reasoning": "<brief explanation>"}
"""

_GEVAL_SYSTEM_PROMPT = """\
/no_think
You are a strict evaluation judge. Do NOT use internal reasoning. Respond directly.

Evaluate an AI agent's output by assessing each criterion in one sentence, then give a final score.

Rules:
1. For each criterion, write one brief sentence of assessment.
2. After all criteria, you MUST write your final score on its own line.
3. The score line format is exactly: SCORE: 0.X (a number between 0.0 and 1.0)

Example output format:
1. The agent attempted the task. Yes, it ran the correct command.
2. The output is accurate. It matches what was expected.
3. No fabricated information. All data came from tool results.
SCORE: 0.85
"""


VERDICT_PARSED = "parsed"
VERDICT_UNPARSEABLE = "unparseable"

# The two forms a judge is asked to reply in.
JSON_REPLY = "json"      # evaluate(): {"score": <0-1>, "reasoning": "..."}
GEVAL_REPLY = "geval"    # evaluate_geval(): one sentence per criterion, then SCORE: <0-1>


def _is_score(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0.0 <= value <= 1.0)


@dataclass
class JudgeVerdict:
    """Result of an LLM judge evaluation.

    ``status`` says whether the judge gave a verdict, so no caller has to
    guess from the number:

    * ``parsed``: ``score`` is the judge's own number, finite, in [0, 1].
    * ``unparseable``: the reply held no verdict. ``score`` is None, never
      0.0 (that reads as a model failure) and never a number found elsewhere
      in the text. ``reasoning`` says why; ``raw_response`` keeps the reply.
    """

    score: float | None
    reasoning: str
    raw_response: str
    status: str = VERDICT_PARSED

    def __post_init__(self) -> None:
        if self.status == VERDICT_PARSED:
            if not _is_score(self.score):
                raise ValueError(
                    f"a parsed verdict needs a finite score in [0, 1], got {self.score!r}")
        elif self.status == VERDICT_UNPARSEABLE:
            if self.score is not None:
                raise ValueError(f"an unparseable verdict has no score, got {self.score!r}")
        else:
            raise ValueError(f"unknown verdict status {self.status!r}")


# A G-Eval score line OPENS with "SCORE:" (any case; markdown emphasis
# allowed), and its value is one plain number with nothing else on the line.
_SCORE_LABEL = re.compile(r"^[\s*_`]*score[\s*_`]*:", re.IGNORECASE)
_SCORE_VALUE = re.compile(r"[\s*_`]*(\d+(?:\.\d*)?|\.\d+)[\s*_`]*")


def parse_judge_reply(raw: str, form: str = JSON_REPLY) -> JudgeVerdict:
    """Read a judge's reply. The one strict parser.

    The evals, and the ladder's ``strict_judge_score``, read replies here.
    A verdict counts only when the judge gave a finite score in [0, 1] in
    the form it was asked for:

    * ``json``: the ``"score"`` key of the reply's JSON object. Markdown
      fences and text around the object are tolerated; the first JSON
      object found decides. A ``true``, a ``"0.9"``, a ``NaN`` or a 7 is
      not a score.
    * ``geval``: the FINAL line that opens with ``SCORE:``, holding a
      single number. The criteria above it are the reasoning.

    Anything else is ``unparseable``. That includes an empty reply, prose
    around a number, "rating: 4", "the score is 0.9", "Final score: 0.8",
    and a JSON reply to a G-Eval prompt (or the reverse). No stray number
    is read and no out-of-range one is clamped. Pure: the caller logs.
    """
    if not raw or not raw.strip():
        return _unparseable(raw, "the reply is empty")
    if form == JSON_REPLY:
        return _read_json_reply(raw)
    if form == GEVAL_REPLY:
        return _read_geval_reply(raw)
    raise ValueError(f"unknown judge reply form {form!r}")


def _unparseable(raw: str, why: str) -> JudgeVerdict:
    return JudgeVerdict(score=None, reasoning=f"unparseable judge reply: {why}",
                        raw_response=raw, status=VERDICT_UNPARSEABLE)


def _read_json_reply(raw: str) -> JudgeVerdict:
    # The ladder's strict_judge_score reads its replies through this, so the
    # candidates and the rules below are the ladder's (#588), unchanged.
    candidates = [raw, re.sub(r"```(?:json)?\s*\n?", "", raw).strip()]
    if "{" in raw and "}" in raw:
        candidates.append(raw[raw.index("{"): raw.rindex("}") + 1])
    for text in candidates:
        try:
            obj = json.loads(text)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        s = obj.get("score")
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            return _unparseable(raw, 'no numeric "score" key')
        s = float(s)
        if not (math.isfinite(s) and 0.0 <= s <= 1.0):
            return _unparseable(raw, f"score {s!r} is not in [0, 1]")
        return JudgeVerdict(score=s, reasoning=str(obj.get("reasoning", "")),
                            raw_response=raw)
    return _unparseable(raw, "no JSON object")


def _read_geval_reply(raw: str) -> JudgeVerdict:
    lines = raw.splitlines()
    labels = [(i, m) for i, line in enumerate(lines) if (m := _SCORE_LABEL.match(line))]
    if not labels:
        return _unparseable(raw, "no SCORE: line")
    final, label = labels[-1]
    value = _SCORE_VALUE.fullmatch(lines[final][label.end():])
    if value is None:
        return _unparseable(raw, "the final SCORE: line is not a single number")
    s = float(value.group(1))
    if not (math.isfinite(s) and 0.0 <= s <= 1.0):
        return _unparseable(raw, f"score {s!r} is not in [0, 1]")
    reasoning = "\n".join(lines[:final]).strip()
    if len(reasoning) > 500:
        reasoning = "..." + reasoning[-500:]
    return JudgeVerdict(score=s, reasoning=reasoning, raw_response=raw)


class PrometheusJudge:
    """Evaluate agent outputs using a local LLM as judge.

    Uses raw httpx calls to /v1/chat/completions — independent of the
    ModelProvider abstraction to avoid circular dependencies.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        # What _detect_model() actually resolved to, so provenance() can report
        # the judge that graded rather than the judge that was requested.
        self._resolved_model: str | None = model

    def provenance(self) -> dict[str, object]:
        """Who graded this — recorded ALONGSIDE every score.

        A score with no judge attribution cannot be compared with another
        score. Before 2026-08-02 two judges ran concurrently: the nightly
        script pinned its model, while the in-daemon GEPA optimizer passed
        ``model=None`` and graded with whatever the endpoint had loaded. The
        numbers were indistinguishable in the artifact, so cross-comparison
        was unsound and there was no way to tell after the fact.

        ``pinned`` is the field that matters and is NOT inferable from
        ``model`` alone: an auto-detected judge that happens to resolve to
        ``qwen2.5:7b-instruct`` records the same model name as one pinned to
        it, but only the pinned run is reproducible.

        ``model`` is None until the first call when auto-detecting — that is
        honest, not a gap: nothing has been graded yet, so no judge identity
        exists to record.
        """
        return {
            "base_url": self._base_url,
            "model": self._resolved_model or self._model,
            "pinned": self._model is not None,
        }

    async def _detect_model(self) -> str:
        """Query /v1/models to find the loaded model."""
        if self._model:
            return self._model
        if self._resolved_model:
            return self._resolved_model
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._base_url}/v1/models")
                resp.raise_for_status()
                models = resp.json().get("data", [])
                if models:
                    detected = models[0].get("id", "unknown")
                    log.debug("Judge detected model: %s", detected)
                    # Cache it so provenance() reports the judge that actually
                    # graded, not just the one that was requested.
                    self._resolved_model = detected
                    return detected
        except Exception as exc:
            log.warning("Could not detect judge model: %s", exc)
        return "unknown"

    async def _call_llm(
        self,
        system: str,
        user: str,
        max_tokens: int = 1024,
        retries: int = 2,
        response_format: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return the response text.

        Args:
            response_format: If provided, passed to llama.cpp to constrain
                output via GBNF grammar (e.g. json_schema mode).
            chat_template_kwargs: If provided, passed to llama.cpp to control
                template behavior (e.g. {"enable_thinking": False}).

        Retries up to `retries` times if the model returns an empty response.
        With constrained decoding, retries should rarely trigger.
        """
        model = await self._detect_model()
        for attempt in range(retries + 1):
            temp = attempt * 0.2  # 0.0 → 0.2 → 0.4
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": temp,
            }
            if response_format is not None:
                payload["response_format"] = response_format
            if chat_template_kwargs is not None:
                payload["chat_template_kwargs"] = chat_template_kwargs

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            content = data["choices"][0]["message"].get("content", "")

            # Qwen3.5 bug: thinking leaks into reasoning_content, content empty
            if not content or not content.strip():
                reasoning = data["choices"][0]["message"].get("reasoning_content", "")
                if reasoning:
                    log.warning("Empty content, extracting from reasoning_content")
                    extracted = self._extract_json_from_reasoning(reasoning)
                    if extracted:
                        return extracted

            if content and content.strip():
                return content
            if attempt < retries:
                log.warning(
                    "Empty LLM response (attempt %d/%d), retrying with temp=%.2f",
                    attempt + 1, retries + 1, temp + 0.2,
                )
        log.warning("Empty LLM response after %d attempts", retries + 1)
        return ""

    def _extract_json_from_reasoning(self, reasoning: str) -> str:
        """Extract JSON from reasoning_content when content field is empty."""
        match = re.search(r'\{[^{}]*"score"[^{}]*\}', reasoning)
        if match:
            try:
                parsed = json.loads(match.group())
                return json.dumps(parsed)
            except json.JSONDecodeError:
                pass
        return ""

    # ------------------------------------------------------------------
    # JSON-based evaluation (original)
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        task_input: str,
        agent_output: str,
        expected_behavior: str,
        tool_trace: list[dict[str, Any]] | None = None,
    ) -> JudgeVerdict:
        """Judge an agent's output against expected behavior.

        Uses constrained decoding (JSON schema mode) to guarantee valid
        output from llama.cpp. The grammar constraint makes parse failures
        impossible — the model can only produce tokens that form valid JSON
        matching JUDGE_SCORE_SCHEMA.

        Returns a JudgeVerdict with score (0.0-1.0) and reasoning. A reply
        that holds no verdict anyway (a server that ignored the schema, an
        empty reply) comes back ``unparseable`` with no score.
        """
        user_prompt = f"Task: {task_input}\n\nExpected behavior: {expected_behavior}\n\n"
        user_prompt += f"Agent output:\n{agent_output[:3000]}\n"

        if tool_trace:
            tools_summary = ", ".join(
                t.get("tool_name", "unknown") for t in tool_trace
            )
            user_prompt += f"\nTools called: {tools_summary}"

        raw = await self._call_llm(
            _JUDGE_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=512,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_score",
                    "strict": True,
                    "schema": JUDGE_SCORE_SCHEMA,
                },
            },
            chat_template_kwargs={"enable_thinking": False},
        )
        return self._parse_verdict(raw)

    # ------------------------------------------------------------------
    # G-Eval: chain-of-thought evaluation (better for local models)
    # ------------------------------------------------------------------

    async def evaluate_geval(
        self,
        criteria: list[str],
        context: str,
    ) -> JudgeVerdict:
        """G-Eval style evaluation with chain-of-thought reasoning.

        The model reasons through each criterion step by step, then
        produces a final score. More reliable than JSON-only prompting
        with local models (Qwen, Gemma) because the model can think
        before scoring.

        Args:
            criteria: Numbered evaluation criteria the model reasons through.
            context: The full evaluation context (task, output, evidence).
        """
        criteria_text = "\n".join(
            f"{i}. {c}" for i, c in enumerate(criteria, 1)
        )

        user_prompt = f"""{context}

---

Evaluate step by step using these criteria:
{criteria_text}

Write one sentence per criterion, then end with SCORE: followed by a number.
Example ending: SCORE: 0.85"""

        raw = await self._call_llm(_GEVAL_SYSTEM_PROMPT, user_prompt, max_tokens=1024)
        return self._parse_geval_verdict(raw)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_verdict(self, raw: str) -> JudgeVerdict:
        """Read a JSON reply (``evaluate()``) with the strict parser."""
        return self._logged(parse_judge_reply(raw, JSON_REPLY))

    def _parse_geval_verdict(self, raw: str) -> JudgeVerdict:
        """Read a G-Eval reply (``evaluate_geval()``) with the strict parser."""
        return self._logged(parse_judge_reply(raw, GEVAL_REPLY))

    @staticmethod
    def _logged(verdict: JudgeVerdict) -> JudgeVerdict:
        if verdict.status != VERDICT_PARSED:
            log.warning("Judge reply holds no verdict (%s): %s",
                        verdict.reasoning, verdict.raw_response[:200])
        return verdict
