"""PrometheusJudge — LLM-as-judge using the local llama.cpp endpoint.

Evaluates agent outputs against expected behavior descriptions.
Uses the same OpenAI-compatible /v1/chat/completions API as the main model.

Key reliability feature (Sprint 14): **Constrained decoding** via llama.cpp's
``response_format`` with ``json_schema`` type. The server converts the schema
to a GBNF grammar and masks invalid tokens at decode time — the model
physically cannot produce invalid JSON. This eliminates parse failures that
plagued G-Eval and raw JSON prompting with local models.

Supports two evaluation modes:
- evaluate(): JSON with constrained decoding (primary, used by metrics)
- evaluate_geval(): G-Eval chain-of-thought (kept for manual/debug use)
"""

from __future__ import annotations

import json
import logging
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


@dataclass
class JudgeVerdict:
    """Result of an LLM judge evaluation."""

    score: float
    reasoning: str
    raw_response: str


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

    async def _detect_model(self) -> str:
        """Query /v1/models to find the loaded model."""
        if self._model:
            return self._model
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._base_url}/v1/models")
                resp.raise_for_status()
                models = resp.json().get("data", [])
                if models:
                    detected = models[0].get("id", "unknown")
                    log.debug("Judge detected model: %s", detected)
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

        Returns a JudgeVerdict with score (0.0-1.0) and reasoning.
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
        """Parse the judge's JSON response into a JudgeVerdict.

        Layered parsing for maximum compatibility:
        1. Direct json.loads — works when grammar-constrained (llama.cpp)
        2. Strip markdown fences — handles ```json ... ``` wrapping (Ollama)
        3. Extract {…} substring — handles preamble/postamble text
        4. Regex fallback — last resort score extraction
        """
        if not raw or not raw.strip():
            log.warning("Empty judge response")
            return JudgeVerdict(score=0.0, reasoning="Empty response", raw_response=raw)

        # 1. Direct parse (constrained decoding gives clean JSON)
        try:
            parsed = json.loads(raw)
            return self._verdict_from_dict(parsed, raw)
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown fences (Ollama, other servers)
        stripped = re.sub(r"```(?:json)?\s*\n?", "", raw).strip()
        if stripped != raw.strip():
            try:
                parsed = json.loads(stripped)
                return self._verdict_from_dict(parsed, raw)
            except json.JSONDecodeError:
                pass

        # 3. Extract {…} JSON substring (preamble text before JSON)
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
            return self._verdict_from_dict(parsed, raw)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        log.warning("Could not parse judge response as JSON: %s", raw[:200])
        return self._fallback_parse(raw)

    def _verdict_from_dict(self, parsed: dict, raw: str) -> JudgeVerdict:
        """Build a JudgeVerdict from a parsed dict, clamping score."""
        return JudgeVerdict(
            score=max(0.0, min(1.0, float(parsed.get("score", 0.0)))),
            reasoning=str(parsed.get("reasoning", "")),
            raw_response=raw,
        )

    def _parse_geval_verdict(self, raw: str) -> JudgeVerdict:
        """Parse G-Eval chain-of-thought response.

        Looks for 'SCORE: X.X' in the response. Falls back to alternative
        patterns like 'score is 0.X', 'rating: 0.X', or a standalone
        decimal on the last non-empty line.
        """
        if not raw or not raw.strip():
            log.warning("Empty G-Eval response")
            return JudgeVerdict(score=0.0, reasoning="Empty response", raw_response=raw)

        # Primary: SCORE: pattern (case-insensitive)
        match = re.search(r"SCORE:\s*(\d+\.?\d*)", raw, re.IGNORECASE)

        # Fallback patterns if SCORE: not found
        if not match:
            match = re.search(
                r"(?:final\s+score|rating|score\s+is|score\s*=)\s*:?\s*(\d+\.?\d*)",
                raw, re.IGNORECASE,
            )

        # Last resort: standalone decimal (0.X or 1.0) on the last non-empty line
        if not match:
            last_lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
            if last_lines:
                last_match = re.search(r"\b(0\.\d+|1\.0)\b", last_lines[-1])
                if last_match:
                    match = last_match

        if match:
            score = max(0.0, min(1.0, float(match.group(1))))
            reasoning_end = match.start()
            reasoning = raw[:reasoning_end].strip() if reasoning_end > 0 else raw.strip()
            if len(reasoning) > 500:
                reasoning = "..." + reasoning[-500:]
            return JudgeVerdict(
                score=score,
                reasoning=reasoning,
                raw_response=raw,
            )

        log.warning("No score found in G-Eval response: %s", raw[:200])
        return self._fallback_parse(raw)

    def _fallback_parse(self, raw: str) -> JudgeVerdict:
        """Last-resort score extraction from unstructured text."""
        match = re.search(r"(\d+\.?\d*)", raw)
        score = float(match.group(1)) if match else 0.0
        score = max(0.0, min(1.0, score))
        return JudgeVerdict(
            score=score,
            reasoning=f"Parse fallback: {raw[:200]}",
            raw_response=raw,
        )
