"""Base class for all LLM-based scoring."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

PROVIDER_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
    "deepseek": "https://api.deepseek.com/v1",
}


def parse_model_string(model_string: str) -> tuple[str, str]:
    """Split 'provider/model' into (provider, model). Defaults to openai."""
    parts = model_string.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "openai", model_string


@dataclass
class JudgeResult:
    score: float
    raw_response: str
    reasoning: str = ""
    parsed: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


class LLMJudge:
    """Wraps any OpenAI-compatible API. Structured rubric in, scored result out."""

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        self.provider, self.model_name = parse_model_string(model)
        self.api_key = api_key or ""
        self.base_url = base_url or PROVIDER_URLS.get(
            self.provider, PROVIDER_URLS["openai"]
        )
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_body(self, system_prompt: str, user_prompt: str) -> dict:
        if self.provider == "anthropic":
            return {
                "model": self.model_name,
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            }
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }

    def _endpoint(self) -> str:
        if self.provider == "anthropic":
            return f"{self.base_url}/messages"
        return f"{self.base_url}/chat/completions"

    def _extract_text(self, body: dict) -> str:
        if self.provider == "anthropic":
            for block in body.get("content", []):
                if block.get("type") == "text":
                    return block.get("text", "")
            return ""
        choices = body.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    def score(self, system_prompt: str, user_prompt: str) -> JudgeResult:
        """Send a rubric prompt to the LLM and return a parsed score."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(
                    self._endpoint(),
                    headers=self._build_headers(),
                    json=self._build_body(system_prompt, user_prompt),
                )
                resp.raise_for_status()
                text = self._extract_text(resp.json())
                return self._parse_score(text)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM judge attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
        logger.error(
            "LLM judge failed after %d retries: %s", self.max_retries, last_error
        )
        return JudgeResult(score=0.0, raw_response=str(last_error), parsed=False)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt and return raw text response (for model under test)."""
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(
                    self._endpoint(),
                    headers=self._build_headers(),
                    json=self._build_body(system_prompt, user_prompt),
                )
                resp.raise_for_status()
                return self._extract_text(resp.json())
            except Exception as exc:
                logger.warning("Generate attempt %d failed: %s", attempt + 1, exc)
        return ""

    _SCORE_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r'"score"\s*:\s*([0-9]*\.?[0-9]+)'),
        re.compile(r"(?:score|rating)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"\b([0-9]*\.?[0-9]+)\s*/\s*(?:10|5|4|1)\b"),
        re.compile(r"\b([0-9]*\.?[0-9]+)\b"),
    ]

    def _parse_score(self, text: str) -> JudgeResult:
        reasoning = ""
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                reasoning = data.get("reasoning", data.get("reason", ""))
                if "score" in data:
                    return JudgeResult(
                        score=float(data["score"]),
                        raw_response=text,
                        reasoning=str(reasoning),
                        metadata=data,
                    )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        for pattern in self._SCORE_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    return JudgeResult(
                        score=float(match.group(1)),
                        raw_response=text,
                        reasoning=text,
                    )
                except ValueError:
                    continue

        return JudgeResult(score=0.0, raw_response=text, parsed=False)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMJudge:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
