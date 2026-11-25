"""
Minimal streaming client for any OpenAI-compatible chat API (Groq by default).

Configured from the environment, so the key only ever lives in the hosting
platform's secret store:
    LLM_API_KEY    required; answers are disabled without it
    ANSWER_MODEL   default openai/gpt-oss-20b (fast, production model on Groq)
    LLM_BASE_URL   default https://api.groq.com/openai/v1
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


class LLMError(RuntimeError):
    """The provider failed or returned something unusable."""


class RateLimited(LLMError):
    """HTTP 429: the provider's (free-tier) quota is used up for now."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class LLMClient:
    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_sec: float = 60.0
    max_tokens: int = 900
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "LLMClient | None":
        key = os.getenv("LLM_API_KEY", "").strip()
        if not key:
            return None
        return cls(api_key=key,
                   model=os.getenv("ANSWER_MODEL") or DEFAULT_MODEL,
                   base_url=os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield the answer text as it is generated."""
        import requests  # imported here so the API (and its tests) load without it

        body = {"model": self.model, "messages": messages, "stream": True,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if "gpt-oss" in self.model:
            # Keep the model's hidden reasoning short and out of the answer stream.
            body.update(reasoning_effort="low", include_reasoning=False)
        try:
            resp = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions", json=body, stream=True,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=(10, self.timeout_sec))
        except requests.RequestException as e:
            raise LLMError(f"LLM provider unreachable: {type(e).__name__}") from None

        with resp:
            if resp.status_code == 429:
                raise RateLimited("LLM rate limit reached", _retry_after(resp.headers))
            if resp.status_code >= 400:
                raise LLMError(f"LLM provider returned HTTP {resp.status_code}: {resp.text[:200]}")
            # Decode the bytes ourselves. requests falls back to ISO-8859-1 for a
            # text/event-stream response without a charset, which turns UTF-8
            # punctuation into mojibake ("receiver’s" -> "receiverâ€™s").
            for raw in resp.iter_lines():
                line = raw.decode("utf-8", errors="replace")
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    raise LLMError(f"LLM stream error: {str(chunk['error'])[:200]}")
                for choice in chunk.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        yield text


def _retry_after(headers) -> float | None:
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
