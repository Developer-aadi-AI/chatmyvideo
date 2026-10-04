from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any, Protocol, TypedDict

from app import config
from app.agent.language import answer_language_instruction

logger = logging.getLogger(__name__)
_LLM_TIMEOUT_SECONDS = 30.0
_LLM_MAX_RETRIES = 2
_LLM_TEMPERATURE = 0.2


class ChatMessage(TypedDict):
    role: str
    content: str


class _Completions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    completions: _Completions


class _GroqClient(Protocol):
    chat: _Chat


_provider: GroqProvider | None = None
_provider_lock = threading.Lock()


class GroqProvider:
    def __init__(self, client: _GroqClient, model: str) -> None:
        self._client = client
        self.model = model

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        transcript_language: str,
        answer_language_question: str | None = None,
        include_language_instruction: bool = True,
    ) -> str:
        """Generate one chat completion using conservative, retryable defaults."""
        completion_messages = list(messages)
        if include_language_instruction:
            language_question = answer_language_question or next(
                (message["content"] for message in reversed(messages) if message["role"] == "user"),
                "",
            )
            completion_messages.append(
                ChatMessage(
                    role="system",
                    content=answer_language_instruction(language_question, transcript_language),
                )
            )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=completion_messages,
                temperature=_LLM_TEMPERATURE,
            )
        except Exception as exc:
            # Groq SDK errors (rate limits, timeouts, auth) become RuntimeError so the API
            # returns a readable 503 instead of an unhandled 500 without CORS headers.
            logger.exception("Groq chat completion failed.")
            raise RuntimeError(
                "The answer service is busy or unavailable right now. Please try again shortly."
            ) from exc
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("Groq returned no completion choices.")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str):
            raise TypeError("Groq returned a completion with invalid content.")
        if not content:
            raise RuntimeError("Groq returned an empty completion.")
        return content


def _create_groq_client() -> _GroqClient:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required to use the Groq chat model.")

    from groq import Groq

    return Groq(
        api_key=config.GROQ_API_KEY,
        timeout=_LLM_TIMEOUT_SECONDS,
        max_retries=_LLM_MAX_RETRIES,
    )


def get_llm_provider() -> GroqProvider:
    """Create the process-wide Groq client once and reuse it for all calls."""
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = GroqProvider(_create_groq_client(), config.LLM_MODEL)
    return _provider
