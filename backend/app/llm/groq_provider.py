from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from typing import Any, Protocol, TypedDict

from app import config
from app.agent.language import answer_language_instruction, language_instruction

logger = logging.getLogger(__name__)
_LLM_TIMEOUT_SECONDS = 30.0
# Groq's free tier allows only ~8,000 tokens per minute per model; the SDK honours
# the retry-after hint, so a few retries ride out short rate-limit windows.
_LLM_MAX_RETRIES = 4
_LLM_TEMPERATURE = 0.2
_EMPTY_COMPLETION_ATTEMPTS = 2
# Output caps (including hidden reasoning) keep answers short and save the free-tier
# token budget; prompts ask for brief answers, so these are only a safety net.
_MAX_ANSWER_TOKENS = 1200
_MAX_BACKGROUND_TOKENS = 700


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
_fast_provider: GroqProvider | None = None
_provider_lock = threading.Lock()


class GroqProvider:
    def __init__(
        self,
        client: _GroqClient,
        model: str,
        *,
        reasoning_effort: str | None = None,
        max_completion_tokens: int | None = None,
    ) -> None:
        self._client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        transcript_language: str,
        answer_language_question: str | None = None,
        answer_language: str | None = None,
        include_language_instruction: bool = True,
    ) -> str:
        """Generate one chat completion using conservative, retryable defaults."""
        completion_messages = list(messages)
        if include_language_instruction and answer_language:
            completion_messages.append(
                ChatMessage(role="system", content=language_instruction(answer_language))
            )
        elif include_language_instruction:
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
        extra: dict[str, Any] = {}
        if self.reasoning_effort:
            extra["reasoning_effort"] = self.reasoning_effort
        if self.max_completion_tokens:
            extra["max_completion_tokens"] = self.max_completion_tokens
        # Reasoning models occasionally spend their whole output on hidden reasoning and
        # return empty content, so retry once before giving up.
        for attempt in range(1, _EMPTY_COMPLETION_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=completion_messages,
                    temperature=_LLM_TEMPERATURE,
                    **extra,
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
            if isinstance(content, str) and content.strip():
                return content
            logger.warning(
                "Groq model %s returned empty content (attempt %d).", self.model, attempt
            )
        raise RuntimeError(
            "The answer service returned an empty response. Please try again shortly."
        )


def _create_groq_client() -> _GroqClient:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required to use the Groq chat model.")

    from groq import Groq

    return Groq(
        api_key=config.GROQ_API_KEY,
        timeout=_LLM_TIMEOUT_SECONDS,
        max_retries=_LLM_MAX_RETRIES,
    )


def _low_effort_for(model: str) -> str | None:
    # Low reasoning effort makes gpt-oss answers faster and spends fewer of the
    # rate-limited tokens on hidden reasoning; other models reject this parameter.
    return "low" if "gpt-oss" in model else None


def get_llm_provider() -> GroqProvider:
    """Create the process-wide Groq client once and reuse it for all calls."""
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = GroqProvider(
                    _create_groq_client(),
                    config.LLM_MODEL,
                    reasoning_effort=_low_effort_for(config.LLM_MODEL),
                    max_completion_tokens=_MAX_ANSWER_TOKENS,
                )
    return _provider


def get_fast_llm_provider() -> GroqProvider:
    """Return a provider for LLM_FAST_MODEL that shares the process-wide Groq client."""
    global _fast_provider
    if _fast_provider is None:
        main_provider = get_llm_provider()
        with _provider_lock:
            if _fast_provider is None:
                _fast_provider = GroqProvider(
                    main_provider._client,
                    config.LLM_FAST_MODEL,
                    reasoning_effort=_low_effort_for(config.LLM_FAST_MODEL),
                    max_completion_tokens=_MAX_BACKGROUND_TOKENS,
                )
    return _fast_provider
