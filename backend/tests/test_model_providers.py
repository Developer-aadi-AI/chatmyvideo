import json
import sys
from types import SimpleNamespace
from typing import Self
from urllib.error import HTTPError

import pytest

from app import config
from app.embeddings import provider as embeddings
from app.llm import groq_provider
from app.llm.groq_provider import ChatMessage


@pytest.fixture(autouse=True)
def reset_model_singletons(monkeypatch) -> None:
    monkeypatch.setattr(config, "EMBED_PROVIDER", "local")
    groq_provider._provider = None
    embeddings._model = None
    embeddings._model_name = None


def test_groq_provider_is_created_once_and_reused(monkeypatch) -> None:
    clients: list[object] = []

    def create_client():
        client = SimpleNamespace(chat=SimpleNamespace(completions=object()))
        clients.append(client)
        return client

    monkeypatch.setattr(groq_provider, "_create_groq_client", create_client)
    first = groq_provider.get_llm_provider()
    second = groq_provider.get_llm_provider()

    assert first is second
    assert len(clients) == 1


def test_groq_completion_uses_configured_model_and_low_temperature() -> None:
    captured: dict[str, object] = {}

    def create(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Answer"))])

    provider = groq_provider.GroqProvider(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        "configured-model",
    )

    answer = provider.complete(
        [ChatMessage(role="user", content="Question")],
        transcript_language="en",
    )

    assert answer == "Answer"
    assert captured["model"] == "configured-model"
    assert captured["temperature"] == 0.2
    completion_messages = captured["messages"]
    assert completion_messages[0] == {"role": "user", "content": "Question"}
    assert completion_messages[-1] == {
        "role": "system",
        "content": (
            "Answer in English. Keep the response natural and fluent in that language. "
            "Do not change languages unless the user asks you to."
        ),
    }


def test_groq_client_uses_timeout_and_retries(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def groq_constructor(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "groq",
        SimpleNamespace(Groq=groq_constructor),
    )

    groq_provider._create_groq_client()

    assert captured["timeout"] > 0
    assert captured["max_retries"] == 4
    assert captured["api_key"] == "test-key"


def test_embedding_model_is_loaded_once_and_reused(monkeypatch) -> None:
    instances: list[object] = []

    def load_model(model_name: str) -> object:
        instance = object()
        instances.append(instance)
        return instance

    monkeypatch.setattr(config, "EMBED_MODEL", "intfloat/multilingual-e5-small")
    monkeypatch.setattr(embeddings, "_load_model", load_model)

    first = embeddings.get_embedding_model()
    second = embeddings.get_embedding_model()

    assert first is second
    assert len(instances) == 1


def test_embedding_helpers_apply_e5_prefix_and_normalize(monkeypatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    class FakeArray(list):
        def tolist(self):
            return list(self)

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            if isinstance(texts, str):
                return FakeArray([0.6, 0.8])
            return FakeArray([FakeArray([0.6, 0.8]) for _ in texts])

    monkeypatch.setattr(embeddings, "_model", FakeModel())
    monkeypatch.setattr(embeddings, "_model_name", "intfloat/multilingual-e5-large")
    monkeypatch.setattr(config, "EMBED_MODEL", "intfloat/multilingual-e5-large")

    query = embeddings.embed_query("How do plants grow?")
    passages = embeddings.embed_documents(["Plants use sunlight to grow."])

    assert calls[0][0] == "query: How do plants grow?"
    assert calls[1][0] == ["passage: Plants use sunlight to grow."]
    assert all(call[1]["normalize_embeddings"] is True for call in calls)
    assert query == [0.6, 0.8]
    assert passages == [[0.6, 0.8]]
    assert embeddings.cosine_similarity(query, passages[0]) == pytest.approx(1.0)


def test_empty_document_list_does_not_load_model() -> None:
    assert embeddings.embed_documents([]) == []
    assert embeddings._model is None


def test_embedding_model_setting_cannot_change_after_load(monkeypatch) -> None:
    monkeypatch.setattr(embeddings, "_model", object())
    monkeypatch.setattr(embeddings, "_model_name", "intfloat/multilingual-e5-large")
    monkeypatch.setattr(config, "EMBED_MODEL", "intfloat/multilingual-e5-small")

    with pytest.raises(RuntimeError, match="restart the process"):
        embeddings.get_embedding_model()


def test_cosine_similarity_requires_equal_dimensions() -> None:
    with pytest.raises(ValueError, match="same number"):
        embeddings.cosine_similarity([1.0], [1.0, 0.0])


def test_groq_api_failure_becomes_friendly_runtime_error() -> None:
    def create(**kwargs: object) -> object:
        raise ConnectionError("rate limited")

    provider = groq_provider.GroqProvider(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        "configured-model",
    )

    with pytest.raises(RuntimeError, match="try again"):
        provider.complete(
            [ChatMessage(role="user", content="Question")],
            transcript_language="en",
        )


class _FakeHFResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _use_hf_api(monkeypatch) -> None:
    monkeypatch.setattr(config, "EMBED_PROVIDER", "huggingface")
    monkeypatch.setattr(config, "HF_TOKEN", "test-token")
    monkeypatch.setattr(config, "EMBED_MODEL", "intfloat/multilingual-e5-small")
    monkeypatch.setattr(embeddings.time, "sleep", lambda seconds: None)


def test_hf_api_embeds_in_batches_with_prefixes_and_normalizes(monkeypatch) -> None:
    _use_hf_api(monkeypatch)
    monkeypatch.setattr(embeddings, "_HF_BATCH_SIZE", 2)
    sent: list[dict[str, object]] = []

    def fake_urlopen(request, timeout):
        assert request.get_header("Authorization") == "Bearer test-token"
        assert "intfloat/multilingual-e5-small" in request.full_url
        body = json.loads(request.data)
        sent.append(body)
        return _FakeHFResponse([[3.0, 4.0] for _ in body["inputs"]])

    monkeypatch.setattr(embeddings, "urlopen", fake_urlopen)

    vectors = embeddings.embed_documents(["a", "b", "c"])
    query = embeddings.embed_query("question")

    assert [body["inputs"] for body in sent] == [
        ["passage: a", "passage: b"],
        ["passage: c"],
        ["query: question"],
    ]
    assert vectors == [[0.6, 0.8]] * 3
    assert query == [0.6, 0.8]
    assert embeddings._model is None


def test_hf_api_retries_while_model_warms_up(monkeypatch) -> None:
    _use_hf_api(monkeypatch)
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HTTPError(request.full_url, 503, "Loading", {}, None)
        return _FakeHFResponse([[1.0, 0.0]])

    monkeypatch.setattr(embeddings, "urlopen", fake_urlopen)

    assert embeddings.embed_query("question") == [1.0, 0.0]
    assert attempts == 2


def test_hf_api_rejected_token_is_a_friendly_runtime_error(monkeypatch) -> None:
    _use_hf_api(monkeypatch)

    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(embeddings, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        embeddings.embed_documents(["text"])


def test_empty_completion_is_retried_once() -> None:
    contents = ["", "Answer"]
    captured: dict[str, object] = {}

    def create(**kwargs: object) -> object:
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=contents.pop(0)))]
        )

    provider = groq_provider.GroqProvider(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        "openai/gpt-oss-20b",
        reasoning_effort="low",
    )

    assert provider.complete([ChatMessage(role="user", content="Q")], transcript_language="en") == (
        "Answer"
    )
    assert captured["reasoning_effort"] == "low"
