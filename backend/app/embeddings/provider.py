from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Protocol

from app import config


class _Encoder(Protocol):
    def encode(
        self,
        sentences: str | list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> object: ...


_model: _Encoder | None = None
_model_name: str | None = None
_model_lock = threading.Lock()


def _load_model(model_name: str) -> _Encoder:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name,
        token=config.HF_TOKEN or None,
    )


def get_embedding_model() -> _Encoder:
    """Load the configured embedding model once for this process."""
    global _model, _model_name
    configured_name = config.EMBED_MODEL
    with _model_lock:
        if _model is None:
            _model = _load_model(configured_name)
            _model_name = configured_name
        elif _model_name != configured_name:
            raise RuntimeError(
                "EMBED_MODEL changed after the embedding model was loaded; "
                "restart the process to use the new model."
            )
        return _model


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Embed passages as normalized vectors, adding the E5 prefix automatically."""
    if not texts:
        return []
    encoded = get_embedding_model().encode(
        [f"passage: {text}" for text in texts],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return encoded.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a query as one normalized vector, adding the E5 prefix automatically."""
    encoded = get_embedding_model().encode(
        f"query: {text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return encoded.tolist()


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Return cosine similarity for the normalized vectors produced by this module."""
    if len(first) != len(second):
        raise ValueError("Embedding vectors must have the same number of dimensions.")
    return sum(left * right for left, right in zip(first, second, strict=True))
