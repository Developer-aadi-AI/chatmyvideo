from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Iterable, Sequence
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from app import config

logger = logging.getLogger(__name__)
_HF_API_URL = (
    "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
)
_HF_BATCH_SIZE = 32
_HF_TIMEOUT_SECONDS = 60.0
_HF_MAX_ATTEMPTS = 3
_EMBEDDING_UNAVAILABLE = (
    "Video indexing is temporarily unavailable. Please try again in a few minutes."
)
# The int8-quantized ONNX export in the model's own Hugging Face repo. It peaks around
# 550 MB of RAM (the full-precision file needs ~1.3 GB), and its vectors match the
# original model with ~0.99 cosine similarity. It also runs on CPUs without AVX-512.
_LOCAL_ONNX_FILE = "onnx/model_qint8_avx512_vnni.onnx"
_LOCAL_EMBEDDING_DIM = 384
# Small batches keep peak memory low; larger ones barely speed things up on a CPU.
_LOCAL_BATCH_SIZE = 4


class _Encoder(Protocol):
    def embed(self, documents: list[str], batch_size: int) -> Iterable[Sequence[float]]: ...


_model: _Encoder | None = None
_model_name: str | None = None
_model_lock = threading.Lock()


def _load_model(model_name: str) -> _Encoder:
    """Load the model with fastembed, which runs ONNX files without PyTorch."""
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    # fastembed doesn't ship multilingual-e5-small, so register its ONNX file ourselves.
    # Mean pooling + normalization is how E5 models produce sentence vectors.
    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(hf=model_name),
        dim=_LOCAL_EMBEDDING_DIM,
        model_file=_LOCAL_ONNX_FILE,
    )
    return TextEmbedding(model_name)


def _embed_locally(texts: list[str]) -> list[list[float]]:
    try:
        vectors = get_embedding_model().embed(texts, batch_size=_LOCAL_BATCH_SIZE)
        return [[float(value) for value in vector] for vector in vectors]
    except Exception as exc:
        logger.exception("Local embedding model failed.")
        raise RuntimeError(_EMBEDDING_UNAVAILABLE) from exc


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


def _uses_hf_api() -> bool:
    return config.EMBED_PROVIDER == "huggingface"


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [float(value) for value in vector]
    return [float(value) / norm for value in vector]


def _embed_with_hf_api(texts: list[str]) -> list[list[float]]:
    """Embed texts with the Hugging Face Inference API, in batches."""
    if not config.HF_TOKEN:
        logger.error("HF_TOKEN is missing; cannot call the Hugging Face Inference API.")
        raise RuntimeError(_EMBEDDING_UNAVAILABLE)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _HF_BATCH_SIZE):
        vectors.extend(_request_hf_embeddings(texts[start : start + _HF_BATCH_SIZE]))
    return vectors


def _request_hf_embeddings(batch: list[str]) -> list[list[float]]:
    url = _HF_API_URL.format(model=quote(config.EMBED_MODEL, safe="/"))
    body = json.dumps({"inputs": batch, "normalize": True}).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {config.HF_TOKEN}",
        "Content-Type": "application/json",
    }
    for attempt in range(1, _HF_MAX_ATTEMPTS + 1):
        try:
            request = Request(url, data=body, headers=headers, method="POST")
            with urlopen(request, timeout=_HF_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            # 503 means the hosted model is still warming up, so wait and retry.
            if exc.code == 503 and attempt < _HF_MAX_ATTEMPTS:
                time.sleep(2 * attempt)
                continue
            if exc.code in {401, 403}:
                logger.error(
                    "Hugging Face rejected HF_TOKEN (HTTP %d). It needs the "
                    "'Make calls to Inference Providers' permission.",
                    exc.code,
                )
            else:
                logger.error("Hugging Face embedding request failed with HTTP %d.", exc.code)
            raise RuntimeError(_EMBEDDING_UNAVAILABLE) from exc
        except (TimeoutError, URLError, OSError, json.JSONDecodeError) as exc:
            if attempt < _HF_MAX_ATTEMPTS:
                time.sleep(2 * attempt)
                continue
            logger.exception("Hugging Face embedding request could not be completed.")
            raise RuntimeError(_EMBEDDING_UNAVAILABLE) from exc

    if (
        not isinstance(payload, list)
        or len(payload) != len(batch)
        or not all(
            isinstance(vector, list) and vector and all(isinstance(v, (int, float)) for v in vector)
            for vector in payload
        )
    ):
        logger.error("Hugging Face returned embeddings in an unexpected shape.")
        raise RuntimeError(_EMBEDDING_UNAVAILABLE)
    # Normalize locally too, so FAISS inner-product scores stay cosine similarities.
    return [_normalize(vector) for vector in payload]


def embed_documents(texts: Sequence[str]) -> list[list[float]]:
    """Embed passages as normalized vectors, adding the E5 prefix automatically."""
    if not texts:
        return []
    prefixed = [f"passage: {text}" for text in texts]
    if _uses_hf_api():
        return _embed_with_hf_api(prefixed)
    return _embed_locally(prefixed)


def embed_query(text: str) -> list[float]:
    """Embed a query as one normalized vector, adding the E5 prefix automatically."""
    if _uses_hf_api():
        return _embed_with_hf_api([f"query: {text}"])[0]
    return _embed_locally([f"query: {text}"])[0]


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Return cosine similarity for the normalized vectors produced by this module."""
    if len(first) != len(second):
        raise ValueError("Embedding vectors must have the same number of dimensions.")
    return sum(left * right for left, right in zip(first, second, strict=True))
