from __future__ import annotations

from time import perf_counter

from app.embeddings.provider import (
    cosine_similarity,
    embed_documents,
    embed_query,
    get_embedding_model,
)
from app.llm.groq_provider import get_llm_provider


def _timed_load(label: str, loader) -> None:
    for attempt in (1, 2):
        started = perf_counter()
        loader()
        elapsed = perf_counter() - started
        print(f"{label} load {attempt}: {elapsed:.3f}s")


def main() -> None:
    _timed_load("Groq client", get_llm_provider)
    _timed_load("Embedding model", get_embedding_model)

    question = "How do plants convert sunlight into energy?"
    passage = (
        "Photosynthesis is the process by which plants use sunlight, water, and "
        "carbon dioxide to make sugars for energy."
    )
    similarity = cosine_similarity(embed_query(question), embed_documents([passage])[0])
    print(f"Question/passage similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
