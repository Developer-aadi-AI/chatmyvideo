import logging

import pytest

from app.ingest.search import VideoSearchService, _is_near_duplicate
from app.transcripts.base import Transcript, TranscriptSegment


def make_transcript(*texts: str) -> Transcript:
    return Transcript(
        language="en",
        segments=[
            TranscriptSegment(text, start=float(index * 10), end=float(index * 10 + 8))
            for index, text in enumerate(texts)
        ],
    )


def test_indexes_once_and_logs_chunk_count_and_elapsed_time(caplog) -> None:
    calls: list[list[str]] = []
    transcript = make_transcript("A" * 900, "B" * 900)

    def embed_documents(texts):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    service = VideoSearchService(
        document_embedder=embed_documents,
        query_embedder=lambda text: [1.0, 0.0],
        transcript_fetcher=lambda url: transcript,
    )

    with caplog.at_level(logging.INFO):
        assert service.index_video("https://youtu.be/dQw4w9WgXcQ")
        assert not service.index_video("dQw4w9WgXcQ")

    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert "2 chunks" in caplog.text
    assert "seconds" in caplog.text


def test_search_deduplicates_near_matches_and_returns_video_order() -> None:
    shared = " ".join(f"shared{index}" for index in range(100))
    early_distinct = " ".join(f"early{index}" for index in range(110))
    near_duplicate = shared + " late-detail"
    second_near_duplicate = shared + " alternate-detail"
    later_distinct = " ".join(f"other{index}" for index in range(110))
    texts = [early_distinct, near_duplicate, second_near_duplicate, later_distinct]
    vectors_by_text = {
        early_distinct: [0.2, 0.98],
        near_duplicate: [1.0, 0.0],
        second_near_duplicate: [0.99, 0.1],
        later_distinct: [0.8, 0.6],
    }

    service = VideoSearchService(
        document_embedder=lambda chunks: [vectors_by_text[text] for text in chunks],
        query_embedder=lambda question: [1.0, 0.0],
    )
    service.index_video("dQw4w9WgXcQ", make_transcript(*texts))

    results = service.search("dQw4w9WgXcQ", "question", limit=3)

    assert [result.text for result in results] == [
        early_distinct,
        near_duplicate,
        later_distinct,
    ]
    assert [result.position for result in results] == [0, 1, 3]
    assert [(result.start, result.end) for result in results] == [
        (0.0, 8.0),
        (10.0, 18.0),
        (30.0, 38.0),
    ]
    assert all(result.video_id == "dQw4w9WgXcQ" for result in results)
    assert all(result.language == "en" for result in results)
    assert results[0].score < results[1].score


def test_similarity_helper_identifies_near_duplicates_but_not_distinct_chunks() -> None:
    shared = " ".join(f"word{index}" for index in range(100))

    assert _is_near_duplicate(shared + " extra", shared + " different")
    assert not _is_near_duplicate(shared, " ".join(f"other{index}" for index in range(100)))


def test_search_requires_existing_index() -> None:
    service = VideoSearchService(
        document_embedder=lambda chunks: [],
        query_embedder=lambda question: [],
    )

    with pytest.raises(LookupError, match="not been indexed"):
        service.search("dQw4w9WgXcQ", "question")


def test_search_rejects_invalid_result_limit() -> None:
    service = VideoSearchService(
        document_embedder=lambda chunks: [],
        query_embedder=lambda question: [],
    )

    with pytest.raises(ValueError, match="greater than zero"):
        service.search("dQw4w9WgXcQ", "question", limit=0)


def test_empty_transcript_is_not_marked_indexed() -> None:
    service = VideoSearchService(
        document_embedder=lambda chunks: [[1.0, 0.0] for _ in chunks],
        query_embedder=lambda question: [1.0, 0.0],
    )

    with pytest.raises(ValueError, match="no transcript text"):
        service.index_video("dQw4w9WgXcQ", Transcript(language="en", segments=[]))
    assert service.index_video("dQw4w9WgXcQ", make_transcript("A" * 900))
