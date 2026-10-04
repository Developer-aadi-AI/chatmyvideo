from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import faiss
import numpy as np

from app.embeddings.provider import embed_documents, embed_query
from app.ingest.chunking import TranscriptChunk, chunk_transcript
from app.ingest.youtube_url import parse_youtube_video_id
from app.transcripts.base import Transcript
from app.transcripts.youtube_transcript import fetch_transcript

logger = logging.getLogger(__name__)
_NEAR_DUPLICATE_THRESHOLD = 0.85


@dataclass(frozen=True)
class SearchResult:
    text: str
    start: float
    end: float
    position: int
    video_id: str
    language: str
    score: float


@dataclass
class _VideoIndex:
    index: Any
    chunks: list[TranscriptChunk]
    transcript: Transcript


class VideoSearchService:
    def __init__(
        self,
        *,
        document_embedder: Callable[[Sequence[str]], list[list[float]]] = embed_documents,
        query_embedder: Callable[[str], list[float]] = embed_query,
        transcript_fetcher: Callable[[str], Transcript] = fetch_transcript,
    ) -> None:
        self._document_embedder = document_embedder
        self._query_embedder = query_embedder
        self._transcript_fetcher = transcript_fetcher
        self._indexes: dict[str, _VideoIndex] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    def index_video(
        self,
        video_url: str,
        transcript: Transcript | None = None,
    ) -> bool:
        """Index a video once per process; return False if it was indexed already."""
        video_id = parse_youtube_video_id(video_url)
        with self._lock:
            video_lock = self._locks.setdefault(video_id, threading.Lock())

        with video_lock:
            with self._lock:
                if video_id in self._indexes:
                    return False

            started_at = time.perf_counter()
            loaded_transcript = transcript or self._transcript_fetcher(video_url)
            chunks = chunk_transcript(
                loaded_transcript.segments,
                video_id=video_id,
                language=loaded_transcript.language,
            )
            if not chunks:
                raise ValueError("This video has no transcript text to index.")

            vectors = self._document_embedder([chunk.text for chunk in chunks])
            if len(vectors) != len(chunks):
                raise RuntimeError("The embedding model returned an unexpected number of vectors.")

            vector_matrix = np.asarray(vectors, dtype=np.float32)
            if vector_matrix.ndim != 2 or vector_matrix.shape[0] != len(chunks):
                raise RuntimeError("The embedding model returned invalid vectors.")

            index = faiss.IndexFlatIP(vector_matrix.shape[1])
            index.add(vector_matrix)
            with self._lock:
                self._indexes[video_id] = _VideoIndex(
                    index=index,
                    chunks=chunks,
                    transcript=loaded_transcript,
                )

            elapsed = time.perf_counter() - started_at
            logger.info(
                "Indexed video %s: %d chunks in %.3f seconds",
                video_id,
                len(chunks),
                elapsed,
            )
            return True

    def search(self, video_id: str, question: str, *, limit: int = 5) -> list[SearchResult]:
        """Search one indexed video and return deduplicated matches in video order."""
        if limit < 1:
            raise ValueError("limit must be greater than zero.")

        parsed_video_id = parse_youtube_video_id(video_id)
        with self._lock:
            video_index = self._indexes.get(parsed_video_id)
        if video_index is None:
            raise LookupError("This video has not been indexed in this session.")

        query_vector = np.asarray([self._query_embedder(question)], dtype=np.float32)
        if query_vector.ndim != 2 or query_vector.shape[1] != video_index.index.d:
            raise RuntimeError("The query embedding has an incompatible dimension.")

        scores, indices = video_index.index.search(query_vector, video_index.index.ntotal)
        candidates = sorted(
            (
                (int(chunk_index), float(score))
                for chunk_index, score in zip(indices[0], scores[0], strict=True)
                if chunk_index >= 0
            ),
            key=lambda match: match[1],
            reverse=True,
        )

        selected: list[tuple[int, float]] = []
        for candidate_index, score in candidates:
            candidate_text = video_index.chunks[candidate_index].text
            if any(
                _is_near_duplicate(candidate_text, video_index.chunks[index].text)
                for index, _ in selected
            ):
                continue
            selected.append((candidate_index, score))
            if len(selected) == limit:
                break

        selected.sort(key=lambda match: video_index.chunks[match[0]].position)
        return [
            SearchResult(
                text=video_index.chunks[index].text,
                start=video_index.chunks[index].start,
                end=video_index.chunks[index].end,
                position=video_index.chunks[index].position,
                video_id=video_index.chunks[index].video_id,
                language=video_index.chunks[index].language,
                score=score,
            )
            for index, score in selected
        ]

    def get_transcript(self, video_id: str) -> Transcript:
        """Return the timestamped transcript held by this process's video index."""
        parsed_video_id = parse_youtube_video_id(video_id)
        with self._lock:
            video_index = self._indexes.get(parsed_video_id)
        if video_index is None:
            raise LookupError("This video has not been indexed in this session.")
        return video_index.transcript


def _is_near_duplicate(first: str, second: str) -> bool:
    first_tokens = first.lower().split()
    second_tokens = second.lower().split()
    if not first_tokens or not second_tokens:
        return False
    return SequenceMatcher(None, first_tokens, second_tokens).ratio() >= _NEAR_DUPLICATE_THRESHOLD


_video_search_service = VideoSearchService()


def get_video_search_service() -> VideoSearchService:
    return _video_search_service
