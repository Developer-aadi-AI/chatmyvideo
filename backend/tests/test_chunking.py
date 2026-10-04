from itertools import pairwise

import pytest

from app.ingest.chunking import TranscriptChunk, chunk_transcript
from app.transcripts.base import TranscriptSegment


def make_segments(*texts: str) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(text=text, start=float(index * 5), end=float(index * 5 + 4))
        for index, text in enumerate(texts)
    ]


def test_empty_segments_make_no_empty_chunks() -> None:
    assert chunk_transcript([], video_id="video-1", language="en") == []
    chunks = chunk_transcript(
        make_segments(" ", "\t"),
        video_id="video-1",
        language="en",
    )
    assert chunks == []


def test_chunks_keep_segments_whole_and_exclude_timestamps_from_text() -> None:
    segments = make_segments("First thought.", "Second thought.", "Third thought.")

    chunks = chunk_transcript(
        segments,
        video_id="video-1",
        language="en",
        target_chars=30,
        overlap_chars=0,
    )

    assert chunks == [
        TranscriptChunk("First thought. Second thought.", 0.0, 9.0, 0, "video-1", "en"),
        TranscriptChunk("Third thought.", 10.0, 14.0, 1, "video-1", "en"),
    ]
    assert all("00:" not in chunk.text for chunk in chunks)


def test_chunk_metadata_and_timing_are_in_video_order() -> None:
    chunks = chunk_transcript(
        make_segments("A" * 20, "B" * 20, "C" * 20, "D" * 20),
        video_id="dQw4w9WgXcQ",
        language="fr",
        target_chars=25,
        overlap_chars=0,
    )

    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.video_id == "dQw4w9WgXcQ" for chunk in chunks)
    assert all(chunk.language == "fr" for chunk in chunks)
    assert all(chunk.start <= chunk.end for chunk in chunks)
    assert all(left.start <= right.start for left, right in pairwise(chunks))
    assert all(left.end <= right.end for left, right in pairwise(chunks))


def test_segments_are_chunked_in_timestamp_order() -> None:
    chunks = chunk_transcript(
        [
            TranscriptSegment("Later segment.", start=10, end=11),
            TranscriptSegment("Earlier segment.", start=2, end=3),
        ],
        video_id="video-1",
        language="en",
        target_chars=10,
        overlap_chars=0,
    )

    assert [chunk.start for chunk in chunks] == [2, 10]
    assert [chunk.text for chunk in chunks] == ["Earlier segment.", "Later segment."]


def test_neighbouring_chunks_overlap_by_about_requested_amount() -> None:
    segments = make_segments(*[f"segment-{index:02d} " + ("x" * 30) for index in range(12)])

    chunks = chunk_transcript(
        segments,
        video_id="video-1",
        language="en",
        target_chars=150,
        overlap_chars=40,
    )

    assert len(chunks) > 1
    chunk_segment_indices = [
        [index for index, segment in enumerate(segments) if segment.text in chunk.text]
        for chunk in chunks
    ]
    for index, (previous, following) in enumerate(pairwise(chunks), start=1):
        previous_indices = chunk_segment_indices[index - 1]
        following_indices = chunk_segment_indices[index]
        assert set(previous_indices).intersection(following_indices)
        assert following.start <= previous.end
        assert following.start > previous.start


def test_all_transcript_content_appears_in_chunks_without_segment_splitting() -> None:
    texts = [f"unique-token-{index} " + ("context " * 12) for index in range(18)]
    segments = make_segments(*texts)

    chunks = chunk_transcript(
        segments,
        video_id="video-1",
        language="en",
        target_chars=250,
        overlap_chars=60,
    )

    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)
    all_chunk_text = "\n".join(chunk.text for chunk in chunks)
    assert all(text.strip() in all_chunk_text for text in texts)
    assert all(
        segment.text in chunk.text
        for chunk in chunks
        for segment in segments
        if segment.text in chunk.text
    )


@pytest.mark.parametrize(
    ("target_chars", "overlap_chars"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_rejects_invalid_chunk_sizes(target_chars: int, overlap_chars: int) -> None:
    with pytest.raises(ValueError):
        chunk_transcript(
            make_segments("text"),
            video_id="video-1",
            language="en",
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
