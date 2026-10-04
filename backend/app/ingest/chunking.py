from collections.abc import Sequence
from dataclasses import dataclass

from app.transcripts.base import TranscriptSegment


@dataclass(frozen=True)
class TranscriptChunk:
    text: str
    start: float
    end: float
    position: int
    video_id: str
    language: str


def chunk_transcript(
    segments: Sequence[TranscriptSegment],
    *,
    video_id: str,
    language: str,
    target_chars: int = 850,
    overlap_chars: int = 150,
) -> list[TranscriptChunk]:
    """Group whole timestamped transcript segments into overlapping search chunks."""
    if target_chars <= 0:
        raise ValueError("target_chars must be greater than zero.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative.")
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be smaller than target_chars.")

    nonempty_segments = [segment for segment in segments if segment.text.strip()]
    if not nonempty_segments:
        return []

    ordered_segments = sorted(nonempty_segments, key=lambda segment: (segment.start, segment.end))
    chunks: list[TranscriptChunk] = []
    start_index = 0

    while start_index < len(ordered_segments):
        chunk_start_index = start_index
        end_index = start_index
        text_length = 0
        while end_index < len(ordered_segments):
            segment_length = len(ordered_segments[end_index].text.strip())
            combined_length = text_length + segment_length
            if end_index > start_index:
                combined_length += 1
            if end_index > start_index and combined_length > target_chars:
                break
            text_length = combined_length
            end_index += 1

        chunk_segments = ordered_segments[start_index:end_index]
        chunks.append(
            TranscriptChunk(
                text=" ".join(segment.text.strip() for segment in chunk_segments),
                start=chunk_segments[0].start,
                end=chunk_segments[-1].end,
                position=len(chunks),
                video_id=video_id,
                language=language,
            )
        )
        if end_index == len(ordered_segments):
            break

        start_index = end_index
        overlap_length = 0
        if overlap_chars:
            for index in range(end_index - 1, chunk_start_index - 1, -1):
                segment_length = len(ordered_segments[index].text.strip())
                new_overlap_length = overlap_length + segment_length + (1 if overlap_length else 0)
                next_segment_length = len(ordered_segments[end_index].text.strip())
                combined_length = new_overlap_length + 1 + next_segment_length
                if combined_length > target_chars:
                    break
                if overlap_length and new_overlap_length > overlap_chars:
                    break
                overlap_length = new_overlap_length
                start_index = index

    return chunks
