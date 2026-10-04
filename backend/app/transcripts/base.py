from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Transcript:
    language: str
    segments: list[TranscriptSegment]

    @property
    def duration(self) -> float:
        return max((segment.end for segment in self.segments), default=0.0)


class TranscriptProvider(Protocol):
    def fetch(self, video_url: str) -> Transcript: ...