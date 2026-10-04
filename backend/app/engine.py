from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.agent.qa import QuestionAnswer, QuestionAnswerService
from app.ingest.search import VideoSearchService, get_video_search_service
from app.ingest.youtube_url import parse_youtube_video_id
from app.llm.groq_provider import ChatMessage
from app.transcripts.base import Transcript
from app.transcripts.youtube_transcript import TranscriptFetchError

logger = logging.getLogger(__name__)
MAX_QUESTION_CHARS = 2000


class EngineError(Exception):
    """A user-facing failure from the video chat engine."""


@dataclass(frozen=True)
class LoadedVideo:
    video_id: str
    language: str
    duration: float
    indexed: bool


class VideoChatEngine:
    """Framework-independent interface for loading a video and asking questions."""

    def __init__(
        self,
        *,
        search_service: VideoSearchService | None = None,
        answer_service: QuestionAnswerService | None = None,
    ) -> None:
        self._search_service = search_service or get_video_search_service()
        self._answer_service = answer_service or QuestionAnswerService(
            search_service=self._search_service
        )
        self._video_id: str | None = None

    def load_video(self, link: str) -> LoadedVideo:
        """Fetch captions and index the linked video for this engine instance."""
        self._video_id = None
        try:
            video_id = parse_youtube_video_id(link)
        except Exception as exc:
            logger.exception("Rejected invalid video link.")
            message = (
                str(exc)
                if isinstance(exc, ValueError)
                else ("Please provide a valid YouTube video link or video ID.")
            )
            raise EngineError(message) from exc

        try:
            indexed = self._search_service.index_video(link)
            transcript: Transcript = self._search_service.get_transcript(video_id)
        except Exception as exc:
            logger.exception("Failed to load video.")
            raise EngineError(_friendly_message(exc, operation="load")) from exc

        self._video_id = video_id
        return LoadedVideo(
            video_id=video_id,
            language=transcript.language,
            duration=transcript.duration,
            indexed=indexed,
        )

    def ask(
        self,
        question: str,
        *,
        history: Sequence[ChatMessage] | None = None,
    ) -> QuestionAnswer:
        """Answer a question about the loaded video with optional conversation context."""
        if self._video_id is None:
            logger.warning("Rejected question because no video has been loaded.")
            raise EngineError("Load a YouTube video before asking a question.")
        if not isinstance(question, str) or not question.strip():
            logger.warning("Rejected an empty question.")
            raise EngineError("Please enter a question.")
        if len(question) > MAX_QUESTION_CHARS:
            logger.warning("Rejected a question exceeding the maximum length.")
            raise EngineError(f"Questions must be {MAX_QUESTION_CHARS:,} characters or fewer.")
        if history is not None and (
            not isinstance(history, Sequence)
            or isinstance(history, (str, bytes))
            or any(
                not isinstance(message, dict)
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message in history
            )
        ):
            logger.warning("Rejected invalid chat history.")
            raise EngineError("Chat history must contain non-empty user or assistant messages.")

        try:
            return self._answer_service.answer(
                self._video_id,
                question.strip(),
                history=history or (),
            )
        except Exception as exc:
            logger.exception("Failed to answer a question.")
            raise EngineError(_friendly_message(exc, operation="answer")) from exc


def _friendly_message(exc: Exception, *, operation: str) -> str:
    if isinstance(exc, TranscriptFetchError):
        return str(exc)
    if (
        operation == "load"
        and isinstance(exc, ValueError)
        and str(exc) == "This video has no transcript text to index."
    ):
        return str(exc)
    if isinstance(exc, LookupError):
        return "The video is not ready yet. Load it again and try your question."
    if operation == "load":
        return "We couldn't load this video. Please check the link and try again."
    return "We couldn't answer that question right now. Please try again later."
