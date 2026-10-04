from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from app.agent.language import ANSWER_LANGUAGE_CODES
from app.agent.qa import get_question_answer_service
from app.config import FRONTEND_ORIGIN, RETRIEVER_K
from app.ingest.search import get_video_search_service
from app.ingest.youtube_url import parse_youtube_video_id
from app.transcripts.youtube_transcript import TranscriptFetchError, fetch_transcript

app = FastAPI(title="ChatMyVideo API", version="0.1.0")


class IndexVideoRequest(BaseModel):
    url: str


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskVideoRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=RETRIEVER_K, ge=1, le=20)
    history: list[ChatHistoryMessage] = Field(default_factory=list, max_length=20)
    # Optional language code chosen in the UI ("en", "hi", "hinglish", "video", ...).
    answer_language: str | None = Field(default=None, max_length=20)

    @field_validator("answer_language")
    @classmethod
    def answer_language_must_be_supported(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        code = value.strip().lower()
        if code not in ANSWER_LANGUAGE_CODES:
            raise ValueError("Please choose a supported answer language.")
        return code

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Please enter a question.")
        return question


app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/transcript")
def get_transcript(url: str = Query(..., description="YouTube video URL or video ID")) -> dict:
    try:
        transcript = fetch_transcript(url)
    except TranscriptFetchError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return {
        "language": transcript.language,
        "duration": transcript.duration,
        "segments": [
            {"text": segment.text, "start": segment.start, "end": segment.end}
            for segment in transcript.segments
        ],
    }


@app.post("/videos/index")
def index_video(request: IndexVideoRequest) -> dict[str, str | bool]:
    try:
        indexed = get_video_search_service().index_video(request.url)
    except TranscriptFetchError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "video_id": parse_youtube_video_id(request.url),
        "indexed": indexed,
    }


@app.get("/videos/{video_id}/search")
def search_video(
    video_id: str,
    question: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=20),
) -> dict[str, list[dict[str, str | float | int]]]:
    try:
        results = get_video_search_service().search(video_id, question, limit=limit)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "results": [
            {
                "text": result.text,
                "start": result.start,
                "end": result.end,
                "position": result.position,
                "video_id": result.video_id,
                "language": result.language,
                "score": result.score,
            }
            for result in results
        ]
    }


@app.post("/videos/{video_id}/ask")
def ask_video(video_id: str, request: AskVideoRequest) -> dict:
    try:
        answer = get_question_answer_service().answer(
            video_id,
            request.question,
            limit=request.limit,
            history=[message.model_dump() for message in request.history],
            # Only pass the language when the user chose one; otherwise it is detected.
            **({"answer_language": request.answer_language} if request.answer_language else {}),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "answer": answer.answer,
        "cited_times": answer.cited_times,
        "source_excerpts": [
            {
                "text": excerpt.text,
                "start": excerpt.start,
                "end": excerpt.end,
                "position": excerpt.position,
            }
            for excerpt in answer.source_excerpts
        ],
    }
