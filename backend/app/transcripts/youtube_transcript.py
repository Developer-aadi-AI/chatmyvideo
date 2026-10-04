from __future__ import annotations

import html
import json
import math
import re
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from app.config import SUPADATA_API_KEY
from app.ingest.youtube_url import parse_youtube_video_id
from app.transcripts.base import Transcript, TranscriptSegment

_API_BASE_URL = "https://api.supadata.ai/v1"
_VIDEO_CUE_PATTERN = re.compile(r"\[[^\]]*\]")
_HTML_TAG_PATTERN = re.compile(r"<[^>]*>")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_PUNCTUATION_SPACE_PATTERN = re.compile(r"\s+([,.!?;:])")
_TRANSCRIPT_CACHE: dict[str, Transcript | TranscriptFetchError] = {}
_TRANSCRIPT_CACHE_LOCK = threading.Lock()
_VIDEO_LOCKS: dict[str, threading.Lock] = {}


class TranscriptFetchError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def clean_segment_text(text: str) -> str:
    """Remove markup, caption cues, HTML entities, and redundant whitespace."""
    cleaned = html.unescape(text)
    cleaned = _HTML_TAG_PATTERN.sub(" ", cleaned)
    cleaned = _VIDEO_CUE_PATTERN.sub(" ", cleaned)
    cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return _PUNCTUATION_SPACE_PATTERN.sub(r"\1", cleaned)


class SupadataTranscriptProvider:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_timeout: float = 20.0,
        poll_timeout: float = 90.0,
        poll_interval: float = 1.0,
    ) -> None:
        self.api_key = SUPADATA_API_KEY if api_key is None else api_key
        self.request_timeout = request_timeout
        self.poll_timeout = poll_timeout
        self.poll_interval = poll_interval

    def fetch(self, video_url: str) -> Transcript:
        try:
            video_id = parse_youtube_video_id(video_url)
        except ValueError as exc:
            raise TranscriptFetchError(str(exc)) from exc

        if not self.api_key:
            raise TranscriptFetchError(
                "Transcript fetching is not configured. Please try again later.",
                status_code=503,
            )

        with _TRANSCRIPT_CACHE_LOCK:
            cached = _TRANSCRIPT_CACHE.get(video_id)
            if isinstance(cached, Transcript):
                return cached
            if isinstance(cached, TranscriptFetchError):
                raise TranscriptFetchError(str(cached), cached.status_code)
            video_lock = _VIDEO_LOCKS.setdefault(video_id, threading.Lock())

        with video_lock:
            with _TRANSCRIPT_CACHE_LOCK:
                cached = _TRANSCRIPT_CACHE.get(video_id)
                if isinstance(cached, Transcript):
                    return cached
                if isinstance(cached, TranscriptFetchError):
                    raise TranscriptFetchError(str(cached), cached.status_code)
            try:
                transcript = self._fetch_uncached(video_id)
            except TranscriptFetchError as exc:
                # Outages and timeouts (5xx) are temporary, so let the user retry them;
                # cache only lasting failures such as missing captions or quota limits.
                if exc.status_code < 500:
                    with _TRANSCRIPT_CACHE_LOCK:
                        _TRANSCRIPT_CACHE[video_id] = exc
                raise
            with _TRANSCRIPT_CACHE_LOCK:
                _TRANSCRIPT_CACHE[video_id] = transcript
            return transcript

    def _fetch_uncached(self, video_id: str) -> Transcript:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        query = urlencode({"url": video_url, "text": "false", "mode": "native"})
        payload = self._request_json(f"{_API_BASE_URL}/transcript?{query}")
        if payload.get("error"):
            self._raise_api_error(payload)

        job_id = payload.get("jobId")
        if isinstance(job_id, str) and job_id:
            payload = self._poll_job(job_id)

        if payload.get("status") == "failed":
            error = payload.get("error")
            self._raise_api_error(error if isinstance(error, Mapping) else payload)

        return self._parse_transcript(payload)

    def _poll_job(self, job_id: str) -> Mapping[str, Any]:
        deadline = time.monotonic() + self.poll_timeout
        job_url = f"{_API_BASE_URL}/transcript/{quote(job_id, safe='')}"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TranscriptFetchError(
                    "The transcript is taking too long to prepare. Please try again later.",
                    status_code=504,
                )
            payload = self._request_json(
                job_url,
                timeout=min(self.request_timeout, remaining),
            )
            if payload.get("status") == "completed":
                return payload
            if payload.get("status") == "failed":
                error = payload.get("error")
                self._raise_api_error(error if isinstance(error, Mapping) else payload)
            if payload.get("status") not in {"queued", "active"}:
                raise TranscriptFetchError(
                    "The transcript service returned an unexpected job status.",
                    status_code=502,
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TranscriptFetchError(
                    "The transcript is taking too long to prepare. Please try again later.",
                    status_code=504,
                )
            time.sleep(min(self.poll_interval, remaining))

    def _request_json(
        self,
        url: str,
        *,
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        request = Request(
            url,
            headers={"x-api-key": self.api_key, "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=timeout or self.request_timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                body = exc.read().decode("utf-8")
                error_payload = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                error_payload = {}
            self._raise_api_error(error_payload, status_code=exc.code)
        except (TimeoutError, URLError, OSError) as exc:
            raise TranscriptFetchError(
                "The transcript service is temporarily unreachable. Please try again later.",
                status_code=503,
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TranscriptFetchError(
                "The transcript service returned an unreadable response.",
                status_code=502,
            ) from exc

        if not isinstance(payload, Mapping):
            raise TranscriptFetchError(
                "The transcript service returned an unexpected response.",
                status_code=502,
            )
        return payload

    def _raise_api_error(
        self,
        payload: Mapping[str, Any],
        *,
        status_code: int | None = None,
    ) -> NoReturn:
        code = str(payload.get("error", "")).lower()
        details = " ".join(str(payload.get(key, "")) for key in ("message", "details")).lower()
        upstream_status = status_code or 0

        if upstream_status == 401 or code == "unauthorized":
            raise TranscriptFetchError(
                "The transcript service rejected its API key. Please contact support.",
                status_code=503,
            )
        if upstream_status in {402, 429} or code in {"limit-exceeded", "upgrade-required"}:
            raise TranscriptFetchError(
                "The transcript service has reached its monthly quota. Please try again later.",
                status_code=429,
            )
        if any(word in details for word in ("private", "region", "country", "blocked")):
            raise TranscriptFetchError(
                "This video is private or blocked in your region. Try a public, accessible video.",
                status_code=422,
            )
        if upstream_status == 403 or code == "forbidden":
            raise TranscriptFetchError(
                "This video is private or blocked in your region. Try a public, accessible video.",
                status_code=422,
            )
        if upstream_status >= 500 or code == "internal-error":
            raise TranscriptFetchError(
                "The transcript service is temporarily unreachable. Please try again later.",
                status_code=503,
            )
        if upstream_status == 404 or code == "not-found":
            raise TranscriptFetchError(
                "This video is private, unavailable, or blocked in your region. "
                "Try a public, accessible video.",
                status_code=422,
            )
        if code == "transcript-unavailable" or upstream_status == 206:
            raise TranscriptFetchError(
                "No captions are available for this video.",
                status_code=422,
            )
        raise TranscriptFetchError(
            "We couldn't fetch captions for this video. Please check the link and try again.",
            status_code=422,
        )

    def _parse_transcript(self, payload: Mapping[str, Any]) -> Transcript:
        content = payload.get("content")
        if not isinstance(content, list) or not content:
            raise TranscriptFetchError(
                "No captions are available for this video.",
                status_code=422,
            )

        language = payload.get("lang")
        if not isinstance(language, str) or not language:
            language = next(
                (
                    chunk.get("lang")
                    for chunk in content
                    if isinstance(chunk, Mapping) and isinstance(chunk.get("lang"), str)
                ),
                None,
            )
        if not language:
            raise TranscriptFetchError(
                "The transcript service did not provide the transcript language.",
                status_code=502,
            )

        segments: list[TranscriptSegment] = []
        for chunk in content:
            if not isinstance(chunk, Mapping):
                raise TranscriptFetchError(
                    "The transcript service returned an invalid caption segment.",
                    status_code=502,
                )
            text = chunk.get("text")
            offset = chunk.get("offset")
            duration = chunk.get("duration")
            if (
                not isinstance(text, str)
                or not isinstance(offset, (int, float))
                or isinstance(offset, bool)
                or not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(offset)
                or not math.isfinite(duration)
                or offset < 0
                or duration < 0
            ):
                raise TranscriptFetchError(
                    "The transcript service returned an invalid caption segment.",
                    status_code=502,
                )
            cleaned_text = clean_segment_text(text)
            if cleaned_text:
                start = offset / 1000
                segments.append(
                    TranscriptSegment(
                        text=cleaned_text,
                        start=start,
                        end=start + duration / 1000,
                    )
                )

        if not segments:
            raise TranscriptFetchError(
                "No readable captions are available for this video.",
                status_code=422,
            )
        return Transcript(language=language, segments=segments)


def fetch_transcript(video_url: str) -> Transcript:
    return SupadataTranscriptProvider().fetch(video_url)


def _format_time(seconds: float) -> str:
    total_seconds = int(seconds)
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m app.transcripts.youtube_transcript <youtube-url>")
        return 2

    try:
        transcript = fetch_transcript(sys.argv[1])
    except TranscriptFetchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Language: {transcript.language}")
    print(f"Segments: {len(transcript.segments)}")
    print(f"Duration: {_format_time(transcript.duration)}")
    print("First segments:")
    for segment in transcript.segments[:5]:
        print(f"[{_format_time(segment.start)}] {segment.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
