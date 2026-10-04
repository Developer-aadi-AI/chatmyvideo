import pytest

from app.transcripts import youtube_transcript
from app.transcripts.base import Transcript, TranscriptSegment
from app.transcripts.youtube_transcript import (
    SupadataTranscriptProvider,
    TranscriptFetchError,
    clean_segment_text,
)


@pytest.fixture(autouse=True)
def clear_transcript_cache() -> None:
    youtube_transcript._TRANSCRIPT_CACHE.clear()


def test_parses_native_response_units_and_cleans_caption_text(monkeypatch) -> None:
    provider = SupadataTranscriptProvider(api_key="test-key")
    response = {
        "lang": "en",
        "availableLangs": ["en"],
        "content": [
            {
                "text": "  <b>Hello&nbsp; world!</b> [Music] ",
                "offset": 181896,
                "duration": 1000,
            },
            {"text": "Second line", "offset": 182896, "duration": 750},
        ],
    }
    monkeypatch.setattr(provider, "_request_json", lambda url, **kwargs: response)

    transcript = provider.fetch("https://youtu.be/dQw4w9WgXcQ")

    assert transcript.language == "en"
    assert transcript.duration == pytest.approx(183.646)
    assert transcript.segments == [
        TranscriptSegment(text="Hello world!", start=181.896, end=182.896),
        TranscriptSegment(text="Second line", start=182.896, end=183.646),
    ]


def test_requests_existing_captions_and_caches_video_by_id(monkeypatch) -> None:
    provider = SupadataTranscriptProvider(api_key="test-key")
    calls: list[str] = []
    response = {
        "lang": "en",
        "content": [{"text": "Hello", "offset": 0, "duration": 1000}],
    }

    def request(url: str, **kwargs):
        calls.append(url)
        return response

    monkeypatch.setattr(provider, "_request_json", request)

    first = provider.fetch("https://youtu.be/dQw4w9WgXcQ")
    second = provider.fetch("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=20")

    assert first is second
    assert len(calls) == 1
    assert "mode=native" in calls[0]
    assert "text=false" in calls[0]


def test_polls_async_jobs_until_completed(monkeypatch) -> None:
    provider = SupadataTranscriptProvider(
        api_key="test-key",
        poll_interval=0,
    )
    responses = iter(
        [
            {"jobId": "job-123"},
            {"status": "active"},
            {
                "status": "completed",
                "lang": "fr",
                "content": [{"text": "Bonjour", "offset": 2500, "duration": 1200}],
            },
        ]
    )
    urls: list[str] = []

    def request(url: str, **kwargs):
        urls.append(url)
        return next(responses)

    monkeypatch.setattr(provider, "_request_json", request)

    transcript = provider.fetch("dQw4w9WgXcQ")

    assert transcript.language == "fr"
    assert transcript.segments == [TranscriptSegment("Bonjour", 2.5, 3.7)]
    assert urls[1].endswith("/transcript/job-123")
    assert urls[2].endswith("/transcript/job-123")


def test_async_job_timeout_is_friendly(monkeypatch) -> None:
    provider = SupadataTranscriptProvider(
        api_key="test-key",
        poll_timeout=0,
    )
    monkeypatch.setattr(provider, "_request_json", lambda url, **kwargs: {"jobId": "slow"})

    with pytest.raises(TranscriptFetchError, match="taking too long"):
        provider.fetch("dQw4w9WgXcQ")


@pytest.mark.parametrize(
    ("error_payload", "expected"),
    [
        ({"error": "transcript-unavailable"}, "No captions are available"),
        ({"error": "not-found"}, "private, unavailable, or blocked"),
        (
            {"error": "forbidden", "message": "Private video"},
            "private or blocked in your region",
        ),
        (
            {"error": "forbidden", "details": "Video blocked in this country"},
            "private or blocked in your region",
        ),
        ({"error": "limit-exceeded"}, "monthly quota"),
        ({"error": "unauthorized"}, "API key"),
        ({"error": "internal-error"}, "temporarily unreachable"),
    ],
)
def test_maps_supadata_failures_to_friendly_errors(
    error_payload: dict[str, str],
    expected: str,
    monkeypatch,
) -> None:
    provider = SupadataTranscriptProvider(api_key="test-key")
    monkeypatch.setattr(
        provider,
        "_request_json",
        lambda url, **kwargs: error_payload,
    )

    with pytest.raises(TranscriptFetchError, match=expected):
        provider.fetch("dQw4w9WgXcQ")


def test_caches_a_failure_to_avoid_retrying_video_in_same_process(monkeypatch) -> None:
    provider = SupadataTranscriptProvider(api_key="test-key")
    calls = 0

    def request(url: str, **kwargs):
        nonlocal calls
        calls += 1
        return {"error": "transcript-unavailable"}

    monkeypatch.setattr(provider, "_request_json", request)
    for _ in range(2):
        with pytest.raises(TranscriptFetchError, match="No captions"):
            provider.fetch("dQw4w9WgXcQ")

    assert calls == 1


def test_cached_quota_failure_keeps_user_facing_status(monkeypatch) -> None:
    provider = SupadataTranscriptProvider(api_key="test-key")
    monkeypatch.setattr(
        provider,
        "_request_json",
        lambda url, **kwargs: {"error": "limit-exceeded"},
    )

    for _ in range(2):
        with pytest.raises(TranscriptFetchError) as error:
            provider.fetch("dQw4w9WgXcQ")
        assert error.value.status_code == 429


def test_invalid_youtube_url_is_rejected_without_request() -> None:
    provider = SupadataTranscriptProvider(api_key="test-key")

    with pytest.raises(TranscriptFetchError, match="YouTube"):
        provider.fetch("https://youtube.com.evil.test/watch?v=dQw4w9WgXcQ")


def test_missing_api_key_is_friendly() -> None:
    provider = SupadataTranscriptProvider(api_key="")

    with pytest.raises(TranscriptFetchError, match="not configured"):
        provider.fetch("dQw4w9WgXcQ")


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (206, "No captions are available"),
        (401, "API key"),
        (402, "monthly quota"),
        (429, "monthly quota"),
        (500, "temporarily unreachable"),
    ],
)
def test_maps_supadata_http_statuses_to_friendly_errors(
    status_code: int,
    expected: str,
) -> None:
    provider = SupadataTranscriptProvider(api_key="test-key")

    with pytest.raises(TranscriptFetchError, match=expected):
        provider._raise_api_error({}, status_code=status_code)


def test_segment_text_cleanup_removes_html_entities_cues_and_spacing() -> None:
    assert clean_segment_text("  &lt;b&gt;Hello&nbsp; there&lt;/b&gt; [MUSIC] ! ") == "Hello there!"


def test_transcript_duration_uses_latest_segment_end() -> None:
    transcript = Transcript(
        language="en",
        segments=[
            TranscriptSegment("One", 0, 2),
            TranscriptSegment("Two", 3, 5),
        ],
    )

    assert transcript.duration == 5


def test_temporary_outage_is_not_cached_so_retry_can_succeed(monkeypatch) -> None:
    provider = SupadataTranscriptProvider(api_key="test-key")
    responses = [
        {"error": "internal-error"},
        {"lang": "en", "content": [{"text": "Hello", "offset": 0, "duration": 1000}]},
    ]
    monkeypatch.setattr(provider, "_request_json", lambda url, **kwargs: responses.pop(0))

    with pytest.raises(TranscriptFetchError) as error:
        provider.fetch("dQw4w9WgXcQ")
    assert error.value.status_code == 503

    assert provider.fetch("dQw4w9WgXcQ").segments[0].text == "Hello"
