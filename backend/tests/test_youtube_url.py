import pytest

from app.ingest.youtube_url import parse_youtube_video_id


@pytest.mark.parametrize(
    "value",
    [
        "dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "youtube.com/watch?feature=share&v=dQw4w9WgXcQ&t=42",
        "http://youtube.com/watch?si=abc123&t=42&v=dQw4w9WgXcQ",
        "www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123",
        "http://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?t=42",
        "youtu.be/dQw4w9WgXcQ",
        "https://www.youtu.be/dQw4w9WgXcQ",
        "https://youtube.com/shorts/dQw4w9WgXcQ",
        "m.youtube.com/shorts/dQw4w9WgXcQ",
        "https://m.youtube.com/live/dQw4w9WgXcQ?feature=share",
        "youtube.com/live/dQw4w9WgXcQ",
        "youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/v/dQw4w9WgXcQ",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
        "//youtube.com/watch?v=dQw4w9WgXcQ",
        "  https://music.youtube.com/watch?v=dQw4w9WgXcQ  ",
    ],
)
def test_parse_youtube_video_id_accepts_common_links(value: str) -> None:
    assert parse_youtube_video_id(value) == "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-video-id",
        "dQw4w9WgXc",
        "dQw4w9WgXcQextra",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com.evil.test/watch?v=dQw4w9WgXcQ",
        "https://youtube.com.attacker.test/watch?v=dQw4w9WgXcQ",
        "https://notyoutube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be.evil.test/dQw4w9WgXcQ",
        "https://youtube.com@evil.test/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?v=invalid",
        "https://youtube.com/watch",
        "https://youtube.com/watch?v=dQw4w9WgXcQ&v=abcdefghijk",
        "https://youtube.com/shorts/invalid",
        "https://youtube.com/channel/dQw4w9WgXcQ",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com:8080/watch?v=dQw4w9WgXcQ",
    ],
)
def test_parse_youtube_video_id_rejects_non_video_inputs(value: str) -> None:
    with pytest.raises(ValueError):
        parse_youtube_video_id(value)
