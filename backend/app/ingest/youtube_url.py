import re
from urllib.parse import parse_qs, urlsplit

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
_VIDEO_PATH_PREFIXES = {
    "embed",
    "live",
    "shorts",
    "v",
}


def parse_youtube_video_id(value: str) -> str:
    """Return the video ID from a YouTube URL or a bare 11-character ID."""
    candidate = value.strip()
    if _VIDEO_ID_PATTERN.fullmatch(candidate):
        return candidate
    if not candidate:
        raise ValueError("Enter a YouTube video URL or 11-character video ID.")

    url = candidate if "://" in candidate or candidate.startswith("//") else f"https://{candidate}"
    if url.startswith("//"):
        url = f"https:{url}"

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Enter a valid YouTube video URL.") from exc

    if parsed.scheme.lower() not in {"http", "https"} or hostname is None:
        raise ValueError("Enter a valid YouTube video URL.")
    if hostname.lower() not in _YOUTUBE_HOSTS or parsed.username or parsed.password:
        raise ValueError("URL must point to a YouTube video.")
    if port is not None and port not in {80, 443}:
        raise ValueError("URL must point to a YouTube video.")

    path_parts = [part for part in parsed.path.split("/") if part]
    if hostname.lower() in {"youtu.be", "www.youtu.be"}:
        video_id = path_parts[0] if len(path_parts) == 1 else ""
    elif path_parts == ["watch"]:
        values = parse_qs(parsed.query, keep_blank_values=True).get("v", [])
        video_id = values[0] if len(values) == 1 else ""
    elif len(path_parts) == 2 and path_parts[0] in _VIDEO_PATH_PREFIXES:
        video_id = path_parts[1]
    else:
        video_id = ""

    if not _VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("URL does not contain a valid YouTube video ID.")
    return video_id