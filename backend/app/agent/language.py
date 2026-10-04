from __future__ import annotations

import re
import unicodedata
from collections import Counter

_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_MIN_QUESTION_WORDS = 3

_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0900, 0x097F, "hi"),
    (0x0980, 0x09FF, "bn"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
    (0x0E00, 0x0E7F, "th"),
    (0x0E80, 0x0EFF, "lo"),
    (0x0F00, 0x0FFF, "bo"),
    (0x1000, 0x109F, "my"),
    (0x10A0, 0x10FF, "ka"),
    (0x1200, 0x137F, "am"),
    (0x13A0, 0x13FF, "chr"),
    (0x1780, 0x17FF, "km"),
    (0x3040, 0x309F, "ja"),
    (0x30A0, 0x30FF, "ja"),
    (0xAC00, 0xD7AF, "ko"),
    (0x0600, 0x06FF, "ar"),
    (0x0750, 0x077F, "ar"),
    (0x0400, 0x052F, "ru"),
    (0x0370, 0x03FF, "el"),
    (0x0590, 0x05FF, "he"),
    (0x1100, 0x11FF, "ko"),
    (0x3400, 0x4DBF, "zh"),
    (0x4E00, 0x9FFF, "zh"),
)

_LANGUAGE_NAMES = {
    "am": "Amharic",
    "ar": "Arabic",
    "bn": "Bengali",
    "bo": "Tibetan",
    "chr": "Cherokee",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hinglish": "Hinglish (Hindi written in Latin script)",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kn": "Kannada",
    "ko": "Korean",
    "km": "Khmer",
    "lo": "Lao",
    "ml": "Malayalam",
    "mr": "Marathi",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "or": "Odia",
    "pa": "Punjabi",
    "pt": "Portuguese",
    "ru": "Russian",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

_LATIN_MARKERS: dict[str, frozenset[str]] = {
    "de": frozenset({"der", "die", "das", "ist", "nicht", "und", "was", "wie", "warum"}),
    "es": frozenset(
        {
            "el",
            "la",
            "los",
            "las",
            "que",
            "como",
            "por",
            "para",
            "una",
            "qué",
            "es",
            "esto",
            "funciona",
        }
    ),
    "fr": frozenset(
        {"le", "les", "des", "est", "pas", "pour", "avec", "que", "comment", "pourquoi"}
    ),
    "id": frozenset({"yang", "dan", "ini", "itu", "adalah", "untuk", "dengan", "bagaimana"}),
    "it": frozenset({"il", "lo", "gli", "che", "come", "perché", "non", "una", "sono"}),
    "nl": frozenset({"het", "een", "is", "niet", "voor", "waarom", "hoe", "deze"}),
    "pt": frozenset({"que", "não", "uma", "para", "como", "porquê", "isso", "está"}),
    "tr": frozenset({"bir", "bu", "için", "nasıl", "neden", "değil", "mı", "ve"}),
    "vi": frozenset({"của", "là", "không", "và", "như", "thế", "nào", "tại", "sao"}),
    "en": frozenset({"the", "is", "are", "what", "how", "why", "does", "this", "that", "with"}),
}

_HINGLISH_MARKERS = frozenset(
    {
        "accha",
        "acha",
        "baare",
        "batao",
        "hai",
        "hain",
        "ka",
        "kar",
        "kaise",
        "kare",
        "kya",
        "kyu",
        "kyun",
        "mein",
        "mujhe",
        "nahi",
        "nahin",
        "raha",
        "rahi",
        "samjhao",
        "video",
        "ye",
        "yeh",
    }
)
_DISTINCT_HINGLISH_MARKERS = frozenset(
    {"baare", "batao", "kaise", "kyu", "kyun", "mujhe", "nahi", "nahin", "samjhao"}
)
_DEVANAGARI_LANGUAGE_MARKERS: dict[str, frozenset[str]] = {
    "mr": frozenset({"आहे", "आहेत", "म्हणजे", "आणि", "नाही", "कशाबद्दल"}),
    "ne": frozenset({"छन्", "गर्नु", "कसरी", "छैन", "भिडियो"}),
}
_URDU_MARKERS = frozenset({"ہے", "ہیں", "کیا", "کیسے", "اور", "نہیں", "یہ"})


def normalize_language(language: str) -> str:
    """Return a readable language name for common ISO codes and YouTube tags."""
    normalized = language.strip().replace("_", "-")
    code = normalized.split("-", maxsplit=1)[0].lower()
    return _LANGUAGE_NAMES.get(code, normalized or "the video's transcript language")


def _normalize_word(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word.casefold())
    return "".join(character for character in decomposed if unicodedata.category(character) != "Mn")


def _script_language(text: str) -> str | None:
    counts: Counter[str] = Counter()
    for character in text:
        codepoint = ord(character)
        for start, end, language in _SCRIPT_RANGES:
            if start <= codepoint <= end and unicodedata.category(character).startswith("L"):
                counts[language] += 1
                break
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def detect_question_language(question: str, transcript_language: str) -> str:
    """Choose a stable answer language, falling back when text is short or ambiguous."""
    script = _script_language(question)
    script_letter_count = sum(
        1
        for character in question
        if unicodedata.category(character).startswith("L")
        and any(start <= ord(character) <= end for start, end, _ in _SCRIPT_RANGES)
    )
    if script and script_letter_count >= _MIN_QUESTION_WORDS:
        if script in {"zh", "ja", "ko"}:
            return _LANGUAGE_NAMES[script]
        if script == "hi":
            normalized_question = unicodedata.normalize("NFC", question)
            if any(marker in normalized_question for marker in _DEVANAGARI_LANGUAGE_MARKERS["mr"]):
                return _LANGUAGE_NAMES["mr"]
            if any(marker in normalized_question for marker in _DEVANAGARI_LANGUAGE_MARKERS["ne"]):
                return _LANGUAGE_NAMES["ne"]
            return _LANGUAGE_NAMES["hi"]
        if script == "ar":
            if any(marker in question for marker in _URDU_MARKERS):
                return _LANGUAGE_NAMES["ur"]
            return _LANGUAGE_NAMES["ar"]
        return normalize_language(script)

    words = [word.lower() for word in _WORD_PATTERN.findall(question)]
    if len(words) < _MIN_QUESTION_WORDS:
        return normalize_language(transcript_language)

    word_set = {_normalize_word(word) for word in words}
    hinglish_hits = word_set & _HINGLISH_MARKERS
    distinctive_hinglish_hits = word_set & _DISTINCT_HINGLISH_MARKERS
    if len(distinctive_hinglish_hits) >= 1 and len(hinglish_hits) >= 2:
        return _LANGUAGE_NAMES["hinglish"]
    if len(hinglish_hits) >= 3:
        return _LANGUAGE_NAMES["hinglish"]

    counts = {
        language: len(word_set & {_normalize_word(marker) for marker in markers})
        for language, markers in _LATIN_MARKERS.items()
    }
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    best_language, best_score = ranked[0]
    second_score = ranked[1][1]
    if best_score >= 2 and best_score > second_score:
        return _LANGUAGE_NAMES[best_language]
    return normalize_language(transcript_language)


def answer_language_instruction(question: str, transcript_language: str) -> str:
    language = detect_question_language(question, transcript_language)
    return (
        f"Answer in {language}. Keep the response natural and fluent in that language. "
        "Do not change languages unless the user asks you to."
    )
