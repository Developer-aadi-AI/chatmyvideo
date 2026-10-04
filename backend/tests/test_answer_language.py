import pytest

from app.agent.language import answer_language_instruction, detect_question_language


@pytest.mark.parametrize(
    ("question", "expected_language"),
    [
        ("What is the purpose of this video?", "English"),
        ("Was ist das und warum funktioniert es so?", "German"),
        ("Pourquoi est-ce que cela fonctionne?", "French"),
        ("¿Qué es esto y cómo funciona?", "Spanish"),
        ("Como isso funciona e porquê?", "Portuguese"),
        ("Yang ini bagaimana dan untuk apa?", "Indonesian"),
        ("ये वीडियो किस बारे में है", "Hindi"),
        ("বাংলায় এই ভিডিওটি কীভাবে কাজ করে?", "Bengali"),
        ("ગુજરાતીમાં આ વિડિયો શું સમજાવે છે?", "Gujarati"),
        ("தமிழில் இந்த வீடியோ எதைப் பற்றி பேசுகிறது?", "Tamil"),
        ("తెలుగులో ఈ వీడియో దేని గురించి?", "Telugu"),
        ("ಕನ್ನಡದಲ್ಲಿ ಈ ವಿಡಿಯೋ ಏನನ್ನು ವಿವರಿಸುತ್ತದೆ?", "Kannada"),
        ("മലയാളത്തിൽ ഈ വീഡിയോ എന്താണ് പറയുന്നത്?", "Malayalam"),
        ("ਪੰਜਾਬੀ ਵਿੱਚ ਇਹ ਵੀਡੀਓ ਕਿਸ ਬਾਰੇ ਹੈ?", "Punjabi"),
        ("मराठीमध्ये हा व्हिडिओ कशाबद्दल आहे?", "Marathi"),
        ("नेपालीमा यो भिडियो के बारेमा छ?", "Nepali"),
        ("这段视频主要讲什么内容？", "Chinese"),
        ("この動画は何について説明していますか？", "Japanese"),
        ("이 영상은 무엇에 대해 설명하나요?", "Korean"),
        ("ما موضوع هذا الفيديو وكيف يشرح الفكرة؟", "Arabic"),
        ("О чём это видео и как оно объясняет тему?", "Russian"),
    ],
)
def test_detects_supported_question_languages(question: str, expected_language: str) -> None:
    assert detect_question_language(question, "en") == expected_language


def test_hinglish_is_not_misidentified_as_another_latin_language() -> None:
    question = "ye video kis baare mein hai"

    assert detect_question_language(question, "fr") == ("Hinglish (Hindi written in Latin script)")


@pytest.mark.parametrize(
    "question",
    [
        "What?",
        "Tell me more about this",
        "Can you explain?",
    ],
)
def test_short_or_uncertain_question_falls_back_to_transcript_language(
    question: str,
) -> None:
    assert detect_question_language(question, "hi") == "Hindi"


def test_detection_is_deterministic() -> None:
    question = "¿Qué es esto y cómo funciona?"

    detections = [detect_question_language(question, "en") for _ in range(10)]

    assert detections == ["Spanish"] * 10


def test_instruction_explicitly_selects_question_language() -> None:
    assert answer_language_instruction(
        "ye video kis baare mein hai",
        "en",
    ) == (
        "Answer in Hinglish (Hindi written in Latin script). Keep the response natural "
        "and fluent in that language. Do not change languages unless the user asks you to."
    )


def test_iso_code_fallback_is_normalized() -> None:
    assert detect_question_language("Can you explain?", "pt-BR") == "Portuguese"


def test_chosen_answer_language_overrides_detection() -> None:
    from app.agent.language import resolve_answer_language

    assert resolve_answer_language("Summarize this video", "ur", "en") == "English"
    assert resolve_answer_language("Summarize this video", "ur", "video") == "Urdu"
    assert resolve_answer_language("Summarize this video", "ur", "hinglish") == (
        "Hinglish (Hindi written in Latin script)"
    )
    assert resolve_answer_language("Summarize this video", "ur", None) == "Urdu"
