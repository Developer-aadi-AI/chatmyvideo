from app.agent.qa import QuestionAnswer
from app.chat import main
from app.engine import LoadedVideo
from app.llm.groq_provider import ChatMessage


def test_terminal_chat_uses_engine_for_loading_and_follow_ups(monkeypatch, capsys) -> None:
    prompts = iter(
        [
            "https://youtu.be/dQw4w9WgXcQ",
            "What is the topic?",
            "Explain that more",
            "quit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(prompts))

    class FakeEngine:
        def __init__(self) -> None:
            self.questions: list[tuple[str, list[ChatMessage]]] = []

        def load_video(self, link: str) -> LoadedVideo:
            assert link == "https://youtu.be/dQw4w9WgXcQ"
            return LoadedVideo("dQw4w9WgXcQ", "en", 92, True)

        def ask(self, question: str, *, history=None) -> QuestionAnswer:
            self.questions.append((question, list(history or [])))
            return QuestionAnswer(
                answer=f"Answer to {question} [00:12].",
                cited_times=["00:12"],
                source_excerpts=[],
            )

    engine = FakeEngine()

    assert main(engine) == 0

    output = capsys.readouterr().out
    assert "Loaded video (en, 01:32)." in output
    assert "Answer to What is the topic? [00:12]." in output
    assert "Timestamps: 00:12" in output
    assert engine.questions[0] == ("What is the topic?", [])
    assert engine.questions[1] == (
        "Explain that more",
        [
            {"role": "user", "content": "What is the topic?"},
            {"role": "assistant", "content": "Answer to What is the topic? [00:12]."},
        ],
    )


def test_terminal_chat_reports_engine_error_and_continues(monkeypatch, capsys) -> None:
    from app.engine import EngineError

    prompts = iter(["dQw4w9WgXcQ", "", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(prompts))

    class FakeEngine:
        def load_video(self, link: str) -> LoadedVideo:
            return LoadedVideo("dQw4w9WgXcQ", "en", 0, False)

        def ask(self, question: str, *, history=None) -> QuestionAnswer:
            raise EngineError("Please enter a question.")

    assert main(FakeEngine()) == 0
    assert "Error: Please enter a question." in capsys.readouterr().out
