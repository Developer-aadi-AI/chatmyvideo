import json
from pathlib import Path

import pytest

from app.agent.qa import QuestionAnswer, SourceExcerpt
from app.engine import EngineError, LoadedVideo
from app.evaluate_engine import (
    EvaluationQuestion,
    EvaluationVideo,
    citation_time_difference,
    judge_answer,
    load_dataset,
    run_video_evaluation,
    validate_video_ready,
)


class FakeEngine:
    def __init__(
        self,
        answer: QuestionAnswer,
        *,
        load_error: EngineError | None = None,
    ) -> None:
        self.answer_result = answer
        self.load_error = load_error
        self.loaded_links: list[str] = []
        self.asked: list[str] = []

    def load_video(self, link: str) -> LoadedVideo:
        if self.load_error:
            raise self.load_error
        self.loaded_links.append(link)
        return LoadedVideo("dQw4w9WgXcQ", "en", 120, True)

    def ask(self, question: str) -> QuestionAnswer:
        self.asked.append(question)
        return self.answer_result


class FakeJudge:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, **kwargs) -> str:
        self.calls.append(messages)
        return self.response


def _ready_video() -> EvaluationVideo:
    return EvaluationVideo(
        id="tutorial-en",
        title="Short English tutorial",
        video_type="short_tutorial",
        caption_type="manual_or_creator_captions",
        url="https://youtu.be/dQw4w9WgXcQ",
        questions=(
            EvaluationQuestion(
                id="steps",
                question="What steps are shown?",
                off_topic=False,
                expected_time_seconds=72,
                expected_points=("First key step", "Final confirmation"),
                expected_answer="The instructor does the first key step then confirms success.",
            ),
            EvaluationQuestion(
                id="unrelated",
                question="What is the capital of Japan?",
                off_topic=True,
                expected_time_seconds=None,
                expected_points=(),
                expected_answer="The video does not cover this question.",
            ),
        ),
    )


def test_dataset_contains_four_templates_with_four_or_five_questions() -> None:
    videos = load_dataset()

    assert [video.id for video in videos] == [
        "tutorial-en",
        "talk-en",
        "hindi-video",
        "auto-captions",
    ]
    assert all(4 <= len(video.questions) <= 5 for video in videos)
    assert videos[-1].caption_type == "auto_generated"
    assert videos[2].questions[0].question.startswith("वक्ता")


def test_unfilled_video_template_is_rejected_before_loading() -> None:
    with pytest.raises(ValueError, match="Fill in the real YouTube link"):
        validate_video_ready(load_dataset()[0])


@pytest.mark.parametrize(
    ("citations", "expected", "pass_result", "closest", "delta"),
    [
        (["01:12"], 72, True, "01:12", 0),
        (["01:31", "02:20"], 72, True, "01:31", 19),
        (["01:33"], 72, False, "01:33", 21),
        ([], 72, False, None, None),
        (["01:12"], None, None, None, None),
    ],
)
def test_citation_timestamp_tolerance(
    citations: list[str],
    expected: float | None,
    pass_result: bool | None,
    closest: str | None,
    delta: float | None,
) -> None:
    assert citation_time_difference(citations, expected) == (pass_result, closest, delta)


def test_judge_parses_a_boolean_result_and_receives_expected_points() -> None:
    question = _ready_video().questions[0]
    judge = FakeJudge('{"passed": true, "reason": "Both expected facts are present."}')

    result = judge_answer(question, "The steps are followed, then success is confirmed.", judge)

    assert result == {"passed": True, "reason": "Both expected facts are present."}
    assert "First key step" in judge.calls[0][1]["content"]
    assert "Final confirmation" in judge.calls[0][1]["content"]


def test_run_saves_per_question_results_and_only_loads_selected_video(
    tmp_path: Path,
) -> None:
    video = _ready_video()
    engine = FakeEngine(
        QuestionAnswer(
            answer="It follows the key step and confirms success [01:12].",
            cited_times=["01:12"],
            source_excerpts=[SourceExcerpt("Evidence", 72, 80, 0)],
        )
    )
    judge = FakeJudge('{"passed": true, "reason": "Correct."}')
    result_path = tmp_path / "runs.jsonl"

    run = run_video_evaluation(
        video,
        engine=engine,
        judge_provider=judge,
        results_path=result_path,
    )

    assert engine.loaded_links == [video.url]
    assert engine.asked == ["What steps are shown?", "What is the capital of Japan?"]
    assert len(judge.calls) == 2
    assert run["status"] == "complete"
    assert [item["citation_within_20_seconds"] for item in run["question_results"]] == [
        True,
        None,
    ]
    assert [item["expected_points_covered"] for item in run["question_results"]] == [
        True,
        None,
    ]
    assert [item["off_topic_refused"] for item in run["question_results"]] == [
        None,
        True,
    ]
    stored = json.loads(result_path.read_text(encoding="utf-8").splitlines()[0])
    assert stored["video_key"] == "tutorial-en"
    assert len(stored["question_results"]) == 2


def test_load_failure_is_saved_as_a_failed_run(tmp_path: Path) -> None:
    video = _ready_video()
    path = tmp_path / "runs.jsonl"
    engine = FakeEngine(
        QuestionAnswer("No.", [], []),
        load_error=EngineError("No captions are available."),
    )

    result = run_video_evaluation(
        video,
        engine=engine,
        judge_provider=FakeJudge('{"passed": true, "reason": "unused"}'),
        results_path=path,
    )

    assert result["status"] == "failed"
    assert all(item["status"] == "not_run" for item in result["question_results"])
    assert json.loads(path.read_text(encoding="utf-8"))["video_error"] == (
        "No captions are available."
    )


def test_judge_rejects_invalid_json_instead_of_returning_fake_pass() -> None:
    with pytest.raises(RuntimeError, match="invalid JSON"):
        judge_answer(_ready_video().questions[0], "An answer.", FakeJudge("not json"))
