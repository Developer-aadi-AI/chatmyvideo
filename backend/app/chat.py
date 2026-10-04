from __future__ import annotations

from app.engine import ChatMessage, EngineError, VideoChatEngine


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def main(engine: VideoChatEngine | None = None) -> int:
    """Run a terminal conversation using only the public engine interface."""
    video_engine = engine or VideoChatEngine()
    while True:
        try:
            link = input("Paste a YouTube link: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        try:
            video = video_engine.load_video(link)
            break
        except EngineError as exc:
            print(f"Error: {exc}")

    print(
        f"Loaded video ({video.language}, "
        f"{_format_duration(video.duration)}). Ask questions; type 'quit' to exit."
    )
    history: list[ChatMessage] = []
    while True:
        try:
            question = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if question.strip().lower() in {"quit", "exit"}:
            print("Goodbye.")
            return 0

        try:
            answer = video_engine.ask(question, history=history)
        except EngineError as exc:
            print(f"Error: {exc}")
            continue

        print(f"Assistant: {answer.answer}")
        if answer.cited_times:
            print(f"Timestamps: {', '.join(answer.cited_times)}")
        history.extend(
            [
                ChatMessage(role="user", content=question),
                ChatMessage(role="assistant", content=answer.answer),
            ]
        )
        history = history[-6:]


if __name__ == "__main__":
    raise SystemExit(main())
