"use client";

import { useEffect, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import Link from "next/link";

import Markdown from "@/components/Markdown";
import TimestampLink from "@/components/TimestampLink";

type ApiState = "checking" | "online" | "offline" | "unconfigured";
type ChatRole = "user" | "assistant";
type ChatMessage = { role: ChatRole; content: string };
type SourceExcerpt = {
  text: string;
  start: number;
  end: number;
  position: number;
};
type ChatTurn = {
  question: string;
  answer: string;
  citedTimes: string[];
  sources: SourceExcerpt[];
};
type LoadedVideo = {
  videoId: string;
  url: string;
  indexed: boolean;
};
type IndexResponse = { video_id: string; indexed: boolean };
type AskResponse = {
  answer: string;
  cited_times: string[];
  source_excerpts: SourceExcerpt[];
};

const MAX_HISTORY_MESSAGES = 6;
// One-click requests; the backend sends these through its whole-video path.
const QUICK_ACTIONS = [
  { label: "Summary", question: "Summarize this video" },
  { label: "Flashcards", question: "Create flashcards for this video" },
] as const;
// Codes must match ANSWER_LANGUAGE_CODES in backend/app/agent/language.py.
const ANSWER_LANGUAGES = [
  { code: "en", label: "English" },
  { code: "hi", label: "Hindi" },
  { code: "hinglish", label: "Hinglish" },
  { code: "ur", label: "Urdu" },
  { code: "bn", label: "Bengali" },
  { code: "mr", label: "Marathi" },
  { code: "gu", label: "Gujarati" },
  { code: "pa", label: "Punjabi" },
  { code: "ta", label: "Tamil" },
  { code: "te", label: "Telugu" },
  { code: "kn", label: "Kannada" },
  { code: "ml", label: "Malayalam" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "pt", label: "Portuguese" },
  { code: "ar", label: "Arabic" },
  { code: "ru", label: "Russian" },
  { code: "zh", label: "Chinese" },
  { code: "ja", label: "Japanese" },
  { code: "ko", label: "Korean" },
  { code: "video", label: "Video's language" },
  { code: "auto", label: "Auto (match my question)" },
] as const;
const DEFAULT_ANSWER_LANGUAGE = "en";
const ANSWER_LANGUAGE_STORAGE_KEY = "chatmyvideo.answerLanguage";
const MAX_HISTORY_CHARS = 4000;

const configuredApiUrl = process.env.NEXT_PUBLIC_API_URL?.trim();
const apiUrl = (
  configuredApiUrl ||
  (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "")
).replace(/\/+$/, "");

async function responseError(response: Response): Promise<string> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    return `The backend returned an error (${response.status}).`;
  }
  if (
    typeof payload === "object" &&
    payload !== null &&
    "detail" in payload &&
    typeof payload.detail === "string"
  ) {
    return payload.detail;
  }
  // FastAPI validation errors (422) send `detail` as a list of { msg } objects.
  if (
    typeof payload === "object" &&
    payload !== null &&
    "detail" in payload &&
    Array.isArray(payload.detail)
  ) {
    const first: unknown = payload.detail[0];
    if (typeof first === "object" && first !== null && "msg" in first && typeof first.msg === "string") {
      return first.msg.replace(/^Value error, /, "");
    }
  }
  return `The backend returned an error (${response.status}).`;
}

function isIndexResponse(value: unknown): value is IndexResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    "video_id" in value &&
    typeof value.video_id === "string" &&
    "indexed" in value &&
    typeof value.indexed === "boolean"
  );
}

function isAskResponse(value: unknown): value is AskResponse {
  return (
    typeof value === "object" &&
    value !== null &&
    "answer" in value &&
    typeof value.answer === "string" &&
    "cited_times" in value &&
    Array.isArray(value.cited_times) &&
    value.cited_times.every((time) => typeof time === "string") &&
    "source_excerpts" in value &&
    Array.isArray(value.source_excerpts) &&
    value.source_excerpts.every(
      (source) =>
        typeof source === "object" &&
        source !== null &&
        "text" in source &&
        typeof source.text === "string" &&
        "start" in source &&
        typeof source.start === "number" &&
        "end" in source &&
        typeof source.end === "number" &&
        "position" in source &&
        typeof source.position === "number",
    )
  );
}

function formatTime(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safeSeconds / 60);
  return `${String(minutes).padStart(2, "0")}:${String(safeSeconds % 60).padStart(2, "0")}`;
}

function friendlyRequestError(error: unknown, fallback: string): string {
  if (error instanceof TypeError) {
    return "We couldn't reach the backend. Check that it is running and that the Vercel API URL is configured.";
  }
  return error instanceof Error ? error.message : fallback;
}

function renderCitedAnswer(
  answer: string,
  citedTimes: string[],
  onSeek: (seconds: number) => void,
): ReactNode[] {
  const citedSet = new Set(citedTimes);
  return answer.split(/(\[\d+:\d{2}\])/g).map((part, index) => {
    const timestamp = /^\[(\d+):([0-5]\d)\]$/.exec(part);
    if (!timestamp) return <span key={index}>{part}</span>;
    const seconds = Number(timestamp[1]) * 60 + Number(timestamp[2]);
    if (!citedSet.has(`${timestamp[1]}:${timestamp[2]}`)) {
      return <span key={index}>{part}</span>;
    }
    return (
      <TimestampLink className="citation-link" key={index} onSeek={onSeek} seconds={seconds}>
        {part}
      </TimestampLink>
    );
  });
}

export default function HomePage() {
  const [apiState, setApiState] = useState<ApiState>("checking");
  const [videoUrl, setVideoUrl] = useState("");
  const [loadedVideo, setLoadedVideo] = useState<LoadedVideo | null>(null);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [isLoadingVideo, setIsLoadingVideo] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState("");
  const playerRef = useRef<HTMLIFrameElement>(null);
  const [answerLanguage, setAnswerLanguage] = useState<string>(DEFAULT_ANSWER_LANGUAGE);

  // Restore the viewer's last language choice (a per-browser convenience only).
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(ANSWER_LANGUAGE_STORAGE_KEY);
      if (saved && ANSWER_LANGUAGES.some((language) => language.code === saved)) {
        setAnswerLanguage(saved);
      }
    } catch {
      // Storage can be unavailable (private mode); the default language still works.
    }
  }, []);

  function chooseAnswerLanguage(code: string): void {
    setAnswerLanguage(code);
    try {
      window.localStorage.setItem(ANSWER_LANGUAGE_STORAGE_KEY, code);
    } catch {
      // Ignore storage failures; the choice still applies for this visit.
    }
  }

  // Seek the embedded player through the YouTube IFrame postMessage API (enabled by
  // `enablejsapi=1` in the embed URL), so no extra script or dependency is needed.
  function seekTo(seconds: number): void {
    const player = playerRef.current;
    if (!player?.contentWindow) return;
    const send = (func: string, args: unknown[] = []) =>
      player.contentWindow?.postMessage(JSON.stringify({ event: "command", func, args }), "*");
    send("seekTo", [seconds, true]);
    send("playVideo");
    player.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  useEffect(() => {
    if (!apiUrl) {
      setApiState("unconfigured");
      return;
    }
    const controller = new AbortController();

    async function checkApi(): Promise<void> {
      try {
        const response = await fetch(`${apiUrl}/health`, {
          cache: "no-store",
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Health check returned ${response.status}`);
        }

        const health: unknown = await response.json();
        if (
          typeof health !== "object" ||
          health === null ||
          !("status" in health) ||
          health.status !== "ok"
        ) {
          throw new Error("Health check returned an unexpected response");
        }

        setApiState("online");
      } catch (error: unknown) {
        if (!controller.signal.aborted) {
          console.error("Backend health check failed.", error);
          setApiState("offline");
        }
      }
    }

    void checkApi();
    return () => controller.abort();
  }, []);

  const statusLabel = {
    checking: "Checking API",
    online: "API connected",
    offline: "API unavailable",
    unconfigured: "Connect a backend",
  }[apiState];

  async function handleLoadVideo(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiUrl) {
      setError(
        "The backend URL is not configured. Add NEXT_PUBLIC_API_URL in the Vercel project settings and redeploy.",
      );
      return;
    }
    setIsLoadingVideo(true);
    setError("");
    setLoadedVideo(null);
    setTurns([]);
    try {
      const response = await fetch(`${apiUrl}/videos/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: videoUrl.trim() }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const payload: unknown = await response.json();
      if (!isIndexResponse(payload)) {
        throw new Error("The backend returned an unexpected video response.");
      }
      setLoadedVideo({
        videoId: payload.video_id,
        url: videoUrl.trim(),
        indexed: payload.indexed,
      });
      setVideoUrl(videoUrl.trim());
    } catch (loadError: unknown) {
      setError(friendlyRequestError(loadError, "We couldn't load this video. Please try again."));
    } finally {
      setIsLoadingVideo(false);
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await askQuestion(question.trim(), { clearInput: true });
  }

  async function askQuestion(askedQuestion: string, { clearInput }: { clearInput: boolean }) {
    if (!loadedVideo || !askedQuestion || isAsking || !apiUrl) return;
    // The backend accepts at most 20 history messages of 4,000 characters each and only
    // uses the last 6, so send just those; sending everything breaks after 10 questions.
    const history: ChatMessage[] = turns
      .flatMap((turn): ChatMessage[] => [
        { role: "user", content: turn.question },
        { role: "assistant", content: turn.answer },
      ])
      .filter((message) => message.content.trim())
      .slice(-MAX_HISTORY_MESSAGES)
      .map((message) => ({ ...message, content: message.content.slice(-MAX_HISTORY_CHARS) }));
    setIsAsking(true);
    setError("");
    try {
      const response = await fetch(
        `${apiUrl}/videos/${encodeURIComponent(loadedVideo.videoId)}/ask`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: askedQuestion,
            history,
            // "auto" lets the backend detect the language from the question.
            ...(answerLanguage === "auto" ? {} : { answer_language: answerLanguage }),
          }),
        },
      );
      if (!response.ok) throw new Error(await responseError(response));
      const payload: unknown = await response.json();
      if (!isAskResponse(payload)) {
        throw new Error("The backend returned an unexpected answer response.");
      }
      setTurns((previous) => [
        ...previous,
        {
          question: askedQuestion,
          answer: payload.answer,
          citedTimes: payload.cited_times,
          sources: payload.source_excerpts,
        },
      ]);
      if (clearInput) setQuestion("");
    } catch (askError: unknown) {
      setError(
        friendlyRequestError(askError, "We couldn't answer this question. Please try again."),
      );
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <main className="site-shell">
      <header className="topbar">
        <Link className="brand" href="/" aria-label="ChatMyVideo home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 20 20" fill="none">
              <path d="M7 5.5 14 10l-7 4.5v-9Z" fill="currentColor" />
            </svg>
          </span>
          <span>chatmyvideo</span>
        </Link>
        <nav className="topbar-right" aria-label="Main navigation">
          <a href="#chat">Ask about a video</a>
          <span className="api-status" data-state={apiState} role="status">
            <span className="status-dot" aria-hidden="true" />
            {statusLabel}
          </span>
        </nav>
      </header>

      <section className="hero" aria-labelledby="hero-title">
        <div className="eyebrow">
          <span className="eyebrow-line" />
          Your video, understood
        </div>
        <h1 id="hero-title">
          Watch less.
          <br />
          <em>Understand</em> more.
        </h1>
        <p className="hero-copy">
          Turn long videos into clear summaries, useful notes, and answers that
          take you straight to the moment that matters.
        </p>
        <div className="hero-actions">
          <a className="primary-link" href="#chat">
            Try it now
            <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <path
                d="M4 10h12m-5-5 5 5-5 5"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.5"
              />
            </svg>
          </a>
          <span className="quiet-link">Grounded answers · Clickable timestamps</span>
        </div>

        <aside className="preview" aria-label="Example video summary preview">
          <div className="preview-video" aria-hidden="true">
            <span className="play-button">
              <svg viewBox="0 0 20 20" fill="none">
                <path d="m7 4.75 8 5.25-8 5.25V4.75Z" fill="currentColor" />
              </svg>
            </span>
          </div>
          <div>
            <div className="preview-label">A little more than a video</div>
            <div className="preview-title">Ideas worth coming back to.</div>
          </div>
          <div className="preview-caption">
            <span>Summary · Notes · Timestamped answers</span>
            <span>01</span>
          </div>
        </aside>
      </section>

      <section className="chat-section" id="chat" aria-labelledby="chat-title">
        <div className="chat-heading">
          <span className="feature-number">Your video, your questions</span>
          <h2 id="chat-title">Start with a link.</h2>
          <p>We’ll find the captions, then answer from the video with moments you can verify.</p>
        </div>

        <div className="video-workspace">
          <div className="workspace-main">
            <form className="video-form" onSubmit={handleLoadVideo}>
              <label htmlFor="video-url">YouTube video link</label>
              <div className="form-row">
                <input
                  autoComplete="url"
                  id="video-url"
                  inputMode="url"
                  onChange={(event) => setVideoUrl(event.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  required
                  type="text"
                  value={videoUrl}
                />
                {/* Disabled while asking so a late answer can't attach to a newly loaded video. */}
                <button
                  className="primary-button"
                  disabled={isLoadingVideo || isAsking}
                  type="submit"
                >
                  {isLoadingVideo ? "Loading captions…" : "Load video"}
                </button>
              </div>
              <p className="form-hint">Only existing captions are used. No AI transcription.</p>
            </form>

            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}

            {loadedVideo && (
              <section className="loaded-video" aria-label="Loaded video chat">
                <div className="loaded-video-header">
                  <div>
                    <span className="feature-number">
                      {loadedVideo.indexed ? "Video ready" : "Video ready · already indexed"}
                    </span>
                    <h3>Ask anything about this video.</h3>
                  </div>
                  <a
                    className="video-open-link"
                    href={`https://www.youtube.com/watch?v=${encodeURIComponent(loadedVideo.videoId)}`}
                    rel="noreferrer"
                    target="_blank"
                  >
                    Open on YouTube ↗
                  </a>
                </div>
                <div className="video-player">
                  <iframe
                    ref={playerRef}
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                    allowFullScreen
                    referrerPolicy="strict-origin-when-cross-origin"
                    src={`https://www.youtube-nocookie.com/embed/${encodeURIComponent(loadedVideo.videoId)}?enablejsapi=1&rel=0`}
                    title="YouTube video player"
                  />
                </div>

                {turns.length > 0 && (
                  <div className="conversation" aria-live="polite">
                    {turns.map((turn, index) => (
                      <article className="chat-turn" key={`${index}-${turn.question}`}>
                        <p className="question-bubble">{turn.question}</p>
                        <div className="answer-block">
                          <Markdown
                            renderText={(text) => renderCitedAnswer(text, turn.citedTimes, seekTo)}
                            text={turn.answer}
                          />
                          {turn.sources.length > 0 && (
                            <details className="source-list">
                              <summary>Sources ({turn.sources.length})</summary>
                              <ul>
                                {turn.sources.map((source) => (
                                  <li key={`${source.position}-${source.start}`}>
                                    <TimestampLink
                                      className="source-time-button"
                                      onSeek={seekTo}
                                      seconds={Math.floor(source.start)}
                                    >
                                      <span className="source-time">
                                        {formatTime(source.start)}–{formatTime(source.end)}
                                      </span>
                                    </TimestampLink>
                                    <span>{source.text}</span>
                                  </li>
                                ))}
                              </ul>
                            </details>
                          )}
                        </div>
                      </article>
                    ))}
                    <div aria-live="polite" className="chat-bottom" />
                  </div>
                )}

                <form className="question-form" onSubmit={handleAsk}>
                  <div className="quick-actions" role="group" aria-label="Quick actions">
                    {QUICK_ACTIONS.map((action) => (
                      <button
                        className="quick-action"
                        disabled={isAsking}
                        key={action.label}
                        onClick={() => void askQuestion(action.question, { clearInput: false })}
                        type="button"
                      >
                        {action.label}
                      </button>
                    ))}
                    <label className="language-picker">
                      <span>Answer in</span>
                      <select
                        disabled={isAsking}
                        onChange={(event) => chooseAnswerLanguage(event.target.value)}
                        value={answerLanguage}
                      >
                        {ANSWER_LANGUAGES.map((language) => (
                          <option key={language.code} value={language.code}>
                            {language.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <label htmlFor="video-question">Ask a question</label>
                  <div className="form-row">
                    <input
                      id="video-question"
                      maxLength={2000}
                      onChange={(event) => setQuestion(event.target.value)}
                      placeholder="What are the main points?"
                      required
                      value={question}
                    />
                    <button
                      className="primary-button"
                      disabled={isAsking || !question.trim()}
                      type="submit"
                    >
                      {isAsking ? "Thinking…" : "Ask"}
                    </button>
                  </div>
                  {turns.length > 0 && (
                    <p className="form-hint">Recent conversation is included for follow-ups.</p>
                  )}
                </form>
              </section>
            )}
          </div>
        </div>
      </section>

      <section className="features" id="how-it-works" aria-label="How it works">
        <div className="feature-grid">
          <article className="feature">
            <span className="feature-number">01 — Make it clear</span>
            <h2>Get the whole picture.</h2>
            <p>Ask broad questions to get a concise, timestamped overview of the video.</p>
          </article>
          <article className="feature">
            <span className="feature-number">02 — Find the moment</span>
            <h2>Answers with receipts.</h2>
            <p>Every answer is grounded in the transcript, with timestamps to verify it.</p>
          </article>
          <article className="feature">
            <span className="feature-number">03 — Stay focused</span>
            <h2>Jump straight in.</h2>
            <p>Follow a citation back to the exact part of the video worth another look.</p>
          </article>
        </div>
      </section>

      <footer className="footer">
        <Link className="brand" href="/" aria-label="ChatMyVideo home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 20 20" fill="none">
              <path d="M7 5.5 14 10l-7 4.5v-9Z" fill="currentColor" />
            </svg>
          </span>
          <span>chatmyvideo</span>
        </Link>
        <span>Understand the video. Keep the insight.</span>
      </footer>
    </main>
  );
}