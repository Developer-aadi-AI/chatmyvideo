"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

type ApiState = "checking" | "online" | "offline";

const apiUrl = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

export default function HomePage() {
  const [apiState, setApiState] = useState<ApiState>("checking");

  useEffect(() => {
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
  }[apiState];

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
          <a href="#how-it-works">How it works</a>
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
          <a className="primary-link" href="#how-it-works">
            See how it works
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
          <span className="quiet-link">Free to use · Built around the transcript</span>
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

      <section className="features" id="how-it-works" aria-label="How it works">
        <div className="feature-grid">
          <article className="feature">
            <span className="feature-number">01 — Make it clear</span>
            <h2>Get the whole picture.</h2>
            <p>Turn a long video into a concise summary and notes you can revisit.</p>
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