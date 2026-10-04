import Link from "next/link";

export default function VideoPage() {
  return (
    <main className="placeholder-page">
      <p className="eyebrow">Video workspace</p>
      <h1>Good insights are on the way.</h1>
      <p className="hero-copy">
        Video summaries, transcript search, and timestamped answers will be
        available here in a later development step.
      </p>
      <Link className="primary-link" href="/">
        Back to home
      </Link>
    </main>
  );
}