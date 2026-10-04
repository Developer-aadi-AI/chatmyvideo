import Link from "next/link";

export default function LibraryPage() {
  return (
    <main className="placeholder-page">
      <p className="eyebrow">Coming soon</p>
      <h1>Your library is taking shape.</h1>
      <p className="hero-copy">
        Saved videos and conversations will live here. The first development
        milestone is connecting this app to its backend.
      </p>
      <Link className="primary-link" href="/">
        Back to home
      </Link>
    </main>
  );
}