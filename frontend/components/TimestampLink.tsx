import type { ReactNode } from "react";

type TimestampLinkProps = {
  seconds: number;
  onSeek: (seconds: number) => void;
  className?: string;
  children: ReactNode;
};

/** A citation button that jumps the embedded player to `seconds` and starts playback. */
export default function TimestampLink({ seconds, onSeek, className, children }: TimestampLinkProps) {
  return (
    <button
      className={className}
      onClick={() => onSeek(seconds)}
      title="Play the video from this moment"
      type="button"
    >
      {children}
    </button>
  );
}
