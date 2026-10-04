import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "ChatMyVideo — Make every video make sense",
  description:
    "Turn YouTube videos into clear summaries, searchable notes, and answers with timestamps.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}