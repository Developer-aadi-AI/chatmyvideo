import { Fragment } from "react";
import type { ReactNode } from "react";

/** Renders plain text (no Markdown) — used for citation links inside Markdown. */
export type TextRenderer = (text: string) => ReactNode[];

type MarkdownProps = {
  text: string;
  renderText: TextRenderer;
};

type Block =
  | { kind: "heading"; level: number; text: string }
  | { kind: "paragraph"; lines: string[] }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "table"; header: string[]; rows: string[][] };

const HEADING = /^\s*(#{1,6})\s+(.*)$/;
const BULLET = /^\s*[-*•]\s+(.*)$/;
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/;
const TABLE_ROW = /^\s*\|.*\|\s*$/;
const TABLE_SEPARATOR = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/;
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|<br\s*\/?>)/gi;

function splitRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function parseBlocks(text: string): Block[] {
  const lines = text.split(/\r?\n/);
  const blocks: Block[] = [];
  let paragraph: string[] = [];
  const flush = () => {
    if (paragraph.length) blocks.push({ kind: "paragraph", lines: paragraph });
    paragraph = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const heading = HEADING.exec(line);
    if (!line.trim()) {
      flush();
    } else if (heading) {
      flush();
      blocks.push({ kind: "heading", level: heading[1].length, text: heading[2] });
    } else if (TABLE_ROW.test(line) && TABLE_SEPARATOR.test(lines[i + 1] ?? "")) {
      flush();
      const header = splitRow(line);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && TABLE_ROW.test(lines[i])) rows.push(splitRow(lines[i++]));
      i--;
      blocks.push({ kind: "table", header, rows });
    } else if (BULLET.test(line) || NUMBERED.test(line)) {
      flush();
      const ordered = !BULLET.test(line);
      const pattern = ordered ? NUMBERED : BULLET;
      const items: string[] = [];
      while (i < lines.length && pattern.test(lines[i])) items.push(pattern.exec(lines[i++])![1]);
      i--;
      blocks.push({ kind: "list", ordered, items });
    } else {
      paragraph.push(line.trim());
    }
  }
  flush();
  return blocks;
}

function renderInline(text: string, renderText: TextRenderer): ReactNode[] {
  return text.split(INLINE).map((part, index) => {
    if (/^<br\s*\/?>$/i.test(part)) return <br key={index} />;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={index}>{renderText(part.slice(2, -2))}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }
    return <Fragment key={index}>{renderText(part)}</Fragment>;
  });
}

/**
 * A small, safe Markdown renderer for model answers: headings, paragraphs, bold,
 * inline code, bullet/numbered lists and tables. It only creates React elements
 * (never raw HTML), so transcript or model text cannot inject markup.
 */
export default function Markdown({ text, renderText }: MarkdownProps) {
  const inline = (value: string) => renderInline(value, renderText);
  return (
    <div className="markdown">
      {parseBlocks(text).map((block, index) => {
        switch (block.kind) {
          case "heading":
            return block.level <= 2 ? (
              <h4 key={index}>{inline(block.text)}</h4>
            ) : (
              <h5 key={index}>{inline(block.text)}</h5>
            );
          case "list": {
            const items = block.items.map((item, itemIndex) => (
              <li key={itemIndex}>{inline(item)}</li>
            ));
            return block.ordered ? <ol key={index}>{items}</ol> : <ul key={index}>{items}</ul>;
          }
          case "table":
            return (
              <div className="markdown-table" key={index}>
                <table>
                  <thead>
                    <tr>
                      {block.header.map((cell, cellIndex) => (
                        <th key={cellIndex}>{inline(cell)}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex}>{inline(cell)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          default:
            return (
              <p key={index}>
                {block.lines.map((line, lineIndex) => (
                  <Fragment key={lineIndex}>
                    {lineIndex > 0 && <br />}
                    {inline(line)}
                  </Fragment>
                ))}
              </p>
            );
        }
      })}
    </div>
  );
}
