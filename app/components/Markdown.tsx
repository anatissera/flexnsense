import { Fragment, type ReactNode } from "react";

// Minimal, dependency-free markdown: headings, bullet/numbered lists, code blocks,
// **bold**, *italic*, `code`, [links](url). Everything is rendered as React nodes (no HTML injection).
function inline(text: string, key = ""): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(__[^_]+__)|(\*[^*\s][^*]*\*)|(\[[^\]]+\]\(https?:\/\/[^)\s]+\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    const k = `${key}${i++}`;
    if (m[1]) out.push(<code key={k} className="rounded bg-(--surface-hover) px-1 font-mono t-small">{t.slice(1, -1)}</code>);
    else if (m[2] || m[3]) out.push(<strong key={k}>{inline(t.slice(2, -2), k)}</strong>);
    else if (m[4]) out.push(<em key={k}>{inline(t.slice(1, -1), k)}</em>);
    else {
      const [, label, href] = /\[([^\]]+)\]\(([^)]+)\)/.exec(t)!;
      out.push(<a key={k} href={href} target="_blank" rel="noreferrer" className="text-(--primary) underline">{label}</a>);
    }
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export default function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r/g, "").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (l.startsWith("```")) {
      const buf: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) buf.push(lines[i++]);
      i++;
      blocks.push(<pre key={blocks.length} className="overflow-x-auto rounded-md bg-black/50 p-2 font-mono t-small">{buf.join("\n")}</pre>);
    } else if (/^\s*([-*+]|\d+[.)])\s+/.test(l)) {
      const ordered = /^\s*\d/.test(l);
      const items: string[] = [];
      while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*([-*+]|\d+[.)])\s+/, ""));
      const Tag = ordered ? "ol" : "ul";
      blocks.push(
        <Tag key={blocks.length} className={`${ordered ? "list-decimal" : "list-disc"} space-y-0.5 pl-4`}>
          {items.map((it, k) => <li key={k}>{inline(it, `l${k}`)}</li>)}
        </Tag>,
      );
    } else if (/^#{1,6}\s/.test(l)) {
      blocks.push(<div key={blocks.length} className="t-section">{inline(l.replace(/^#+\s/, ""))}</div>);
      i++;
    } else if (!l.trim()) {
      i++;
    } else {
      const buf: string[] = [];
      while (i < lines.length && lines[i].trim() && !lines[i].startsWith("```") && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i]) && !/^#{1,6}\s/.test(lines[i])) buf.push(lines[i++]);
      blocks.push(
        <p key={blocks.length}>
          {buf.map((b, k) => <Fragment key={k}>{k > 0 && <br />}{inline(b, `p${k}`)}</Fragment>)}
        </p>,
      );
    }
  }
  return <div className="space-y-1.5">{blocks}</div>;
}
