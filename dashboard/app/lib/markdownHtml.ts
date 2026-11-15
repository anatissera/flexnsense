// Minimal markdown to standalone HTML page, for the subset our reports use:
// headings, paragraphs, bullet and numbered lists, tables, blockquotes, rules,
// **bold**, *italic* and `code`. No dependencies.

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function inline(s: string): string {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
}

const cells = (row: string) =>
  row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());

export function markdownToHtml(md: string): string {
  const lines = md.split(/\r?\n/);
  const out: string[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    let m: RegExpMatchArray | null;
    if (!line.trim()) { i++; continue; }
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
      out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`); i++;
    } else if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
      out.push("<hr>"); i++;
    } else if (line.trim().startsWith("|") && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1] ?? "")) {
      const head = cells(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(cells(lines[i++]));
      out.push(
        `<table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>` +
          rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("") +
          "</tbody></table>",
      );
    } else if (/^\s*>/.test(line)) {
      const buf: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
      out.push(`<blockquote>${inline(buf.join(" "))}</blockquote>`);
    } else if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const tag = /^\s*\d+\./.test(line) ? "ol" : "ul";
      const items: string[] = [];
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i]))
        items.push(lines[i++].replace(/^\s*([-*]|\d+\.)\s+/, ""));
      out.push(`<${tag}>${items.map((t) => `<li>${inline(t)}</li>`).join("")}</${tag}>`);
    } else {
      const buf: string[] = [];
      while (
        i < lines.length && lines[i].trim() &&
        !/^(#{1,6}\s|\s*>|\s*([-*]|\d+\.)\s+|\s*\|)/.test(lines[i])
      ) buf.push(lines[i++]);
      if (!buf.length) { buf.push(lines[i++]); }
      out.push(`<p>${inline(buf.join(" "))}</p>`);
    }
  }
  return out.join("\n");
}

export function reportPage(title: string, md: string): string {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(title)}</title>
<style>
  body{font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;color:#1a1d21;background:#fff;max-width:820px;margin:2rem auto;padding:0 1.25rem}
  h1{font-size:1.6rem;border-bottom:2px solid #1a1d21;padding-bottom:.3rem}
  h2{font-size:1.2rem;margin-top:2rem;border-bottom:1px solid #ccc;padding-bottom:.2rem}
  h3{font-size:1rem;margin-top:1.4rem}
  table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}
  th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left}
  th{background:#f2f2f2}
  blockquote{margin:1.2rem 0;padding:.6rem 1rem;border-left:4px solid #999;background:#f7f7f7;color:#444}
  code{background:#f2f2f2;padding:0 .25rem;border-radius:3px}
  @media print{body{margin:0}}
</style></head><body>
${markdownToHtml(md)}
</body></html>`;
}
