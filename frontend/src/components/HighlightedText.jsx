import { useMemo } from "react";

/** Escape a string for use inside a RegExp. */
function esc(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Render `text` with two kinds of highlights:
 *  - statute section refs (green)    e.g. "s. 302", "Section 88", "Article 21"
 *  - citation party keywords (amber) e.g. "Bolam", "Hunter"
 */
export default function HighlightedText({ text, sections = [], partyKeywords = [] }) {
  const regex = useMemo(() => {
    const parts = [];
    if (sections.length) {
      const nums = [...new Set(sections)].map((n) => esc(String(n))).join("|");
      parts.push(`(?<sec>\\b(?:s\\.?|ss\\.?|sec\\.?|sections?|arts?\\.?|articles?)\\s*(?:${nums})\\b)`);
    }
    if (partyKeywords.length) {
      const kw = [...new Set(partyKeywords)].map(esc).join("|");
      parts.push(`(?<party>\\b(?:${kw})\\b)`);
    }
    return parts.length ? new RegExp(parts.join("|"), "gi") : null;
  }, [sections, partyKeywords]);

  if (!regex) return <span className="prose-legal">{text}</span>;

  const out = [];
  let last = 0;
  let m;
  regex.lastIndex = 0;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const isSec = m.groups?.sec != null;
    out.push(
      <mark
        key={m.index}
        className={isSec ? "rounded bg-green-200 px-0.5" : "rounded bg-amber-200 px-0.5"}
      >
        {m[0]}
      </mark>,
    );
    last = m.index + m[0].length;
    if (m[0].length === 0) regex.lastIndex++;
  }
  out.push(text.slice(last));
  return <span className="prose-legal">{out}</span>;
}
