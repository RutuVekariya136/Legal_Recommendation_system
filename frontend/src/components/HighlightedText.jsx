import { useMemo } from "react";

/** Escape a string for use inside a RegExp. */
function esc(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Render `text` with three kinds of highlights:
 *  - statute section refs (green)      e.g. "s. 302", "Section 88", "Article 21"
 *  - citation party keywords (amber)   e.g. "Bolam", "Hunter"
 *  - shared wording/phrases (blue)     the terms actually driving the text-similarity score
 */
export default function HighlightedText({ text, sections = [], partyKeywords = [], sharedTerms = [] }) {
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
    if (sharedTerms.length) {
      const terms = [...new Set(sharedTerms)].map(esc).join("|");
      parts.push(`(?<term>\\b(?:${terms})\\b)`);
    }
    return parts.length ? new RegExp(parts.join("|"), "gi") : null;
  }, [sections, partyKeywords, sharedTerms]);

  if (!regex) return <span className="prose-legal">{text}</span>;

  const out = [];
  let last = 0;
  let m;
  regex.lastIndex = 0;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tone = m.groups?.sec != null ? "sec" : m.groups?.term != null ? "term" : "party";
    const cls = {
      sec: "rounded bg-green-200 px-0.5",
      party: "rounded bg-amber-200 px-0.5",
      term: "rounded bg-blue-200 px-0.5",
    }[tone];
    out.push(
      <mark key={m.index} className={cls}>
        {m[0]}
      </mark>,
    );
    last = m.index + m[0].length;
    if (m[0].length === 0) regex.lastIndex++;
  }
  out.push(text.slice(last));
  return <span className="prose-legal">{out}</span>;
}
