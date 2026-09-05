import { useState } from "react";
import { api } from "../services/api.js";
import { Loading } from "./ui.jsx";
import HighlightedText from "./HighlightedText.jsx";

/** Party keywords worth highlighting from a normalised "a v b" citation string. */
function partyKeywords(rawCitations) {
  const stop = new Set(["state", "union", "india", "another", "others", "and", "ltd", "the", "of"]);
  const out = new Set();
  for (const c of rawCitations || []) {
    for (const tok of c.split(/\s+/)) {
      if (tok.length >= 5 && !stop.has(tok)) out.add(tok);
    }
  }
  return [...out].slice(0, 40);
}

export default function SideBySide({ ex, kind, uploadId }) {
  const [texts, setTexts] = useState(null);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    try {
      const left =
        kind === "upload"
          ? await api.getUpload(uploadId).then((u) => ({ title: u.filename, text: u.preview + "…" }))
          : await api.getCaseText(ex.query_id).then((r) => ({ title: ex.query_id, text: r.text }));
      const right = await api.getCaseText(ex.target_id).then((r) => ({ title: ex.target_id, text: r.text }));
      setTexts({ left, right });
    } finally {
      setLoading(false);
    }
  }

  const sections = (ex.shared_statutes || []).filter((s) => s.section_number).map((s) => s.section_number);
  const kws = partyKeywords(ex.shared_raw_citations);

  if (!texts) {
    return (
      <button
        onClick={load}
        disabled={loading}
        className="rounded border border-gray-300 px-4 py-2 text-sm hover:bg-gray-50 disabled:opacity-50"
      >
        {loading ? "Loading…" : "Read the two judgments side by side"}
      </button>
    );
  }

  return (
    <div>
      <p className="mb-2 text-xs text-gray-500">
        <mark className="bg-green-200 px-0.5">shared statute refs</mark>{" "}
        <mark className="bg-amber-200 px-0.5">shared cited-party names</mark>
      </p>
      {loading && <Loading />}
      <div className="grid grid-cols-2 gap-4">
        {[texts.left, texts.right].map((c, i) => (
          <div key={i} className="max-h-[70vh] overflow-auto rounded-lg border border-gray-200 bg-white p-3 text-sm text-gray-700">
            <div className="mb-2 font-medium text-slate-700">{c.title}</div>
            <HighlightedText text={c.text} sections={sections} partyKeywords={kws} />
          </div>
        ))}
      </div>
    </div>
  );
}
