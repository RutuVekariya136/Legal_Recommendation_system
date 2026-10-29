import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import { Badge, ErrorBox, Loading } from "../components/ui.jsx";

function TreatmentBanner({ treatment }) {
  if (!treatment) return null;
  const { overall_status: status, signals = [], coverage_note } = treatment;
  if (status === "negative" || status === "distinguished") {
    return (
      <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
        <p className="font-medium">
          ⚠ {status === "negative"
            ? `A later case may have overruled or doubted this one — ${signals.length} citing case(s) found.`
            : `A later case distinguished this one on its facts — ${signals.length} citing case(s) found.`}
        </p>
        <ul className="mt-2 space-y-1">
          {signals.map((s) => (
            <li key={s.citing_doc_id} className="text-xs text-red-700">
              <Link to={`/case/${s.citing_doc_id}`} className="font-medium hover:underline">
                {s.citing_doc_id}
              </Link>
              {s.citing_year ? ` (${s.citing_year})` : ""}: "{s.context}"
            </li>
          ))}
        </ul>
        <p className="mt-2 text-xs text-red-600">{coverage_note}</p>
      </div>
    );
  }
  if (status === "positive") {
    return (
      <p className="mt-3 text-xs text-green-700">
        ✓ Affirmed/followed by {signals.length} later case(s). {coverage_note}
      </p>
    );
  }
  return null;
}

export default function CasePage() {
  const { docId } = useParams();
  const { data: c, loading, error } = useAsync(() => api.getCase(docId), [docId]);
  const { data: treatment } = useAsync(() => api.getCaseTreatment(docId), [docId]);
  const [full, setFull] = useState(null);

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;
  if (!c) return null;

  return (
    <div>
      <Link to="/search" className="text-sm text-blue-600 hover:underline">
        ← back to search
      </Link>
      <div className="mt-2 flex items-center gap-2">
        <h1 className="text-xl font-semibold text-slate-800">{c.title || c.doc_id}</h1>
        {c.year && <Badge>{c.year}</Badge>}
        {(c.splits || []).map((s) => (
          <Badge key={s} tone="blue">{s}</Badge>
        ))}
      </div>
      <p className="mt-1 text-xs text-gray-400">
        {c.doc_id} · {c.char_len.toLocaleString()} chars
      </p>
      <TreatmentBanner treatment={treatment} />

      <div className="mt-4 flex gap-3">
        <Link
          to={`/recommend/case/${c.doc_id}`}
          className="rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          Find similar cases
        </Link>
        <Link
          to={`/case/${c.doc_id}/graph`}
          className="rounded border border-gray-300 px-4 py-2 text-sm hover:bg-gray-50"
        >
          Citation graph
        </Link>
        <button
          onClick={() => api.getCaseText(c.doc_id).then((r) => setFull(r.text))}
          className="rounded border border-gray-300 px-4 py-2 text-sm hover:bg-gray-50"
        >
          {full ? "Full text loaded" : "Load full text"}
        </button>
      </div>

      <div className="prose-legal mt-5 max-h-[60vh] overflow-auto rounded-lg border border-gray-200 bg-white p-4 text-sm text-gray-700">
        {full ?? c.preview}
        {!full && "…"}
      </div>
    </div>
  );
}
