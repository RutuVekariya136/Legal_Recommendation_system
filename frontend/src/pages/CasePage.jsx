import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import { Badge, ErrorBox, Loading } from "../components/ui.jsx";

export default function CasePage() {
  const { docId } = useParams();
  const { data: c, loading, error } = useAsync(() => api.getCase(docId), [docId]);
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
