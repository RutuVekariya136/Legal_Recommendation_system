import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import { ErrorBox, Loading } from "../components/ui.jsx";
import RecommendationList from "../components/RecommendationList.jsx";

export default function RecommendPage({ kind }) {
  const { docId, uploadId } = useParams();
  const [k, setK] = useState(10);
  const [yearMin, setYearMin] = useState("");

  const { data, loading, error } = useAsync(() => {
    const opts = { k, yearMin: yearMin || undefined };
    return kind === "upload" ? api.recommendUpload(uploadId, opts) : api.recommendCase(docId, opts);
  }, [kind, docId, uploadId, k, yearMin]);

  const compareHref =
    kind === "upload"
      ? (target) => `/compare/upload/${uploadId}?target=${target}`
      : (target) => `/compare?a=${docId}&b=${target}`;

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-gray-400">
            Similar cases {kind === "upload" ? "for uploaded document" : "to"}
          </p>
          <h1 className="text-lg font-semibold text-slate-800">
            {kind === "upload" ? (data?.query || uploadId) : docId}
          </h1>
          {data && (
            <p className="text-xs text-gray-400">
              method: {data.method} · {data.n_candidates} candidates · {data.n_filtered_out} filtered
              {data.run_id ? ` · run ${data.run_id.slice(0, 8)}` : ""}
            </p>
          )}
        </div>
        {kind === "case" && (
          <Link to={`/case/${docId}`} className="text-sm text-blue-600 hover:underline">
            view query case
          </Link>
        )}
      </div>

      <div className="mt-3 flex items-center gap-3 text-sm">
        <label className="flex items-center gap-1">
          top&nbsp;
          <select
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
            className="rounded border border-gray-300 px-1 py-0.5"
          >
            {[5, 10, 20, 30].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <input
          type="number"
          placeholder="year ≥"
          value={yearMin}
          onChange={(e) => setYearMin(e.target.value)}
          className="w-24 rounded border border-gray-300 px-2 py-0.5"
        />
      </div>

      <ErrorBox error={error} />
      {loading ? <Loading /> : <div className="mt-4"><RecommendationList results={data?.results} compareHref={compareHref} /></div>}
    </div>
  );
}
