import { useState } from "react";
import { Link, useLocation, useParams, useSearchParams } from "react-router-dom";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import { ErrorBox, Loading } from "../components/ui.jsx";
import RecommendationList from "../components/RecommendationList.jsx";
import ModelAccuracyBanner from "../components/ModelAccuracyBanner.jsx";
import PrecedentSummary from "../components/PrecedentSummary.jsx";
import CaseDetailsForm from "../components/CaseDetailsForm.jsx";

export default function RecommendPage({ kind }) {
  const { docId, uploadId } = useParams();
  const [searchParams] = useSearchParams();
  const [k, setK] = useState(10);
  const [yearMin, setYearMin] = useState("");
  // "" = use the category chosen at upload time (backend default); "any" = explicitly clear it
  const [category, setCategory] = useState(searchParams.get("category") || "");
  const [categoryQuery, setCategoryQuery] = useState("");
  const { data: categories } = useAsync(() => api.categories(), []);
  // manually entered case details (place, weapon, time ...) - applied on demand
  const location = useLocation();
  const [details, setDetails] = useState(location.state?.details || {});
  const { data: paramData } = useAsync(
    () => (kind === "upload" ? api.uploadParameters(uploadId, category === "any" ? undefined : category || undefined) : Promise.resolve(null)),
    [kind, uploadId, category],
  );
  const { data: impact } = useAsync(() => (kind === "upload" ? api.parameterImpact().catch(() => null) : Promise.resolve(null)), [kind]);
  const detailLabels = Object.fromEntries((paramData?.parameters || []).map((p) => [p.id, p.label]));

  const { data, loading, error } = useAsync(() => {
    const opts = { k, yearMin: yearMin || undefined };
    if (kind === "upload") {
      opts.category = category === "any" ? "" : category || undefined;
      opts.categoryQuery = category === "other" ? categoryQuery || undefined : undefined;
      opts.details = details;
    }
    return kind === "upload" ? api.recommendUpload(uploadId, opts) : api.recommendCase(docId, opts);
  }, [kind, docId, uploadId, k, yearMin, category, categoryQuery, JSON.stringify(details)]);

  const compareHref =
    kind === "upload"
      ? (target) => `/compare/upload/${uploadId}?target=${target}`
      : (target) => `/compare?a=${docId}&b=${target}`;

  return (
    <div>
      <ModelAccuracyBanner />
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
              Searched {data.n_candidates.toLocaleString()} case
              {data.n_candidates === 1 ? "" : "s"}
              {kind === "upload" && category && category !== "any"
                ? ` in "${categories?.find((c) => c.id === category)?.label || category}"`
                : ""}
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
          Show top&nbsp;
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
          placeholder="from year (optional)"
          value={yearMin}
          onChange={(e) => setYearMin(e.target.value)}
          className="w-36 rounded border border-gray-300 px-2 py-0.5"
        />
        {kind === "upload" && (
          <label className="flex items-center gap-1">
            Case type&nbsp;
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="rounded border border-gray-300 px-1 py-0.5"
            >
              <option value="">(as chosen when uploaded)</option>
              <option value="any">Search all case types</option>
              {categories?.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
          </label>
        )}
        {kind === "upload" && category === "other" && (
          <input
            type="text"
            placeholder="phrase to match"
            value={categoryQuery}
            onChange={(e) => setCategoryQuery(e.target.value)}
            className="w-48 rounded border border-gray-300 px-2 py-0.5"
          />
        )}
      </div>

      {kind === "upload" && (
        <CaseDetailsForm
          parameters={paramData}
          applied={details}
          issues={data?.detail_issues}
          comparison={data?.comparison}
          topAccuracy={data?.results?.[0]?.reliability}
          onApply={setDetails}
        />
      )}

      {data?.fallback_note && (
        <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">{data.fallback_note}</p>
      )}
      <ErrorBox error={error} />
      {loading ? (
        <Loading />
      ) : !data?.results?.length && kind === "upload" && category && category !== "any" ? (
        <p className="mt-4 text-sm text-gray-500">
          No cases in "{categories?.find((c) => c.id === category)?.label || category}" closely
          resemble this document's wording or citations. Try{" "}
          <button onClick={() => setCategory("any")} className="text-blue-600 hover:underline">
            searching all case types
          </button>{" "}
          instead.
        </p>
      ) : !data?.results?.length ? (
        <p className="mt-4 text-sm text-gray-500">
          No cases in this corpus share any detectable wording, statute, or citation overlap with
          this document.
        </p>
      ) : (
        <div className="mt-4">
          <PrecedentSummary summary={data?.precedent_summary} />
          <RecommendationList results={data?.results} compareHref={compareHref} detailLabels={detailLabels} />
        </div>
      )}
    </div>
  );
}
