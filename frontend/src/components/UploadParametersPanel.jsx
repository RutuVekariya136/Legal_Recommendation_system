import { useState } from "react";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import ParameterFields from "./ParameterFields.jsx";

/** "Additional case details" step on the upload result: shows the category's
 * parameters, which are already found in the document and which are missing,
 * and passes only the details the user actually typed to the search. Optional:
 * the search button is always enabled. */
export default function UploadParametersPanel({ uploadId, category, onContinue }) {
  const [draft, setDraft] = useState({});
  const { data, error } = useAsync(
    () => api.uploadParameters(uploadId, category || undefined),
    [uploadId, category],
  );
  if (error || !data?.parameters?.length) return null;

  const params = data.parameters;
  const typed = (id) => Boolean(draft[id] && String(draft[id]).trim());
  const have = (p) => p.present || typed(p.id);
  const found = params.filter(have);
  const missing = params.filter((p) => !have(p));
  const share = params.length ? found.length / params.length : 0;
  const details = Object.fromEntries(Object.entries(draft).filter(([, v]) => v && String(v).trim()));

  return (
    <div className="mt-4 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-sm font-semibold text-slate-800">
          Additional case details{data.category_label ? ` — ${data.category_label}` : ""}
        </p>
        <p className="text-xs text-gray-500">{found.length} of {params.length} details provided</p>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded bg-gray-100">
        <div className="h-full bg-slate-700 transition-all" style={{ width: `${Math.round(share * 100)}%` }} />
      </div>

      <p className="mt-3 text-xs text-gray-500">
        These additional case details are optional. Providing more relevant information may improve the relevance
        of the recommended cases. Details already found in your document are kept; anything you type is added as
        extra context.
      </p>

      <div className="mt-3">
        <ParameterFields params={params} draft={draft} onDraft={setDraft} />
      </div>

      <ul className="mt-4 grid gap-1 text-xs sm:grid-cols-2">
        {params.map((p) => (
          <li key={p.id} className={have(p) ? "text-green-700" : "text-gray-400"}>
            {have(p) ? "✓" : "○"} {p.label}
          </li>
        ))}
      </ul>

      {missing.length > 0 && (
        <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {missing.length} detail{missing.length === 1 ? " is" : "s are"} missing:{" "}
          {missing.map((p) => p.label).join(", ")}. You can still continue without them.
        </p>
      )}

      <button
        type="button"
        onClick={() => onContinue(details)}
        className="mt-4 rounded-lg bg-slate-800 px-5 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-slate-700"
      >
        Find similar cases with details →
      </button>
    </div>
  );
}
