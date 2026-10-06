import { useState } from "react";
import ParameterFields from "./ParameterFields.jsx";

const pts = (x) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)} pts`;
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const pct1 = (x) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);

/** Results-page version: re-runs the search with typed details and reports the
 * recomputed accuracy of the top match. Same fields as the upload step. */
export default function CaseDetailsForm({ parameters, applied, issues, comparison, topAccuracy, onApply }) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState({});
  const params = parameters?.parameters;
  if (!params?.length) return null;
  const nApplied = Object.keys(applied || {}).length;
  const filled = Object.values(draft).some((v) => v && v.trim());
  const b = comparison?.before, a = comparison?.after;
  const delta = b?.top_accuracy != null && a?.top_accuracy != null ? a.top_accuracy - b.top_accuracy : null;

  return (
    <div className="mt-3 rounded-lg border border-gray-200 bg-white">
      <button onClick={() => setOpen((v) => !v)} className="flex w-full items-center justify-between px-4 py-2 text-left text-sm">
        <span className="font-medium text-slate-700">
          Additional case details{parameters.category_label ? ` — ${parameters.category_label}` : ""}
          {nApplied > 0 && <span className="ml-2 text-xs text-green-700">{nApplied} applied</span>}
        </span>
        <span className="text-xs text-gray-400">{open ? "Hide ▲" : "Optional ▼"}</span>
      </button>

      <div className="border-t border-gray-100 bg-gray-50 px-4 py-2 text-xs text-gray-700">
        {comparison ? (
          <>
            Top match accuracy: <strong>{pct1(b.top_accuracy)}</strong> from your document alone →{" "}
            <strong>{pct1(a.top_accuracy)}</strong> with your details
            {delta != null && (
              <span className={delta > 0.0005 ? "text-green-700" : delta < -0.0005 ? "text-amber-700" : "text-gray-500"}>
                {" "}({delta >= 0 ? "+" : ""}{(delta * 100).toFixed(2)} pts)
              </span>
            )}
            .{b.top_doc !== a.top_doc && " The best match changed."}{" "}
            <span className="text-gray-400">
              Recomputed by the model from the details you gave. The effect is often small; it can also be zero or negative.
            </span>
          </>
        ) : (
          <>
            Top match accuracy from your document alone: <strong>{pct1(topAccuracy)}</strong>. Add details above and
            the figure is recomputed with them, so you can compare.
          </>
        )}
      </div>

      {open && (
        <div className="border-t border-gray-100 px-4 py-3">
          <p className="text-xs text-gray-500">
            These additional case details are optional. Providing more relevant information may improve the
            relevance of the recommended cases. Anything already found in your document is kept; what you type here
            is added as extra context.
          </p>
          <div className="mt-3">
            <ParameterFields params={params} draft={draft} onDraft={setDraft} issues={issues} />
          </div>
          <div className="mt-3 flex items-center gap-3">
            <button
              onClick={() => onApply(Object.fromEntries(Object.entries(draft).filter(([, v]) => v && v.trim())))}
              disabled={!filled}
              className="rounded-lg bg-slate-800 px-3 py-1.5 text-sm text-white disabled:bg-gray-300"
            >
              Update results with these details
            </button>
            {nApplied > 0 && (
              <button onClick={() => { setDraft({}); onApply({}); }} className="text-sm text-gray-500 hover:underline">
                Clear details
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
