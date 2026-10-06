import { readableAct } from "../utils/similarity.js";

/** "What was applied in similar cases" - a tally of shared statute sections
 * across the Top-K results, not a verdict/outcome prediction. Purely a
 * frequency summary over data the recommend response already computes per
 * result; makes the pattern visible without implying the model judged an
 * outcome for the uploaded document itself. */
export default function PrecedentSummary({ summary }) {
  if (!summary?.length) return null;

  return (
    <div className="mb-4 rounded-lg border border-gray-200 bg-gray-50 px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-wide text-gray-500">
        Statutes applied in similar cases
      </p>
      <p className="mt-0.5 text-xs text-gray-400">
        How often each provision appears among the similar cases below - not a predicted outcome
        for this document.
      </p>
      <ul className="mt-2 flex flex-wrap gap-2">
        {summary.map((s) => (
          <li
            key={`${s.act_id}-${s.section_number || "act"}`}
            className="rounded-full border border-gray-200 bg-white px-3 py-1 text-xs text-slate-700"
            title={s.section_title || undefined}
          >
            {s.act_name || readableAct(s.act_id)}
            {s.section_number ? ` §${s.section_number}` : ""}
            <span className="ml-1.5 font-medium text-blue-600">
              {s.count}/{s.of_k}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
