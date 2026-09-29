import { Link } from "react-router-dom";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";

// The config the deployed RecommenderService actually serves recommendations
// with - keep in sync with GOLD_THRESHOLD's config name in scripts/evaluate.py.
const DEPLOYED_CONFIG = "hybrid: learned linear (text+net)";

/** Compact "how accurate is this model overall" strip, so a judge sees the
 * headline number without having to find the Evaluation page. Links there
 * for the full picture (including the honest test-split caveat). */
export default function ModelAccuracyBanner() {
  const { data } = useAsync(() => api.evalSummary(), []);
  const row = data?.rows?.find((r) => r.config === DEPLOYED_CONFIG);
  if (!row?.validation?.classification) return null;
  const c = row.validation.classification;

  return (
    <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-gray-200 bg-gray-50 px-4 py-2 text-xs text-gray-600">
      <span className="font-medium text-gray-700">Overall model accuracy</span>
      <span title="On the 100 expert-scored validation pairs, at a 0.5 similarity threshold.">
        Accuracy <strong className="text-slate-700">{Math.round(c.accuracy * 100)}%</strong>
      </span>
      <span title="Correlation with expert similarity judgments (Spearman rho) - the primary metric for this task.">
        Correlation (ρ) <strong className="text-slate-700">{row.validation.spearman.toFixed(2)}</strong>
      </span>
      <span title="Area under the ROC curve - doesn't depend on a threshold, the fairer number to compare methods on.">
        ROC-AUC <strong className="text-slate-700">{c.roc_auc.toFixed(2)}</strong>
      </span>
      <Link to="/evaluation" className="ml-auto text-blue-600 hover:underline">
        Full evaluation, incl. limitations →
      </Link>
    </div>
  );
}
