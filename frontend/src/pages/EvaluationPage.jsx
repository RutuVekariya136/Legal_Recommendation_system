import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import { ErrorBox, Loading } from "../components/ui.jsx";

function Cell({ v, signed = true }) {
  if (v == null || Number.isNaN(v)) return <td className="px-2 text-right text-gray-300">—</td>;
  const strong = v >= 0.6;
  return (
    <td className={`px-2 text-right tabular-nums ${strong ? "font-semibold text-slate-800" : "text-gray-600"}`}>
      {signed && v >= 0 ? "+" : ""}
      {v.toFixed(2)}
    </td>
  );
}

export default function EvaluationPage() {
  const { data, loading, error } = useAsync(() => api.evalSummary(), []);

  return (
    <div>
      <h1 className="text-lg font-semibold text-slate-800">Evaluation</h1>
      <p className="text-sm text-gray-500">
        Correlation of each method's pairwise similarity with the expert scores
        (100 validation + 90 test pairs). See <code>EVALUATION.md</code> for the full analysis.
      </p>
      <ErrorBox error={error} />
      {loading && <Loading />}

      {data && (
        <>
          <table className="mt-4 w-full text-sm">
            <thead>
              <tr className="text-xs uppercase text-gray-400">
                <th className="py-1 text-left">Config</th>
                <th className="px-2 text-right">val ρ</th>
                <th className="px-2 text-right">val r</th>
                <th className="px-2 text-right">val MSE</th>
                <th className="px-2 text-right">test ρ</th>
                <th className="px-2 text-right">test MSE</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <tr key={row.config} className="border-t border-gray-100">
                  <td className="py-1 pr-2">
                    {row.config} <span className="text-xs text-gray-400">({row.kind})</span>
                  </td>
                  <Cell v={row.validation.spearman} />
                  <Cell v={row.validation.pearson} />
                  <Cell v={row.validation.mse} />
                  <Cell v={row.test.spearman} />
                  <Cell v={row.test.mse} />
                </tr>
              ))}
            </tbody>
          </table>

          {data.rows.some((r) => r.validation.classification) && (
            <>
              <h2 className="mt-6 text-sm font-semibold uppercase tracking-wide text-gray-500">
                Classification metrics
              </h2>
              <p className="text-xs text-gray-500">
                Binarizing the continuous similarity score into "similar / not similar" at a
                threshold of 0.5 on both the expert score and the model's score (a fixed,
                a-priori choice — not tuned on this data). ROC-AUC needs no threshold and is the
                more honest single number for a continuous score; the rest are shown because
                they're the standard, requested metrics for a yes/no framing of this task.
              </p>
              <table className="mt-2 w-full text-sm">
                <thead>
                  <tr className="text-xs uppercase text-gray-400">
                    <th className="py-1 text-left">Config</th>
                    <th className="px-2 text-right">val acc</th>
                    <th className="px-2 text-right">val prec</th>
                    <th className="px-2 text-right">val recall</th>
                    <th className="px-2 text-right">val F1</th>
                    <th className="px-2 text-right">val AUC</th>
                    <th className="px-2 text-right">test acc</th>
                    <th className="px-2 text-right">test F1</th>
                    <th className="px-2 text-right">test AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((row) => {
                    const v = row.validation.classification, t = row.test.classification;
                    if (!v) return null;
                    return (
                      <tr key={row.config} className="border-t border-gray-100">
                        <td className="py-1 pr-2">
                          {row.config} <span className="text-xs text-gray-400">({row.kind})</span>
                        </td>
                        <Cell v={v.accuracy} signed={false} />
                        <Cell v={v.precision} signed={false} />
                        <Cell v={v.recall} signed={false} />
                        <Cell v={v.f1} signed={false} />
                        <Cell v={v.roc_auc} signed={false} />
                        <Cell v={t?.accuracy} signed={false} />
                        <Cell v={t?.f1} signed={false} />
                        <Cell v={t?.roc_auc} signed={false} />
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </>
          )}

          {data.diagnosis?.test && (
            <div className="mt-5 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
              <strong>Why is test-split correlation ≈ 0 for every method (incl. baselines)?</strong>
              <ul className="mt-1 list-disc pl-5">
                <li>
                  Test scores span only [{data.diagnosis.test.score_min?.toFixed(2)},{" "}
                  {data.diagnosis.test.score_max?.toFixed(2)}] with{" "}
                  {data.diagnosis.test.distinct_scores} distinct values (range restriction).
                </li>
                <li>
                  {data.diagnosis.test.pairs_with_isolated_case}/{data.diagnosis.test.n_pairs} pairs
                  involve a case with no citation-network node.
                </li>
                {data.diagnosis.test_subset_no_isolated_case && (
                  <li>
                    Even on the {data.diagnosis.test_subset_no_isolated_case.n_pairs} pairs with no
                    isolated case, hybrid ρ ={" "}
                    {data.diagnosis.test_subset_no_isolated_case["hybrid_weighted_0.6_0.4"]?.spearman?.toFixed(2)}.
                  </li>
                )}
              </ul>
            </div>
          )}
          <p className="mt-3 text-xs text-gray-400">Protocol: {data.protocol}</p>
        </>
      )}
    </div>
  );
}
