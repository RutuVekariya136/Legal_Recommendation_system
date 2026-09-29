import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, ScoreBar, Section } from "./ui.jsx";
import { plainExplanation, bestSimilarityLevel, reliabilityCaption, distinctiveness } from "../utils/similarity.js";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";

const COMPONENT_INFO = {
  text_score: {
    label: "Wording similarity",
    color: "#3b82f6",
    blurb: "how closely the two judgments are worded",
  },
  text: { label: "Wording similarity", color: "#3b82f6", blurb: "how closely the two judgments are worded" },
  network_score: {
    label: "Shared law & citations",
    color: "#64748b",
    blurb: "shared statutes, sections, and case citations between the two",
  },
  network: {
    label: "Shared law & citations",
    color: "#64748b",
    blurb: "shared statutes, sections, and case citations between the two",
  },
  intercept: {
    label: "Baseline",
    color: "#cbd5e1",
    blurb: "a fixed starting value the model always adds, the same for every pair of cases",
  },
};

function componentInfo(name) {
  return COMPONENT_INFO[name] || { label: name, color: "#94a3b8", blurb: "" };
}

function Contributions({ contributions, sharedTerms = [] }) {
  const rows = contributions.filter((c) => c.contribution !== 0 || c.component === "intercept");
  const data = rows.map((c) => ({
    name: componentInfo(c.component).label,
    key: c.component,
    pct: c.pct_of_score,
    contribution: c.contribution,
  }));
  return (
    <div>
      <p className="mb-2 text-sm text-gray-600">
        The model turns wording similarity and shared law/citations into one final score. This
        shows how much each factor pushed that score up (or down).
      </p>
      <div className="h-40 w-full">
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 20, right: 30 }}>
            <XAxis type="number" tickFormatter={(v) => `${v}%`} fontSize={11} />
            <YAxis type="category" dataKey="name" width={140} fontSize={11} />
            <Tooltip formatter={(v, k) => (k === "pct" ? `${v}% of score` : v.toFixed(3))} />
            <Bar dataKey="pct" radius={3}>
              {data.map((d) => (
                <Cell key={d.key} fill={componentInfo(d.key).color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <ul className="mt-3 space-y-1 text-xs text-gray-500">
        {rows.map((c) => {
          const info = componentInfo(c.component);
          return (
            <li key={c.component} className="flex items-start gap-2">
              <span
                className="mt-0.5 inline-block h-2.5 w-2.5 shrink-0 rounded-sm"
                style={{ backgroundColor: info.color }}
              />
              <span>
                <strong className="text-gray-700">{info.label}</strong> ({c.pct_of_score}% of the
                score) — {info.blurb}.
                {c.component.includes("text") && sharedTerms.length > 0 && (
                  <>
                    {" "}Evidence:{" "}
                    {sharedTerms.slice(0, 8).map((t, i) => (
                      <span key={t}>
                        <mark className="rounded bg-blue-100 px-1 text-gray-700">{t}</mark>
                        {i < Math.min(sharedTerms.length, 8) - 1 ? " " : ""}
                      </span>
                    ))}
                    {" "}— see these highlighted in the side-by-side text below.
                  </>
                )}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default function ExplanationPanel({ ex }) {
  const { data: stats } = useAsync(() => api.stats(), []);
  if (!ex) return null;
  const level = bestSimilarityLevel(ex.hybrid_score, ex.reliability);
  const reliabilityCap = reliabilityCaption(ex.reliability, ex.reliability_n);
  return (
    <div>
      <div className="mb-6 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">
            Why these cases are similar
          </h2>
          <div className="flex flex-col items-end gap-1">
            <Badge tone={level.tone}>{level.text}</Badge>
            {reliabilityCap && (
              <span
                className={`text-xs ${reliabilityCap.weak ? "font-medium text-amber-700" : "text-gray-400"}`}
                title={reliabilityCap.title}
              >
                {reliabilityCap.text}
              </span>
            )}
          </div>
        </div>
        <p className="mt-2 text-base text-slate-800">{plainExplanation(ex)}</p>
      </div>

      <Section title="Scores">
        <div className="max-w-sm">
          <ScoreBar label="Final score (hybrid)" value={ex.hybrid_score} tone="violet" />
          <ScoreBar label="Wording similarity" value={ex.text_score} tone="blue" />
          <ScoreBar label="Shared law & citations" value={ex.network_score} tone="slate" />
          <div className="mt-1 text-xs text-gray-400">
            Shared law &amp; citations = statute overlap {ex.statute_overlap?.toFixed(2)} · citation overlap{" "}
            {ex.citation_overlap?.toFixed(2)}{ex.embedding_score > 0 ? ` · graph embedding ${ex.embedding_score.toFixed(2)}` : ""}
          </div>
        </div>
      </Section>

      <Section title={`Contribution to the hybrid score (${ex.method})`}>
        <Contributions contributions={ex.contributions} sharedTerms={ex.shared_terms} />
      </Section>

      {ex.shared_statutes?.length > 0 && (
        <Section title={`Shared statutes (${ex.shared_statutes.length})`}>
          <p className="mb-2 text-xs text-gray-500">
            <strong>ICF (Inverse Citation Frequency)</strong> is a rarity score: the fewer cases in the corpus
            cite a provision, the higher its ICF. A rare shared provision is stronger evidence of similarity
            than one almost every case cites (like Article 226), so higher-ICF provisions count for more in the score.
          </p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase text-gray-400">
                <th className="py-1">Act</th>
                <th>Provision</th>
                <th className="text-right">ICF <span className="font-normal normal-case text-gray-400">(rarity score)</span></th>
              </tr>
            </thead>
            <tbody>
              {ex.shared_statutes.map((s, i) => (
                <tr key={i} className="border-t border-gray-100">
                  <td className="py-1 pr-2">
                    {s.act_name || s.act_id}{" "}
                    {!s.in_curated_hierarchy && (
                      <span title="This Act is not in the curated statute list, so its sections have no titles"><Badge tone="amber">no section titles</Badge></span>
                    )}
                  </td>
                  <td className="pr-2 text-gray-600">
                    {s.kind === "section" ? (
                      <>
                        s.{s.section_number}
                        {s.section_title ? ` — ${s.section_title}` : ""}
                      </>
                    ) : (
                      "(Act cited)"
                    )}
                  </td>
                  <td className="text-right tabular-nums text-gray-500">
                    {s.icf?.toFixed(2)}
                    {(() => {
                      const d = distinctiveness(s.icf, stats?.searchable_cases);
                      return d ? (
                        <div className="text-xs font-normal text-gray-400">
                          {d.label.toLowerCase()} · cited in ~{d.df.toLocaleString()} cases
                        </div>
                      ) : null;
                    })()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Section>
      )}

      {(ex.shared_cited_cases?.length > 0 || ex.shared_raw_citations?.length > 0) && (
        <Section title="Shared citations">
          {ex.shared_cited_cases?.length > 0 && (
            <ul className="mb-2 list-disc pl-5 text-sm">
              {ex.shared_cited_cases.map((c) => (
                <li key={c.doc_id}>
                  {c.title || c.doc_id} {c.year ? `(${c.year})` : ""}
                </li>
              ))}
            </ul>
          )}
          {ex.shared_raw_citations?.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {ex.shared_raw_citations.map((r) => (
                <Badge key={r}>{r}</Badge>
              ))}
            </div>
          )}
          {ex.cocitation_count > 0 && (
            <p className="mt-2 text-sm text-gray-500">Co-cited by {ex.cocitation_count} other case(s).</p>
          )}
        </Section>
      )}
    </div>
  );
}
