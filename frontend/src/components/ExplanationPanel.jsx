import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, ScoreBar, Section } from "./ui.jsx";

const COLORS = { text_score: "#3b82f6", network_score: "#64748b", intercept: "#a78bfa", text: "#3b82f6", network: "#64748b" };

function Contributions({ contributions }) {
  const data = contributions
    .filter((c) => c.contribution !== 0 || c.component === "intercept")
    .map((c) => ({ name: c.component, pct: c.pct_of_score, contribution: c.contribution }));
  return (
    <div className="h-48 w-full">
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ left: 20, right: 30 }}>
          <XAxis type="number" tickFormatter={(v) => `${v}%`} fontSize={11} />
          <YAxis type="category" dataKey="name" width={90} fontSize={11} />
          <Tooltip formatter={(v, k) => (k === "pct" ? `${v}% of score` : v.toFixed(3))} />
          <Bar dataKey="pct" radius={3}>
            {data.map((d) => (
              <Cell key={d.name} fill={COLORS[d.name] || "#94a3b8"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function ExplanationPanel({ ex }) {
  if (!ex) return null;
  return (
    <div>
      <Section title="Scores">
        <div className="max-w-sm">
          <ScoreBar label="hybrid similarity" value={ex.hybrid_score} tone="violet" />
          <ScoreBar label="text" value={ex.text_score} tone="blue" />
          <ScoreBar label="network" value={ex.network_score} tone="slate" />
          <div className="mt-1 text-xs text-gray-400">
            network = embedding {ex.embedding_score?.toFixed(2)} · statute overlap{" "}
            {ex.statute_overlap?.toFixed(2)} · citation overlap {ex.citation_overlap?.toFixed(2)}
          </div>
        </div>
        <p className="mt-3 text-sm text-gray-700">{ex.summary}</p>
      </Section>

      <Section title={`Contribution to the hybrid score (${ex.method})`}>
        <Contributions contributions={ex.contributions} />
      </Section>

      {ex.shared_statutes?.length > 0 && (
        <Section title={`Shared statutes (${ex.shared_statutes.length})`}>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase text-gray-400">
                <th className="py-1">Act</th>
                <th>Provision</th>
                <th className="text-right">ICF</th>
              </tr>
            </thead>
            <tbody>
              {ex.shared_statutes.map((s, i) => (
                <tr key={i} className="border-t border-gray-100">
                  <td className="py-1 pr-2">
                    {s.act_name || s.act_id}{" "}
                    {!s.in_curated_hierarchy && <Badge tone="amber">bare node</Badge>}
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
                  <td className="text-right tabular-nums text-gray-500">{s.icf?.toFixed(2)}</td>
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
