import { Link } from "react-router-dom";
import { Badge, ScoreBar } from "./ui.jsx";

/** One recommendation card. `compareHref` builds the compare link for this pair. */
function RecoCard({ item, compareHref }) {
  return (
    <li className="rounded-lg border border-gray-200 bg-white p-4 transition-shadow hover:shadow-md">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-gray-400">#{item.rank}</span>
            <Link to={`/case/${item.doc_id}`} className="font-medium text-blue-700 hover:underline">
              {item.title || item.doc_id}
            </Link>
            {item.year && <Badge>{item.year}</Badge>}
          </div>
          <p className="mt-1 text-sm text-gray-600">{item.rationale}</p>
        </div>
        <div className="w-40 shrink-0">
          <ScoreBar label="hybrid" value={item.hybrid_score} tone="violet" />
          <ScoreBar label="text" value={item.text_score} tone="blue" />
          <ScoreBar label="network" value={item.network_score} tone="slate" />
        </div>
      </div>

      {(item.shared_sections?.length > 0 || item.shared_acts?.length > 0) && (
        <div className="mt-2 flex flex-wrap gap-1">
          {item.shared_sections?.slice(0, 5).map((s) => (
            <Badge key={s.section_key} tone="green">s.{s.number}</Badge>
          ))}
          {item.shared_acts?.slice(0, 3).map((a) => (
            <Badge key={a.act_id} tone="blue">{a.act_id}</Badge>
          ))}
          {item.cocitation_count > 0 && <Badge tone="amber">co-cited ×{item.cocitation_count}</Badge>}
        </div>
      )}

      <div className="mt-3 flex gap-3 text-sm">
        <Link to={compareHref(item.doc_id)} className="text-blue-600 hover:underline">
          Why similar →
        </Link>
        <Link to={`/recommend/case/${item.doc_id}`} className="text-gray-500 hover:underline">
          Similar to this
        </Link>
      </div>
    </li>
  );
}

export default function RecommendationList({ results, compareHref }) {
  if (!results?.length) return <p className="text-sm text-gray-500">No results.</p>;
  return (
    <ul className="space-y-3">
      {results.map((item) => (
        <RecoCard key={item.doc_id} item={item} compareHref={compareHref} />
      ))}
    </ul>
  );
}
