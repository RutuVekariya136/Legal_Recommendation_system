import { useState } from "react";
import { Link } from "react-router-dom";
import { Badge, ScoreBar } from "./ui.jsx";
import { readableAct, bestSimilarityLevel, reliabilityCaption } from "../utils/similarity.js";

/** One recommendation card. `compareHref` builds the compare link for this pair. */
function RecoCard({ item, compareHref, detailLabels }) {
  const [showDetails, setShowDetails] = useState(false);
  const level = bestSimilarityLevel(item.hybrid_score, item.reliability);
  const reliabilityCap = reliabilityCaption(item.reliability, item.reliability_n);
  const uniqueActs = [...new Map(item.shared_acts?.map((a) => [a.act_id, a])).values()];
  const uniqueSections = [
    ...new Map(item.shared_sections?.map((s) => [s.section_key, s])).values(),
  ];

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
            {item.exact_match && <Badge tone="purple">✓ Exact match</Badge>}
          </div>
          <p className="mt-1 text-sm text-gray-600">{item.rationale}</p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
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

      {(uniqueSections.length > 0 || uniqueActs.length > 0) && (
        <div className="mt-2 flex flex-wrap gap-1">
          {uniqueActs.slice(0, 2).map((a) => (
            <Badge key={a.act_id} tone="blue">
              {readableAct(a.act_id)}
              {uniqueSections.find((s) => s.section_key.startsWith(a.act_id))
                ? ` §${uniqueSections.find((s) => s.section_key.startsWith(a.act_id)).number}`
                : ""}
            </Badge>
          ))}
          {item.cocitation_count > 0 && <Badge tone="amber">co-cited ×{item.cocitation_count}</Badge>}
        </div>
      )}

      {detailLabels && Object.keys(item.detail_matches || {}).length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-1 text-xs">
          <span className="text-gray-400">Your details this case also mentions:</span>
          {Object.entries(item.detail_matches).map(([id, ok]) => (
            <span
              key={id}
              className={`rounded-full px-2 py-0.5 ${ok ? "bg-green-50 text-green-700" : "bg-gray-100 text-gray-400"}`}
            >
              {ok ? "✓" : "✗"} {detailLabels[id] || id}
            </span>
          ))}
        </div>
      )}

      <div className="mt-3 flex items-center gap-3 text-sm">
        <Link to={compareHref(item.doc_id)} className="text-blue-600 hover:underline">
          See why it's similar →
        </Link>
        <Link to={`/recommend/case/${item.doc_id}`} className="text-gray-500 hover:underline">
          Find cases like this one
        </Link>
        <button
          onClick={() => setShowDetails((v) => !v)}
          className="ml-auto text-xs text-gray-400 hover:text-gray-600 hover:underline"
        >
          {showDetails ? "Hide score details" : "Show score details"}
        </button>
      </div>

      {showDetails && (
        <div className="mt-2 w-full max-w-xs border-t border-gray-100 pt-2">
          <ScoreBar label="Final score (hybrid)" value={item.hybrid_score} tone="violet" />
          <ScoreBar label="Wording similarity" value={item.text_score} tone="blue" />
          <ScoreBar label="Shared law & citations" value={item.network_score} tone="slate" />
        </div>
      )}
    </li>
  );
}

export default function RecommendationList({ results, compareHref, detailLabels }) {
  if (!results?.length) return <p className="text-sm text-gray-500">No results.</p>;
  return (
    <ul className="space-y-3">
      {results.map((item) => (
        <RecoCard key={item.doc_id} item={item} compareHref={compareHref} detailLabels={detailLabels} />
      ))}
    </ul>
  );
}
