import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import { Badge, ErrorBox, Loading } from "../components/ui.jsx";

const MODES = ["semantic", "title", "hybrid"];

export default function SearchPage() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") || "";
  const mode = params.get("mode") || "semantic";
  const yearMin = params.get("year_min") || "";
  const [input, setInput] = useState(q);

  const { data, loading, error } = useAsync(
    () => (q ? api.search(q, { mode, yearMin: yearMin || undefined }) : Promise.resolve(null)),
    [q, mode, yearMin],
  );

  function update(next) {
    const merged = { q, mode, ...(yearMin ? { year_min: yearMin } : {}), ...next };
    Object.keys(merged).forEach((k) => merged[k] === "" && delete merged[k]);
    setParams(merged);
  }

  return (
    <div>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          update({ q: input.trim() });
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Search the corpus…"
          className="flex-1 rounded border border-gray-300 px-4 py-2 outline-none focus:border-blue-500"
        />
        <button className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-700">Search</button>
      </form>

      <div className="mt-3 flex items-center gap-2 text-sm">
        {MODES.map((m) => (
          <button
            key={m}
            onClick={() => update({ mode: m })}
            className={`rounded px-2 py-1 ${m === mode ? "bg-slate-800 text-white" : "bg-gray-100 text-gray-600"}`}
          >
            {m}
          </button>
        ))}
        <input
          type="number"
          placeholder="year ≥"
          defaultValue={yearMin}
          onBlur={(e) => update({ year_min: e.target.value })}
          className="w-24 rounded border border-gray-300 px-2 py-1"
        />
      </div>

      <ErrorBox error={error} />
      {loading && <Loading />}

      {data && (
        <div className="mt-4 grid grid-cols-[1fr_180px] gap-6">
          <div>
            <p className="mb-2 text-xs text-gray-400">{data.total} result(s)</p>
            <ul className="space-y-2">
              {data.items.map((h) => (
                <li key={h.doc_id} className="rounded-lg border border-gray-200 bg-white p-3">
                  <div className="flex items-center gap-2">
                    <Link to={`/case/${h.doc_id}`} className="font-medium text-blue-700 hover:underline">
                      {h.title || h.doc_id}
                    </Link>
                    {h.year && <Badge>{h.year}</Badge>}
                    {h.title_match && <Badge tone="green">title</Badge>}
                    <span className="ml-auto text-xs tabular-nums text-gray-400">
                      {h.score.toFixed(3)}
                    </span>
                  </div>
                  <p className="mt-1 line-clamp-2 text-sm text-gray-600">{h.snippet}</p>
                  <Link
                    to={`/recommend/case/${h.doc_id}`}
                    className="mt-1 inline-block text-xs text-blue-600 hover:underline"
                  >
                    find similar cases →
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {data.facets && (
            <aside className="text-sm">
              <h3 className="mb-1 text-xs font-semibold uppercase text-gray-400">By decade</h3>
              {Object.entries(data.facets.years).map(([decade, n]) => (
                <div key={decade} className="flex items-center gap-2">
                  <span className="w-12 text-gray-500">{decade}</span>
                  <div className="h-2 flex-1 rounded bg-gray-100">
                    <div
                      className="h-full rounded bg-blue-400"
                      style={{ width: `${(n / data.facets.total) * 100 * 3}%` }}
                    />
                  </div>
                  <span className="w-6 text-right text-gray-400">{n}</span>
                </div>
              ))}
            </aside>
          )}
        </div>
      )}
    </div>
  );
}
