import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import { ErrorBox } from "../components/ui.jsx";

export default function HomePage() {
  const [q, setQ] = useState("");
  const navigate = useNavigate();
  const { data: stats } = useAsync(() => api.stats(), []);

  return (
    <div>
      <h1 className="text-2xl font-semibold text-slate-800">Find similar legal cases</h1>
      <p className="mt-2 max-w-2xl text-gray-600">
        Combines <strong>text similarity</strong> and <strong>citation/statute-network similarity</strong>{" "}
        (Hier-SPCNet) into a hybrid score, with an explainable breakdown for every match.
      </p>

      <form
        className="mt-6 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (q.trim()) navigate(`/search?q=${encodeURIComponent(q.trim())}`);
        }}
      >
        <input
          autoFocus
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder='e.g. "medical negligence", "land acquisition compensation"'
          className="flex-1 rounded border border-gray-300 px-4 py-2.5 outline-none focus:border-blue-500"
        />
        <button className="rounded bg-blue-600 px-5 py-2.5 font-medium text-white hover:bg-blue-700">
          Search
        </button>
      </form>

      <div className="mt-4 flex gap-3 text-sm">
        <a href="/search?q=criminal+appeal+murder" className="text-blue-600 hover:underline">
          browse examples
        </a>
        <a href="/upload" className="text-blue-600 hover:underline">
          upload a judgment (PDF/txt)
        </a>
      </div>

      {stats && (
        <div className="mt-8 grid grid-cols-3 gap-4">
          <Stat label="Indexed cases" value={stats.cases} />
          <Stat label="Year range" value={`${stats.year_min}–${stats.year_max}`} />
          <Stat
            label="Model"
            value={`${stats.model?.text_method} + ${stats.model?.network_method} → ${stats.model?.hybrid_method}`}
          />
        </div>
      )}
      <ErrorBox error={null} />
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="text-xs uppercase tracking-wide text-gray-400">{label}</div>
      <div className="mt-1 text-lg font-semibold text-slate-800">{value}</div>
    </div>
  );
}
