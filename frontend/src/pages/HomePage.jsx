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
      <section className="rounded-2xl border border-gray-200 bg-white px-6 py-10 shadow-sm sm:px-10">
        <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
          Find similar legal cases
        </h1>
        <p className="mt-3 max-w-2xl text-gray-600">
          Combines <strong className="text-slate-800">text similarity</strong> and{" "}
          <strong className="text-slate-800">citation/statute-network similarity</strong>{" "}
          (Hier-SPCNet) into a hybrid score, with an explainable breakdown for every match.
        </p>

        <form
          className="mt-6 flex max-w-2xl gap-2"
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
            className="flex-1 rounded-lg border border-gray-300 px-4 py-2.5 outline-none transition-colors focus:border-blue-500 focus:ring-2 focus:ring-blue-100"
          />
          <button className="rounded-lg bg-blue-600 px-5 py-2.5 font-medium text-white shadow-sm transition-colors hover:bg-blue-700">
            Search
          </button>
        </form>

        <div className="mt-4 flex gap-4 text-sm">
          <a href="/search?q=criminal+appeal+murder" className="text-blue-600 hover:underline">
            Browse examples →
          </a>
          <a href="/upload" className="text-blue-600 hover:underline">
            Upload a judgment (PDF/txt) →
          </a>
        </div>
      </section>

      {stats && (
        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Stat icon="📚" label="Indexed cases" value={stats.cases.toLocaleString()} />
          <Stat icon="🗓️" label="Year range" value={`${stats.year_min}–${stats.year_max}`} />
          <Stat
            icon="🧠"
            label="Model"
            value={`${stats.model?.text_method} + ${stats.model?.network_method} → ${stats.model?.hybrid_method}`}
          />
        </div>
      )}

      <section className="mt-10">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">
          How it works
        </h2>
        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Step
            n={1}
            title="Text similarity"
            body="TF-IDF (default), with Doc2Vec and Sentence-BERT available as alternative embedders."
          />
          <Step
            n={2}
            title="Network similarity"
            body="A Hier-SPCNet graph of cases, Acts, chapters and sections, embedded via metapath2vec."
          />
          <Step
            n={3}
            title="Hybrid ranking"
            body="A learned combiner blends both signals into one score, with a shared-statute / shared-citation explanation."
          />
        </div>
      </section>

      <ErrorBox error={null} />
    </div>
  );
}

function Stat({ icon, label, value }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-gray-200 bg-white p-4 transition-shadow hover:shadow-md">
      <span className="text-2xl leading-none" aria-hidden>
        {icon}
      </span>
      <div>
        <div className="text-xs uppercase tracking-wide text-gray-400">{label}</div>
        <div className="mt-1 text-lg font-semibold text-slate-800">{value}</div>
      </div>
    </div>
  );
}

function Step({ n, title, body }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-xs font-semibold text-white">
        {n}
      </div>
      <div className="mt-2 font-medium text-slate-800">{title}</div>
      <p className="mt-1 text-sm text-gray-600">{body}</p>
    </div>
  );
}
