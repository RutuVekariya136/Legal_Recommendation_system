import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import { ErrorBox } from "../components/ui.jsx";

export default function HomePage() {
  const [q, setQ] = useState("");
  const navigate = useNavigate();
  const { data: stats } = useAsync(() => api.stats(), []);
  const { data: facets } = useAsync(() => api.facets(), []);

  const decadeData = facets
    ? Object.entries(facets.years).map(([decade, count]) => ({ decade, count }))
    : [];

  return (
    <div>
      <section className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-indigo-950 px-6 py-12 text-white shadow-lg sm:px-10">
        <div
          className="pointer-events-none absolute inset-0 opacity-20"
          style={{
            backgroundImage:
              "radial-gradient(circle at 20% 20%, #6366f1 0%, transparent 40%), radial-gradient(circle at 85% 75%, #a78bfa 0%, transparent 45%)",
          }}
        />
        <div className="relative">
          <span className="inline-block rounded-full bg-indigo-500/20 px-3 py-1 text-xs font-medium tracking-wide text-indigo-200 uppercase">
            Text similarity × citation/statute network similarity
          </span>
          <h1 className="mt-4 text-3xl font-semibold tracking-tight sm:text-4xl">
            Find similar legal cases in seconds, not hours
          </h1>
          <p className="mt-3 max-w-2xl text-slate-300">
            A hybrid recommender over Indian Supreme Court judgments — combining{" "}
            <strong className="text-white">text similarity</strong> with a{" "}
            <strong className="text-white">Hier-SPCNet citation/statute graph</strong>, and an
            explainable score breakdown for every match.
          </p>

          <form
            className="mt-7 flex max-w-2xl gap-2"
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
              className="flex-1 rounded-lg border border-white/10 bg-white/10 px-4 py-2.5 text-white placeholder-slate-400 outline-none backdrop-blur-sm transition-colors focus:border-indigo-400 focus:bg-white/15"
            />
            <button className="rounded-lg bg-indigo-500 px-5 py-2.5 font-medium text-white shadow-sm transition-colors hover:bg-indigo-400">
              Search
            </button>
          </form>

          <div className="mt-4 flex gap-4 text-sm">
            <a href="/search?q=criminal+appeal+murder" className="text-indigo-300 hover:text-indigo-200 hover:underline">
              Browse examples →
            </a>
            <a href="/upload" className="text-indigo-300 hover:text-indigo-200 hover:underline">
              Upload a case (PDF/txt) →
            </a>
          </div>
        </div>
      </section>

      {stats && (
        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Stat
            icon="📚"
            accent="border-l-indigo-500"
            label="Searchable cases"
            value={stats.searchable_cases.toLocaleString()}
            note={
              stats.cases > stats.searchable_cases
                ? `${stats.cases.toLocaleString()} ingested; the rest aren't in the active search index yet`
                : undefined
            }
          />
          <Stat
            icon="🗓️"
            accent="border-l-violet-500"
            label="Year range"
            value={`${stats.year_min}–${stats.year_max}`}
          />
          <Stat
            icon="🧠"
            accent="border-l-emerald-500"
            label="Model"
            value={`${stats.model?.text_method} + ${stats.model?.network_method} → ${stats.model?.hybrid_method}`}
          />
        </div>
      )}

      {decadeData.length > 0 && (
        <section className="mt-6 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">
            Corpus coverage by decade
          </h2>
          <p className="mt-1 text-xs text-gray-400">
            {facets.total.toLocaleString()} judgments indexed, 1947–2020
          </p>
          <div className="mt-3 h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={decadeData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <XAxis dataKey="decade" tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} />
                <Tooltip
                  cursor={{ fill: "#f1f5f9" }}
                  formatter={(v) => [v.toLocaleString(), "cases"]}
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid #e2e8f0" }}
                />
                <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      <section className="mt-10">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">
          How it works
        </h2>
        <div className="relative mt-4 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <Step
            n={1}
            color="bg-indigo-600"
            title="Text similarity"
            body="TF-IDF (default), with Doc2Vec and Sentence-BERT available as alternative embedders."
          />
          <Step
            n={2}
            color="bg-violet-600"
            title="Network similarity"
            body="A Hier-SPCNet graph of cases, Acts, chapters and sections, embedded via metapath2vec."
          />
          <Step
            n={3}
            color="bg-emerald-600"
            title="Hybrid ranking"
            body="A learned combiner blends both signals into one score, with a shared-statute / shared-citation explanation."
          />
        </div>
      </section>

      <ErrorBox error={null} />
    </div>
  );
}

function Stat({ icon, label, value, note, accent = "border-l-gray-300" }) {
  return (
    <div
      className={`flex items-start gap-3 rounded-lg border border-gray-200 border-l-4 ${accent} bg-white p-4 transition-shadow hover:shadow-md`}
    >
      <span className="text-2xl leading-none" aria-hidden>
        {icon}
      </span>
      <div>
        <div className="text-xs uppercase tracking-wide text-gray-400">{label}</div>
        <div className="mt-1 text-lg font-semibold text-slate-800">{value}</div>
        {note && <div className="mt-1 text-xs text-gray-400">{note}</div>}
      </div>
    </div>
  );
}

function Step({ n, title, body, color }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 transition-shadow hover:shadow-md">
      <div className={`flex h-7 w-7 items-center justify-center rounded-full ${color} text-xs font-semibold text-white shadow-sm`}>
        {n}
      </div>
      <div className="mt-2 font-medium text-slate-800">{title}</div>
      <p className="mt-1 text-sm text-gray-600">{body}</p>
    </div>
  );
}
