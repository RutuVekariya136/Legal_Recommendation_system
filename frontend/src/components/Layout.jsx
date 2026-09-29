import { useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";

const navClass = ({ isActive }) =>
  `px-3 py-1.5 rounded text-sm font-medium transition-colors ${
    isActive ? "bg-white/15 text-white" : "text-blue-100/80 hover:bg-white/10 hover:text-white"
  }`;

export default function Layout({ children }) {
  const [q, setQ] = useState("");
  const navigate = useNavigate();
  const { data: stats } = useAsync(() => api.stats(), []);

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950 text-white shadow-md">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3">
          <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <span aria-hidden className="text-lg">⚖️</span> Legal Case Similarity
            {stats?.searchable_cases && (
              <span className="hidden rounded-full bg-indigo-500/20 px-2 py-0.5 text-[11px] font-medium text-indigo-200 sm:inline">
                {stats.searchable_cases.toLocaleString()} cases indexed
              </span>
            )}
          </Link>
          <nav className="flex gap-1">
            <NavLink to="/" end className={navClass}>Home</NavLink>
            <NavLink to="/search" className={navClass}>Search</NavLink>
            <NavLink to="/upload" className={navClass}>Upload</NavLink>
            <NavLink to="/evaluation" className={navClass}>Evaluation</NavLink>
            <NavLink to="/about" className={navClass}>About</NavLink>
          </nav>
          <form
            className="ml-auto"
            onSubmit={(e) => {
              e.preventDefault();
              if (q.trim()) navigate(`/search?q=${encodeURIComponent(q.trim())}`);
            }}
          >
            <input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search cases…"
              className="w-56 rounded-md border border-white/10 bg-white/10 px-3 py-1.5 text-sm text-white placeholder-blue-200/70 outline-none transition-colors focus:border-white/30 focus:bg-white/20"
            />
          </form>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-6">{children}</main>

      <footer className="mt-8 border-t border-gray-200 bg-white">
        <div className="mx-auto max-w-5xl px-4 py-6 text-xs text-gray-400">
          Academic project — hybrid text + citation/statute network similarity over{" "}
          {stats?.searchable_cases ? stats.searchable_cases.toLocaleString() : "…"} Indian Supreme
          Court judgments. Not legal advice.
        </div>
      </footer>
    </div>
  );
}
