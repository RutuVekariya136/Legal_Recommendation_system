import { useState } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";

const navClass = ({ isActive }) =>
  `px-3 py-1.5 rounded text-sm ${isActive ? "bg-white/20 text-white" : "text-blue-100 hover:text-white"}`;

export default function Layout({ children }) {
  const [q, setQ] = useState("");
  const navigate = useNavigate();

  return (
    <div className="min-h-screen">
      <header className="bg-slate-800 text-white">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3">
          <Link to="/" className="font-semibold">
            ⚖️ Legal Case Similarity
          </Link>
          <nav className="flex gap-1">
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
              className="w-56 rounded bg-white/10 px-3 py-1.5 text-sm placeholder-blue-200 outline-none focus:bg-white/20"
            />
          </form>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-6">{children}</main>

      <footer className="mx-auto max-w-5xl px-4 py-8 text-xs text-gray-400">
        Academic project — hybrid text + citation/statute network similarity over ~295 Indian
        Supreme Court judgments. Not legal advice.
      </footer>
    </div>
  );
}
