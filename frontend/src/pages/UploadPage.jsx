import UploadParametersPanel from "../components/UploadParametersPanel.jsx";
import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, apiError } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import { Badge, ErrorBox } from "../components/ui.jsx";

const STEPS = [
  { n: 1, title: "Upload", body: "Drop a case as PDF or plain text." },
  { n: 2, title: "Extract", body: "The text layer is read and cleaned; statutes and citations are detected." },
  { n: 3, title: "Compare", body: "It is matched against 33,000+ Supreme Court cases and explained." },
];

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [year, setYear] = useState("");
  const [category, setCategory] = useState("");
  const [categoryQuery, setCategoryQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const navigate = useNavigate();
  const { data: categories } = useAsync(() => api.categories(), []);

  function pick(f) {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
  }

  async function submit(e) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setResult(
        await api.upload(file, year || undefined, category || undefined,
                         category === "other" ? categoryQuery || undefined : undefined),
      );
    } catch (err) {
      setError(apiError(err));
    } finally {
      setBusy(false);
    }
  }

  const chip = (active) =>
    `rounded-full border px-3 py-1.5 text-sm transition-colors ${
      active
        ? "border-indigo-500 bg-indigo-600 text-white shadow-sm"
        : "border-gray-300 bg-white text-gray-700 hover:border-indigo-300 hover:bg-indigo-50"
    }`;

  return (
    <div>
      <section className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-slate-900 via-slate-900 to-indigo-950 px-6 py-9 text-white shadow-lg sm:px-10">
        <div
          className="pointer-events-none absolute inset-0 opacity-20"
          style={{
            backgroundImage:
              "radial-gradient(circle at 15% 25%, #6366f1 0%, transparent 40%), radial-gradient(circle at 90% 80%, #a78bfa 0%, transparent 45%)",
          }}
        />
        <div className="relative">
          <span className="inline-block rounded-full bg-indigo-500/20 px-3 py-1 text-xs font-medium uppercase tracking-wide text-indigo-200">
            Find precedents for your case
          </span>
          <h1 className="mt-3 text-2xl font-semibold tracking-tight sm:text-3xl">Upload a case</h1>
          <p className="mt-2 max-w-2xl text-slate-300">
            Upload a case and get the most similar cases, with a plain-English explanation of
            exactly why each one matches.
          </p>
        </div>
      </section>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
        <form onSubmit={submit} className="space-y-5 rounded-xl border border-gray-200 bg-white p-5 shadow-sm lg:col-span-2">
          <div>
            <div className="mb-2 text-sm font-semibold text-slate-800">1. Choose the case</div>
            <div
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files?.[0]); }}
              className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-9 text-center transition-colors ${
                dragging ? "border-indigo-500 bg-indigo-50" : file ? "border-emerald-300 bg-emerald-50/50" : "border-gray-300 bg-gray-50 hover:border-indigo-300 hover:bg-indigo-50/40"
              }`}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".pdf,.txt"
                className="hidden"
                onChange={(e) => pick(e.target.files?.[0] || null)}
              />
              {file ? (
                <>
                  <div className="text-3xl" aria-hidden>📄</div>
                  <div className="mt-2 font-medium text-slate-800">{file.name}</div>
                  <div className="text-xs text-gray-500">{formatSize(file.size)} · click to choose a different file</div>
                </>
              ) : (
                <>
                  <div className="text-3xl" aria-hidden>⬆️</div>
                  <div className="mt-2 font-medium text-slate-800">Drag &amp; drop a case here</div>
                  <div className="text-xs text-gray-500">or click to browse · PDF or .txt · up to 20 MB</div>
                </>
              )}
            </div>
          </div>

          <div>
            <div className="mb-2 text-sm font-semibold text-slate-800">
              2. Case type <span className="font-normal text-gray-400">(optional)</span>
            </div>
            <p className="mb-2 text-xs text-gray-500">
              Restricts the search to one kind of case. Leave on "Any" to search everything.
            </p>
            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={() => setCategory("")} className={chip(category === "")}>
                Any
              </button>
              {categories?.map((c) => (
                <button key={c.id} type="button" onClick={() => setCategory(c.id)} className={chip(category === c.id)}>
                  {c.label}
                </button>
              ))}
            </div>
            {category === "other" && (
              <input
                type="text"
                placeholder="Describe the case type, or a distinctive phrase (e.g. a statute name)"
                value={categoryQuery}
                onChange={(e) => setCategoryQuery(e.target.value)}
                className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
              />
            )}
          </div>

          <div>
            <div className="mb-2 text-sm font-semibold text-slate-800">
              3. Year of the case <span className="font-normal text-gray-400">(optional)</span>
            </div>
            <input
              type="number"
              placeholder="e.g. 2015"
              value={year}
              onChange={(e) => setYear(e.target.value)}
              className="w-40 rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
            />
          </div>

          <button
            disabled={!file || busy}
            className="rounded-lg bg-indigo-600 px-6 py-2.5 font-medium text-white shadow-sm transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? "Extracting…" : "Upload & extract"}
          </button>
          <ErrorBox error={error} />
        </form>

        <aside className="space-y-4">
          <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
            <div className="text-sm font-semibold uppercase tracking-wide text-gray-500">What happens next</div>
            <ol className="mt-3 space-y-4">
              {STEPS.map((s) => (
                <li key={s.n} className="flex gap-3">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-xs font-semibold text-white">
                    {s.n}
                  </span>
                  <div>
                    <div className="text-sm font-medium text-slate-800">{s.title}</div>
                    <div className="text-xs text-gray-500">{s.body}</div>
                  </div>
                </li>
              ))}
            </ol>
          </div>
          <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs text-amber-900">
            Scanned or image-only PDFs are detected but can't be read (no OCR). Your file is used only to
            find similar cases. This is a research aid, not legal advice.
          </div>
        </aside>
      </div>

      {result && (
        <div className="mt-6 rounded-xl border border-emerald-200 bg-white p-5 shadow-sm">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-lg" aria-hidden>✅</span>
            <span className="font-medium text-slate-800">{result.filename}</span>
            <Badge tone="blue">{result.source_type}</Badge>
            {result.needs_ocr && <Badge tone="amber">needs OCR</Badge>}
            {(result.category || result.category_query) && (
              <Badge tone="gray">
                {result.category
                  ? categories?.find((c) => c.id === result.category)?.label || result.category
                  : `"${result.category_query}"`}
              </Badge>
            )}
          </div>
          <p className="mt-1 text-sm text-gray-500">
            {result.char_count.toLocaleString()} characters
            {result.page_count ? ` · ${result.page_count} pages` : ""} extracted
          </p>
          {result.warnings?.length > 0 && (
            <ul className="mt-1 text-xs text-amber-700">
              {result.warnings.map((w) => (
                <li key={w}>⚠ {w}</li>
              ))}
            </ul>
          )}
          <p className="prose-legal mt-3 max-h-40 overflow-auto rounded-lg bg-gray-50 p-3 text-xs text-gray-600">
            {result.preview}…
          </p>

          <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-gray-100 pt-4">
            <p className="text-xs text-gray-500">
              Text extracted successfully. Ready to search across 33,000+ Supreme Court cases.
            </p>
            <button
              type="button"
              onClick={() => navigate(`/recommend/upload/${result.id}`)}
              className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-indigo-500"
            >
              Find similar cases →
            </button>
          </div>
        </div>
      )}
      {result && (
        <UploadParametersPanel
          uploadId={result.id}
          category={result.category}
          onContinue={(details) => navigate(`/recommend/upload/${result.id}`, { state: { details } })}
        />
      )}
    </div>
  );
}
