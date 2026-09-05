import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, apiError } from "../services/api.js";
import { Badge, ErrorBox } from "../components/ui.jsx";

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [year, setYear] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const navigate = useNavigate();

  async function submit(e) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.upload(file, year || undefined));
    } catch (err) {
      setError(apiError(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-xl">
      <h1 className="text-xl font-semibold text-slate-800">Upload a judgment</h1>
      <p className="mt-1 text-sm text-gray-600">
        PDF or plain text. The text layer is extracted and run through the same pipeline as the
        indexed corpus. Scanned/image-only PDFs are detected (no OCR).
      </p>

      <form onSubmit={submit} className="mt-5 space-y-3">
        <input
          type="file"
          accept=".pdf,.txt"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="block w-full rounded border border-gray-300 p-2 text-sm"
        />
        <input
          type="number"
          placeholder="year of the case (optional)"
          value={year}
          onChange={(e) => setYear(e.target.value)}
          className="w-56 rounded border border-gray-300 px-3 py-2 text-sm"
        />
        <button
          disabled={!file || busy}
          className="block rounded bg-blue-600 px-5 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {busy ? "Extracting…" : "Upload & extract"}
        </button>
      </form>

      <ErrorBox error={error} />

      {result && (
        <div className="mt-6 rounded-lg border border-gray-200 bg-white p-4">
          <div className="flex items-center gap-2">
            <span className="font-medium">{result.filename}</span>
            <Badge tone="blue">{result.source_type}</Badge>
            {result.needs_ocr && <Badge tone="amber">needs OCR</Badge>}
          </div>
          <p className="mt-1 text-sm text-gray-500">
            {result.char_count.toLocaleString()} characters
            {result.page_count ? ` · ${result.page_count} pages` : ""}
          </p>
          {result.warnings?.length > 0 && (
            <ul className="mt-1 text-xs text-amber-700">
              {result.warnings.map((w) => (
                <li key={w}>⚠ {w}</li>
              ))}
            </ul>
          )}
          <p className="prose-legal mt-3 max-h-40 overflow-auto rounded bg-gray-50 p-2 text-xs text-gray-600">
            {result.preview}…
          </p>
          <button
            onClick={() => navigate(`/recommend/upload/${result.id}`)}
            className="mt-3 rounded bg-slate-800 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
          >
            Find similar cases →
          </button>
        </div>
      )}
    </div>
  );
}
