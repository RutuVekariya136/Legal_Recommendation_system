import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import { ErrorBox, Loading, Section } from "../components/ui.jsx";
import ExplanationPanel from "../components/ExplanationPanel.jsx";
import SideBySide from "../components/SideBySide.jsx";

export default function ComparePage({ kind }) {
  const { uploadId } = useParams();
  const [params] = useSearchParams();
  const a = params.get("a");
  const b = params.get("b");
  const target = params.get("target");

  const { data: ex, loading, error } = useAsync(
    () => (kind === "upload" ? api.compareUpload(uploadId, target) : api.compare(a, b)),
    [kind, uploadId, target, a, b],
  );

  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-gray-400">Why are these similar?</p>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-lg font-semibold text-slate-800">
        <span>{kind === "upload" ? "Uploaded document" : <Link className="text-blue-700 hover:underline" to={`/case/${a}`}>{a}</Link>}</span>
        <span className="text-gray-300">vs</span>
        <Link className="text-blue-700 hover:underline" to={`/case/${kind === "upload" ? target : b}`}>
          {kind === "upload" ? target : b}
        </Link>
      </div>

      <ErrorBox error={error} />
      {loading ? (
        <Loading />
      ) : (
        ex && (
          <div className="mt-5">
            <ExplanationPanel ex={ex} />
            <Section title="Side-by-side text">
              <SideBySide ex={ex} kind={kind} uploadId={uploadId} />
            </Section>
          </div>
        )
      )}
    </div>
  );
}
