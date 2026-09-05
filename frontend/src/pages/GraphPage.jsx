import { Link, useParams } from "react-router-dom";
import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";
import { ErrorBox, Loading } from "../components/ui.jsx";
import GraphView from "../components/GraphView.jsx";

export default function GraphPage() {
  const { docId } = useParams();
  const { data, loading, error } = useAsync(() => api.egoGraph(docId), [docId]);

  return (
    <div>
      <Link to={`/case/${docId}`} className="text-sm text-blue-600 hover:underline">
        ← {docId}
      </Link>
      <h1 className="mt-1 text-lg font-semibold text-slate-800">Citation / statute network</h1>
      <p className="text-sm text-gray-500">
        <span className="text-violet-600">■</span> this case &nbsp;
        <span className="text-blue-600">■</span> similar cases (thicker = higher hybrid score) &nbsp;
        <span className="text-teal-700">■</span> Act &nbsp;
        <span className="text-green-700">■</span> section
      </p>

      <ErrorBox error={error} />
      {loading ? <Loading /> : <div className="mt-3"><GraphView graph={data} /></div>}

      <p className="mt-2 text-xs text-gray-400">
        Click a blue node to open that case. A section shared by two cases links to both.
      </p>
    </div>
  );
}
