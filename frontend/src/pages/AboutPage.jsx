import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";

export default function AboutPage() {
  const { data: ready } = useAsync(() => api.ready(), []);

  return (
    <div className="prose max-w-2xl text-sm text-gray-700">
      <h1 className="text-xl font-semibold text-slate-800">About this project</h1>
      <p className="mt-2">
        An implementation of Bhattacharya, Ghosh, Pal &amp; Ghosh (2022),{" "}
        <em>“Legal Case Document Similarity: You Need Both Network and Text”</em> (IP&amp;M 59(6),
        103069), built from scratch — the paper released the dataset but no model code.
      </p>
      <ul className="mt-3 list-disc pl-5">
        <li><strong>Text similarity:</strong> TF-IDF (baseline), Doc2Vec (the paper), Sentence-BERT (our option).</li>
        <li><strong>Network similarity:</strong> Hier-SPCNet (cases + Acts + statute hierarchy), node2vec / metapath2vec embeddings, ICF weighting.</li>
        <li><strong>Hybrid:</strong> weighted sum, or a learned combiner in the spirit of the paper's NN-Map.</li>
      </ul>
      <p className="mt-3 text-gray-500">
        Demo corpus: ~295 Indian Supreme Court judgments (the paper used 53,210). Citation
        resolution is fuzzy title matching, so case→case links are sparse; the statute layer
        carries most of the network signal. This is an academic project — not legal advice.
      </p>
      {ready && (
        <pre className="mt-4 overflow-auto rounded bg-slate-800 p-3 text-xs text-slate-100">
{JSON.stringify(ready, null, 2)}
        </pre>
      )}
    </div>
  );
}
