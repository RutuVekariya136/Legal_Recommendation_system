import { api } from "../services/api.js";
import { useAsync } from "../hooks/useAsync.js";

function Box({ x, y, w, h, fill, text, textColor = "#fff", fontSize = 13 }) {
  const lines = text.split("\n");
  const lineHeight = fontSize + 4;
  const startY = y + h / 2 - ((lines.length - 1) * lineHeight) / 2 + fontSize / 3;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={8} fill={fill} />
      {lines.map((line, i) => (
        <text
          key={i}
          x={x + w / 2}
          y={startY + i * lineHeight}
          textAnchor="middle"
          fontSize={fontSize}
          fontWeight={600}
          fill={textColor}
        >
          {line}
        </text>
      ))}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2 }) {
  return (
    <line
      x1={x1} y1={y1} x2={x2} y2={y2}
      stroke="#94a3b8" strokeWidth={2} markerEnd="url(#arrowhead)"
    />
  );
}

/** The system's actual pipeline: Upload -> Extraction/Preprocessing -> [Text | Network]
 * analysis branches -> Hybrid -> Ranking -> Top-K -> Explanation/Comparison. */
function ArchitectureDiagram() {
  const NAVY = "#1e3a5f";
  const SLATE = "#334155";
  const GOLD = "#a67c3d";
  const INDIGO = "#4f46e5";
  const VIOLET = "#7c3aed";

  const cx = 350; // overall center
  const boxW = 300, boxH = 44;
  const colW = 240, colH = 40;

  let y = 20;
  const rows = [];

  const userY = y; rows.push(userY); y += boxH + 22;
  const extractY = y; rows.push(extractY); y += boxH + 14;
  const preprocessY = y; rows.push(preprocessY); y += boxH + 30;

  const splitTopY = y;
  const leftX = cx - colW - 20, rightX = cx + 20;
  const textEmbedY = splitTopY, textSimY = splitTopY + colH + 12;
  const citeY = splitTopY, hierY = splitTopY + colH + 12,
        netEmbedY = splitTopY + 2 * (colH + 12), netSimY = splitTopY + 3 * (colH + 12);
  const branchBottomY = Math.max(textSimY, netSimY) + colH;

  y = branchBottomY + 30;
  const hybridY = y; y += boxH + 14;
  const rankY = y; y += boxH + 14;
  const topkY = y; y += boxH + 14;
  const explainY = y; y += boxH + 10;

  const totalHeight = explainY + boxH + 20;

  return (
    <svg viewBox={`0 0 700 ${totalHeight}`} className="w-full" role="img" aria-label="System architecture pipeline">
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8" />
        </marker>
      </defs>

      <Box x={cx - boxW / 2} y={userY} w={boxW} h={boxH} fill={NAVY} text="User → Legal Case (search or upload)" fontSize={12.5} />
      <Arrow x1={cx} y1={userY + boxH} x2={cx} y2={extractY} />

      <Box x={cx - boxW / 2} y={extractY} w={boxW} h={boxH} fill={SLATE} text="Text Extraction" />
      <Arrow x1={cx} y1={extractY + boxH} x2={cx} y2={preprocessY} />

      <Box x={cx - boxW / 2} y={preprocessY} w={boxW} h={boxH} fill={SLATE} text="Text Preprocessing" />

      {/* split */}
      <Arrow x1={cx} y1={preprocessY + boxH} x2={leftX + colW / 2} y2={splitTopY - 14} />
      <Arrow x1={cx} y1={preprocessY + boxH} x2={rightX + colW / 2} y2={splitTopY - 14} />

      <Box x={leftX} y={splitTopY - 14} w={colW} h={26} fill={GOLD} text="TEXT ANALYSIS" fontSize={11} />
      <Box x={rightX} y={splitTopY - 14} w={colW} h={26} fill={GOLD} text="NETWORK ANALYSIS" fontSize={11} />

      <Arrow x1={leftX + colW / 2} y1={splitTopY + 12} x2={leftX + colW / 2} y2={textEmbedY + 18} />
      <Arrow x1={rightX + colW / 2} y1={splitTopY + 12} x2={rightX + colW / 2} y2={citeY + 18} />

      <Box x={leftX} y={textEmbedY + 18} w={colW} h={colH} fill="#eef2ff" textColor={INDIGO} text="Text Embedding" fontSize={12} />
      <Arrow x1={leftX + colW / 2} y1={textEmbedY + 18 + colH} x2={leftX + colW / 2} y2={textSimY + 18} />
      <Box x={leftX} y={textSimY + 18} w={colW} h={colH} fill="#eef2ff" textColor={INDIGO} text="Text Similarity" fontSize={12} />

      <Box x={rightX} y={citeY + 18} w={colW} h={colH} fill="#f5f3ff" textColor={VIOLET} text="Citation + Statute Extraction" fontSize={11.5} />
      <Arrow x1={rightX + colW / 2} y1={citeY + 18 + colH} x2={rightX + colW / 2} y2={hierY + 18} />
      <Box x={rightX} y={hierY + 18} w={colW} h={colH} fill="#f5f3ff" textColor={VIOLET} text="Hier-SPCNet" fontSize={12} />
      <Arrow x1={rightX + colW / 2} y1={hierY + 18 + colH} x2={rightX + colW / 2} y2={netEmbedY + 18} />
      <Box x={rightX} y={netEmbedY + 18} w={colW} h={colH} fill="#f5f3ff" textColor={VIOLET} text="Network Embedding" fontSize={12} />
      <Arrow x1={rightX + colW / 2} y1={netEmbedY + 18 + colH} x2={rightX + colW / 2} y2={netSimY + 18} />
      <Box x={rightX} y={netSimY + 18} w={colW} h={colH} fill="#f5f3ff" textColor={VIOLET} text="Network Similarity (ICF-weighted)" fontSize={11} />

      {/* merge */}
      <Arrow x1={leftX + colW / 2} y1={textSimY + 18 + colH} x2={cx} y2={hybridY} />
      <Arrow x1={rightX + colW / 2} y1={netSimY + 18 + colH} x2={cx} y2={hybridY} />

      <Box x={cx - boxW / 2} y={hybridY} w={boxW} h={boxH} fill={NAVY} text="Hybrid Similarity" />
      <Arrow x1={cx} y1={hybridY + boxH} x2={cx} y2={rankY} />
      <Box x={cx - boxW / 2} y={rankY} w={boxW} h={boxH} fill={NAVY} text="Case Ranking" />
      <Arrow x1={cx} y1={rankY + boxH} x2={cx} y2={topkY} />
      <Box x={cx - boxW / 2} y={topkY} w={boxW} h={boxH} fill={GOLD} text="Top-K Case Recommendation" />
      <Arrow x1={cx} y1={topkY + boxH} x2={cx} y2={explainY} />
      <Box x={cx - boxW / 2} y={explainY} w={boxW} h={boxH} fill={GOLD} text="Explanation + Case Comparison" fontSize={12.5} />
    </svg>
  );
}

export default function AboutPage() {
  const { data: ready } = useAsync(() => api.ready(), []);
  const { data: stats } = useAsync(() => api.stats(), []);

  return (
    <div className="max-w-3xl">
      <h1 className="text-2xl font-semibold tracking-tight text-slate-900">About this project</h1>
      <p className="mt-2 text-sm text-gray-600">
        An implementation of Bhattacharya, Ghosh, Pal &amp; Ghosh (2022),{" "}
        <em>“Legal Case Document Similarity: You Need Both Network and Text”</em> (IP&amp;M 59(6),
        103069), built from scratch — the paper released the dataset but no model code.
      </p>

      <section className="mt-6 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">
          System architecture
        </h2>
        <div className="mt-3">
          <ArchitectureDiagram />
        </div>
      </section>

      <section className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
        <InfoCard
          accent="border-l-indigo-500"
          title="Text similarity"
          body="TF-IDF (baseline), Doc2Vec (the paper's method), Sentence-BERT (our option)."
        />
        <InfoCard
          accent="border-l-violet-500"
          title="Network similarity"
          body="Hier-SPCNet (cases + Acts + statute hierarchy), node2vec / metapath2vec embeddings, ICF weighting so generic statutes don't dominate."
        />
        <InfoCard
          accent="border-l-emerald-500"
          title="Hybrid combiner"
          body="Weighted sum, or a learned combiner in the spirit of the paper's NN-Map."
        />
      </section>

      <p className="mt-6 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
        <strong>Corpus:</strong> {stats?.searchable_cases?.toLocaleString() ?? "…"} Indian Supreme
        Court judgments are indexed for search/recommendation (1947–2020, bulk-ingested from
        ILDC). Expert-scored evaluation is only available on the original{" "}
        <strong>296-document</strong> paper corpus (the paper itself used 53,210 cases). Citation
        resolution is fuzzy title matching, so case→case links stay sparse relative to the
        statute layer, which carries most of the network signal. This is an academic project —
        not legal advice.
      </p>

      {ready && (
        <pre className="mt-4 overflow-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-100">
{JSON.stringify(ready, null, 2)}
        </pre>
      )}
    </div>
  );
}

function InfoCard({ title, body, accent }) {
  return (
    <div className={`rounded-lg border border-gray-200 border-l-4 ${accent} bg-white p-4 shadow-sm`}>
      <div className="font-medium text-slate-800">{title}</div>
      <p className="mt-1 text-sm text-gray-600">{body}</p>
    </div>
  );
}
