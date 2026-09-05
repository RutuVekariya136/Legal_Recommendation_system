import { useMemo } from "react";
import { useNavigate } from "react-router-dom";

const W = 760;
const H = 520;
const CX = W / 2;
const CY = H / 2;

/** Simple dependency-free radial layout: center case, similar cases on the left
 *  arc, statutes on the right arc. */
export default function GraphView({ graph }) {
  const navigate = useNavigate();

  const positioned = useMemo(() => {
    if (!graph) return null;
    const center = graph.nodes.find((n) => n.center);
    const cases = graph.nodes.filter((n) => n.type === "case" && !n.center);
    const statutes = graph.nodes.filter((n) => n.type !== "case");
    const pos = { [center.id]: { x: CX, y: CY } };

    const place = (arr, angleStart, angleEnd, radius) => {
      arr.forEach((n, i) => {
        const t = arr.length === 1 ? 0.5 : i / (arr.length - 1);
        const a = angleStart + t * (angleEnd - angleStart);
        pos[n.id] = { x: CX + radius * Math.cos(a), y: CY + radius * Math.sin(a) };
      });
    };
    place(cases, Math.PI * 0.6, Math.PI * 1.4, 210); // left arc
    place(statutes, -Math.PI * 0.4, Math.PI * 0.4, 230); // right arc
    return { center, pos };
  }, [graph]);

  if (!graph || !positioned) return null;
  const { pos } = positioned;
  const nodeById = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-gray-200 bg-white">
      {graph.edges.map((e, i) => {
        const a = pos[e.source];
        const b = pos[e.target];
        if (!a || !b) return null;
        return (
          <line
            key={i}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            stroke={e.type === "similar" ? "#a78bfa" : "#cbd5e1"}
            strokeWidth={e.type === "similar" ? 1 + 3 * (e.weight || 0) : 1}
          />
        );
      })}
      {graph.nodes.map((n) => {
        const p = pos[n.id];
        if (!p) return null;
        const isCase = n.type === "case";
        const fill = n.center ? "#7c3aed" : isCase ? "#3b82f6" : n.type === "act" ? "#0f766e" : "#15803d";
        const r = n.center ? 10 : isCase ? 7 : 5;
        return (
          <g
            key={n.id}
            transform={`translate(${p.x},${p.y})`}
            className={isCase ? "cursor-pointer" : ""}
            onClick={() => isCase && !n.center && navigate(`/case/${n.id.replace("case:", "")}`)}
          >
            <circle r={r} fill={fill} />
            <text
              x={p.x > CX ? r + 4 : -(r + 4)}
              y={4}
              textAnchor={p.x > CX ? "start" : "end"}
              fontSize={n.center ? 12 : 10}
              fill="#334155"
            >
              {(n.label || n.id).slice(0, 42)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
