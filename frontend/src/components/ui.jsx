/** Small shared UI primitives. */

export function Loading({ label = "Loading…" }) {
  return <div className="py-10 text-center text-sm text-gray-500 animate-pulse">{label}</div>;
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return (
    <div className="my-4 rounded border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      {String(error)}
    </div>
  );
}

export function Badge({ children, tone = "gray" }) {
  const tones = {
    gray: "bg-gray-100 text-gray-700",
    blue: "bg-blue-100 text-blue-700",
    green: "bg-green-100 text-green-700",
    amber: "bg-amber-100 text-amber-800",
    purple: "bg-purple-100 text-purple-700",
    red: "bg-red-100 text-red-700",
  };
  return (
    <span className={`inline-block rounded px-1.5 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
}

/** Horizontal 0..1 score bar with a label. */
export function ScoreBar({ label, value, tone = "blue" }) {
  const pct = Math.max(0, Math.min(1, value || 0)) * 100;
  const bar = { blue: "bg-blue-500", violet: "bg-violet-500", slate: "bg-slate-500" }[tone];
  return (
    <div className="mb-1.5">
      <div className="flex justify-between text-xs text-gray-500">
        <span>{label}</span>
        <span className="tabular-nums">{(value ?? 0).toFixed(3)}</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded bg-gray-100">
        <div className={`h-full ${bar}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function Section({ title, children, right }) {
  return (
    <section className="mb-6">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-500">{title}</h2>
        {right}
      </div>
      {children}
    </section>
  );
}
