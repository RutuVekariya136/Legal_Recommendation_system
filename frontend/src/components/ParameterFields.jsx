// Renders the generalized case parameters for one category. Driven entirely by
// the server's parameter list (id, label, ftype, options, placeholder), so adding
// or changing a category's parameters never touches this component.

const inputClass = "mt-1 w-full rounded-lg border border-gray-300 px-2.5 py-1.5 text-sm focus:border-slate-500 focus:outline-none";

function Field({ p, value, onChange, issue }) {
  const common = { id: `cp-${p.id}`, className: inputClass, value: value || "" };
  if (p.ftype === "select") {
    return (
      <select {...common} onChange={(e) => onChange(e.target.value)}>
        <option value="">— not specified —</option>
        {p.options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    );
  }
  if (p.ftype === "date") {
    return <input {...common} type="text" inputMode="numeric" placeholder="DD/MM/YYYY" onChange={(e) => onChange(e.target.value)} />;
  }
  if (p.ftype === "number") {
    return <input {...common} type="text" inputMode="decimal" placeholder={p.placeholder} onChange={(e) => onChange(e.target.value)} />;
  }
  return <input {...common} type="text" maxLength={300} placeholder={p.placeholder} onChange={(e) => onChange(e.target.value)} />;
}

export default function ParameterFields({ params, draft, onDraft, issues = {} }) {
  return (
    <div className="grid gap-x-4 gap-y-3 sm:grid-cols-2">
      {params.map((p) => (
        <div key={p.id}>
          <label htmlFor={`cp-${p.id}`} className="flex items-center gap-2 text-xs">
            <span className="font-medium text-slate-700">{p.label}</span>
            {p.present && <span className="rounded-full bg-green-50 px-1.5 text-green-700" title={p.evidence || ""}>in your document</span>}
          </label>
          <Field p={p} value={draft[p.id]} onChange={(v) => onDraft({ ...draft, [p.id]: v })} />
          {issues[p.id] && <span className="mt-0.5 block text-xs text-red-600">{issues[p.id]}</span>}
          <span className="mt-0.5 block text-xs text-gray-400">{p.why}</span>
        </div>
      ))}
    </div>
  );
}
