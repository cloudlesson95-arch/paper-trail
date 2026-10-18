function Cell({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span className="text-muted">null</span>;
  if (typeof value === "number") {
    return <>{Number.isInteger(value) ? value.toLocaleString("en-US") : value.toLocaleString("en-US", { maximumFractionDigits: 4 })}</>;
  }
  // Full-text snippets mark their hits with « and ».
  const text = String(value);
  return (
    <>
      {text.split(/(«[^»]*»)/).map((part, i) =>
        part.startsWith("«") ? (
          <mark key={i} className="hit">
            {part.slice(1, -1)}
          </mark>
        ) : (
          part
        ),
      )}
    </>
  );
}

export default function ResultTable({ columns, rows }: { columns: string[]; rows: unknown[][] }) {
  if (rows.length === 0) return <p className="text-sm text-ink-2">No rows.</p>;
  const numeric = columns.map((_, j) => rows.some((row) => typeof row[j] === "number"));
  return (
    <div className="max-h-96 overflow-auto rounded border border-line">
      <table className="w-full border-collapse text-sm">
        <thead className="sticky top-0 bg-surface">
          <tr>
            {columns.map((c, j) => (
              <th key={c} className={`border-b border-line px-2 py-1.5 font-medium text-ink-2 ${numeric[j] ? "text-right" : "text-left"}`}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-line last:border-0">
              {row.map((value, j) => (
                <td key={j} className={`px-2 py-1 align-top ${typeof value === "number" ? "num text-right" : ""}`}>
                  <Cell value={value} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
