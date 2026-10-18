/**
 * A confusion matrix as a heatmap: one sequential hue, each cell's strength the share of
 * its true class (row) predicted as that column, so the diagonal reads as per-class recall.
 */
export default function ConfusionMatrix({
  labels,
  matrix,
  names,
}: {
  labels: string[];
  matrix: number[][];
  names: Record<string, string>;
}) {
  const name = (key: string) => names[key] ?? key;
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-0.5 text-sm">
        <thead>
          <tr>
            <th className="px-2 py-1 text-left text-xs font-normal text-ink-2">true ↓ / predicted →</th>
            {labels.map((l) => (
              <th key={l} className="px-2 py-1 text-xs font-medium text-ink-2">
                {name(l)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={labels[i]}>
                <th className="px-2 py-1 text-left text-xs font-medium text-ink-2">{name(labels[i])}</th>
                {row.map((count, j) => {
                  const share = count / total;
                  return (
                    <td
                      key={labels[j]}
                      title={`True ${name(labels[i])}, predicted ${name(labels[j])}: ${count} of ${total}`}
                      className="num h-10 min-w-16 rounded px-2 text-center"
                      style={{
                        background: count ? `rgb(var(--heat) / ${0.12 + 0.88 * share})` : "transparent",
                        color: share > 0.5 ? "#ffffff" : "var(--ink)",
                        outline: "1px solid var(--line)",
                      }}
                    >
                      {count}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
