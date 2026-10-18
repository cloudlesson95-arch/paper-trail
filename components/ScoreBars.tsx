import { label } from "@/lib/format";

/** Class probabilities as thin bars: the predicted class in the accent colour, the rest grey. */
export default function ScoreBars({ scores, names }: { scores: Record<string, number>; names: Record<string, string> }) {
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const top = entries[0]?.[0];
  return (
    <ul className="space-y-1 text-xs">
      {entries.map(([key, p]) => (
        <li key={key} className="grid grid-cols-[7.5rem_1fr_3.5rem] items-center gap-2" title={`${label(names, key)}: ${(p * 100).toFixed(1)}%`}>
          <span className={key === top ? "font-medium" : "text-ink-2"}>{label(names, key)}</span>
          <span className="h-2 rounded-full bg-line">
            <span
              className="block h-2 rounded-full"
              style={{ width: `${Math.max(p * 100, 1)}%`, background: key === top ? "var(--accent)" : "var(--axis)" }}
            />
          </span>
          <span className="num text-right text-ink-2">{(p * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}
