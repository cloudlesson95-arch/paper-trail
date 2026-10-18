import { ENTITY_LABELS } from "@/lib/format";

export type Entity = { label: string; text: string; start_char: number; end_char: number };

const COLOR: Record<string, string> = {
  ORG: "var(--ent-org)",
  PERSON: "var(--ent-person)",
  DATE: "var(--ent-date)",
  MONEY: "var(--ent-money)",
  DOC_ID: "var(--ent-doc-id)",
  EMAIL: "var(--ent-email)",
};

const swatch = (label: string) => ({ "--ent": COLOR[label] }) as React.CSSProperties;

export function EntityLegend() {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
      {Object.entries(ENTITY_LABELS).map(([key, name]) => (
        <li key={key}>
          <mark className="entity" style={swatch(key)}>
            {name}
          </mark>
        </li>
      ))}
    </ul>
  );
}

/** OCR text with each entity marked in its type's colour; the type is also in the tooltip. */
export function EntityText({ text, entities }: { text: string; entities: Entity[] }) {
  const parts: React.ReactNode[] = [];
  let at = 0;
  for (const e of [...entities].sort((a, b) => a.start_char - b.start_char)) {
    if (e.start_char < at) continue; // overlapping spans: keep the first
    parts.push(text.slice(at, e.start_char));
    parts.push(
      <mark key={e.start_char} className="entity" style={swatch(e.label)} title={ENTITY_LABELS[e.label] ?? e.label}>
        {text.slice(e.start_char, e.end_char)}
      </mark>,
    );
    at = e.end_char;
  }
  parts.push(text.slice(at));
  return <pre className="whitespace-pre-wrap font-mono text-[13px] leading-6">{parts}</pre>;
}

/** The same entities as a list grouped by type, so nothing depends on colour alone. */
export function EntityList({ entities }: { entities: Entity[] }) {
  const groups = new Map<string, Set<string>>();
  for (const e of entities) groups.set(e.label, (groups.get(e.label) ?? new Set()).add(e.text));
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs">
      {Object.keys(ENTITY_LABELS)
        .filter((key) => groups.has(key))
        .map((key) => (
          <div key={key} className="contents">
            <dt className="text-ink-2">{ENTITY_LABELS[key]}</dt>
            <dd>{[...groups.get(key)!].join(" · ")}</dd>
          </div>
        ))}
    </dl>
  );
}
