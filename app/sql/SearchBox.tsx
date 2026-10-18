"use client";

import { useState } from "react";
import ResultTable from "@/components/ResultTable";

type SearchResult = { columns: string[]; rows: unknown[][]; match: string | null };

const EXAMPLES = ["toner cartridge", "hotel", "direct debit", "INV-2026"];

export default function SearchBox({ sql }: { sql: string }) {
  const [text, setText] = useState("");
  const [result, setResult] = useState<SearchResult | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function run(words: string) {
    setText(words);
    setBusy(true);
    setMessage(null);
    try {
      const response = await fetch(`/api/query?q=${encodeURIComponent(words)}`);
      const data = await response.json();
      if (!response.ok) {
        setMessage(data.error ?? `Search failed with HTTP ${response.status}.`);
        setResult(null);
      } else {
        setResult(data);
        if (!data.match) setMessage("Type at least one word or number.");
      }
    } catch {
      setMessage("The search could not reach the server. Try again in a moment.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-3">
      <form
        className="flex flex-wrap gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          run(text);
        }}
      >
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          maxLength={200}
          placeholder="Search the OCR text, e.g. toner cartridge"
          aria-label="Search words"
          className="min-w-0 flex-1 rounded border border-line bg-page px-3 py-1.5 text-sm"
        />
        <button type="submit" disabled={busy} className="rounded bg-accent px-4 py-1.5 text-sm font-medium text-white disabled:opacity-60">
          {busy ? "Searching…" : "Search"}
        </button>
      </form>
      <p className="text-xs text-ink-2">
        Try:{" "}
        {EXAMPLES.map((example, i) => (
          <span key={example}>
            {i > 0 && " · "}
            <button type="button" onClick={() => run(example)} className="underline hover:text-ink">
              {example}
            </button>
          </span>
        ))}
      </p>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="space-y-1">
          <pre className="overflow-auto rounded border border-line bg-page p-3 font-mono text-[12.5px] leading-5">{sql}</pre>
          {result?.match && (
            <p className="text-xs text-ink-2">
              :q = <code>{result.match}</code> (each word quoted as a prefix term)
            </p>
          )}
        </div>
        <div className="space-y-1">
          {message && <p className="text-sm">{message}</p>}
          {result?.match && <ResultTable columns={result.columns} rows={result.rows} />}
          {result?.match && (
            <p className="text-xs text-ink-2">
              {result.rows.length} {result.rows.length === 1 ? "match" : "matches"}
            </p>
          )}
          {!result && !message && <p className="text-sm text-ink-2">Results appear here.</p>}
        </div>
      </div>
    </div>
  );
}
