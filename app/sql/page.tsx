import type { Metadata } from "next";
import ResultTable from "@/components/ResultTable";
import { QUERIES, SEARCH, runQuery } from "@/lib/queries";
import SearchBox from "./SearchBox";

export const metadata: Metadata = { title: "SQL explorer" };

export default function SqlPage() {
  return (
    <div className="space-y-8">
      <section className="max-w-3xl space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">SQL explorer</h1>
        <p className="text-ink-2">
          The pipeline loads everything into one SQLite file: <code>documents</code>, their text <code>blocks</code>,
          extracted <code>entities</code>, per-scan <code>ocr_runs</code> and evaluation <code>metrics</code>, with
          foreign keys, indexes and an FTS5 index over the OCR text. Each named query below is a file in{" "}
          <code>sql/queries/</code>, shown next to its result. The database is opened read-only and only these files
          can run; each is also available as JSON from <code>/api/query?id=…</code>.
        </p>
      </section>

      <section className="card space-y-3 p-4">
        <h2 className="text-lg font-semibold">{SEARCH.title}</h2>
        <p className="text-sm text-ink-2">{SEARCH.description}</p>
        <SearchBox sql={SEARCH.sql} />
      </section>

      {QUERIES.map((query) => {
        const result = runQuery(query.id)!;
        return (
          <section key={query.id} id={query.id} className="card space-y-3 p-4">
            <div className="flex flex-wrap items-baseline gap-x-3">
              <h2 className="text-lg font-semibold">{query.title}</h2>
              <a href={`/api/query?id=${query.id}`} className="text-xs text-ink-2 hover:underline">
                {query.id}.sql · JSON
              </a>
            </div>
            <p className="text-sm text-ink-2">{query.description}</p>
            <div className="grid gap-4 lg:grid-cols-2">
              <pre className="overflow-auto rounded border border-line bg-page p-3 font-mono text-[12.5px] leading-5">
                {query.sql}
              </pre>
              <div className="space-y-1">
                <ResultTable columns={result.columns} rows={result.rows} />
                <p className="text-xs text-ink-2">
                  {result.rows.length} {result.rows.length === 1 ? "row" : "rows"}
                </p>
              </div>
            </div>
          </section>
        );
      })}
    </div>
  );
}
