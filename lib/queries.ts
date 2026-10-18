import fs from "node:fs";
import path from "node:path";
import { getDb } from "./db";

export type NamedQuery = { id: string; title: string; description: string; sql: string };
export type QueryResult = { columns: string[]; rows: unknown[][] };

const DIR = path.join(process.cwd(), "sql", "queries");

/** A query file starts with its title as a comment line, then comment lines describing it. */
function parse(id: string, sql: string): NamedQuery {
  const comments: string[] = [];
  for (const line of sql.split(/\r?\n/)) {
    if (!line.startsWith("--")) break;
    comments.push(line.replace(/^--\s?/, ""));
  }
  return { id, title: comments[0] ?? id, description: comments.slice(1).join(" "), sql: sql.trim() };
}

const ALL: NamedQuery[] = fs
  .readdirSync(DIR)
  .filter((file) => file.endsWith(".sql"))
  .sort()
  .map((file) => parse(file.slice(0, -4), fs.readFileSync(path.join(DIR, file), "utf8")));

const ORDER = ["spend_by_category_month", "top_vendors", "ner_misses", "entity_cooccurrence", "deskew_cer_gain"];

/** The named analytics queries, in display order; the search template is separate. */
export const QUERIES = ALL.filter((q) => q.id !== "search").sort(
  (a, b) => (ORDER.indexOf(a.id) + 1 || 99) - (ORDER.indexOf(b.id) + 1 || 99),
);
export const SEARCH = ALL.find((q) => q.id === "search")!;

function execute(sql: string, params?: Record<string, string>): QueryResult {
  const statement = getDb().prepare(sql).raw(true);
  const rows = (params ? statement.all(params) : statement.all()) as unknown[][];
  return { columns: statement.columns().map((c) => c.name), rows };
}

/** Run a named query by id; only the files in sql/queries can run. */
export function runQuery(id: string): QueryResult | undefined {
  const query = QUERIES.find((q) => q.id === id);
  return query && execute(query.sql);
}

/**
 * Turn free text into a safe FTS5 query: each word becomes a quoted prefix term, so
 * FTS5 syntax in the input can't break or widen the search.
 */
export function ftsQuery(input: string): string | null {
  const words = input.match(/[\p{L}\p{N}]+/gu)?.slice(0, 8) ?? [];
  return words.length ? words.map((w) => `"${w}"*`).join(" ") : null;
}

export function search(input: string): QueryResult & { match: string | null } {
  const match = ftsQuery(input);
  if (!match) return { columns: [], rows: [], match };
  return { ...execute(SEARCH.sql, { q: match }), match };
}
