import type { NextRequest } from "next/server";
import { QUERIES, SEARCH, runQuery, search } from "@/lib/queries";

const meta = ({ id, title, description, sql }: (typeof QUERIES)[number]) => ({ id, title, description, sql });

/**
 * GET /api/query            -> the list of named queries
 * GET /api/query?id=<id>    -> one named query's SQL and result
 * GET /api/query?q=<words>  -> full-text search over the OCR text
 *
 * Only the fixed SQL files run, on a read-only connection; search words are bound as a
 * parameter, so a request can neither change the database nor run its own SQL.
 */
export async function GET(request: NextRequest) {
  const params = request.nextUrl.searchParams;
  const q = params.get("q");
  if (q !== null) {
    if (q.length > 200) return Response.json({ error: "Search text is limited to 200 characters." }, { status: 400 });
    return Response.json({ ...meta(SEARCH), ...search(q) });
  }

  const id = params.get("id");
  if (id === null) return Response.json({ queries: QUERIES.map(({ id, title, description }) => ({ id, title, description })) });

  const query = QUERIES.find((query) => query.id === id);
  if (!query) {
    return Response.json(
      { error: `Unknown query "${id}".`, ids: QUERIES.map((query) => query.id) },
      { status: 404 },
    );
  }
  return Response.json({ ...meta(query), ...runQuery(id) });
}
