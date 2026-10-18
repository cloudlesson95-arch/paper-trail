import Database from "better-sqlite3";
import path from "node:path";

let db: Database.Database | undefined;

/** The database built by the pipeline (data/paper_trail.db), opened read-only. */
export function getDb(): Database.Database {
  if (!db) {
    db = new Database(path.join(process.cwd(), "data", "paper_trail.db"), {
      readonly: true,
      fileMustExist: true,
    });
    db.pragma("query_only = ON");
  }
  return db;
}
