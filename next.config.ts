import type { NextConfig } from "next";

// The database and query files are read at runtime by /api/query (and by pages when a
// request renders them dynamically), so they must be traced into the server functions.
const DATA_FILES = ["./data/paper_trail.db", "./sql/queries/*.sql"];

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  outputFileTracingIncludes: {
    "/api/query": DATA_FILES,
    "/*": DATA_FILES,
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
