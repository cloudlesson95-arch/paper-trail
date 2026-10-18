import type { Metadata } from "next";
import samples from "@/data/samples.json";
import type { Analysis } from "@/lib/format";
import TryDocument from "./TryDocument";

export const metadata: Metadata = { title: "Try a document" };

export default function TryPage() {
  return (
    <div className="space-y-6">
      <section className="max-w-3xl space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Try a document</h1>
        <p className="text-ink-2">
          Pick a sample to see its stored analysis, or upload a scan of your own (PNG or JPEG, 1 MB at most). An
          upload runs live: OpenCV deskews it and finds the text blocks, the exported HOG model classifies its type,
          Gemini reads it and the exported TF-IDF model picks the expense category. Live analyses are limited to 5 an
          hour per visitor and a daily total; past that you get a stored example, clearly marked.
        </p>
      </section>
      <TryDocument samples={samples as Analysis[]} />
    </div>
  );
}
