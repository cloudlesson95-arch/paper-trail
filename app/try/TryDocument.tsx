"use client";

import { useState } from "react";
import ScoreBars from "@/components/ScoreBars";
import { CATEGORY_LABELS, TYPE_LABELS, label, type Analysis } from "@/lib/format";

const MAX_BYTES = 1024 * 1024;

type Status = { kind: "idle" } | { kind: "busy"; file: string } | { kind: "error"; message: string };

export default function TryDocument({ samples }: { samples: Analysis[] }) {
  const [result, setResult] = useState<Analysis | null>(null);
  const [fromSample, setFromSample] = useState(false);
  const [status, setStatus] = useState<Status>({ kind: "idle" });

  function pick(sample: Analysis) {
    setResult(sample);
    setFromSample(true);
    setStatus({ kind: "idle" });
  }

  async function upload(file: File) {
    if (file.type !== "image/png" && file.type !== "image/jpeg") {
      setStatus({ kind: "error", message: "Choose a PNG or JPEG image." });
      return;
    }
    if (file.size > MAX_BYTES) {
      setStatus({ kind: "error", message: `That file is ${(file.size / MAX_BYTES).toFixed(1)} MB; the limit is 1 MB.` });
      return;
    }
    setStatus({ kind: "busy", file: file.name });
    setResult(null);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": file.type },
        body: file,
      });
      const data = await response.json().catch(() => null);
      if (!response.ok || !data) {
        setStatus({ kind: "error", message: data?.error ?? `The analysis service answered with HTTP ${response.status}.` });
        return;
      }
      setResult(data as Analysis);
      setFromSample(false);
      setStatus({ kind: "idle" });
    } catch {
      setStatus({ kind: "error", message: "The analysis service could not be reached. Try again in a moment." });
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <div className="space-y-5">
        <section className="card space-y-3 p-4">
          <h2 className="font-medium">Upload a scan</h2>
          <input
            type="file"
            accept="image/png,image/jpeg"
            disabled={status.kind === "busy"}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) upload(file);
              e.target.value = "";
            }}
            className="block w-full text-sm file:mr-3 file:rounded file:border file:border-line file:bg-page file:px-3 file:py-1.5 file:text-ink"
          />
          <p className="text-xs text-ink-2">
            PNG or JPEG, up to 1 MB. The image goes to Google&apos;s Gemini API for OCR and this site keeps no copy;
            don&apos;t upload anything private.
          </p>
        </section>

        <section className="card space-y-3 p-4">
          <h2 className="font-medium">Or pick a sample</h2>
          <ul className="grid grid-cols-3 gap-2 sm:grid-cols-4">
            {samples.map((s) => (
              <li key={s.sample_id}>
                <button
                  type="button"
                  onClick={() => pick(s)}
                  className={`w-full rounded border p-1 text-left text-xs hover:border-accent ${
                    result?.sample_id === s.sample_id && fromSample ? "border-accent" : "border-line"
                  }`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element -- pre-sized gallery PNGs */}
                  <img src={s.original} alt={`Sample ${s.sample_id}`} className="aspect-[3/4] w-full rounded object-cover object-top" />
                  <span className="mt-1 block text-ink-2">{label(TYPE_LABELS, s.type)}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <section className="card min-h-64 space-y-4 p-4" aria-live="polite">
        {status.kind === "busy" && (
          <p className="text-sm">
            Analysing <span className="font-medium">{status.file}</span>: deskewing, classifying and reading the text
            with Gemini…
          </p>
        )}
        {status.kind === "error" && (
          <p className="rounded border-l-4 px-3 py-2 text-sm" style={{ background: "var(--error-bg)", borderColor: "var(--error-line)" }}>
            {status.message}
          </p>
        )}
        {status.kind === "idle" && !result && (
          <p className="text-sm text-ink-2">Upload a scan or pick a sample to see its analysis here.</p>
        )}
        {result && <AnalysisView result={result} fromSample={fromSample} />}
      </section>
    </div>
  );
}

function AnalysisView({ result, fromSample }: { result: Analysis; fromSample: boolean }) {
  return (
    <div className="space-y-4">
      {fromSample && (
        <p className="rounded border-l-4 border-accent px-3 py-2 text-sm" style={{ background: "var(--page)" }}>
          Cached result for sample <span className="font-medium">{result.sample_id}</span>: computed offline by the
          same pipeline, so no API call was made.
        </p>
      )}
      {!fromSample && result.cached && (
        <p className="rounded border-l-4 px-3 py-2 text-sm" style={{ background: "var(--notice-bg)", borderColor: "var(--notice-line)" }}>
          <span className="font-medium">Cached example, not your upload.</span> {result.note} (sample {result.sample_id})
        </p>
      )}
      {!fromSample && !result.cached && (
        <p className="rounded border-l-4 px-3 py-2 text-sm" style={{ background: "var(--page)", borderColor: "var(--ent-date)" }}>
          Live analysis of your upload.
        </p>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <figure className="space-y-1">
          {/* eslint-disable-next-line @next/next/no-img-element -- data URL or pre-sized PNG */}
          <img src={result.overlay} alt="Deskewed page with text blocks outlined" className="w-full rounded border border-line" />
          <figcaption className="text-xs text-ink-2">
            Deskewed by <span className="num">{result.skew_deg.toFixed(2)}°</span>, {result.blocks} text blocks
          </figcaption>
        </figure>
        <div className="space-y-4">
          <div>
            <h3 className="mb-1 text-sm">
              Document type: <span className="font-semibold">{label(TYPE_LABELS, result.type)}</span>
            </h3>
            <ScoreBars scores={result.type_scores} names={TYPE_LABELS} />
          </div>
          <div>
            <h3 className="mb-1 text-sm">
              Expense category: <span className="font-semibold">{label(CATEGORY_LABELS, result.category)}</span>
            </h3>
            <ScoreBars scores={result.category_scores} names={CATEGORY_LABELS} />
          </div>
        </div>
      </div>

      <div>
        <h3 className="mb-1 text-xs font-medium uppercase tracking-wide text-ink-2">OCR text ({result.model})</h3>
        <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded border border-line p-2 font-mono text-[13px] leading-6">
          {result.text || "(no text found)"}
        </pre>
      </div>
    </div>
  );
}
