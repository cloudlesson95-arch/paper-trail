import { EntityLegend, EntityList, EntityText, type Entity } from "@/components/EntityText";
import { getDb } from "@/lib/db";
import { CATEGORY_LABELS, TYPE_LABELS, label, money, percent } from "@/lib/format";

type GalleryDoc = {
  id: string;
  type: string;
  category: string;
  vendor: string;
  number: string;
  total: number;
  issued_on: string;
  ocr_text: string;
  summary: string;
  cer: number;
  skew_deg: number;
  blocks: number;
};

function loadGallery() {
  const db = getDb();
  const docs = db
    .prepare(
      `SELECT d.id, d.type, d.category, d.vendor, d.number, d.total, d.issued_on, d.ocr_text, d.summary,
              d.cer, d.skew_deg, COUNT(b.id) AS blocks
       FROM documents AS d LEFT JOIN blocks AS b ON b.doc_id = d.id
       WHERE d.in_gallery = 1
       GROUP BY d.id
       ORDER BY d.type, d.id`,
    )
    .all() as GalleryDoc[];
  const entities = db
    .prepare(
      `SELECT doc_id, label, text, start_char, end_char FROM entities
       WHERE doc_id IN (SELECT id FROM documents WHERE in_gallery = 1)`,
    )
    .all() as (Entity & { doc_id: string })[];
  return docs.map((doc) => ({ ...doc, entities: entities.filter((e) => e.doc_id === doc.id) }));
}

export default function GalleryPage() {
  const docs = loadGallery();
  return (
    <div className="space-y-8">
      <section className="max-w-3xl space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight">Scanned paperwork, read and indexed</h1>
        <p className="text-ink-2">
          Each document is a synthetic scan: skewed, blurred, noisy and unevenly lit. OpenCV binarises it with an
          adaptive threshold, measures the skew with <code>minAreaRect</code> and rotates it back, and finds text
          blocks with dilation and contours. A HOG + logistic regression classifier names the document type, Gemini
          reads the text, spaCy and an EntityRuler mark the entities, and TextRank picks a two-sentence summary.
          Everything lands in SQLite.
        </p>
        <EntityLegend />
      </section>

      {docs.map((doc) => (
        <article key={doc.id} className="card space-y-4 p-4" id={doc.id}>
          <header className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
            <h2 className="text-lg font-semibold">{doc.vendor}</h2>
            <span className="rounded bg-line px-2 py-0.5 text-xs font-medium">{label(TYPE_LABELS, doc.type)}</span>
            <span className="text-sm text-ink-2">
              {label(CATEGORY_LABELS, doc.category)} · {doc.number} · {doc.issued_on} ·{" "}
              <span className="num">{money(doc.total)}</span>
            </span>
            <span className="ml-auto text-xs text-ink-2">{doc.id}</span>
          </header>

          <div className="grid gap-4 lg:grid-cols-[1fr_1fr_1.3fr]">
            <figure className="space-y-1">
              {/* eslint-disable-next-line @next/next/no-img-element -- pre-sized gallery PNGs */}
              <img src={`/samples/${doc.id}_original.png`} alt={`Original scan of ${doc.id}`} loading="lazy" className="w-full rounded border border-line" />
              <figcaption className="text-xs text-ink-2">Original scan</figcaption>
            </figure>
            <figure className="space-y-1">
              {/* eslint-disable-next-line @next/next/no-img-element -- pre-sized gallery PNGs */}
              <img src={`/samples/${doc.id}_overlay.png`} alt={`Deskewed ${doc.id} with text blocks outlined`} loading="lazy" className="w-full rounded border border-line" />
              <figcaption className="text-xs text-ink-2">
                Deskewed by <span className="num">{doc.skew_deg.toFixed(2)}°</span>, {doc.blocks} text blocks
              </figcaption>
            </figure>
            <div className="min-w-0 space-y-3">
              <div>
                <h3 className="text-xs font-medium uppercase tracking-wide text-ink-2">Summary (TextRank)</h3>
                <p className="text-sm">{doc.summary || "—"}</p>
              </div>
              <div>
                <h3 className="text-xs font-medium uppercase tracking-wide text-ink-2">
                  OCR text · character error rate <span className="num">{percent(doc.cer, 2)}</span>
                </h3>
                <div className="max-h-80 overflow-auto rounded border border-line p-2">
                  <EntityText text={doc.ocr_text} entities={doc.entities} />
                </div>
              </div>
              <EntityList entities={doc.entities} />
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}
