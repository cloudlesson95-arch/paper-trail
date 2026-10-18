import type { Metadata } from "next";
import ConfusionMatrix from "@/components/ConfusionMatrix";
import ResultTable from "@/components/ResultTable";
import { getDb } from "@/lib/db";
import { CATEGORY_LABELS, ENTITY_LABELS, TYPE_LABELS, label, percent } from "@/lib/format";
import { runQuery } from "@/lib/queries";

export const metadata: Metadata = { title: "Metrics" };

type Metric = { value: number; detail: Record<string, unknown> };

function loadMetrics(): Record<string, Metric> {
  const rows = getDb().prepare("SELECT name, value, detail FROM metrics").all() as {
    name: string;
    value: number;
    detail: string | null;
  }[];
  return Object.fromEntries(rows.map((r) => [r.name, { value: r.value, detail: r.detail ? JSON.parse(r.detail) : {} }]));
}

function Tile({ title, value, note }: { title: string; value: string; note: string }) {
  return (
    <div className="card p-4">
      <p className="text-sm text-ink-2">{title}</p>
      <p className="mt-1 text-3xl font-semibold">{value}</p>
      <p className="mt-1 text-xs text-ink-2">{note}</p>
    </div>
  );
}

const fixed = (value: number, digits = 3) => value.toFixed(digits);

export default function MetricsPage() {
  const m = loadMetrics();
  const doctype = m.doctype_cv_accuracy;
  const category = m.category_cv_accuracy;
  const nerLabels = [...Object.keys(ENTITY_LABELS), "ALL"];
  const ner = nerLabels.map((key) => ({ key, scores: (m[`ner_f1_${key}`]?.detail ?? {}) as Record<string, number> }));
  const cerByType = runQuery("deskew_cer_gain")!;
  const settings = [
    { key: "standard", name: "Standard scans (±8°, full resolution)" },
    { key: "hard", name: "Hard scans (12–25°, 60% resolution, more blur and noise)" },
  ];

  return (
    <div className="space-y-8">
      <section className="max-w-3xl space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Metrics</h1>
        <p className="text-ink-2">
          Measured by the offline pipeline on 120 synthetic documents (30 per type, 24 per expense category) against
          their generated ground truth. Classifiers are scored with stratified 5-fold cross-validation.
        </p>
      </section>

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Tile title="Document type" value={percent(doctype.value)} note="5-fold CV accuracy, HOG + logistic regression" />
        <Tile title="Expense category" value={percent(category.value)} note="5-fold CV accuracy, TF-IDF + logistic regression" />
        <Tile title="Entity F1" value={fixed(m.ner_f1_ALL.value, 2)} note="all six types, spaCy + EntityRuler" />
        <Tile title="OCR error rate" value={percent(m.ocr_cer_standard_deskewed.value, 2)} note="characters, deskewed standard scans" />
        <Tile title="Deskew error" value={`${fixed(m.deskew_error_deg_standard.value, 2)}°`} note="mean, against the known rotation" />
      </section>

      <section className="card space-y-4 p-4">
        <h2 className="text-lg font-semibold">Document type classifier</h2>
        <p className="max-w-3xl text-sm text-ink-2">
          {String(doctype.detail.features)} HOG features from each binarised page (cropped to its ink, resized to
          128 × 160), standardised, then multinomial logistic regression. Accuracy is{" "}
          <span className="num">{fixed(doctype.value)}</span> ± <span className="num">{fixed(doctype.detail.std as number)}</span>{" "}
          with deskewing and <span className="num">{fixed(m.doctype_cv_accuracy_raw_scans.value)}</span> on the raw scans
          {doctype.value >= 0.99 && m.doctype_cv_accuracy_raw_scans.value >= 0.99
            ? ": the four layouts differ so much that this task saturates either way."
            : "."}
        </p>
        <ConfusionMatrix labels={doctype.detail.labels as string[]} matrix={doctype.detail.confusion as number[][]} names={TYPE_LABELS} />
      </section>

      <section className="card space-y-4 p-4">
        <h2 className="text-lg font-semibold">Deskewing and OCR</h2>
        <div className="overflow-x-auto">
          <table className="text-sm">
            <thead className="text-left text-ink-2">
              <tr>
                <th className="py-1 pr-6 font-medium">Scan setting</th>
                <th className="py-1 pr-6 text-right font-medium">Deskew error (mean / max)</th>
                <th className="py-1 pr-6 text-right font-medium">OCR CER raw</th>
                <th className="py-1 pr-6 text-right font-medium">OCR CER deskewed</th>
              </tr>
            </thead>
            <tbody>
              {settings.map(({ key, name }) => (
                <tr key={key} className="border-t border-line">
                  <td className="py-1 pr-6">{name}</td>
                  <td className="num py-1 pr-6 text-right">
                    {fixed(m[`deskew_error_deg_${key}`].value, 2)}° / {fixed(m[`deskew_error_deg_${key}`].detail.max as number, 2)}°
                  </td>
                  <td className="num py-1 pr-6 text-right">{percent(m[`ocr_cer_${key}_raw`].value, 2)}</td>
                  <td className="num py-1 pr-6 text-right">{percent(m[`ocr_cer_${key}_deskewed`].value, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="max-w-3xl text-sm text-ink-2">
          Character error rate of {String(m.ocr_cer_standard_raw.detail.model)} against the ground-truth text, after
          collapsing whitespace. By document type (from the <code>deskew_cer_gain</code> query; gain is in percentage
          points):
        </p>
        <ResultTable columns={cerByType.columns} rows={cerByType.rows} />
      </section>

      <section className="card space-y-4 p-4">
        <h2 className="text-lg font-semibold">Entity extraction</h2>
        <p className="max-w-3xl text-sm text-ink-2">
          Precision, recall and F1 per entity type on the OCR text, matching each document&apos;s extracted values
          against its ground truth. spaCy&apos;s small English model alone, then with an EntityRuler handling amounts,
          dates, document numbers and e-mails (the statistical model keeps people and organisations).
        </p>
        <div className="overflow-x-auto">
          <table className="text-sm">
            <thead className="text-ink-2">
              <tr>
                <th rowSpan={2} className="py-1 pr-6 text-left font-medium">Type</th>
                <th rowSpan={2} className="py-1 pr-6 text-right font-medium">Support</th>
                <th colSpan={3} className="px-3 py-1 text-center font-medium">spaCy only</th>
                <th colSpan={3} className="px-3 py-1 text-center font-medium">spaCy + EntityRuler</th>
              </tr>
              <tr>
                {["P", "R", "F1", "P", "R", "F1"].map((h, i) => (
                  <th key={i} className="px-3 py-1 text-right font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ner.map((row) => (
                <tr key={row.key} className={`border-t border-line ${row.key === "ALL" ? "font-semibold" : ""}`}>
                  <td className="py-1 pr-6">{row.key === "ALL" ? "All types" : label(ENTITY_LABELS, row.key)}</td>
                  <td className="num py-1 pr-6 text-right">{row.scores.support}</td>
                  {(["spacy_only", "with_rules"] as const).flatMap((setup) =>
                    (["precision", "recall", "f1"] as const).map((metric) => (
                      <td key={`${setup}-${metric}`} className="num px-3 py-1 text-right">
                        {fixed(row.scores[`${setup}_${metric}`] ?? 0, 2)}
                      </td>
                    )),
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="card space-y-4 p-4">
        <h2 className="text-lg font-semibold">Expense category classifier</h2>
        <p className="max-w-3xl text-sm text-ink-2">
          TF-IDF ({String(category.detail.vocabulary)} terms, sublinear term frequency, terms in at least two documents)
          and logistic regression on the OCR text. Accuracy <span className="num">{fixed(category.value)}</span> ±{" "}
          <span className="num">{fixed(category.detail.std as number)}</span>.
        </p>
        <ConfusionMatrix labels={category.detail.labels as string[]} matrix={category.detail.confusion as number[][]} names={CATEGORY_LABELS} />
      </section>
    </div>
  );
}
