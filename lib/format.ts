export const TYPE_LABELS: Record<string, string> = {
  receipt: "Receipt",
  invoice: "Invoice",
  letter: "Letter",
  order_form: "Order form",
};

export const CATEGORY_LABELS: Record<string, string> = {
  office_supplies: "Office supplies",
  travel: "Travel",
  meals: "Meals",
  utilities: "Utilities",
  software: "Software",
};

export const ENTITY_LABELS: Record<string, string> = {
  ORG: "Organisation",
  PERSON: "Person",
  DATE: "Date",
  MONEY: "Amount",
  DOC_ID: "Document number",
  EMAIL: "E-mail",
};

export const money = (value: number) =>
  value.toLocaleString("en-US", { style: "currency", currency: "USD" });

export const percent = (value: number, digits = 1) => `${(value * 100).toFixed(digits)}%`;

export const label = (labels: Record<string, string>, key: string) => labels[key] ?? key;

/** Analysis of one document, as returned by /api/analyze and cached in data/samples.json. */
export type Analysis = {
  cached?: boolean;
  note?: string | null;
  sample_id: string | null;
  type: string;
  type_scores: Record<string, number>;
  category: string;
  category_scores: Record<string, number>;
  skew_deg: number;
  blocks: number;
  text: string;
  overlay: string;
  original?: string;
  model: string;
};
