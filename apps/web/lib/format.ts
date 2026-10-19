export const money = (v: number | null | undefined, cur = "EUR") =>
  v == null ? "—" : new Intl.NumberFormat(undefined, { style: "currency", currency: cur, maximumFractionDigits: v % 1 ? 2 : 0 }).format(v);

export const usd = (v: number | null | undefined) => (v == null ? "—" : `$${v < 0.01 && v > 0 ? v.toFixed(4) : v.toFixed(2)}`);

export const label = (s: string | null | undefined) => (s ? s.replace(/_/g, " ").replace(/-/g, " ") : "");

export const pct = (v: number) => `${Math.round(v * 100)}`;

export function garmentTitle(g: { name: string | null; subcategory: string | null; category: string; colors: string[] }) {
  return g.name || [g.colors[0], g.subcategory || (g.category !== "unknown" ? g.category : "garment")].filter(Boolean).join(" ");
}

export function timeAgo(iso: string) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}
