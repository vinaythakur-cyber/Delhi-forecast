// Presentation helpers: the site always shows Delhi local time (IST); the API sends UTC.
const tz = "Asia/Kolkata";

export function ist(iso: string | null | undefined, opts: Intl.DateTimeFormatOptions): string {
  if (!iso) return "–";
  return new Intl.DateTimeFormat("en-IN", { timeZone: tz, ...opts }).format(new Date(iso));
}
export const istFull = (iso: string | null | undefined) => ist(iso, { day: "numeric", month: "short", hour: "numeric", minute: "2-digit", hour12: true }) + " IST";
export const istDay = (iso: string | null | undefined) => ist(iso, { weekday: "short", day: "numeric", month: "short" });
export const istHour = (iso: string | null | undefined) => ist(iso, { weekday: "short", hour: "numeric", hour12: true });

export function ago(minutes: number | null | undefined): string {
  if (minutes === null || minutes === undefined) return "unknown";
  if (minutes < 60) return `${minutes} min ago`;
  const h = Math.floor(minutes / 60);
  if (h < 48) return `${h} h ago`;
  return `${Math.floor(h / 24)} days ago`;
}
export const fixed = (v: number | null | undefined, d = 1) => (v === null || v === undefined ? "–" : v.toFixed(d));
export const pct = (v: number | null | undefined, d = 0) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(d)}%`);
