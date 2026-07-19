/** Formatters shared across the dashboard. */

export type CostTier = "good" | "warn" | "crit";

/** <$0.01 green, <$0.10 yellow, ≥$0.10 red. */
export function costTier(usd: number): CostTier {
  if (usd < 0.01) return "good";
  if (usd < 0.1) return "warn";
  return "crit";
}

/** Adaptive-precision dollars: big values coarse, sub-cent values exact. */
export function fmtUSD(value: number | string | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const n = typeof value === "string" ? Number(value) : value;
  if (!Number.isFinite(n)) return "—";
  if (n === 0) return "$0";
  const abs = Math.abs(n);
  let digits: number;
  if (abs >= 100) digits = 0;
  else if (abs >= 1) digits = 2;
  else if (abs >= 0.01) digits = 4;
  else digits = 6;
  return `$${n.toLocaleString("en-US", {
    minimumFractionDigits: Math.min(digits, 2),
    maximumFractionDigits: digits,
  })}`;
}

export function fmtTokens(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(n >= 10_000 ? 0 : 1)}k`;
  return String(n);
}

export function fmtDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1) return `${ms.toFixed(2)}ms`;
  if (ms < 1_000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1_000).toFixed(2)}s`;
  const m = Math.floor(ms / 60_000);
  return `${m}m ${Math.round((ms % 60_000) / 1_000)}s`;
}

export function relTime(iso: string, now: number = Date.now()): string {
  const t = new Date(iso).getTime();
  const s = Math.max(0, Math.round((now - t) / 1_000));
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3_600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86_400) return `${Math.floor(s / 3_600)}h ago`;
  if (s < 7 * 86_400) return `${Math.floor(s / 86_400)}d ago`;
  return new Date(iso).toLocaleDateString();
}

export function fmtClock(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return `${d.toLocaleTimeString()} · ${d.toLocaleDateString()}`;
}

export function fmtPercent(fraction: number): string {
  return `${(fraction * 100).toFixed(fraction > 0 && fraction < 0.01 ? 2 : 1)}%`;
}
