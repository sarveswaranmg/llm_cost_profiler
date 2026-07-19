/** Tiny API client. Base URL is configurable via VITE_TOKENLENS_API;
 * the default (empty string) means same-origin relative URLs, which works
 * both in production (the FastAPI server serves this app) and in dev (the
 * Vite proxy forwards /api and /ws to http://localhost:8321).
 */

const API_BASE: string =
  (import.meta.env.VITE_TOKENLENS_API as string | undefined)?.replace(/\/$/, "") ?? "";

export function apiUrl(
  path: string,
  params?: Record<string, string | number | undefined>,
): string {
  const query = params
    ? Object.entries(params)
        .filter(([, v]) => v !== undefined && v !== "")
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
        .join("&")
    : "";
  return `${API_BASE}${path}${query ? `?${query}` : ""}`;
}

export async function getJSON<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`GET ${url} → ${resp.status} ${resp.statusText}`);
  }
  return (await resp.json()) as T;
}

export function liveWsUrl(): string {
  if (API_BASE) {
    return `${API_BASE.replace(/^http/, "ws")}/ws/live`;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/live`;
}
