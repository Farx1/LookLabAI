export class ApiError extends Error {
  status: number;
  code?: string;
  detail: unknown;
  constructor(status: number, detail: unknown) {
    const d = detail as { message?: string; code?: string } | string | undefined;
    const msg = typeof d === "string" ? d : d?.message ?? (Array.isArray(detail) ? "Invalid input" : `Request failed (${status})`);
    super(msg);
    this.status = status;
    this.detail = detail;
    this.code = typeof d === "object" && d ? d.code : undefined;
  }
}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { "X-Requested-With": "looklab" };
  let payload: BodyInit | undefined;
  if (body instanceof FormData) payload = body;
  else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(url, { method, headers, body: payload, credentials: "same-origin" });
  const text = await res.text();
  const data = text ? (() => { try { return JSON.parse(text); } catch { return text; } })() : null;
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? data);
  return data as T;
}

export const api = {
  get: <T>(url: string) => request<T>("GET", url),
  post: <T>(url: string, body?: unknown) => request<T>("POST", url, body ?? {}),
  put: <T>(url: string, body?: unknown) => request<T>("PUT", url, body),
  patch: <T>(url: string, body?: unknown) => request<T>("PATCH", url, body),
  del: <T>(url: string) => request<T>("DELETE", url),
};

export const fetcher = <T,>(url: string) => api.get<T>(url);

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) {
    const d = e.detail as { errors?: { filename: string; error: string }[] } | undefined;
    if (d && typeof d === "object" && "errors" in d && d.errors) return d.errors.map((x) => `${x.filename}: ${x.error}`).join("; ");
    if (Array.isArray(e.detail)) return (e.detail as { msg: string; loc: string[] }[]).map((x) => `${x.loc.at(-1)}: ${x.msg}`).join("; ");
    return e.message;
  }
  return e instanceof Error ? e.message : String(e);
}
