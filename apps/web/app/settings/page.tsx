"use client";
import { useEffect, useState } from "react";
import useSWR from "swr";
import { Download, Loader2, ShieldCheck, Trash2 } from "lucide-react";
import type { CostReport, SystemInfo, UserSettings } from "@looklab/shared-types";
import { Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, Input, Label, Select, Switch, Tabs, TabsContent, TabsList, TabsTrigger, cn } from "@looklab/ui";
import { api, errorMessage, fetcher } from "@/lib/api";
import { useVocabulary } from "@/lib/hooks";
import { label, timeAgo, usd } from "@/lib/format";
import { Notice, PageHeader } from "@/components/page-header";

function Chips({ options, value, onChange, hex }: { options: string[]; value: string[]; onChange: (v: string[]) => void; hex?: Record<string, string> }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((o) => {
        const on = value.includes(o);
        return (
          <button key={o} type="button" onClick={() => onChange(on ? value.filter((x) => x !== o) : [...value, o])}
            className={cn("flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs capitalize", on ? "border-chocolate bg-chocolate text-cream" : "border-line text-muted")}>
            {hex && <span className="size-2.5 rounded-full ring-1 ring-line" style={{ background: hex[o] }} />}
            {o}
          </button>
        );
      })}
    </div>
  );
}

function Preferences({ s, save }: { s: UserSettings; save: (b: Partial<UserSettings>) => Promise<void> }) {
  const { data: vocab } = useVocabulary();
  const [p, setP] = useState(s.preferences);
  const { data: learned, mutate } = useSWR<{ events: number; top_features: [string, number][] }>("/api/preferences/learned", fetcher);
  useEffect(() => setP(s.preferences), [s.preferences]);
  const set = <K extends keyof typeof p>(k: K, v: (typeof p)[K]) => setP((x) => ({ ...x, [k]: v }));
  return (
    <div className="grid gap-5 *:min-w-0 lg:grid-cols-[1fr_340px]">
      <Card>
        <CardHeader><CardTitle>Your style</CardTitle><CardDescription>Used as soft preferences; disliked colors are never suggested.</CardDescription></CardHeader>
        <CardContent className="space-y-5">
          <div className="grid gap-2"><Label>Colors you love</Label><Chips options={vocab?.colors ?? []} hex={vocab?.color_hex} value={p.liked_colors} onChange={(v) => set("liked_colors", v)} /></div>
          <div className="grid gap-2"><Label>Colors to avoid</Label><Chips options={vocab?.colors ?? []} hex={vocab?.color_hex} value={p.disliked_colors} onChange={(v) => set("disliked_colors", v)} /></div>
          <div className="grid gap-2"><Label>Styles you love</Label><Chips options={vocab?.styles ?? []} value={p.liked_styles} onChange={(v) => set("liked_styles", v)} /></div>
          <div className="grid gap-2"><Label>Styles to avoid</Label><Chips options={vocab?.styles ?? []} value={p.disliked_styles} onChange={(v) => set("disliked_styles", v)} /></div>
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="grid gap-1.5"><Label>Usual fit</Label>
              <Select value={p.fit_preference ?? ""} onChange={(e) => set("fit_preference", e.target.value || null)}>
                <option value="">No preference</option>
                {vocab?.fits.map((f) => <option key={f} value={f}>{f}</option>)}
              </Select>
            </div>
            <div className="grid gap-1.5"><Label>Default budget</Label><Input inputMode="decimal" value={p.default_budget ?? ""} onChange={(e) => set("default_budget", e.target.value ? Number(e.target.value) : null)} placeholder="none" /></div>
            <div className="grid gap-1.5"><Label>Currency</Label><Input value={p.currency} maxLength={3} onChange={(e) => set("currency", e.target.value.toUpperCase())} /></div>
          </div>
          <div className="grid gap-4 sm:grid-cols-3">
            {["top", "bottom", "shoes"].map((k) => (
              <div key={k} className="grid gap-1.5"><Label>{k} size</Label>
                <Input value={p.sizes[k] ?? ""} onChange={(e) => set("sizes", { ...p.sizes, [k]: e.target.value })} /></div>
            ))}
          </div>
          <Button onClick={() => save({ preferences: p })}>Save preferences</Button>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>What I learned</CardTitle><CardDescription>From {learned?.events ?? 0} likes, dislikes and saves.</CardDescription></CardHeader>
        <CardContent className="space-y-1.5">
          {learned?.top_features.length ? learned.top_features.map(([f, w]) => (
            <div key={f} className="flex items-center gap-2 text-sm">
              <span className="w-32 truncate capitalize">{f.split(":")[1]}</span>
              <span className="text-[10px] uppercase text-muted">{f.split(":")[0]}</span>
              <span className="ml-auto h-1.5 w-20 overflow-hidden rounded-full bg-sand">
                <span className={cn("block h-full", w > 0 ? "bg-chocolate" : "bg-burgundy")} style={{ width: `${Math.min(100, Math.abs(w) * 50)}%` }} />
              </span>
            </div>
          )) : <p className="text-sm text-muted">Like or dislike a few outfits to teach me.</p>}
          {!!learned?.events && <Button size="sm" variant="ghost" className="mt-2" onClick={async () => { await api.del("/api/preferences/learned"); mutate(); }}>Reset learning</Button>}
        </CardContent>
      </Card>
    </div>
  );
}

function Privacy({ s, save }: { s: UserSettings; save: (b: Partial<UserSettings>) => Promise<void> }) {
  const [pv, setPv] = useState(s.privacy);
  const [sp, setSp] = useState(s.spending.monthly_limit_usd);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setPv(s.privacy); setSp(s.spending.monthly_limit_usd); }, [s]);
  async function deleteAll() {
    if (!confirm("Delete ALL your photos, garments, outfits, try-on images and history? This cannot be undone.")) return;
    setBusy(true);
    await api.del("/api/me/data");
    setBusy(false);
    location.href = "/";
  }
  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2"><ShieldCheck className="size-5" /> Privacy</CardTitle>
          <CardDescription>Photos are stored privately on this machine with location data removed. Nothing is public.</CardDescription></CardHeader>
        <CardContent className="space-y-4">
          <label className="flex items-start justify-between gap-4 rounded-xl bg-cream p-4 text-sm">
            <span><b>Allow processing outside this computer</b><br /><span className="text-muted">Required for remote GPU workers and paid APIs. Each paid request still needs your approval.</span></span>
            <Switch checked={pv.allow_external_processing} onCheckedChange={(v) => setPv({ ...pv, allow_external_processing: v })} />
          </label>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="grid gap-1.5"><Label>Auto-delete unsaved try-ons after</Label>
              <Select value={pv.retention_days_generated ?? ""} onChange={(e) => setPv({ ...pv, retention_days_generated: e.target.value ? Number(e.target.value) : null })}>
                <option value="">Keep until I delete</option>
                {[1, 7, 30, 90].map((d) => <option key={d} value={d}>{d} day{d > 1 ? "s" : ""}</option>)}
              </Select></div>
            <div className="grid gap-1.5"><Label>Auto-delete my photos after</Label>
              <Select value={pv.retention_days_photos ?? ""} onChange={(e) => setPv({ ...pv, retention_days_photos: e.target.value ? Number(e.target.value) : null })}>
                <option value="">Keep until I delete</option>
                {[7, 30, 90, 365].map((d) => <option key={d} value={d}>{d} days</option>)}
              </Select></div>
          </div>
          <Button onClick={() => save({ privacy: pv })}>Save privacy settings</Button>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Spending limit</CardTitle>
          <CardDescription>Paid AI is off by default. The effective limit is the lower of yours and the server cap ({usd(s.spending.global_limit_usd)}).</CardDescription></CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-1.5"><Label>My monthly limit (USD)</Label><Input inputMode="decimal" value={sp} onChange={(e) => setSp(Number(e.target.value) || 0)} /></div>
          <p className="text-sm">Effective limit: <b>{usd(s.spending.effective_limit_usd)}</b>/month</p>
          <Button variant="outline" onClick={() => save({ spending: { ...s.spending, monthly_limit_usd: sp } })}>Save limit</Button>
        </CardContent>
      </Card>
      <Card className="lg:col-span-2">
        <CardHeader><CardTitle>Your data</CardTitle></CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          <Button variant="outline" asChild><a href="/api/me/export" download="looklab-export.json"><Download /> Export my data (JSON)</a></Button>
          <Button variant="danger" onClick={deleteAll} disabled={busy}>{busy ? <Loader2 className="animate-spin" /> : <Trash2 />} Delete everything</Button>
        </CardContent>
      </Card>
    </div>
  );
}

function Engines() {
  const { data: sys } = useSWR<SystemInfo>("/api/system", fetcher);
  if (!sys) return <Loader2 className="animate-spin" />;
  const hw = sys.hardware;
  return (
    <div className="space-y-5">
      <Card>
        <CardHeader><CardTitle>This machine</CardTitle></CardHeader>
        <CardContent className="grid gap-2 text-sm sm:grid-cols-4">
          <p><span className="text-muted">Mode</span><br />{hw.execution_mode === "local_gpu" ? "Local GPU" : "CPU only"}</p>
          <p><span className="text-muted">CPU</span><br />{hw.cpu_count} cores · {hw.ram_gb ?? "?"} GB</p>
          <p><span className="text-muted">GPU</span><br />{hw.torch.cuda ? `${hw.torch.gpu_name} (${Math.round((hw.torch.gpu_total_mb ?? 0) / 1024)} GB)` : hw.torch.torch ? "none detected" : "PyTorch not installed"}</p>
          <p><span className="text-muted">FASHN parser mode</span><br />{sys.fashn.parser_mode === "parser_free" ? "parser-free (flat-lay only)" : "full (non-commercial)"}</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Virtual try-on engines</CardTitle><CardDescription>Automatic mode only uses free, local, open-source engines. Anything billed must be chosen per request.</CardDescription></CardHeader>
        <CardContent className="divide-y divide-line">
          {sys.tryon_providers.map((p) => (
            <div key={p.name} className="flex flex-col gap-1 py-3 sm:flex-row sm:items-center sm:gap-4">
              <div className="flex-1">
                <p className="flex flex-wrap items-center gap-1.5 text-sm font-medium">{label(p.name)}
                  <Badge variant="outline">{p.role}</Badge>
                  {p.paid && <Badge variant="accent">paid</Badge>}
                  {p.sends_images_externally && <Badge variant="warning">sends photos out</Badge>}
                </p>
                <p className="text-xs text-muted">{p.model} · {p.description}</p>
              </div>
              <Badge variant={!p.enabled ? "outline" : p.available ? "success" : "warning"}>{!p.enabled ? "disabled in config" : p.available ? "ready" : "not available"}</Badge>
              {p.enabled && !p.available && <p className="text-xs text-muted sm:w-64">{p.reason}</p>}
            </div>
          ))}
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Garment analysis</CardTitle><CardDescription>Colors are measured from pixels; local vision models add style details when installed.</CardDescription></CardHeader>
        <CardContent className="divide-y divide-line">
          {sys.analyzers.map((a) => (
            <div key={a.name} className="flex items-center justify-between gap-4 py-2.5 text-sm">
              <span>{label(a.name)}</span>
              <span className="text-right text-xs text-muted">{a.reason}</span>
              <Badge variant={a.available ? "success" : "outline"}>{a.available ? "ready" : "off"}</Badge>
            </div>
          ))}
        </CardContent>
      </Card>
      <Notice>Setup guides: <code>docs/TRYON_SETUP.md</code> (local GPU, remote GPU worker) and <code>docs/VISION_MODELS.md</code> (Ollama Qwen3-VL, Florence-2). Policy lives in <code>config/ai.yaml</code>.</Notice>
    </div>
  );
}

function Costs() {
  const { data } = useSWR<CostReport>("/api/costs?days=30", fetcher);
  if (!data) return <Loader2 className="animate-spin" />;
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-2xl border border-line bg-paper p-4"><p className="text-xs text-muted">Paid spend this month</p><p className="font-serif text-3xl">{usd(data.month_spend_usd)}</p></div>
        <div className="rounded-2xl border border-line bg-paper p-4"><p className="text-xs text-muted">Monthly limit</p><p className="font-serif text-3xl">{usd(data.effective_limit_usd)}</p></div>
        <div className="rounded-2xl border border-line bg-paper p-4"><p className="text-xs text-muted">AI runs (30 d)</p><p className="font-serif text-3xl">{data.by_provider.reduce((s, r) => s + r.runs, 0)}</p></div>
        <div className="rounded-2xl border border-line bg-paper p-4"><p className="text-xs text-muted">Served from cache</p><p className="font-serif text-3xl">{data.by_provider.reduce((s, r) => s + r.cache_hits, 0)}</p></div>
      </div>
      <Card>
        <CardHeader><CardTitle>By engine (last 30 days)</CardTitle></CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-muted"><tr><th className="py-2">Task</th><th>Engine</th><th>Runs</th><th>Cache hits</th><th>Avg time</th><th>Peak VRAM</th><th>Cost</th></tr></thead>
            <tbody>
              {data.by_provider.map((r) => (
                <tr key={r.task + r.provider} className="border-t border-line">
                  <td className="py-2">{label(r.task)}</td><td>{label(r.provider)}</td><td>{r.runs}</td><td>{r.cache_hits}</td>
                  <td>{r.avg_execution_seconds != null ? `${r.avg_execution_seconds.toFixed(2)} s` : "—"}</td>
                  <td>{r.max_gpu_memory_mb ? `${Math.round(r.max_gpu_memory_mb)} MB` : "—"}</td><td>{usd(r.estimated_cost_usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle>Recent runs</CardTitle><CardDescription>Measured on this machine — not estimates.</CardDescription></CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-muted"><tr><th className="py-2">When</th><th>Task</th><th>Engine</th><th>Device</th><th>Time</th><th>Load</th><th>VRAM</th><th>Cost</th><th>Status</th></tr></thead>
            <tbody>
              {data.recent.slice(0, 50).map((r) => (
                <tr key={r.id} className="border-t border-line">
                  <td className="py-1.5">{timeAgo(r.created_at)}</td><td>{label(r.task)}</td><td>{r.provider}</td><td>{r.device}</td>
                  <td>{r.execution_time != null ? `${r.execution_time.toFixed(2)} s` : r.cache_hit ? "cache" : "—"}</td>
                  <td>{r.model_load_seconds ? `${r.model_load_seconds.toFixed(1)} s` : "—"}</td>
                  <td>{r.gpu_memory_peak_mb ? `${Math.round(r.gpu_memory_peak_mb)} MB` : "—"}</td>
                  <td>{usd(r.actual_cost_usd ?? r.estimated_cost_usd)}</td><td>{r.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </div>
  );
}

export default function SettingsPage() {
  const { data: s, mutate } = useSWR<UserSettings>("/api/settings", fetcher);
  const [msg, setMsg] = useState<{ tone: "success" | "error"; text: string } | null>(null);
  async function save(body: Partial<UserSettings>) {
    try {
      await mutate(await api.put<UserSettings>("/api/settings", body), false);
      setMsg({ tone: "success", text: "Saved." });
    } catch (e) {
      setMsg({ tone: "error", text: errorMessage(e) });
    }
  }
  return (
    <div>
      <PageHeader eyebrow="Settings & privacy" title="Make it yours" />
      {msg && <div className="mb-4"><Notice tone={msg.tone}>{msg.text}</Notice></div>}
      <Tabs defaultValue="style">
        <TabsList className="scrollbar-none mb-6 max-w-full overflow-x-auto">
          <TabsTrigger value="style">Style</TabsTrigger>
          <TabsTrigger value="privacy">Privacy & spending</TabsTrigger>
          <TabsTrigger value="engines">AI engines</TabsTrigger>
          <TabsTrigger value="costs">Costs & performance</TabsTrigger>
        </TabsList>
        <TabsContent value="style">{s ? <Preferences s={s} save={save} /> : <Loader2 className="animate-spin" />}</TabsContent>
        <TabsContent value="privacy">{s ? <Privacy s={s} save={save} /> : <Loader2 className="animate-spin" />}</TabsContent>
        <TabsContent value="engines"><Engines /></TabsContent>
        <TabsContent value="costs"><Costs /></TabsContent>
      </Tabs>
    </div>
  );
}
