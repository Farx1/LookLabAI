"use client";
import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { Bookmark, Download, Loader2, Lock, RefreshCw, Trash2, WandSparkles } from "lucide-react";
import type { Garment, Job, PersonPhoto, ProviderStatus, TryOnResult } from "@looklab/shared-types";
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Dialog, DialogContent, Input, Label, Select, cn } from "@looklab/ui";
import { ApiError, api, errorMessage, fetcher } from "@/lib/api";
import { garmentTitle, label, usd } from "@/lib/format";
import { EmptyState, Notice, PageHeader } from "@/components/page-header";
import { CompareSlider } from "@/components/compare-slider";
import { Dropzone } from "@/components/uploader";

const TRYABLE = new Set(["top", "bottom", "dress", "jumpsuit", "outerwear"]);
type ProvidersResp = { providers: ProviderStatus[]; spending: { month_spend_usd: number; effective_limit_usd: number }; allow_external_processing: boolean };
type Estimate = { provider: string; estimated_cost_usd: number; paid: boolean; fidelity: string; sends_images_externally: boolean; category: string; garment_photo_type: string };

function PhotoConsentDialog({ open, onOpenChange, onUploaded }: { open: boolean; onOpenChange: (o: boolean) => void; onUploaded: (p: PersonPhoto) => void }) {
  const [subject, setSubject] = useState<"self" | "consenting_person">("self");
  const [agree, setAgree] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function upload(files: File[]) {
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      fd.append("file", files[0]);
      fd.append("consent_confirmed", String(agree));
      fd.append("consent_subject", subject);
      onUploaded(await api.post<PersonPhoto>("/api/photos", fd));
      onOpenChange(false);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title="Add a photo" description="Use a well-lit, front-facing photo where the area of the garment is visible. Full-body works best.">
        <div className="space-y-4">
          <div className="grid gap-1.5">
            <Label>Who is in the photo?</Label>
            <Select value={subject} onChange={(e) => setSubject(e.target.value as typeof subject)}>
              <option value="self">Me</option>
              <option value="consenting_person">Someone who agreed to this</option>
            </Select>
          </div>
          <label className="flex gap-3 rounded-xl bg-cream p-4 text-sm leading-relaxed">
            <input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} className="mt-1 accent-[#7b1e30]" />
            <span>
              {subject === "self" ? "This is a photo of me." : "The person in this photo has agreed to its use for virtual try-on."} The photo is stored
              privately on this device, stripped of location data, and is never sent anywhere unless I explicitly choose a remote provider.
            </span>
          </label>
          {agree ? <Dropzone onFiles={upload} multiple={false} busy={busy} title="Choose your photo" hint="JPEG, PNG or WEBP" capture="user" /> : <p className="text-center text-xs text-muted">Confirm consent to continue.</p>}
          {err && <Notice tone="error">{err}</Notice>}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function StudioInner() {
  const params = useSearchParams();
  const { data: photos, mutate: mutatePhotos } = useSWR<PersonPhoto[]>("/api/photos", fetcher);
  const { data: garments } = useSWR<Garment[]>("/api/garments", fetcher);
  const { data: prov, mutate: mutateProv } = useSWR<ProvidersResp>("/api/tryon/providers", fetcher);
  const [photoId, setPhotoId] = useState<string | null>(null);
  const [garmentId, setGarmentId] = useState<string | null>(params.get("garment"));
  const [provider, setProvider] = useState("auto");
  const [steps, setSteps] = useState(30);
  const [seed, setSeed] = useState(42);
  const [estimate, setEstimate] = useState<Estimate | null>(null);
  const [approve, setApprove] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [consentOpen, setConsentOpen] = useState(false);
  const { data: results, mutate: mutateResults } = useSWR<TryOnResult[]>(photoId ? `/api/tryon/results?person_photo_id=${photoId}` : null, fetcher);
  const [current, setCurrent] = useState<TryOnResult | null>(null);

  const outfitIds = params.get("outfit")?.split(",") ?? [];
  const tryable = useMemo(() => (garments ?? []).filter((g) => TRYABLE.has(g.category)), [garments]);
  const photo = photos?.find((p) => p.id === photoId);
  const garment = garments?.find((g) => g.id === garmentId);
  const realAvailable = prov?.providers.some((p) => p.name === "local_fashn" && p.available);

  useEffect(() => {
    if (!photoId && photos?.length) setPhotoId(photos[0].id);
  }, [photos, photoId]);
  useEffect(() => {
    if (prov && !realAvailable && provider === "auto") setProvider(prov.providers.find((p) => p.name === "mock" && p.enabled) ? "mock" : "auto");
  }, [prov, realAvailable, provider]);

  // live estimate whenever the selection changes
  useEffect(() => {
    setEstimate(null);
    setApprove(false);
    if (!photoId || !garmentId) return;
    api
      .post<Estimate>("/api/tryon/estimate", { person_photo_id: photoId, garment_id: garmentId, provider, params: { num_timesteps: steps, seed } })
      .then(setEstimate)
      .catch((e) => setErr(errorMessage(e)));
    setErr(null);
  }, [photoId, garmentId, provider, steps, seed]);

  // poll the job
  useEffect(() => {
    if (!job || ["succeeded", "failed", "cancelled"].includes(job.status)) return;
    const t = setInterval(async () => {
      const j = await api.get<Job>(`/api/jobs/${job.id}`);
      setJob(j);
      if (j.status === "succeeded") {
        const rs = await mutateResults();
        setCurrent(rs?.find((r) => r.id === j.result.result_id) ?? null);
        mutateProv();
      }
    }, 1200);
    return () => clearInterval(t);
  }, [job, mutateResults, mutateProv]);

  async function run(newSeed?: number) {
    if (!photoId || !garmentId) return;
    setErr(null);
    const s = newSeed ?? seed;
    if (newSeed != null) setSeed(newSeed);
    try {
      const j = await api.post<Job>("/api/tryon/jobs", {
        person_photo_id: photoId,
        garment_id: garmentId,
        provider,
        params: { num_timesteps: steps, seed: s },
        approve_paid: approve,
        max_cost_usd: approve && estimate ? estimate.estimated_cost_usd : null,
      });
      setJob(j);
      if (j.status === "succeeded") {
        const rs = await mutateResults();
        setCurrent(rs?.find((r) => r.id === j.result.result_id) ?? null);
      }
    } catch (e) {
      setErr(e instanceof ApiError && e.code === "no_free_provider_available" ? "No free try-on engine is available on this machine. Pick a provider below." : errorMessage(e));
    }
  }

  async function toggleSave(r: TryOnResult) {
    const u = await api.patch<TryOnResult>(`/api/tryon/results/${r.id}`, { saved: !r.saved });
    setCurrent(u);
    mutateResults();
  }
  async function removeResult(r: TryOnResult) {
    await api.del(`/api/tryon/results/${r.id}`);
    setCurrent(null);
    mutateResults();
  }
  async function deletePhoto(p: PersonPhoto) {
    if (!confirm("Delete this photo and every try-on image generated from it?")) return;
    await api.del(`/api/photos/${p.id}`);
    setPhotoId(null);
    setCurrent(null);
    mutatePhotos();
  }

  const running = job && ["queued", "running"].includes(job.status);
  const selectedProvider = prov?.providers.find((p) => p.name === (provider === "auto" ? "local_fashn" : provider));

  return (
    <div>
      <PageHeader eyebrow="Virtual try-on studio" title="See it on you">
        One garment at a time, on your own photo. Results keep your body and face as they are — the model only changes the clothes.
      </PageHeader>

      {prov && !realAvailable && (
        <div className="mb-6">
          <Notice tone="warning">
            Photorealistic try-on (FASHN VTON v1.5) needs an NVIDIA GPU with the model installed, or your own remote GPU worker. On this
            machine you can run the <b>labelled mock preview</b> to test the workflow — it is not a try-on. Setup steps are in Settings → AI engines.
          </Notice>
        </div>
      )}

      <div className="grid gap-6 *:min-w-0 lg:grid-cols-[360px_1fr]">
        <div className="space-y-5">
          <Card>
            <CardHeader className="flex-row items-center justify-between">
              <CardTitle className="text-lg">1 · Your photo</CardTitle>
              <Button size="sm" variant="outline" onClick={() => setConsentOpen(true)}>Add</Button>
            </CardHeader>
            <CardContent>
              {photos?.length ? (
                <div className="grid grid-cols-4 gap-2">
                  {photos.map((p) => (
                    <button key={p.id} onClick={() => setPhotoId(p.id)} className={cn("aspect-[3/4] overflow-hidden rounded-xl border-2", p.id === photoId ? "border-burgundy" : "border-transparent")}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={p.thumb_url} alt="Your photo" className="h-full w-full object-cover" />
                    </button>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted">No photo yet.</p>
              )}
              {photo && (
                <div className="mt-3 space-y-1.5">
                  {photo.quality.issues.filter((i) => i.severity !== "info").map((i) => (
                    <p key={i.code} className={cn("text-xs", i.severity === "error" ? "text-red-700" : "text-amber-800")}>• {i.message}</p>
                  ))}
                  <div className="flex items-center justify-between text-xs text-muted">
                    <span className="flex items-center gap-1"><Lock className="size-3" /> stored privately · {photo.width}×{photo.height}</span>
                    <button onClick={() => deletePhoto(photo)} className="text-red-700 hover:underline">delete</button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-lg">2 · Garment</CardTitle></CardHeader>
            <CardContent>
              {outfitIds.length > 1 && (
                <p className="mb-3 text-xs text-muted">
                  From your outfit: the try-on model dresses one garment per run. Pick the piece to try; layering several pieces in sequence is
                  not guaranteed to keep earlier ones intact.
                </p>
              )}
              <div className="grid max-h-72 grid-cols-4 gap-2 overflow-y-auto">
                {(outfitIds.length > 1 ? tryable.filter((g) => outfitIds.includes(g.id)) : tryable).map((g) => (
                  <button key={g.id} onClick={() => setGarmentId(g.id)} title={garmentTitle(g)} className={cn("aspect-square overflow-hidden rounded-xl border-2 bg-white", g.id === garmentId ? "border-burgundy" : "border-line")}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={g.thumb_url} alt={garmentTitle(g)} className="h-full w-full object-contain p-1" />
                  </button>
                ))}
              </div>
              {tryable.length === 0 && <p className="text-sm text-muted">Add tops, bottoms or dresses (with a category) to your wardrobe.</p>}
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-lg">3 · Engine</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              {prov?.providers.filter((p) => p.enabled || p.name === "local_fashn").map((p) => {
                const id = p.name === "local_fashn" ? "auto" : p.name;
                return (
                  <label key={p.name} className={cn("flex cursor-pointer gap-3 rounded-xl border p-3 text-sm", provider === id ? "border-chocolate bg-cream" : "border-line", !p.available && "opacity-60")}>
                    <input type="radio" name="provider" checked={provider === id} disabled={!p.available} onChange={() => setProvider(id)} className="mt-1 accent-[#3b241c]" />
                    <span className="flex-1">
                      <span className="flex flex-wrap items-center gap-1.5 font-medium">
                        {p.name === "local_fashn" ? "On this computer" : label(p.name)}
                        {p.kind === "mock" && <Badge variant="warning">mock · not a try-on</Badge>}
                        {p.kind === "open_source" && <Badge variant="success">open source</Badge>}
                        {p.paid && <Badge variant="accent">paid</Badge>}
                        {p.fidelity === "generative_edit" && <Badge variant="outline">generative edit</Badge>}
                      </span>
                      <span className="mt-0.5 block text-xs text-muted">{p.available ? p.description : p.reason}</span>
                    </span>
                  </label>
                );
              })}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="grid min-w-0 gap-1.5">
                  <Label>Quality</Label>
                  <Select value={steps} onChange={(e) => setSteps(Number(e.target.value))}>
                    <option value={20}>Fast (20 steps)</option>
                    <option value={30}>Balanced (30)</option>
                    <option value={50}>Best (50)</option>
                  </Select>
                </div>
                <div className="grid min-w-0 gap-1.5">
                  <Label>Seed</Label>
                  <Input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
                </div>
              </div>
            </CardContent>
          </Card>

          {estimate && (
            <div className="rounded-2xl border border-line bg-paper p-4 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted">Estimated cost</span>
                <span className="font-serif text-2xl">{usd(estimate.estimated_cost_usd)}</span>
              </div>
              <p className="mt-1 text-xs text-muted">
                As {label(estimate.category)} · {estimate.garment_photo_type} photo
                {estimate.sends_images_externally && " · your photo will be sent to this provider"}
              </p>
              {estimate.paid && (
                <label className="mt-3 flex gap-2 rounded-xl bg-blush/50 p-3 text-xs leading-relaxed">
                  <input type="checkbox" checked={approve} onChange={(e) => setApprove(e.target.checked)} className="mt-0.5 accent-[#7b1e30]" />
                  I approve spending up to {usd(estimate.estimated_cost_usd)} for this single request. It will not be retried automatically.
                  {prov && ` This month: ${usd(prov.spending.month_spend_usd)} of ${usd(prov.spending.effective_limit_usd)}.`}
                </label>
              )}
            </div>
          )}
          {err && <Notice tone="error">{err}</Notice>}
          <Button size="lg" className="w-full" onClick={() => run()} disabled={!photoId || !garmentId || !!running || (estimate?.paid && !approve)}>
            {running ? <Loader2 className="animate-spin" /> : <WandSparkles />}
            {running ? (job?.status === "queued" ? "Queued…" : "Generating…") : provider === "mock" ? "Run mock preview" : "Try it on"}
          </Button>
          {running && job && (
            <Button variant="ghost" size="sm" className="w-full" onClick={async () => setJob(await api.post<Job>(`/api/jobs/${job.id}/cancel`))}>Cancel</Button>
          )}
          {job?.status === "failed" && <Notice tone="error">{job.error}</Notice>}
        </div>

        <div className="space-y-6">
          {!photo ? (
            <EmptyState title="Start with a photo" action={<Button onClick={() => setConsentOpen(true)}>Add your photo</Button>}>
              Your photo stays on this device. You can delete it, and every image made from it, at any time.
            </EmptyState>
          ) : current ? (
            <div className="grid gap-5 xl:grid-cols-[1fr_220px]">
              <div>
                <CompareSlider before={photo.image_url} after={current.image_url} afterLabel={current.is_mock ? "Mock preview" : "Try-on"} />
              </div>
              <div className="space-y-4">
                {garment && (
                  <div className="rounded-2xl border border-line bg-white p-3">
                    <p className="eyebrow mb-2">Garment</p>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={garments?.find((g) => g.id === current.garment_ids[0])?.thumb_url ?? garment.thumb_url} alt="" className="aspect-square w-full object-contain" />
                  </div>
                )}
                <div className="space-y-1 text-xs text-muted">
                  <p><span className="text-chocolate">{current.is_mock ? "Mock — no model run" : current.model}</span></p>
                  <p>via {label(current.provider)} · {label(current.fidelity)}</p>
                  {job?.result?.elapsed_seconds != null && job.result.result_id === current.id && <p>{Number(job.result.elapsed_seconds).toFixed(1)} s · {usd(Number(job.result.estimated_cost_usd ?? 0))}</p>}
                  {job?.cache_hit && <p>Reused a previous identical result (free).</p>}
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button size="sm" variant={current.saved ? "subtle" : "outline"} onClick={() => toggleSave(current)}><Bookmark className={cn(current.saved && "fill-current")} /> {current.saved ? "Saved" : "Save"}</Button>
                  <Button size="sm" variant="outline" asChild><a href={`${current.image_url}?download=true`}><Download /> Download</a></Button>
                  <Button size="sm" variant="outline" onClick={() => run(Math.floor(Math.random() * 1e6))} disabled={!!running}><RefreshCw /> Regenerate</Button>
                  <Button size="sm" variant="ghost" className="text-red-700" onClick={() => removeResult(current)}><Trash2 /> Delete</Button>
                </div>
              </div>
            </div>
          ) : (
            <div className="grid gap-5 sm:grid-cols-2">
              <div className="overflow-hidden rounded-3xl border border-line bg-white">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={photo.image_url} alt="Your photo" className="aspect-[2/3] w-full object-contain" />
              </div>
              <div className="grid place-items-center rounded-3xl border border-dashed border-line p-6 text-center text-sm text-muted">
                {running ? <Loader2 className="size-8 animate-spin" /> : garment ? `Ready to try ${garmentTitle(garment)}.` : "Pick a garment."}
                {selectedProvider && !selectedProvider.available && <p className="mt-2 text-xs">{selectedProvider.reason}</p>}
              </div>
            </div>
          )}

          {results && results.length > 0 && (
            <div>
              <p className="eyebrow mb-3">Results for this photo</p>
              <div className="flex gap-3 overflow-x-auto pb-2">
                {results.map((r) => (
                  <button key={r.id} onClick={() => setCurrent(r)} className={cn("relative w-28 shrink-0 overflow-hidden rounded-2xl border-2", current?.id === r.id ? "border-burgundy" : "border-line")}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={r.image_url} alt="" className="aspect-[2/3] w-full object-cover" loading="lazy" />
                    {r.is_mock && <span className="absolute inset-x-0 bottom-0 bg-burgundy/90 py-0.5 text-center text-[10px] text-white">mock</span>}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      <PhotoConsentDialog open={consentOpen} onOpenChange={setConsentOpen} onUploaded={(p) => { mutatePhotos(); setPhotoId(p.id); }} />
    </div>
  );
}

export default function StudioPage() {
  return (
    <Suspense>
      <StudioInner />
    </Suspense>
  );
}
