"use client";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { Bookmark, ExternalLink, Loader2, Trash2, WandSparkles } from "lucide-react";
import type { AnalyzedPiece, Job, OutfitAnalysis } from "@looklab/shared-types";
import { Badge, Button, cn } from "@looklab/ui";
import { api, errorMessage } from "@/lib/api";
import { garmentTitle, label, money, pct } from "@/lib/format";
import { Notice, PageHeader } from "@/components/page-header";
import { Dropzone } from "@/components/uploader";
import { useVariations } from "@/components/outfit-card";

const TRYABLE = new Set(["top", "bottom", "dress", "jumpsuit", "outerwear"]);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function Pieces({ pieces, picked, onPick }: { pieces: AnalyzedPiece[]; picked: Record<number, string>; onPick: (index: number, id: string) => void }) {
  return (
    <ol className="space-y-4">
      {pieces.map((p) => (
        <li key={p.index}>
          <fieldset className="rounded-2xl border border-line bg-paper p-4">
            <legend className="flex items-center gap-2 px-1 text-sm font-medium">
              <span className="grid size-6 place-items-center rounded-full bg-chocolate text-xs text-cream">{p.index}</span>
              <span className="capitalize">{p.name}</span>
              <span className="flex gap-1" aria-label={`Colors: ${p.colors.join(", ")}`}>
                {p.color_hex.slice(0, 3).map((h) => <span key={h} className="size-3 rounded-full border border-line" style={{ background: h }} />)}
              </span>
              <span className="text-xs font-normal text-muted">{label(p.category)}</span>
            </legend>
            {p.matches.length === 0 ? (
              <p className="text-sm text-muted">Nothing similar in your wardrobe yet.</p>
            ) : (
              <ul className="grid gap-2 sm:grid-cols-3">
                {p.matches.map(({ garment: g, score }) => (
                  <li key={g.id}>
                    <label className={cn("flex h-full cursor-pointer gap-3 rounded-xl border p-2 text-sm has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-burgundy", picked[p.index] === g.id ? "border-burgundy bg-blush/30" : "border-line")}>
                      <input type="radio" name={`piece-${p.index}`} className="sr-only" checked={picked[p.index] === g.id} onChange={() => onPick(p.index, g.id)} />
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={g.thumb_url} alt="" className="size-16 shrink-0 rounded-lg bg-white object-contain" />
                      <span className="min-w-0 space-y-0.5">
                        <span className="block truncate font-medium capitalize">{garmentTitle(g)}</span>
                        <span className="block text-xs text-muted">{[g.brand, g.price != null && money(g.price, g.currency)].filter(Boolean).join(" · ") || "No brand or price"}</span>
                        <span className="flex flex-wrap items-center gap-1.5 text-xs">
                          <Badge variant={g.is_owned ? "success" : "accent"}>{g.is_owned ? "Owned" : "Not owned"}</Badge>
                          <span className="text-muted">{pct(score)}% match</span>
                          {g.product_url?.startsWith("http") && (
                            <a href={g.product_url} target="_blank" rel="noopener noreferrer" aria-label={`Product page: ${garmentTitle(g)}`} className="text-muted hover:text-chocolate"><ExternalLink className="size-3" /></a>
                          )}
                        </span>
                      </span>
                    </label>
                  </li>
                ))}
              </ul>
            )}
          </fieldset>
        </li>
      ))}
    </ol>
  );
}

function Actions({ ids, tryIds, title }: { ids: string[]; tryIds: string[]; title: string }) {
  const [saved, setSaved] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const variations = useVariations(ids);
  async function save() {
    try {
      await api.post("/api/outfits/saved", { garment_ids: ids, title });
      setSaved(true);
    } catch (e) {
      setErr(errorMessage(e));
    }
  }
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <Button onClick={save} disabled={saved || !ids.length}><Bookmark className={cn(saved && "fill-current")} /> {saved ? "Outfit saved" : "Save outfit"}</Button>
        {tryIds.length ? (
          <Button variant="outline" asChild><Link href={`/studio?garments=${tryIds.join(",")}`}><WandSparkles /> Try on</Link></Button>
        ) : (
          <Button variant="outline" disabled><WandSparkles /> Try on</Button>
        )}
        {variations.button}
      </div>
      {err && <p role="alert" className="text-xs text-red-700">{err}</p>}
      {variations.panel}
    </div>
  );
}

export default function AnalyzePage() {
  const [person, setPerson] = useState(true);
  const [agree, setAgree] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [picked, setPicked] = useState<Record<number, string>>({});
  const alive = useRef(true);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []); // strict mode runs the cleanup once on mount
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const result = job?.status === "succeeded" ? (job.result as unknown as OutfitAnalysis) : null;

  async function analyze(files: File[]) {
    setBusy(true);
    setErr(null);
    setJob(null);
    setPicked({});
    setPreview(URL.createObjectURL(files[0])); // shown from the local file: the server deletes its copy after the analysis
    try {
      const fd = new FormData();
      fd.append("file", files[0]);
      fd.append("person_visible", String(person));
      fd.append("consent_confirmed", String(agree));
      let j = await api.post<Job>("/api/analyze", fd);
      setJob(j);
      while (alive.current && (j.status === "queued" || j.status === "running")) {
        await sleep(1000);
        j = await api.get<Job>(`/api/jobs/${j.id}`);
        setJob(j);
      }
      if (j.status === "succeeded") {
        const first: Record<number, string> = {};
        for (const p of (j.result as unknown as OutfitAnalysis).pieces) if (p.matches[0]) first[p.index] = p.matches[0].garment.id;
        setPicked(first);
      } else if (j.status !== "cancelled") setErr(j.error ?? "Analysis failed");
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function forget() {
    if (job) await api.del(`/api/analyze/${job.id}`).catch((e) => setErr(errorMessage(e)));
    setJob(null);
    setPreview(null);
  }

  const ids = useMemo(() => [...new Set(Object.values(picked))], [picked]);
  const tryIds = useMemo(() => {
    const cat = new Map((result?.pieces ?? []).flatMap((p) => p.matches.map((m) => [m.garment.id, m.garment.category] as const)));
    return ids.filter((id) => TRYABLE.has(cat.get(id) ?? ""));
  }, [ids, result]);
  const canStart = !person || agree;

  return (
    <div>
      <PageHeader eyebrow="Analyze an outfit" title="What is in this outfit?">
        Upload a photo of an outfit. The pieces are detected on this machine, then matched to your wardrobe.
      </PageHeader>

      <div className="mb-6 max-w-xl space-y-3">
        <label className="flex items-center gap-3 text-sm">
          <input type="checkbox" checked={person} onChange={(e) => setPerson(e.target.checked)} className="accent-[#7b1e30]" />
          A person is visible in the photo
        </label>
        {person && (
          <label className="flex gap-3 rounded-xl bg-cream p-4 text-sm leading-relaxed">
            <input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} className="mt-1 accent-[#7b1e30]" />
            <span>
              This is a photo of me, or of a person who agreed to its analysis. The photo is analyzed locally, never sent anywhere, and deleted from the
              server as soon as the analysis ends.
            </span>
          </label>
        )}
        {canStart ? (
          <Dropzone onFiles={analyze} multiple={false} busy={busy} title="Choose an outfit photo" hint="JPEG, PNG or WEBP" />
        ) : (
          <p className="text-xs text-muted">Confirm consent to continue.</p>
        )}
      </div>

      {err && <Notice tone="error">{err}</Notice>}
      {busy && !result && (
        <p role="status" className="flex items-center gap-2 text-sm text-muted"><Loader2 className="size-4 animate-spin" /> {job?.status === "running" ? "Detecting pieces…" : "Waiting for the analysis engine…"}</p>
      )}

      {result && preview && (
        <div className="grid gap-8 lg:grid-cols-[minmax(0,22rem)_1fr]">
          <figure className="relative h-fit self-start">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={preview} alt="Analyzed outfit" className="w-full rounded-2xl" />
            {result.pieces.map((p) => (
              <span
                key={p.index}
                aria-hidden="true"
                className={"pointer-events-none absolute rounded border-2 border-white/90 shadow-[0_0_0_1px_rgba(0,0,0,.4)]"}
                style={{ left: `${p.bbox[0] / 10}%`, top: `${p.bbox[1] / 10}%`, width: `${(p.bbox[2] - p.bbox[0]) / 10}%`, height: `${(p.bbox[3] - p.bbox[1]) / 10}%` }}
              >
                <span className="absolute -left-px -top-px grid size-5 place-items-center rounded-br bg-chocolate text-[11px] text-cream">{p.index}</span>
              </span>
            ))}
            <figcaption className="mt-2 flex items-center justify-between text-xs text-muted">
              <span>{result.pieces.length} piece(s) · {Object.values(result.timings).reduce((a, b) => a + b, 0).toFixed(1)} s</span>
              <button className="flex items-center gap-1 hover:text-chocolate" onClick={forget}><Trash2 className="size-3" /> Forget this analysis</button>
            </figcaption>
          </figure>
          <div className="space-y-6">
            {result.pieces.length === 0 ? (
              <Notice>No clothing detected. Try a sharper photo where the outfit is fully visible.</Notice>
            ) : (
              <>
                <Pieces pieces={result.pieces} picked={picked} onPick={(i, id) => setPicked((s) => ({ ...s, [i]: id }))} />
                <Actions key={ids.join()} ids={ids} tryIds={tryIds} title={`Analyzed outfit: ${ids.length} pieces`} />
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
