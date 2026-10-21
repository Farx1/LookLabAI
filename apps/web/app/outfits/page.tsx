"use client";
import { useState } from "react";
import Link from "next/link";
import { Loader2, Shuffle, SlidersHorizontal, Sparkles } from "lucide-react";
import type { RecommendationResponse, SurpriseResponse } from "@looklab/shared-types";
import { Badge, Button, Input, Label, Select, Switch, Textarea, cn } from "@looklab/ui";
import { api, errorMessage } from "@/lib/api";
import { useVocabulary } from "@/lib/hooks";
import { label } from "@/lib/format";
import { EmptyState, Notice, PageHeader } from "@/components/page-header";
import { OutfitCard } from "@/components/outfit-card";

const EXAMPLES = [
  "Find 15 outfits for a romantic anniversary dinner. I prefer loose-fitting, comfortable clothes, neutral colors, Korean-inspired fashion and a budget of €100.",
  "Create 10 looks under €80 for work, no black",
  "A Korean-inspired outfit with a wide skirt",
  "5 casual weekend outfits using only clothes I own",
];

const FITS = [
  { v: "", t: "Any fit" },
  { v: "0.5", t: "Fitted" },
  { v: "1.5", t: "Regular" },
  { v: "2.5", t: "Loose & comfortable" },
  { v: "3", t: "Oversized" },
];

export default function OutfitsPage() {
  const { data: vocab } = useVocabulary();
  const [prompt, setPrompt] = useState("");
  const [advanced, setAdvanced] = useState(false);
  const [count, setCount] = useState(10);
  const [occasion, setOccasion] = useState("");
  const [season, setSeason] = useState("");
  const [budget, setBudget] = useState("");
  const [ownedOnly, setOwnedOnly] = useState(false);
  const [fit, setFit] = useState("");
  const [styles, setStyles] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [res, setRes] = useState<RecommendationResponse | null>(null);
  const [surprise, setSurprise] = useState<SurpriseResponse | null>(null);

  async function generate() {
    setBusy(true);
    setErr(null);
    try {
      const constraints: Record<string, unknown> = {};
      if (advanced) {
        constraints.count = count;
        if (occasion) constraints.occasion = occasion;
        if (season) constraints.season = season;
        if (budget) constraints.budget = Number(budget);
        if (ownedOnly) constraints.owned_only = true;
        if (fit) constraints.target_looseness = Number(fit);
        if (styles.length) constraints.preferred_styles = styles;
      }
      const r = await api.post<RecommendationResponse>("/api/outfits/generate", {
        prompt,
        constraints: advanced ? constraints : null,
        count: advanced ? count : null,
      });
      setRes(r);
      setSurprise(null);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  async function surpriseMe() {
    setBusy(true);
    setErr(null);
    try {
      // no seed sent: the server draws a fresh one, so "Again" always differs
      setSurprise(await api.post<SurpriseResponse>("/api/outfits/surprise", { count: 1 }));
      setRes(null);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  const rejected = res ? Object.entries(res.rejected) : [];

  return (
    <div>
      <PageHeader eyebrow="Outfit generator" title="What are you dressing for?">
        Describe the moment in your own words. Previews are collages of your real garment photos — instant and free. Photorealistic try-on
        only runs for the looks you choose.
      </PageHeader>

      <div className="mb-8 rounded-3xl border border-line bg-paper p-4 sm:p-6">
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g. 8 outfits for a summer wedding, elegant but comfortable, under €120"
          className="min-h-28 border-0 bg-cream/60 text-base"
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) generate(); }}
        />
        <div className="scrollbar-none -mx-1 mt-3 flex gap-2 overflow-x-auto px-1">
          {EXAMPLES.map((ex) => (
            <button key={ex} onClick={() => setPrompt(ex)} className="shrink-0 rounded-full border border-line px-3 py-1.5 text-xs text-muted hover:bg-cream hover:text-chocolate">
              {ex.length > 52 ? ex.slice(0, 50) + "…" : ex}
            </button>
          ))}
        </div>
        {advanced && (
          <div className="mt-5 grid gap-4 border-t border-line pt-5 sm:grid-cols-2 lg:grid-cols-4">
            <div className="grid gap-1.5">
              <Label>How many</Label>
              <Input type="number" min={1} max={30} value={count} onChange={(e) => setCount(Number(e.target.value))} />
            </div>
            <div className="grid gap-1.5">
              <Label>Occasion</Label>
              <Select value={occasion} onChange={(e) => setOccasion(e.target.value)}>
                <option value="">Any</option>
                {vocab?.occasions.map((o) => <option key={o} value={o}>{label(o)}</option>)}
              </Select>
            </div>
            <div className="grid gap-1.5">
              <Label>Season</Label>
              <Select value={season} onChange={(e) => setSeason(e.target.value)}>
                <option value="">Any</option>
                {vocab?.seasons.map((o) => <option key={o} value={o}>{o}</option>)}
              </Select>
            </div>
            <div className="grid gap-1.5">
              <Label>Budget for new pieces</Label>
              <Input inputMode="decimal" value={budget} onChange={(e) => setBudget(e.target.value)} placeholder="no limit" />
            </div>
            <div className="grid gap-1.5">
              <Label>Fit</Label>
              <Select value={fit} onChange={(e) => setFit(e.target.value)}>
                {FITS.map((f) => <option key={f.v} value={f.v}>{f.t}</option>)}
              </Select>
            </div>
            <label className="flex items-center justify-between gap-3 self-end rounded-xl bg-cream px-4 py-2.5 text-sm">
              Only clothes I own <Switch checked={ownedOnly} onCheckedChange={setOwnedOnly} />
            </label>
            <div className="grid gap-1.5 sm:col-span-2">
              <Label>Styles</Label>
              <div className="flex flex-wrap gap-1.5">
                {vocab?.styles.map((s) => (
                  <button
                    key={s}
                    onClick={() => setStyles((v) => (v.includes(s) ? v.filter((x) => x !== s) : [...v, s]))}
                    className={cn("rounded-full border px-2.5 py-1 text-xs capitalize", styles.includes(s) ? "border-chocolate bg-chocolate text-cream" : "border-line text-muted")}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
          <Button variant="ghost" size="sm" onClick={() => setAdvanced((v) => !v)}>
            <SlidersHorizontal /> {advanced ? "Hide filters" : "Filters"}
          </Button>
          <div className="flex flex-wrap gap-2">
            <Button size="lg" variant="outline" onClick={surpriseMe} disabled={busy} aria-busy={busy} aria-label="Surprise me: a new outfit from clothes you own">
              {busy ? <Loader2 className="animate-spin" /> : <Shuffle />} Surprise me
            </Button>
            <Button size="lg" onClick={generate} disabled={busy || (!prompt.trim() && !advanced)} aria-busy={busy}>
              {busy ? <Loader2 className="animate-spin" /> : <Sparkles />} Generate looks
            </Button>
          </div>
        </div>
      </div>

      {err && <div className="mb-6"><Notice tone="error">{err}</Notice></div>}

      {surprise && (
        <div className="mb-8 space-y-6" aria-live="polite">
          <Notice>
            <span className="font-medium">Trend FW26: {surprise.trend.nom}.</span> {surprise.trend.description}{" "}
            {surprise.trend.sources.slice(0, 3).map((u, i) => (
              <a key={u} href={u} target="_blank" rel="noreferrer" className="underline">
                {i > 0 && ", "}source {i + 1}
              </a>
            ))}
          </Notice>
          {surprise.history_reset && <Notice tone="warning">You have seen every combination: history reset.</Notice>}
          <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
            {surprise.outfits.map((o, i) => (
              <OutfitCard key={o.garment_ids.join()} outfit={o} rank={i + 1} occasion={surprise.constraints.occasion} constraints={surprise.constraints} />
            ))}
          </div>
          <Button variant="outline" onClick={surpriseMe} disabled={busy} aria-busy={busy}>
            {busy ? <Loader2 className="animate-spin" /> : <Shuffle />} Again
          </Button>
        </div>
      )}

      {res && (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center gap-2">
            {res.understood.length > 0 && <span className="text-sm text-muted">Understood:</span>}
            {res.understood.map((u) => <Badge key={u} variant="dark">{u}</Badge>)}
          </div>
          {res.notes.map((n) => <Notice key={n} tone="warning">{n}</Notice>)}
          {rejected.length > 0 && (
            <p className="text-xs text-muted">
              Considered {res.pool_size} pieces; set aside {rejected.map(([k, v]) => `${v} (${k})`).join(", ")}.
            </p>
          )}
          {res.suggestions.length > 0 && (
            <Notice>
              <span className="font-medium">Shopping ideas:</span>{" "}
              {res.suggestions.map((s) => `${s.description} — ${s.reason}`).join("; ")}. Add one to your wishlist from a product link in{" "}
              <Link href="/wardrobe" className="underline">your wardrobe</Link>.
            </Notice>
          )}
          {res.outfits.length === 0 ? (
            <EmptyState title="No outfits yet" action={<Button asChild variant="outline"><Link href="/wardrobe">Go to wardrobe</Link></Button>}>
              Add more garments or relax a constraint.
            </EmptyState>
          ) : (
            <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
              {res.outfits.map((o, i) => (
                <OutfitCard key={o.garment_ids.join()} outfit={o} rank={i + 1} occasion={res.constraints.occasion} constraints={res.constraints} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
