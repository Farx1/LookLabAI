"use client";
import Link from "next/link";
import { useState } from "react";
import { Bookmark, Check, ThumbsDown, ThumbsUp, WandSparkles } from "lucide-react";
import type { Outfit } from "@looklab/shared-types";
import { Badge, Button, cn } from "@looklab/ui";
import { api, errorMessage } from "@/lib/api";
import { label, money, pct } from "@/lib/format";

const TRYABLE = new Set(["top", "bottom", "dress", "jumpsuit", "outerwear"]);

export function ScoreBreakdownBars({ score }: { score: Outfit["score"] }) {
  const rows: [string, number][] = [
    ["Color", score.color], ["Style", score.style], ["Occasion", score.formality], ["Fit", score.fit],
    ["Your taste", score.preference], ["Your request", score.request],
  ];
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
      {rows.map(([k, v]) => (
        <div key={k} className="flex items-center gap-2 text-[11px] text-muted">
          <span className="w-20 shrink-0">{k}</span>
          <span className="h-1 flex-1 overflow-hidden rounded-full bg-sand">
            <span className="block h-full rounded-full bg-chocolate" style={{ width: `${pct(v)}%` }} />
          </span>
        </div>
      ))}
    </div>
  );
}

export function OutfitCard({ outfit, rank, occasion, constraints, onFocus }: { outfit: Outfit; rank?: number; occasion?: string | null; constraints?: unknown; onFocus?: () => void }) {
  const [saved, setSaved] = useState(false);
  const [vote, setVote] = useState<"like" | "dislike" | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showScore, setShowScore] = useState(false);
  const currency = outfit.garments[0]?.currency ?? "EUR";
  const tryable = outfit.garments.filter((g) => TRYABLE.has(g.category));

  async function save() {
    try {
      await api.post("/api/outfits/saved", {
        garment_ids: outfit.garment_ids,
        occasion: occasion ?? null,
        constraints: constraints ?? {},
        score: outfit.score,
        explanation: outfit.explanation,
      });
      setSaved(true);
    } catch (e) {
      setErr(errorMessage(e));
    }
  }
  async function feedback(event: "like" | "dislike") {
    setVote(event);
    await api.post("/api/outfits/feedback", { garment_ids: outfit.garment_ids, event }).catch((e) => setErr(errorMessage(e)));
  }

  return (
    <article className="flex flex-col overflow-hidden rounded-3xl border border-line bg-paper" onMouseEnter={onFocus}>
      <div className="relative aspect-[4/5] bg-cream">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={outfit.collage_url} alt={outfit.garments.map((g) => g.name ?? g.category).join(", ")} loading="lazy" className="h-full w-full object-cover" />
        {rank != null && <span className="absolute left-3 top-3 rounded-full bg-paper/90 px-2.5 py-1 font-serif text-sm">Look {rank}</span>}
        <span className="absolute right-3 top-3 rounded-full bg-chocolate px-2.5 py-1 text-xs text-cream">{pct(outfit.score.total)}% match</span>
      </div>
      <div className="flex flex-1 flex-col gap-3 p-4">
        <ul className="flex flex-wrap gap-1.5">
          {outfit.garments.map((g) => (
            <li key={g.id} className="flex items-center gap-1.5 rounded-full border border-line py-0.5 pl-0.5 pr-2.5 text-xs">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={g.thumb_url} alt="" className="size-6 rounded-full bg-white object-contain" />
              <span className="capitalize">{g.name ?? label(g.subcategory ?? g.category)}</span>
              {!g.is_owned && <span className="text-burgundy">· buy</span>}
            </li>
          ))}
        </ul>
        <ul className="space-y-1 text-sm leading-snug">
          {outfit.explanation.map((e) => (
            <li key={e} className="flex gap-2"><Check className="mt-0.5 size-3.5 shrink-0 text-burgundy" /> <span>{e}</span></li>
          ))}
        </ul>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {outfit.required_purchases.length > 0 ? (
            <Badge variant="accent">To buy: {money(outfit.known_cost, currency)}{outfit.unknown_price_items.length ? " + unknown" : ""}</Badge>
          ) : (
            <Badge variant="success">All owned · {money(0, currency)}</Badge>
          )}
          {outfit.warnings.map((w) => <Badge key={w} variant="warning">{w}</Badge>)}
          <button className="ml-auto text-muted underline-offset-2 hover:underline" onClick={() => setShowScore((v) => !v)}>
            {showScore ? "hide" : "why this score"}
          </button>
        </div>
        {showScore && <ScoreBreakdownBars score={outfit.score} />}
        {outfit.suggestions.length > 0 && (
          <p className="text-xs text-muted">Missing piece: {outfit.suggestions.map((s) => s.description).join(", ")}</p>
        )}
        {err && <p className="text-xs text-red-700">{err}</p>}
        <div className="mt-auto flex items-center gap-1.5 pt-1">
          <Button size="sm" variant={saved ? "subtle" : "default"} onClick={save} disabled={saved}>
            <Bookmark className={cn(saved && "fill-current")} /> {saved ? "Saved" : "Save"}
          </Button>
          {tryable.length > 0 && (
            <Button size="sm" variant="outline" asChild>
              <Link href={`/studio?garment=${tryable[0].id}&outfit=${outfit.garment_ids.join(",")}`}><WandSparkles /> Try on</Link>
            </Button>
          )}
          <div className="ml-auto flex">
            <Button size="icon" variant="ghost" aria-label="More like this" onClick={() => feedback("like")}>
              <ThumbsUp className={cn(vote === "like" && "fill-chocolate")} />
            </Button>
            <Button size="icon" variant="ghost" aria-label="Less like this" onClick={() => feedback("dislike")}>
              <ThumbsDown className={cn(vote === "dislike" && "fill-chocolate")} />
            </Button>
          </div>
        </div>
      </div>
    </article>
  );
}
