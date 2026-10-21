"use client";
import { useState } from "react";
import Link from "next/link";
import useSWR from "swr";
import { Columns2, ThumbsDown, ThumbsUp, Trash2, WandSparkles } from "lucide-react";
import type { SavedOutfit } from "@looklab/shared-types";
import { Badge, Button, cn } from "@looklab/ui";
import { api, fetcher } from "@/lib/api";
import { label, money, pct, timeAgo } from "@/lib/format";
import { EmptyState, PageHeader } from "@/components/page-header";
import { ScoreBreakdownBars, useVariations } from "@/components/outfit-card";

function SavedVariations({ ids }: { ids: string[] }) {
  const { button, panel } = useVariations(ids);
  return <div className="space-y-3"><div>{button}</div>{panel}</div>;
}

export default function SavedPage() {
  const { data, mutate } = useSWR<SavedOutfit[]>("/api/outfits/saved", fetcher);
  const [compare, setCompare] = useState<string[]>([]);

  async function feedback(o: SavedOutfit, event: "like" | "dislike") {
    await api.post(`/api/outfits/saved/${o.id}/feedback`, { event });
    mutate();
  }
  async function remove(o: SavedOutfit) {
    await api.del(`/api/outfits/saved/${o.id}`);
    setCompare((c) => c.filter((x) => x !== o.id));
    mutate();
  }
  const toggleCompare = (id: string) => setCompare((c) => (c.includes(id) ? c.filter((x) => x !== id) : [...c.slice(-1), id]));
  const pair = (data ?? []).filter((o) => compare.includes(o.id));

  return (
    <div>
      <PageHeader eyebrow="Saved outfits" title="Your shortlist">
        Likes and dislikes teach the stylist your taste — see what it learned in Settings.
      </PageHeader>

      {pair.length === 2 && (
        <section className="mb-10 rounded-3xl border border-line bg-paper p-4 sm:p-6">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="font-serif text-2xl">Side by side</h2>
            <Button size="sm" variant="ghost" onClick={() => setCompare([])}>Close</Button>
          </div>
          <div className="grid grid-cols-2 gap-4">
            {pair.map((o) => (
              <div key={o.id} className="space-y-3">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {o.collage_url ? <img src={o.collage_url} alt={o.title} className="aspect-[4/5] w-full rounded-2xl object-cover" /> : <div className="flex aspect-[4/5] w-full rounded-2xl items-center justify-center bg-cream p-3 text-center text-xs text-muted">Pieces deleted</div>}
                <p className="text-sm font-medium capitalize">{o.title}</p>
                {o.score.total != null && <p className="text-xs text-muted">{pct(o.score.total)}% match</p>}
                {o.score.total != null && <ScoreBreakdownBars score={o.score as never} />}
                <ul className="space-y-1 text-xs text-muted">{o.explanation.map((e) => <li key={e}>· {e}</li>)}</ul>
              </div>
            ))}
          </div>
        </section>
      )}

      {data && data.length === 0 ? (
        <EmptyState title="Nothing saved yet" action={<Button asChild><Link href="/outfits">Generate outfits</Link></Button>} />
      ) : (
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {data?.map((o) => (
            <article key={o.id} className={cn("overflow-hidden rounded-3xl border bg-paper", compare.includes(o.id) ? "border-burgundy" : "border-line")}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              {o.collage_url ? <img src={o.collage_url} alt={o.title} className="aspect-[4/5] w-full object-cover" loading="lazy" /> : <div className="flex aspect-[4/5] w-full items-center justify-center bg-cream p-3 text-center text-xs text-muted">Pieces deleted</div>}
              <div className="space-y-3 p-4">
                <div className="flex items-start justify-between gap-2">
                  <p className="font-medium capitalize leading-snug">{o.title}</p>
                  {o.occasion && <Badge>{label(o.occasion)}</Badge>}
                </div>
                <p className="text-xs text-muted">
                  {timeAgo(o.created_at)} · {o.garments.filter((g) => !g.is_owned).length ? `to buy ${money(o.garments.filter((g) => !g.is_owned).reduce((s, g) => s + (g.price ?? 0), 0))}` : "all owned"}
                  {o.missing_garments.length > 0 && ` · ${o.missing_garments.length} piece(s) deleted`}
                </p>
                <div className="flex items-center gap-1">
                  <Button size="icon" variant="ghost" aria-label="Like" onClick={() => feedback(o, "like")}><ThumbsUp className={cn(o.feedback === "like" && "fill-chocolate")} /></Button>
                  <Button size="icon" variant="ghost" aria-label="Dislike" onClick={() => feedback(o, "dislike")}><ThumbsDown className={cn(o.feedback === "dislike" && "fill-chocolate")} /></Button>
                  <Button size="icon" variant={compare.includes(o.id) ? "subtle" : "ghost"} aria-label="Compare" onClick={() => toggleCompare(o.id)}><Columns2 /></Button>
                  <Button size="icon" variant="ghost" aria-label="Try on" asChild>
                    <Link href={`/studio?outfit=${o.garment_ids.join(",")}&garment=${o.garment_ids[0]}`}><WandSparkles /></Link>
                  </Button>
                  <Button size="icon" variant="ghost" aria-label="Delete" className="ml-auto text-red-700" onClick={() => remove(o)}><Trash2 /></Button>
                </div>
                <SavedVariations ids={o.garment_ids} />
              </div>
            </article>
          ))}
        </div>
      )}
      {compare.length === 1 && <p className="fixed bottom-24 left-1/2 -translate-x-1/2 rounded-full bg-chocolate px-4 py-2 text-xs text-cream shadow-lg lg:bottom-6">Pick one more look to compare</p>}
    </div>
  );
}
