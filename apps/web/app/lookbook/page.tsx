"use client";
import Link from "next/link";
import useSWR from "swr";
import type { SavedOutfit, TryOnResult } from "@looklab/shared-types";
import { Button } from "@looklab/ui";
import { fetcher } from "@/lib/api";
import { label } from "@/lib/format";
import { EmptyState, PageHeader } from "@/components/page-header";

type Lookbook = { boards: { occasion: string; outfits: SavedOutfit[] }[]; tryons: TryOnResult[]; mock_tryons: number };

export default function LookbookPage() {
  const { data } = useSWR<Lookbook>("/api/lookbook", fetcher);
  const empty = data && data.boards.length === 0 && data.tryons.length === 0;
  return (
    <div>
      <PageHeader eyebrow="Inspiration lookbook" title="Your looks, by moment">
        Boards build themselves from the outfits you save and the try-ons you keep.
      </PageHeader>
      {empty && (
        <EmptyState title="Your lookbook is empty" action={<Button asChild><Link href="/outfits">Create some looks</Link></Button>}>
          Save outfits from the generator or the stylist and they will appear here, grouped by occasion.
        </EmptyState>
      )}
      {data && data.tryons.length > 0 && (
        <section className="mb-12">
          <h2 className="mb-4 font-serif text-3xl">On you</h2>
          <div className="columns-2 gap-4 sm:columns-3 lg:columns-4">
            {data.tryons.map((r) => (
              <Link key={r.id} href="/studio" className="mb-4 block break-inside-avoid overflow-hidden rounded-2xl border border-line bg-white">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={r.image_url} alt="Try-on result" className="w-full" loading="lazy" />
                <p className="px-3 py-2 text-[11px] text-muted">{label(r.provider)}{r.saved ? " · saved" : ""}</p>
              </Link>
            ))}
          </div>
        </section>
      )}
      {data?.boards.map((b) => (
        <section key={b.occasion} className="mb-12">
          <div className="mb-4 flex items-baseline gap-3">
            <h2 className="font-serif text-3xl capitalize">{label(b.occasion)}</h2>
            <span className="text-sm text-muted">{b.outfits.length} look{b.outfits.length === 1 ? "" : "s"}</span>
          </div>
          <div className="columns-2 gap-4 sm:columns-3 lg:columns-4">
            {b.outfits.map((o) => (
              <figure key={o.id} className="mb-4 break-inside-avoid overflow-hidden rounded-2xl border border-line bg-paper">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {o.collage_url ? <img src={o.collage_url} alt={o.title} className="w-full" loading="lazy" /> : <div className="flex aspect-[4/5] w-full items-center justify-center bg-cream p-3 text-center text-xs text-muted">Pieces deleted</div>}
                <figcaption className="px-3 py-2 text-xs capitalize">{o.title}</figcaption>
              </figure>
            ))}
          </div>
        </section>
      ))}
      {data && data.mock_tryons > 0 && <p className="text-xs text-muted">{data.mock_tryons} mock preview(s) are hidden from the lookbook.</p>}
    </div>
  );
}
