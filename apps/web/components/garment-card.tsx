"use client";
import { Heart } from "lucide-react";
import type { Garment } from "@looklab/shared-types";
import { Badge, cn } from "@looklab/ui";
import { garmentTitle, label, money } from "@/lib/format";

export function ColorDots({ colors, hex }: { colors: string[]; hex?: Record<string, string> }) {
  return (
    <span className="flex -space-x-1">
      {colors.slice(0, 3).map((c) => (
        <span key={c} title={c} className="size-3.5 rounded-full border border-white ring-1 ring-line" style={{ background: hex?.[c] ?? "#ccc" }} />
      ))}
    </span>
  );
}

export function GarmentCard({
  g,
  hex,
  onOpen,
  onFavorite,
  selected,
}: {
  g: Garment;
  hex?: Record<string, string>;
  onOpen?: () => void;
  onFavorite?: () => void;
  selected?: boolean;
}) {
  const needsInfo = g.category === "unknown";
  return (
    <div className={cn("group relative overflow-hidden rounded-2xl border bg-paper transition", selected ? "border-burgundy ring-2 ring-burgundy/30" : "border-line hover:shadow-md")}>
      <button onClick={onOpen} className="block w-full text-left" aria-label={`Open ${garmentTitle(g)}`}>
        <div className="aspect-[4/5] bg-white">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={g.thumb_url} alt={garmentTitle(g)} loading="lazy" className="h-full w-full object-contain p-3 transition group-hover:scale-[1.02]" />
        </div>
        <div className="space-y-1.5 border-t border-line p-3">
          <p className="line-clamp-1 text-sm font-medium capitalize">{garmentTitle(g)}</p>
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs capitalize text-muted">{needsInfo ? "needs category" : label(g.subcategory || g.category)}</span>
            <ColorDots colors={g.colors} hex={hex} />
          </div>
          <div className="flex flex-wrap gap-1">
            {!g.is_owned && <Badge variant="accent">wishlist {g.price != null && `· ${money(g.price, g.currency)}`}</Badge>}
            {needsInfo && <Badge variant="warning">add details</Badge>}
          </div>
        </div>
      </button>
      {onFavorite && (
        <button
          onClick={onFavorite}
          aria-label={g.favorite ? "Remove favorite" : "Mark favorite"}
          className="absolute right-2 top-2 rounded-full bg-paper/90 p-2 shadow-sm"
        >
          <Heart className={cn("size-4", g.favorite ? "fill-burgundy text-burgundy" : "text-muted")} />
        </button>
      )}
    </div>
  );
}
