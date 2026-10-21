"use client";
import Link from "next/link";
import useSWR from "swr";
import { ArrowRight, Cpu, MessageCircle, Shirt, Sparkles, WandSparkles } from "lucide-react";
import type { Garment, PersonPhoto, SavedOutfit, SystemInfo } from "@looklab/shared-types";
import { Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle } from "@looklab/ui";
import { fetcher } from "@/lib/api";
import { Notice } from "@/components/page-header";

export default function Dashboard() {
  const { data: garments } = useSWR<Garment[]>("/api/garments", fetcher);
  const { data: photos } = useSWR<PersonPhoto[]>("/api/photos", fetcher);
  const { data: saved } = useSWR<SavedOutfit[]>("/api/outfits/saved", fetcher);
  const { data: sys, error } = useSWR<SystemInfo>("/api/system", fetcher);
  const realTryOn = sys?.tryon_providers.find((p) => p.name === "local_fashn")?.available;
  const steps = [
    { done: (photos?.length ?? 0) > 0, label: "Add a photo of yourself", href: "/studio" },
    { done: (garments?.length ?? 0) >= 5, label: "Add at least 5 garments", href: "/wardrobe" },
    { done: (saved?.length ?? 0) > 0, label: "Save your first outfit", href: "/outfits" },
  ];

  return (
    <div className="space-y-10">
      <section className="grid gap-8 *:min-w-0 lg:grid-cols-[1.3fr_1fr] lg:items-end">
        <div>
          <p className="eyebrow mb-3">Your private stylist</p>
          <h1 className="font-serif text-5xl leading-[1.02] tracking-tight sm:text-6xl">
            Dress for the moment,<br />from the closet you have.
          </h1>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-muted">
            Build your wardrobe, describe the occasion, and get outfit ideas that respect your taste, your budget and your body as it is.
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            <Button size="lg" asChild><Link href="/outfits"><Sparkles /> Find outfits</Link></Button>
            <Button size="lg" variant="outline" asChild><Link href="/stylist"><MessageCircle /> Ask the stylist</Link></Button>
          </div>
        </div>
        <Card>
          <CardHeader>
            <CardTitle>Getting started</CardTitle>
            <CardDescription>Everything here runs free on this computer.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {steps.map((s) => (
              <Link key={s.label} href={s.href} className="flex items-center justify-between rounded-xl px-3 py-2.5 text-sm hover:bg-cream">
                <span className="flex items-center gap-3">
                  <span className={`grid size-5 place-items-center rounded-full border text-[10px] ${s.done ? "border-burgundy bg-burgundy text-white" : "border-line"}`}>{s.done ? "✓" : ""}</span>
                  <span className={s.done ? "text-muted line-through" : ""}>{s.label}</span>
                </span>
                <ArrowRight className="size-4 text-muted" />
              </Link>
            ))}
          </CardContent>
        </Card>
      </section>

      {error && <Notice tone="error">Cannot reach the LookLab API. Start it with <code>uv run looklab-api</code>.</Notice>}

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: "Garments", value: garments?.length, href: "/wardrobe", icon: Shirt },
          { label: "Wishlist items", value: garments?.filter((g) => !g.is_owned).length, href: "/wardrobe?owned=false", icon: Shirt },
          { label: "Saved looks", value: saved?.length, href: "/saved", icon: Sparkles },
          { label: "Your photos", value: photos?.length, href: "/studio", icon: WandSparkles },
        ].map(({ label, value, href, icon: Icon }) => (
          <Link key={label} href={href} className="rounded-2xl border border-line bg-paper p-4 transition hover:shadow-md">
            <Icon className="size-4 text-muted" />
            <p className="mt-3 font-serif text-4xl">{value ?? "–"}</p>
            <p className="text-xs text-muted">{label}</p>
          </Link>
        ))}
      </section>

      {sys && (
        <section className="flex flex-col gap-3 rounded-3xl border border-line bg-paper p-5 sm:flex-row sm:items-center">
          <Cpu className="size-5 shrink-0 text-muted" />
          <div className="flex-1 text-sm">
            <p className="font-medium">
              {sys.hardware.execution_mode === "local_gpu" ? `GPU mode · ${sys.hardware.torch.gpu_name}` : "CPU mode"} · {sys.hardware.cpu_count} cores
              {sys.hardware.ram_gb ? ` · ${sys.hardware.ram_gb} GB RAM` : ""}
            </p>
            <p className="text-muted">
              Wardrobe, recommendations and collages run locally for free.{" "}
              {realTryOn ? "Photorealistic try-on is ready on your GPU." : "Photorealistic try-on needs an NVIDIA GPU or your own remote GPU worker — see Settings."}
            </p>
          </div>
          <Badge variant={realTryOn ? "success" : "outline"}>{realTryOn ? "Try-on ready" : "Try-on: not set up"}</Badge>
        </section>
      )}

      {saved && saved.length > 0 && (
        <section>
          <div className="mb-4 flex items-end justify-between">
            <h2 className="font-serif text-3xl">Recently saved</h2>
            <Link href="/saved" className="text-sm text-muted hover:text-chocolate">All saved →</Link>
          </div>
          <div className="scrollbar-none -mx-4 flex snap-x gap-4 overflow-x-auto px-4 pb-2">
            {saved.slice(0, 8).map((o) => (
              <Link key={o.id} href="/saved" className="w-48 shrink-0 snap-start overflow-hidden rounded-2xl border border-line bg-paper">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {o.collage_url ? <img src={o.collage_url} alt={o.title} className="aspect-[4/5] w-full object-cover" loading="lazy" /> : <div className="flex aspect-[4/5] w-full items-center justify-center bg-cream p-3 text-center text-xs text-muted">Pieces deleted</div>}
                <p className="line-clamp-1 p-3 text-sm capitalize">{o.title}</p>
              </Link>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
