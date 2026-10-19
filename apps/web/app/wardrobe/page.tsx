"use client";
import { Suspense, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import useSWR from "swr";
import { Link2, Plus, Search } from "lucide-react";
import type { Garment } from "@looklab/shared-types";
import { Button, Dialog, DialogContent, Input, Label, Select, Switch, Tabs, TabsList, TabsTrigger } from "@looklab/ui";
import { api, errorMessage, fetcher } from "@/lib/api";
import { useVocabulary } from "@/lib/hooks";
import { label } from "@/lib/format";
import { EmptyState, Notice, PageHeader } from "@/components/page-header";
import { GarmentCard } from "@/components/garment-card";
import { GarmentEditor } from "@/components/garment-editor";
import { Dropzone } from "@/components/uploader";

type Collection = { id: string; name: string; garment_ids: string[] };

function WardrobeInner() {
  const params = useSearchParams();
  const { data: vocab } = useVocabulary();
  const { data: garments, mutate } = useSWR<Garment[]>("/api/garments", fetcher);
  const { data: collections, mutate: mutateCols } = useSWR<Collection[]>("/api/collections", fetcher);
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("");
  const [color, setColor] = useState("");
  const [owned, setOwned] = useState<string>(params.get("owned") ?? "all");
  const [favOnly, setFavOnly] = useState(false);
  const [col, setCol] = useState("");
  const [editing, setEditing] = useState<Garment | null>(null);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState<{ tone: "success" | "error" | "warning"; text: string } | null>(null);
  const [addOpen, setAddOpen] = useState(false);
  const [uploadOwned, setUploadOwned] = useState(true);
  const [url, setUrl] = useState("");
  const [selectMode, setSelectMode] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);

  const filtered = useMemo(() => {
    const cs = collections?.find((c) => c.id === col);
    return (garments ?? []).filter((g) => {
      if (cat && g.category !== cat) return false;
      if (color && !g.colors.includes(color)) return false;
      if (owned === "true" && !g.is_owned) return false;
      if (owned === "false" && g.is_owned) return false;
      if (favOnly && !g.favorite) return false;
      if (cs && !cs.garment_ids.includes(g.id)) return false;
      if (q) {
        const t = `${g.name ?? ""} ${g.subcategory ?? ""} ${g.brand ?? ""} ${g.style.join(" ")} ${g.colors.join(" ")}`.toLowerCase();
        if (!t.includes(q.toLowerCase())) return false;
      }
      return true;
    });
  }, [garments, collections, q, cat, color, owned, favOnly, col]);

  async function upload(files: File[]) {
    setUploading(true);
    setNotice(null);
    try {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      fd.append("is_owned", String(uploadOwned));
      const r = await api.post<{ created: Garment[]; errors: { filename: string; error: string }[] }>("/api/garments", fd);
      await mutate();
      const unknown = r.created.filter((g) => g.category === "unknown").length;
      setNotice({
        tone: r.errors.length ? "warning" : "success",
        text:
          `Added ${r.created.length} garment${r.created.length === 1 ? "" : "s"}.` +
          (unknown ? ` ${unknown} need a category — tap them to add details.` : "") +
          (r.errors.length ? ` Skipped: ${r.errors.map((e) => `${e.filename} (${e.error})`).join(", ")}` : ""),
      });
      setAddOpen(false);
    } catch (e) {
      setNotice({ tone: "error", text: errorMessage(e) });
    } finally {
      setUploading(false);
    }
  }

  async function importUrl() {
    setUploading(true);
    setNotice(null);
    try {
      const g = await api.post<Garment>("/api/garments/import-url", { url, is_owned: false });
      await mutate();
      setUrl("");
      setAddOpen(false);
      setEditing(g);
    } catch (e) {
      setNotice({ tone: "error", text: errorMessage(e) });
    } finally {
      setUploading(false);
    }
  }

  async function toggleFav(g: Garment) {
    await api.patch(`/api/garments/${g.id}`, { favorite: !g.favorite });
    mutate();
  }

  async function makeCollection() {
    const name = prompt("Collection name");
    if (!name) return;
    await api.post("/api/collections", { name, garment_ids: selected });
    setSelected([]);
    setSelectMode(false);
    mutateCols();
  }

  return (
    <div>
      <PageHeader
        eyebrow="My wardrobe"
        title={`${garments?.length ?? 0} pieces`}
        actions={
          <>
            <Button variant={selectMode ? "subtle" : "outline"} onClick={() => { setSelectMode((v) => !v); setSelected([]); }}>
              {selectMode ? "Cancel" : "Select"}
            </Button>
            {selectMode && selected.length > 0 && <Button variant="outline" onClick={makeCollection}>New collection ({selected.length})</Button>}
            <Button onClick={() => setAddOpen(true)}><Plus /> Add garments</Button>
          </>
        }
      >
        Photograph garments flat on a plain background for the best automatic color detection. Anything the app is unsure about stays
        “unknown” until you fill it in.
      </PageHeader>

      {notice && <div className="mb-5"><Notice tone={notice.tone}>{notice.text}</Notice></div>}

      <div className="mb-5 flex flex-col gap-3">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted" />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search by name, color, style, brand…" className="pl-9" />
        </div>
        <div className="scrollbar-none -mx-4 flex gap-2 overflow-x-auto px-4">
          <Select value={cat} onChange={(e) => setCat(e.target.value)} className="w-auto shrink-0" aria-label="Category">
            <option value="">All categories</option>
            {vocab?.categories.map((c) => <option key={c} value={c}>{label(c)}</option>)}
            <option value="unknown">Needs details</option>
          </Select>
          <Select value={color} onChange={(e) => setColor(e.target.value)} className="w-auto shrink-0" aria-label="Color">
            <option value="">All colors</option>
            {vocab?.colors.map((c) => <option key={c} value={c}>{c}</option>)}
          </Select>
          <Select value={col} onChange={(e) => setCol(e.target.value)} className="w-auto shrink-0" aria-label="Collection">
            <option value="">All collections</option>
            {collections?.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </Select>
          <Tabs value={owned} onValueChange={setOwned} className="shrink-0">
            <TabsList>
              <TabsTrigger value="all">All</TabsTrigger>
              <TabsTrigger value="true">Owned</TabsTrigger>
              <TabsTrigger value="false">Wishlist</TabsTrigger>
            </TabsList>
          </Tabs>
          <label className="flex shrink-0 items-center gap-2 rounded-full border border-line bg-paper px-3 text-sm">
            Favorites <Switch checked={favOnly} onCheckedChange={setFavOnly} />
          </label>
        </div>
      </div>

      {garments && garments.length === 0 ? (
        <EmptyState title="Your wardrobe is empty" action={<Button onClick={() => setAddOpen(true)}><Plus /> Add your first garments</Button>}>
          Upload photos of clothes you own, or import pieces you are considering from a shop page.
        </EmptyState>
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {filtered.map((g) => (
            <GarmentCard
              key={g.id}
              g={g}
              hex={vocab?.color_hex}
              selected={selected.includes(g.id)}
              onOpen={() => (selectMode ? setSelected((s) => (s.includes(g.id) ? s.filter((x) => x !== g.id) : [...s, g.id])) : setEditing(g))}
              onFavorite={() => toggleFav(g)}
            />
          ))}
        </div>
      )}

      <Dialog open={addOpen} onOpenChange={setAddOpen}>
        <DialogContent title="Add garments" description="Photos are cleaned of location and camera metadata before they are stored.">
          <div className="space-y-5">
            <label className="flex items-center justify-between rounded-xl bg-cream px-4 py-3 text-sm">
              These are clothes I already own
              <Switch checked={uploadOwned} onCheckedChange={setUploadOwned} />
            </label>
            <Dropzone onFiles={upload} busy={uploading} />
            <div className="space-y-2">
              <Label>Or import from a product page</Label>
              <div className="flex gap-2">
                <Input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://shop.example.com/product/…" inputMode="url" />
                <Button variant="outline" onClick={importUrl} disabled={!url || uploading}><Link2 /> Import</Button>
              </div>
              <p className="text-xs text-muted">Reads the page’s public product title, price and main image. Imported items go to your wishlist.</p>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <GarmentEditor
        garment={editing}
        vocab={vocab}
        open={!!editing}
        onOpenChange={(o) => !o && setEditing(null)}
        onSaved={() => mutate()}
        onDeleted={() => mutate()}
      />
    </div>
  );
}

export default function WardrobePage() {
  return (
    <Suspense>
      <WardrobeInner />
    </Suspense>
  );
}
