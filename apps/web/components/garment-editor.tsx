"use client";
import { useEffect, useState } from "react";
import { ExternalLink, Sparkles, Trash2 } from "lucide-react";
import type { Garment, Vocabulary } from "@looklab/shared-types";
import { Badge, Button, Dialog, DialogContent, Input, Label, Select, Switch, Textarea, cn } from "@looklab/ui";
import { api, errorMessage } from "@/lib/api";
import { garmentTitle, label } from "@/lib/format";

const SOURCE_LABEL: Record<string, string> = { user: "you", cv: "seen in photo", text: "from title", model: "AI guess" };

function SourceTag({ g, field }: { g: Garment; field: string }) {
  const s = g.field_sources[field];
  if (!s) return <span className="text-[10px] text-muted/70">unknown</span>;
  const kind = s.source.split(":")[0];
  return (
    <span title={s.source} className={cn("text-[10px]", kind === "user" ? "text-emerald-700" : kind === "model" ? "text-amber-700" : "text-muted")}>
      {SOURCE_LABEL[kind] ?? kind}
    </span>
  );
}

function MultiChips({ options, value, onChange, hex }: { options: string[]; value: string[]; onChange: (v: string[]) => void; hex?: Record<string, string> }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((o) => {
        const on = value.includes(o);
        return (
          <button
            type="button"
            key={o}
            onClick={() => onChange(on ? value.filter((x) => x !== o) : [...value, o])}
            className={cn("flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs capitalize transition", on ? "border-chocolate bg-chocolate text-cream" : "border-line bg-paper text-muted hover:text-chocolate")}
          >
            {hex && <span className="size-2.5 rounded-full ring-1 ring-line" style={{ background: hex[o] }} />}
            {label(o)}
          </button>
        );
      })}
    </div>
  );
}

type Draft = Partial<Garment>;

export function GarmentEditor({
  garment,
  vocab,
  open,
  onOpenChange,
  onSaved,
  onDeleted,
}: {
  garment: Garment | null;
  vocab?: Vocabulary;
  open: boolean;
  onOpenChange: (o: boolean) => void;
  onSaved: (g: Garment) => void;
  onDeleted: (id: string) => void;
}) {
  const [d, setD] = useState<Draft>({});
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  useEffect(() => {
    setD({});
    setMsg(null);
  }, [garment?.id]);
  if (!garment) return null;
  const g = { ...garment, ...d } as Garment;
  const set = (k: keyof Garment, v: unknown) => setD((x) => ({ ...x, [k]: v }));

  async function save() {
    setBusy(true);
    setMsg(null);
    try {
      const body: Record<string, unknown> = { ...d };
      if (body.price === "") body.price = null;
      if (typeof body.price === "string") body.price = Number(body.price);
      if (body.formality === "") body.formality = null;
      if (typeof body.formality === "string") body.formality = Number(body.formality);
      if (body.product_url === "") body.product_url = null;
      const saved = await api.patch<Garment>(`/api/garments/${garment!.id}`, body);
      onSaved(saved);
      setD({});
      onOpenChange(false);
    } catch (e) {
      setMsg(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  async function analyze() {
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.post<{ job?: { id: string } }>(`/api/garments/${garment!.id}/analyze`, { analyzer: "auto" });
      setMsg(r.job ? "Analysis queued on your local model. Details will update shortly." : "Analysis complete.");
    } catch (e) {
      setMsg(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  async function remove() {
    if (!confirm("Delete this garment and its photo?")) return;
    await api.del(`/api/garments/${garment!.id}`);
    onDeleted(garment!.id);
    onOpenChange(false);
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent title={garmentTitle(g)} description="Correct anything that looks wrong. Your edits always win over automatic analysis." className="sm:max-w-3xl">
        <div className="grid gap-6 sm:grid-cols-[220px_1fr]">
          <div className="space-y-3">
            <div className="aspect-[4/5] overflow-hidden rounded-2xl border border-line bg-white">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={garment.image_url} alt="" className="h-full w-full object-contain p-2" />
            </div>
            {garment.description && <p className="text-xs italic text-muted">“{garment.description}”</p>}
            <div className="flex flex-wrap gap-1">
              {garment.analyzers_run.map((a) => (
                <Badge key={a} variant="outline">{a}</Badge>
              ))}
            </div>
            <Button variant="outline" size="sm" onClick={analyze} disabled={busy} className="w-full">
              <Sparkles /> Analyze with local AI
            </Button>
          </div>
          <div className="grid gap-4">
            <div className="grid gap-1.5">
              <Label>Name</Label>
              <Input value={g.name ?? ""} onChange={(e) => set("name", e.target.value)} placeholder="e.g. Cream silk blouse" />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="grid gap-1.5">
                <div className="flex justify-between"><Label>Category</Label><SourceTag g={garment} field="category" /></div>
                <Select value={g.category} onChange={(e) => set("category", e.target.value)}>
                  <option value="unknown">unknown</option>
                  {vocab?.categories.map((c) => <option key={c} value={c}>{c}</option>)}
                </Select>
              </div>
              <div className="grid gap-1.5">
                <div className="flex justify-between"><Label>Type</Label><SourceTag g={garment} field="subcategory" /></div>
                <Input value={g.subcategory ?? ""} onChange={(e) => set("subcategory", e.target.value)} placeholder="blouse, midi skirt…" />
              </div>
              <div className="grid gap-1.5">
                <div className="flex justify-between"><Label>Fit</Label><SourceTag g={garment} field="fit" /></div>
                <Select value={g.fit} onChange={(e) => set("fit", e.target.value)}>
                  <option value="unknown">unknown</option>
                  {vocab?.fits.map((c) => <option key={c} value={c}>{c}</option>)}
                </Select>
              </div>
              <div className="grid gap-1.5">
                <div className="flex justify-between"><Label>Pattern</Label><SourceTag g={garment} field="pattern" /></div>
                <Select value={g.pattern} onChange={(e) => set("pattern", e.target.value)}>
                  <option value="unknown">unknown</option>
                  {vocab?.patterns.map((c) => <option key={c} value={c}>{label(c)}</option>)}
                </Select>
              </div>
              <div className="grid gap-1.5">
                <div className="flex justify-between"><Label>Formality 1–5</Label><SourceTag g={garment} field="formality" /></div>
                <Select value={g.formality ?? ""} onChange={(e) => set("formality", e.target.value)}>
                  <option value="">unknown</option>
                  {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n} · {["lounge", "casual", "smart casual", "dressy", "formal"][n - 1]}</option>)}
                </Select>
              </div>
              <div className="grid gap-1.5">
                <Label>Photo type</Label>
                <Select value={g.garment_photo_type} onChange={(e) => set("garment_photo_type", e.target.value)}>
                  <option value="flat-lay">flat-lay / product shot</option>
                  <option value="model">worn by a person</option>
                </Select>
              </div>
            </div>
            <div className="grid gap-1.5">
              <div className="flex justify-between"><Label>Colors (max 3)</Label><SourceTag g={garment} field="colors" /></div>
              <MultiChips options={vocab?.colors ?? []} value={g.colors} hex={vocab?.color_hex} onChange={(v) => set("colors", v.slice(-3))} />
            </div>
            <div className="grid gap-1.5">
              <div className="flex justify-between"><Label>Style</Label><SourceTag g={garment} field="style" /></div>
              <MultiChips options={vocab?.styles ?? []} value={g.style} onChange={(v) => set("style", v)} />
            </div>
            <div className="grid gap-1.5">
              <Label>Season</Label>
              <MultiChips options={vocab?.seasons ?? []} value={g.season} onChange={(v) => set("season", v)} />
            </div>
            <div className="grid gap-1.5">
              <Label>Occasions</Label>
              <MultiChips options={vocab?.occasions ?? []} value={g.occasion} onChange={(v) => set("occasion", v)} />
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="col-span-2 grid gap-1.5 sm:col-span-1">
                <Label>Price</Label>
                <Input inputMode="decimal" value={g.price ?? ""} onChange={(e) => set("price", e.target.value)} />
              </div>
              <div className="grid gap-1.5">
                <Label>Currency</Label>
                <Input value={g.currency} maxLength={3} onChange={(e) => set("currency", e.target.value.toUpperCase())} />
              </div>
              <div className="grid gap-1.5">
                <Label>Size</Label>
                <Input value={g.size ?? ""} onChange={(e) => set("size", e.target.value)} />
              </div>
              <div className="grid gap-1.5">
                <Label>Brand</Label>
                <Input value={g.brand ?? ""} onChange={(e) => set("brand", e.target.value)} />
              </div>
            </div>
            <div className="grid gap-1.5">
              <Label>Product link</Label>
              <div className="flex gap-2">
                <Input value={g.product_url ?? ""} onChange={(e) => set("product_url", e.target.value)} placeholder="https://" />
                {garment.product_url && (
                  <a className="grid place-items-center rounded-xl border border-line px-3" href={garment.product_url} target="_blank" rel="noreferrer noopener" aria-label="Open product page">
                    <ExternalLink className="size-4" />
                  </a>
                )}
              </div>
            </div>
            <label className="flex items-center justify-between rounded-xl bg-cream px-4 py-3 text-sm">
              I already own this
              <Switch checked={g.is_owned} onCheckedChange={(v) => set("is_owned", v)} />
            </label>
            <div className="grid gap-1.5">
              <Label>Notes</Label>
              <Textarea value={g.notes ?? ""} onChange={(e) => set("notes", e.target.value)} className="min-h-16" />
            </div>
            {msg && <p className="text-sm text-burgundy">{msg}</p>}
            <div className="flex items-center justify-between gap-2 pt-1">
              <Button variant="ghost" onClick={remove} className="text-red-700"><Trash2 /> Delete</Button>
              <Button onClick={save} disabled={busy || Object.keys(d).length === 0}>Save changes</Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
