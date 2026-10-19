"use client";
import { useRef, useState } from "react";
import { Camera, ImagePlus, Loader2 } from "lucide-react";
import { Button, cn } from "@looklab/ui";
import { downscale } from "@/lib/image";

export function Dropzone({
  onFiles,
  multiple = true,
  busy,
  title = "Drop photos here",
  hint = "JPEG, PNG or WEBP · up to 40 at a time",
  capture,
}: {
  onFiles: (files: File[]) => void | Promise<void>;
  multiple?: boolean;
  busy?: boolean;
  title?: string;
  hint?: string;
  capture?: "user" | "environment";
}) {
  const input = useRef<HTMLInputElement>(null);
  const cam = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  async function handle(list: FileList | null) {
    if (!list?.length) return;
    const files = await Promise.all(Array.from(list).map((f) => downscale(f)));
    await onFiles(files);
    if (input.current) input.current.value = "";
    if (cam.current) cam.current.value = "";
  }
  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); handle(e.dataTransfer.files); }}
      className={cn("flex flex-col items-center gap-3 rounded-3xl border-2 border-dashed px-6 py-8 text-center transition", over ? "border-burgundy bg-blush/40" : "border-line bg-paper/60")}
    >
      {busy ? <Loader2 className="size-7 animate-spin text-muted" /> : <ImagePlus className="size-7 text-muted" />}
      <div>
        <p className="font-medium">{title}</p>
        <p className="text-xs text-muted">{hint}</p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        <Button size="sm" onClick={() => input.current?.click()} disabled={busy}>Choose files</Button>
        <Button size="sm" variant="outline" onClick={() => cam.current?.click()} disabled={busy} className="sm:hidden">
          <Camera /> Camera
        </Button>
      </div>
      <input ref={input} type="file" accept="image/jpeg,image/png,image/webp,image/heic,image/heif" multiple={multiple} hidden onChange={(e) => handle(e.target.files)} />
      <input ref={cam} type="file" accept="image/*" capture={capture ?? "environment"} hidden onChange={(e) => handle(e.target.files)} />
    </div>
  );
}
