"use client";
import { useRef, useState } from "react";

/** Before/after comparison. Drag, tap, or use arrow keys. */
export function CompareSlider({ before, after, beforeLabel = "Original", afterLabel = "Try-on" }: { before: string; after: string; beforeLabel?: string; afterLabel?: string }) {
  const [pos, setPos] = useState(50);
  const ref = useRef<HTMLDivElement>(null);
  const move = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (r) setPos(Math.max(0, Math.min(100, ((clientX - r.left) / r.width) * 100)));
  };
  return (
    <div
      ref={ref}
      className="relative aspect-[2/3] w-full touch-none select-none overflow-hidden rounded-3xl border border-line bg-white"
      onPointerDown={(e) => { (e.target as HTMLElement).setPointerCapture?.(e.pointerId); move(e.clientX); }}
      onPointerMove={(e) => e.buttons && move(e.clientX)}
      role="slider"
      tabIndex={0}
      aria-label="Compare original and result"
      aria-valuenow={Math.round(pos)}
      aria-valuemin={0}
      aria-valuemax={100}
      onKeyDown={(e) => {
        if (e.key === "ArrowLeft") setPos((p) => Math.max(0, p - 5));
        if (e.key === "ArrowRight") setPos((p) => Math.min(100, p + 5));
      }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={after} alt={afterLabel} className="absolute inset-0 h-full w-full object-contain" draggable={false} />
      <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={before} alt={beforeLabel} className="absolute inset-0 h-full w-full object-contain" draggable={false} />
      </div>
      <div className="absolute inset-y-0 w-0.5 bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.15)]" style={{ left: `${pos}%` }}>
        <div className="absolute top-1/2 -ml-5 -mt-5 grid size-10 place-items-center rounded-full bg-white text-xs text-chocolate shadow-lg">⇆</div>
      </div>
      <span className="absolute left-3 top-3 rounded-full bg-paper/90 px-2.5 py-1 text-xs">{beforeLabel}</span>
      <span className="absolute right-3 top-3 rounded-full bg-paper/90 px-2.5 py-1 text-xs">{afterLabel}</span>
    </div>
  );
}
