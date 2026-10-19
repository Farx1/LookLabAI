"use client";
import { useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { Loader2, Plus, Send } from "lucide-react";
import type { ChatMessage, ChatSession, Constraints } from "@looklab/shared-types";
import { Badge, Button, Textarea, cn } from "@looklab/ui";
import { api, errorMessage, fetcher } from "@/lib/api";
import { label, money } from "@/lib/format";
import { Notice } from "@/components/page-header";
import { OutfitCard } from "@/components/outfit-card";

const STARTERS = [
  "Find 15 outfits for a romantic anniversary dinner. I prefer loose-fitting, comfortable clothes, neutral colors, Korean-inspired fashion and a budget of €100.",
  "Make them less fitted.",
  "Use the clothes I already own.",
  "Show me something more sophisticated.",
  "Keep the same shoes but change the dress.",
  "Create 10 looks under €80.",
];

function ConstraintChips({ c }: { c: Partial<Constraints> }) {
  const chips: string[] = [];
  if (c.count) chips.push(`${c.count} looks`);
  if (c.occasion) chips.push(label(c.occasion));
  if (c.season) chips.push(c.season);
  if (c.budget != null) chips.push(`≤ ${money(c.budget, c.currency)}`);
  if (c.owned_only) chips.push("owned only");
  if (c.target_looseness != null) chips.push(c.target_looseness >= 2 ? "relaxed fit" : c.target_looseness <= 1 ? "fitted" : "regular fit");
  if (c.formality_shift) chips.push(c.formality_shift > 0 ? "dressier" : "more casual");
  c.preferred_styles?.forEach((s) => chips.push(s));
  c.preferred_colors?.forEach((s) => chips.push(s));
  c.excluded_colors?.forEach((s) => chips.push(`no ${s}`));
  c.required_subcategories?.forEach((s) => chips.push(`with ${[...(c.required_details ?? []), s].join(" ")}`));
  if (c.locked_garment_ids?.length) chips.push(`${c.locked_garment_ids.length} kept item(s)`);
  if (!chips.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {chips.map((x) => <Badge key={x} variant="outline" className="capitalize">{x}</Badge>)}
    </div>
  );
}

export default function StylistPage() {
  const { data: sessions, mutate: mutateSessions } = useSWR<ChatSession[]>("/api/stylist/sessions", fetcher);
  const [sid, setSid] = useState<string | null>(null);
  const { data: session, mutate } = useSWR<ChatSession>(sid ? `/api/stylist/sessions/${sid}` : null, fetcher);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!sid && sessions?.length) setSid(sessions[0].id);
  }, [sessions, sid]);
  const first = useRef(true);
  useEffect(() => {
    if (!session?.messages?.length && !pending) return;
    end.current?.scrollIntoView({ behavior: first.current ? "auto" : "smooth" });
    first.current = false;
  }, [session?.messages?.length, pending]);

  async function newChat() {
    const s = await api.post<ChatSession>("/api/stylist/sessions");
    await mutateSessions();
    setSid(s.id);
  }

  async function send(content: string) {
    if (!content.trim()) return;
    setBusy(true);
    setErr(null);
    setPending(content);
    setText("");
    try {
      let id = sid;
      if (!id) {
        id = (await api.post<ChatSession>("/api/stylist/sessions")).id;
        setSid(id);
      }
      await api.post(`/api/stylist/sessions/${id}/messages`, { content });
      await mutate();
      mutateSessions();
    } catch (e) {
      setErr(errorMessage(e));
      setText(content);
    } finally {
      setBusy(false);
      setPending(null);
    }
  }

  const messages: ChatMessage[] = session?.messages ?? [];
  return (
    <div className="grid gap-6 *:min-w-0 lg:grid-cols-[240px_1fr]">
      <aside className="hidden lg:block">
        <Button variant="outline" className="mb-4 w-full" onClick={newChat}><Plus /> New conversation</Button>
        <div className="space-y-1">
          {sessions?.map((s) => (
            <button key={s.id} onClick={() => setSid(s.id)} className={cn("block w-full truncate rounded-xl px-3 py-2 text-left text-sm", s.id === sid ? "bg-chocolate text-cream" : "text-muted hover:bg-paper")}>
              {s.title}
            </button>
          ))}
        </div>
      </aside>
      <section className="flex min-h-[70dvh] min-w-0 flex-col">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <p className="eyebrow mb-1">AI stylist</p>
            <h1 className="font-serif text-4xl">Talk it through</h1>
          </div>
        </div>
        {session?.constraints && Object.keys(session.constraints).length > 0 && (
          <div className="sticky top-14 z-10 -mx-4 mb-4 border-b border-line bg-cream px-4 py-2.5 lg:top-0">
            <p className="eyebrow mb-1.5">Remembering</p>
            <ConstraintChips c={session.constraints} />
          </div>
        )}
        <div className="flex-1 space-y-6">
          {messages.length === 0 && !pending && (
            <div className="rounded-3xl border border-line bg-paper p-6">
              <p className="font-serif text-2xl">Tell me where you are going.</p>
              <p className="mt-1 text-sm text-muted">I remember what you said earlier in the conversation, so you can refine step by step.</p>
              <div className="mt-4 flex flex-wrap gap-2">
                {STARTERS.map((s) => (
                  <button key={s} onClick={() => send(s)} className="rounded-full border border-line px-3 py-1.5 text-left text-xs text-muted hover:bg-cream hover:text-chocolate">
                    {s.length > 70 ? s.slice(0, 68) + "…" : s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, mi) =>
            m.role === "user" ? (
              <div key={m.id} className="flex justify-end">
                <p className="max-w-[85%] rounded-3xl rounded-br-md bg-chocolate px-4 py-2.5 text-sm text-cream">{m.content}</p>
              </div>
            ) : (
              <div key={m.id} className="space-y-4">
                <p className="max-w-3xl text-[15px] leading-relaxed">{m.content}</p>
                {m.payload.outfits && m.payload.outfits.length > 0 && (
                  <div className="scrollbar-none -mx-4 flex snap-x gap-4 overflow-x-auto px-4 pb-2">
                    {m.payload.outfits.map((o, i) => (
                      <div key={o.garment_ids.join()} className="w-[290px] shrink-0 snap-start">
                        <OutfitCard
                          outfit={o}
                          rank={i + 1}
                          occasion={m.payload.constraints?.occasion}
                          constraints={m.payload.constraints}
                          onFocus={mi === messages.length - 1 && sid ? () => api.post(`/api/stylist/sessions/${sid}/focus`, { index: i }).catch(() => {}) : undefined}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ),
          )}
          {pending && (
            <>
              <div className="flex justify-end"><p className="max-w-[85%] rounded-3xl rounded-br-md bg-chocolate/80 px-4 py-2.5 text-sm text-cream">{pending}</p></div>
              <p className="flex items-center gap-2 text-sm text-muted"><Loader2 className="size-4 animate-spin" /> Styling…</p>
            </>
          )}
          <div ref={end} />
        </div>
        {err && <div className="mt-4"><Notice tone="error">{err}</Notice></div>}
        <form
          onSubmit={(e) => { e.preventDefault(); send(text); }}
          className="sticky bottom-20 mt-6 flex items-end gap-2 rounded-3xl border border-line bg-paper p-2 shadow-sm lg:bottom-4"
        >
          <Button type="button" size="icon" variant="ghost" className="lg:hidden" aria-label="New conversation" onClick={newChat}><Plus /></Button>
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(text); } }}
            placeholder="e.g. Keep the same shoes but change the dress"
            className="min-h-11 resize-none border-0 bg-transparent focus:ring-0"
            rows={1}
          />
          <Button type="submit" size="icon" disabled={busy || !text.trim()} aria-label="Send">{busy ? <Loader2 className="animate-spin" /> : <Send />}</Button>
        </form>
        <p className="mt-2 text-center text-[11px] text-muted">Tip: hover or tap a look before saying “keep the shoes…” to work from that look.</p>
      </section>
    </div>
  );
}
