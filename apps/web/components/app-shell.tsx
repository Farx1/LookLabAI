"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { BookOpen, Heart, House, Menu, MessageCircle, Settings, Shirt, Sparkles, WandSparkles, X } from "lucide-react";
import { cn } from "@looklab/ui";

const NAV = [
  { href: "/", label: "Dashboard", icon: House, mobile: true },
  { href: "/wardrobe", label: "My Wardrobe", short: "Wardrobe", icon: Shirt, mobile: true },
  { href: "/outfits", label: "Outfit Generator", short: "Outfits", icon: Sparkles, mobile: true },
  { href: "/studio", label: "Try-On Studio", short: "Try-On", icon: WandSparkles, mobile: true },
  { href: "/stylist", label: "AI Stylist", short: "Stylist", icon: MessageCircle, mobile: true },
  { href: "/lookbook", label: "Lookbook", icon: BookOpen },
  { href: "/saved", label: "Saved Outfits", icon: Heart },
  { href: "/settings", label: "Settings & Privacy", icon: Settings },
];

function active(path: string, href: string) {
  return href === "/" ? path === "/" : path.startsWith(href);
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  useEffect(() => setOpen(false), [path]);
  useEffect(() => {
    if ("serviceWorker" in navigator && process.env.NODE_ENV === "production") navigator.serviceWorker.register("/sw.js").catch(() => {});
  }, []);

  return (
    <div className="flex min-h-dvh">
      <aside className="sticky top-0 hidden h-dvh w-64 shrink-0 flex-col border-r border-line bg-paper/70 px-5 py-7 lg:flex">
        <Link href="/" className="mb-10 block">
          <span className="font-serif text-3xl tracking-tight">LookLab</span>
          <span className="eyebrow ml-2">AI</span>
        </Link>
        <nav className="flex flex-col gap-1">
          {NAV.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition",
                active(path, href) ? "bg-chocolate text-cream" : "text-muted hover:bg-cream hover:text-chocolate",
              )}
            >
              <Icon className="size-4" />
              {label}
            </Link>
          ))}
        </nav>
        <p className="mt-auto text-xs leading-relaxed text-muted">Private by default. Your photos stay on this device unless you choose otherwise.</p>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center justify-between border-b border-line bg-cream/90 px-4 py-3 backdrop-blur lg:hidden" style={{ paddingTop: "max(0.75rem, env(safe-area-inset-top))" }}>
          <Link href="/" className="font-serif text-2xl">LookLab</Link>
          <button aria-label="Menu" onClick={() => setOpen((v) => !v)} className="rounded-full p-2 hover:bg-sand">
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </header>
        {open && (
          <div className="fixed inset-x-0 top-14 z-30 border-b border-line bg-paper p-3 shadow-lg lg:hidden">
            {NAV.filter((n) => !n.mobile).map(({ href, label, icon: Icon }) => (
              <Link key={href} href={href} className="flex items-center gap-3 rounded-xl px-3 py-3 text-sm hover:bg-cream">
                <Icon className="size-4" /> {label}
              </Link>
            ))}
          </div>
        )}
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 pb-28 pt-6 sm:px-6 lg:px-10 lg:pb-12 lg:pt-10">{children}</main>
        <nav
          className="fixed inset-x-0 bottom-0 z-30 grid grid-cols-5 border-t border-line bg-paper/95 backdrop-blur lg:hidden"
          style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
        >
          {NAV.filter((n) => n.mobile).map(({ href, short, label, icon: Icon }) => (
            <Link key={href} href={href} className={cn("flex flex-col items-center gap-1 py-2.5 text-[11px]", active(path, href) ? "text-burgundy" : "text-muted")}>
              <Icon className="size-5" />
              {short ?? label.split(" ")[0]}
            </Link>
          ))}
        </nav>
      </div>
    </div>
  );
}
