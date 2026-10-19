"use client";
import * as React from "react";
import * as D from "@radix-ui/react-dialog";
import { cn } from "./cn";

export const Dialog = D.Root;
export const DialogTrigger = D.Trigger;
export const DialogClose = D.Close;

export function DialogContent({ className, children, title, description, ...p }: D.DialogContentProps & { title: string; description?: string }) {
  return (
    <D.Portal>
      <D.Overlay className="fixed inset-0 z-40 bg-chocolate/40 backdrop-blur-[2px]" />
      <D.Content
        className={cn(
          "fixed inset-x-0 bottom-0 z-50 max-h-[92dvh] overflow-y-auto rounded-t-3xl bg-paper p-6 shadow-2xl sm:inset-auto sm:left-1/2 sm:top-1/2 sm:w-full sm:max-w-2xl sm:-translate-x-1/2 sm:-translate-y-1/2 sm:rounded-3xl",
          className,
        )}
        {...p}
      >
        <D.Title className="font-serif text-2xl text-chocolate">{title}</D.Title>
        {description ? <D.Description className="mt-1 text-sm text-muted">{description}</D.Description> : <D.Description className="sr-only">{title}</D.Description>}
        <div className="mt-5">{children}</div>
        <D.Close className="absolute right-4 top-4 rounded-full p-2 text-muted hover:bg-cream" aria-label="Close">
          ✕
        </D.Close>
      </D.Content>
    </D.Portal>
  );
}
