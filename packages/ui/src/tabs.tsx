"use client";
import * as React from "react";
import * as T from "@radix-ui/react-tabs";
import { cn } from "./cn";

export const Tabs = T.Root;
export function TabsList({ className, ...p }: T.TabsListProps) {
  return <T.List className={cn("inline-flex rounded-full bg-cream p-1", className)} {...p} />;
}
export function TabsTrigger({ className, ...p }: T.TabsTriggerProps) {
  return (
    <T.Trigger
      className={cn(
        "rounded-full px-4 py-1.5 text-sm text-muted transition data-[state=active]:bg-paper data-[state=active]:text-chocolate data-[state=active]:shadow-sm",
        className,
      )}
      {...p}
    />
  );
}
export const TabsContent = T.Content;
