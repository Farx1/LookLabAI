"use client";
import * as React from "react";
import * as S from "@radix-ui/react-switch";
import { cn } from "./cn";

export function Switch({ className, ...p }: S.SwitchProps) {
  return (
    <S.Root
      className={cn(
        "relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full bg-sand transition data-[state=checked]:bg-burgundy disabled:opacity-50",
        className,
      )}
      {...p}
    >
      <S.Thumb className="block h-5 w-5 translate-x-0.5 translate-y-0.5 rounded-full bg-white shadow transition data-[state=checked]:translate-x-[22px]" />
    </S.Root>
  );
}
