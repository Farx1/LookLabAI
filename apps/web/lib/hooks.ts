"use client";
import useSWR from "swr";
import type { Vocabulary } from "@looklab/shared-types";
import { fetcher } from "./api";

export function useVocabulary() {
  return useSWR<Vocabulary>("/api/garments/vocabulary", fetcher, { revalidateOnFocus: false });
}
