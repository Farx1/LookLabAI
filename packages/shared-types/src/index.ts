// Domain types mirroring services/api/looklab_api/serializers.py.
// Request bodies are additionally generated from OpenAPI into ./openapi.d.ts (pnpm gen:types).

export type FieldSource = { source: string; confidence: number };

export interface Garment {
  id: string;
  name: string | null;
  category: string;
  subcategory: string | null;
  colors: string[];
  color_hex: string[];
  pattern: string;
  fit: string;
  style: string[];
  season: string[];
  occasion: string[];
  formality: number | null;
  details: string[];
  brand: string | null;
  size: string | null;
  price: number | null;
  currency: string;
  product_url: string | null;
  is_owned: boolean;
  favorite: boolean;
  garment_photo_type: "flat-lay" | "model";
  notes: string | null;
  description: string | null;
  field_sources: Record<string, FieldSource>;
  analyzers_run: string[];
  image_url: string;
  thumb_url: string;
  created_at: string;
  updated_at: string;
}

export interface GarmentBrief {
  id: string;
  name: string | null;
  category: string;
  subcategory: string | null;
  colors: string[];
  price: number | null;
  currency: string;
  is_owned: boolean;
  thumb_url: string;
  product_url: string | null;
}

export interface Vocabulary {
  categories: string[];
  fits: string[];
  patterns: string[];
  styles: string[];
  seasons: string[];
  occasions: string[];
  colors: string[];
  color_hex: Record<string, string>;
}

export interface QualityIssue {
  code: string;
  severity: "error" | "warning" | "info";
  message: string;
}

export interface PersonPhoto {
  id: string;
  label: string;
  width: number;
  height: number;
  consent_subject: "self" | "consenting_person";
  consent_at: string;
  quality: { ok: boolean; issues: QualityIssue[]; brightness: number; sharpness: number };
  image_url: string;
  thumb_url: string;
  created_at: string;
}

export interface ScoreBreakdown {
  color: number;
  style: number;
  formality: number;
  layering: number;
  fit: number;
  preference: number;
  request: number;
  pattern: number;
  total: number;
}

export interface ShoppingSuggestion {
  category: string;
  description: string;
  reason: string;
  search_query: string;
}

export interface Outfit {
  garment_ids: string[];
  garments: GarmentBrief[];
  score: ScoreBreakdown;
  explanation: string[];
  known_cost: number;
  unknown_price_items: string[];
  required_purchases: string[];
  suggestions: ShoppingSuggestion[];
  warnings: string[];
  collage_url: string;
}

export interface Constraints {
  count: number;
  occasion: string | null;
  season: string | null;
  budget: number | null;
  currency: string;
  owned_only: boolean;
  target_looseness: number | null;
  preferred_colors: string[];
  preferred_styles: string[];
  formality_shift: number;
  required_colors: string[];
  excluded_colors: string[];
  required_subcategories: string[];
  required_categories: string[];
  excluded_subcategories: string[];
  excluded_categories: string[];
  locked_garment_ids: string[];
  excluded_garment_ids: string[];
  required_details: string[];
}

export interface RecommendationResponse {
  outfits: Outfit[];
  constraints: Constraints;
  pool_size: number;
  rejected: Record<string, number>;
  notes: string[];
  suggestions: ShoppingSuggestion[];
  understood: string[];
}

export interface SavedOutfit {
  id: string;
  title: string;
  occasion: string | null;
  garment_ids: string[];
  garments: GarmentBrief[];
  missing_garments: string[];
  constraints: Partial<Constraints>;
  score: Partial<ScoreBreakdown>;
  explanation: string[];
  feedback: "like" | "dislike" | null;
  collage_url: string;
  created_at: string;
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface Job {
  id: string;
  kind: string;
  status: JobStatus;
  provider: string | null;
  result: Record<string, unknown> & { result_id?: string; is_mock?: boolean; elapsed_seconds?: number };
  error: string | null;
  cancel_requested: boolean;
  estimated_cost_usd: number;
  paid: boolean;
  cache_hit: boolean;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface ProviderStatus {
  name: string;
  model: string;
  kind: "mock" | "open_source" | "paid_api";
  location: "local" | "remote_self_hosted" | "third_party";
  fidelity: "mock" | "garment_tryon" | "generative_edit";
  sends_images_externally: boolean;
  paid: boolean;
  description: string;
  enabled: boolean;
  available: boolean;
  reason: string;
  role: string;
}

export interface TryOnResult {
  id: string;
  kind: string;
  job_id: string | null;
  person_photo_id: string | null;
  garment_ids: string[];
  provider: string;
  model: string;
  fidelity: string;
  is_mock: boolean;
  saved: boolean;
  params: Record<string, unknown>;
  image_url: string;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  payload: Partial<RecommendationResponse> & { understood?: string[] };
  created_at: string;
}

export interface ChatSession {
  id: string;
  title: string;
  constraints: Partial<Constraints>;
  updated_at: string;
  messages?: ChatMessage[];
}

export interface SystemInfo {
  hardware: {
    platform: string;
    cpu_count: number;
    ram_gb: number | null;
    execution_mode: "cpu_only" | "local_gpu";
    torch: { torch: boolean; cuda: boolean; gpu_name?: string; gpu_total_mb?: number };
  };
  auth_mode: string;
  storage_backend: string;
  policy: Record<string, unknown>;
  tryon_providers: ProviderStatus[];
  analyzers: { name: string; available: boolean; reason: string }[];
  fashn: { parser_mode: string; weights_dir: string; loaded: boolean };
  creative_editing: { enabled: boolean; available: [boolean, string] };
}

export interface UserSettings {
  preferences: {
    liked_colors: string[];
    disliked_colors: string[];
    liked_styles: string[];
    disliked_styles: string[];
    fit_preference: string | null;
    sizes: Record<string, string>;
    default_budget: number | null;
    currency: string;
  };
  privacy: { allow_external_processing: boolean; retention_days_generated: number | null; retention_days_photos: number | null };
  spending: { monthly_limit_usd: number; global_limit_usd: number; effective_limit_usd: number };
}

export interface CostReport {
  month_spend_usd: number;
  effective_limit_usd: number;
  by_provider: {
    task: string;
    provider: string;
    runs: number;
    cache_hits: number;
    avg_execution_seconds: number | null;
    max_gpu_memory_mb: number | null;
    estimated_cost_usd: number;
  }[];
  recent: {
    id: string;
    task: string;
    provider: string;
    model: string;
    device: string;
    execution_time: number | null;
    model_load_seconds: number | null;
    gpu_memory_peak_mb: number | null;
    estimated_cost_usd: number;
    actual_cost_usd: number | null;
    paid: boolean;
    cache_hit: boolean;
    status: string;
    created_at: string;
  }[];
}
