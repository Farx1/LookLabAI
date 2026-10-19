"""Runtime configuration.

Environment variables (prefix ``LOOKLAB_``) configure infrastructure; the
AI routing policy lives in ``config/ai.yaml`` and is validated against the
providers that actually exist in code, so a typo fails at startup instead
of silently disabling a provider.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]

TRYON_PROVIDERS = ("local_fashn", "remote_fashn", "fashn_api", "openai_image_edit", "mock")
PAID_TRYON_PROVIDERS = ("remote_fashn", "fashn_api", "openai_image_edit")
ANALYSIS_PROVIDERS = ("cv_basic", "local_qwen3_vl", "local_florence2", "none")
CREATIVE_PROVIDERS = ("local_qwen_image_edit", "openai_image_edit", "disabled")
RECOMMENDATION_ENGINES = ("deterministic_hybrid",)


class ImageAnalysisPolicy(BaseModel):
    primary: Literal["cv_basic"] = "cv_basic"
    model: Literal["local_qwen3_vl", "local_florence2", "none"] = "local_qwen3_vl"
    fallback: Literal["local_qwen3_vl", "local_florence2", "none"] = "local_florence2"
    auto_run_model: bool = False  # run the VLM on upload, or only on request


class RecommendationPolicy(BaseModel):
    primary: Literal["deterministic_hybrid"] = "deterministic_hybrid"
    llm_explanations: Literal["none", "ollama"] = "none"
    ollama_text_model: str = "qwen3:1.7b"


class TryOnPolicy(BaseModel):
    primary: Literal["local_fashn"] = "local_fashn"
    secondary: Literal["remote_fashn", "none"] = "remote_fashn"
    paid_fallback: Literal["disabled", "fashn_api", "openai_image_edit"] = "disabled"
    enabled_paid_providers: list[Literal["remote_fashn", "fashn_api", "openai_image_edit"]] = Field(default_factory=list)
    allow_mock: bool = True

    @field_validator("enabled_paid_providers")
    @classmethod
    def _dedupe(cls, v):
        return list(dict.fromkeys(v))


class CreativePolicy(BaseModel):
    enabled: bool = False
    primary: Literal["local_qwen_image_edit", "disabled"] = "local_qwen_image_edit"
    paid_fallback: Literal["disabled", "openai_image_edit"] = "disabled"


class SpendingPolicy(BaseModel):
    monthly_limit_usd: float = Field(0.0, ge=0)
    require_explicit_approval: bool = True
    auto_retry_paid: Literal[False] = False  # not configurable on purpose

    @model_validator(mode="after")
    def _approval_required(self):
        if not self.require_explicit_approval:
            raise ValueError("require_explicit_approval cannot be disabled; per-request approval is mandatory")
        return self


class AIPolicy(BaseModel):
    default_policy: Literal["open_source_first"] = "open_source_first"
    image_analysis: ImageAnalysisPolicy = ImageAnalysisPolicy()
    outfit_recommendation: RecommendationPolicy = RecommendationPolicy()
    virtual_tryon: TryOnPolicy = TryOnPolicy()
    creative_editing: CreativePolicy = CreativePolicy()
    spending: SpendingPolicy = SpendingPolicy()

    @model_validator(mode="after")
    def _paid_fallback_must_be_enabled(self):
        pf = self.virtual_tryon.paid_fallback
        if pf != "disabled" and pf not in self.virtual_tryon.enabled_paid_providers:
            raise ValueError(f"paid_fallback {pf!r} must also be listed in enabled_paid_providers")
        return self


def load_ai_policy(path: str | os.PathLike | None) -> AIPolicy:
    if not path or not Path(path).exists():
        return AIPolicy()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return AIPolicy.model_validate(raw.get("ai", raw))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOOKLAB_", env_file=".env", extra="ignore")

    env: Literal["development", "production", "test"] = "development"
    data_dir: Path = REPO_ROOT / "data"
    database_url: str = ""  # default: sqlite in data_dir
    storage_backend: Literal["local", "s3"] = "local"
    s3_bucket: str = ""
    s3_endpoint_url: str = ""
    s3_region: str = ""
    ai_config_path: Path = REPO_ROOT / "config" / "ai.yaml"

    auth_mode: Literal["single_user", "multi_user"] = "single_user"
    session_days: int = 30
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    cookie_secure: bool = False

    worker_enabled: bool = True
    worker_poll_seconds: float = 0.5
    max_upload_mb: int = 15
    max_image_side: int = 2048

    # models
    fashn_weights_dir: Path = REPO_ROOT / "weights" / "fashn-vton-1.5"
    fashn_parser_mode: Literal["parser_free", "full"] = "parser_free"
    fashn_allow_cpu: bool = False
    fashn_idle_unload_seconds: int = 600
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_vision_model: str = "qwen3-vl:2b"
    florence_model_id: str = "microsoft/Florence-2-base"

    # remote GPU (open-source model, self-hosted)
    remote_gpu_url: str = ""
    remote_gpu_token: str = ""
    remote_gpu_protocol: Literal["http", "runpod"] = "http"
    remote_gpu_hourly_usd: float = 0.0
    remote_gpu_cold_start_seconds: float = 60.0

    # paid APIs (disabled unless listed in ai.yaml enabled_paid_providers)
    fashn_api_key: str = ""
    fashn_api_price_usd: float = 0.075
    openai_api_key: str = ""
    openai_image_model: str = "gpt-image-1.5"
    openai_price_usd: float = 0.08

    # privacy
    default_generated_retention_days: int | None = None

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{self.data_dir / 'looklab.db'}"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
