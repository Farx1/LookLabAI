"""Central provider registry and routing (open-source first, paid never implicit)."""

from __future__ import annotations

from dataclasses import asdict

from looklab_ml.creative.qwen_image_edit import QwenImageEditProvider
from looklab_ml.tryon import (
    FashnApiProvider,
    FashnRuntime,
    LocalFashnProvider,
    MockTryOnProvider,
    OpenAIImageEditProvider,
    RemoteFashnProvider,
    RemoteGpuConfig,
)
from looklab_ml.tryon.layers import LayerPlanner
from looklab_ml.vision import build_analyzer
from looklab_ml.vision.outfit import analyze_outfit

from .config import PAID_TRYON_PROVIDERS, TRYON_PROVIDERS, AIPolicy, Settings


class ProviderError(Exception):
    def __init__(self, status: int, code: str, message: str, extra: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra or {}


class ProviderRegistry:
    def __init__(self, settings: Settings, policy: AIPolicy, overrides: dict | None = None):
        self.settings = settings
        self.policy = policy
        self.fashn_runtime = FashnRuntime(
            weights_dir=str(settings.fashn_weights_dir),
            parser_mode=settings.fashn_parser_mode,
            idle_unload_seconds=settings.fashn_idle_unload_seconds,
            allow_cpu=settings.fashn_allow_cpu,
        )
        planner = LayerPlanner(settings.ollama_url, settings.ollama_vision_model, str(settings.sam2_weights))
        self.outfit_analyze = lambda image, person: analyze_outfit(planner, image, person)  # tests swap it for a fake
        self.tryon = {
            "local_fashn": LocalFashnProvider(self.fashn_runtime, planner),
            "remote_fashn": RemoteFashnProvider(
                RemoteGpuConfig(
                    url=settings.remote_gpu_url,
                    token=settings.remote_gpu_token,
                    protocol=settings.remote_gpu_protocol,
                    gpu_hourly_usd=settings.remote_gpu_hourly_usd,
                    expected_cold_start_seconds=settings.remote_gpu_cold_start_seconds,
                )
            ),
            "fashn_api": FashnApiProvider(settings.fashn_api_key, price_per_image_usd=settings.fashn_api_price_usd),
            "openai_image_edit": OpenAIImageEditProvider(
                settings.openai_api_key, model=settings.openai_image_model, price_per_image_usd=settings.openai_price_usd
            ),
            "mock": MockTryOnProvider(),
        }
        if overrides:
            self.tryon.update(overrides)
        assert set(self.tryon) == set(TRYON_PROVIDERS), "provider registry out of sync with config"
        analyzer_settings = {
            "ollama_url": settings.ollama_url,
            "ollama_vision_model": settings.ollama_vision_model,
            "florence_model_id": settings.florence_model_id,
        }
        self.analyzers = {n: build_analyzer(n, analyzer_settings) for n in ("cv_basic", "local_qwen3_vl", "local_florence2")}
        self.creative = QwenImageEditProvider()

    # ------------------------------------------------------------ try-on
    def is_enabled(self, name: str) -> bool:
        if name in PAID_TRYON_PROVIDERS:
            return name in self.policy.virtual_tryon.enabled_paid_providers
        if name == "mock":
            return self.policy.virtual_tryon.allow_mock
        return name in self.tryon

    def tryon_status(self) -> list[dict]:
        out = []
        for name, p in self.tryon.items():
            ok, why = p.available()
            out.append(
                {
                    **asdict(p.info),
                    "enabled": self.is_enabled(name),
                    "available": ok,
                    "reason": why,
                    "role": self._role(name),
                }
            )
        return out

    def layer_status(self) -> dict:
        """Qwen3-VL (Ollama) and SAM2 availability for the layered try-on mode."""
        return self.tryon["local_fashn"].layer_status() if "local_fashn" in self.tryon else {}

    def _role(self, name: str) -> str:
        vt = self.policy.virtual_tryon
        if name == vt.primary:
            return "primary"
        if name == vt.secondary:
            return "secondary"
        if name == vt.paid_fallback:
            return "paid_fallback (manual only)"
        if name == "mock":
            return "development"
        return "optional"

    def resolve_tryon(self, requested: str):
        """Return the provider to use. ``auto`` only ever picks free,
        local, open-source execution; anything billed must be named."""
        if requested == "auto":
            vt = self.policy.virtual_tryon
            for name in (vt.primary,):
                p = self.tryon[name]
                ok, _ = p.available()
                if ok:
                    return p
            raise ProviderError(
                409,
                "no_free_provider_available",
                "No free try-on engine is available on this machine. Choose one explicitly.",
                {"providers": self.tryon_status()},
            )
        if requested not in self.tryon:
            raise ProviderError(400, "unknown_provider", f"unknown provider {requested!r}")
        if not self.is_enabled(requested):
            raise ProviderError(
                403,
                "provider_disabled",
                f"{requested} is disabled. Paid providers must be listed in ai.virtual_tryon.enabled_paid_providers.",
            )
        p = self.tryon[requested]
        ok, why = p.available()
        if not ok:
            raise ProviderError(409, "provider_unavailable", f"{requested}: {why}")
        return p

    def analysis_status(self) -> list[dict]:
        rows = []
        for name, a in self.analyzers.items():
            ok, why = a.available()
            rows.append({"name": name, "available": ok, "reason": why})
        return rows

    def model_analyzer(self) -> str | None:
        """Registry key of the configured VLM analyzer if available, else its fallback, else None."""
        ia = self.policy.image_analysis
        for name in (ia.model, ia.fallback):
            if name and name != "none" and self.analyzers[name].available()[0]:
                return name
        return None
