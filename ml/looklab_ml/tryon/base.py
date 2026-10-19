"""Provider-neutral virtual try-on interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Literal, Protocol

TryOnCategory = Literal["tops", "bottoms", "one-pieces"]
GarmentPhotoType = Literal["model", "flat-lay"]


@dataclass
class TryOnRequest:
    person_image_path: str
    garment_image_path: str
    category: TryOnCategory
    garment_photo_type: GarmentPhotoType = "flat-lay"
    num_timesteps: int = 30
    guidance_scale: float = 1.5
    seed: int = 42
    num_samples: int = 1
    output_path: str = ""

    def params(self) -> dict:
        d = asdict(self)
        for k in ("person_image_path", "garment_image_path", "output_path"):
            d.pop(k)
        return d


@dataclass
class TryOnResult:
    output_image_path: str
    provider: str
    model: str
    elapsed_seconds: float
    estimated_cost_usd: float
    device: str = "unknown"
    gpu_memory_peak_mb: float | None = None
    model_load_seconds: float | None = None
    actual_cost_usd: float | None = None
    is_mock: bool = False
    notes: list[str] = field(default_factory=list)


@dataclass
class ProviderInfo:
    name: str
    model: str
    kind: Literal["mock", "open_source", "paid_api"]
    location: Literal["local", "remote_self_hosted", "third_party"]
    fidelity: Literal["mock", "garment_tryon", "generative_edit"]
    sends_images_externally: bool
    paid: bool
    description: str


class CancelledError(Exception):
    pass


class ProviderUnavailable(Exception):
    """Raised when a provider cannot run here (missing GPU, weights, key...)."""


CancelCheck = Callable[[], bool]


class VirtualTryOnProvider(Protocol):
    info: ProviderInfo

    def available(self) -> tuple[bool, str]: ...

    def estimate_cost_usd(self, request: TryOnRequest) -> float: ...

    async def generate(self, request: TryOnRequest, should_cancel: CancelCheck | None = None) -> TryOnResult: ...


class AsyncJobProvider(Protocol):
    """Providers whose backend is itself a job queue (FASHN API, RunPod)."""

    async def submit(self, request: TryOnRequest) -> str: ...

    async def poll(self, external_id: str) -> dict: ...

    async def cancel(self, external_id: str) -> None: ...
