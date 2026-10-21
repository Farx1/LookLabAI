"""Try-on provider adapters.

| name               | model                    | runs where            | cost            |
|--------------------|--------------------------|-----------------------|-----------------|
| mock               | none (labelled preview)  | local CPU             | 0               |
| local_fashn        | FASHN VTON v1.5          | local CUDA GPU        | 0 (electricity) |
| remote_fashn       | FASHN VTON v1.5          | your GPU worker       | GPU-seconds     |
| fashn_api          | FASHN hosted API         | third party (paid)    | per image       |
| openai_image_edit  | OpenAI image edit        | third party (paid)    | per image       |

Paid adapters never decide to run by themselves: the API layer's spending
guard approves each request explicitly and never retries them automatically.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from dataclasses import dataclass

import httpx
from PIL import Image, ImageDraw, ImageOps

from .base import (
    CancelCheck,
    CancelledError,
    ProviderInfo,
    ProviderUnavailable,
    TryOnRequest,
    TryOnResult,
)
from .fashn_runtime import FashnRuntime
from .layers import LayerPlanner

log = logging.getLogger(__name__)


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_b64(path: str, max_side: int = 1536) -> str:
    img = _load_rgb(path)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- mock
class MockTryOnProvider:
    """Deterministic stand-in used in CPU-only development and tests.

    The output is visibly labelled as a mock so it can never be mistaken for
    a try-on result.
    """

    info = ProviderInfo(
        name="mock",
        model="none",
        kind="mock",
        location="local",
        fidelity="mock",
        sends_images_externally=False,
        paid=False,
        description="Labelled side-by-side preview for development. No AI model is run.",
    )

    def __init__(self, delay_seconds: float = 0.0):
        self.delay_seconds = delay_seconds

    def available(self) -> tuple[bool, str]:
        return True, "always available (development only)"

    def estimate_cost_usd(self, request: TryOnRequest) -> float:
        return 0.0

    async def generate(self, request: TryOnRequest, should_cancel: CancelCheck | None = None) -> TryOnResult:
        t0 = time.perf_counter()
        steps = max(1, int(self.delay_seconds / 0.05))
        for _ in range(steps):
            if should_cancel and should_cancel():
                raise CancelledError()
            await asyncio.sleep(self.delay_seconds / steps if self.delay_seconds else 0)
        person = _load_rgb(request.person_image_path)
        garment = _load_rgb(request.garment_image_path)
        person.thumbnail((576, 864))
        canvas = Image.new("RGB", (576, 864), (247, 241, 232))
        canvas.paste(person, ((576 - person.width) // 2, (864 - person.height) // 2))
        garment.thumbnail((200, 200))
        canvas.paste(garment, (576 - garment.width - 16, 864 - garment.height - 16))
        d = ImageDraw.Draw(canvas)
        d.rectangle((0, 0, 576, 34), fill=(110, 20, 40))
        d.text((12, 10), "MOCK PREVIEW - no try-on model was run", fill=(255, 255, 255))
        os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
        canvas.save(request.output_path, format="PNG")
        return TryOnResult(
            output_image_path=request.output_path,
            provider="mock",
            model="none",
            elapsed_seconds=time.perf_counter() - t0,
            estimated_cost_usd=0.0,
            device="cpu",
            is_mock=True,
            notes=["mock output: not a virtual try-on"],
        )


# ----------------------------------------------------------------- local FASHN
# shortcut: fixed share of the old outer layer left bare by the new one (0.16 for a waist-length biker over a hip-length puffer), tune on more photos
UNCOVERED_RATIO = 0.10


class LocalFashnProvider:
    info = ProviderInfo(
        name="local_fashn",
        model="fashn-ai/fashn-vton-1.5",
        kind="open_source",
        location="local",
        fidelity="garment_tryon",
        sends_images_externally=False,
        paid=False,
        description="FASHN VTON v1.5 on this machine's NVIDIA GPU. Images never leave the machine.",
    )

    def __init__(self, runtime: FashnRuntime, planner: LayerPlanner | None = None):
        self.runtime = runtime
        self.planner = planner

    def available(self) -> tuple[bool, str]:
        return self.runtime.available()

    def layer_status(self) -> dict:
        """Availability of the layered-mode models. Local only: Ollama is never started from here."""
        if self.planner is None:
            return {}
        return {k: {"available": ok, "reason": why} for k, (ok, why) in self.planner.available().items()}

    def estimate_cost_usd(self, request: TryOnRequest) -> float:
        return 0.0

    async def generate(self, request: TryOnRequest, should_cancel: CancelCheck | None = None) -> TryOnResult:
        ok, why = self.available()
        if not ok:
            raise ProviderUnavailable(why)
        if should_cancel and should_cancel():
            raise CancelledError()
        t0 = time.perf_counter()
        person = await asyncio.to_thread(_load_rgb, request.person_image_path)  # decode / PNG encode of a 3 MP photo would stall the event loop
        garment = await asyncio.to_thread(_load_rgb, request.garment_image_path)
        plan, reason, stages = None, "layered mode not configured", {}
        if self.planner is not None:
            down = [f"{k}: {why}" for k, (ok, why) in self.planner.available().items() if not ok]
            reason = "; ".join(down)
            if not down:
                try:
                    # Qwen (3 GB) + SAM2 + FASHN (5.5 GB) exceed 8 GB: FASHN leaves the GPU while they run
                    # ponytail: FASHN is reloaded (~10 s) per step. Measured 2026-10-20: FASHN 5.6 GB + Qwen 3.4 GB > 8.2 GB, so Qwen cannot stay beside it; upgrade = a bigger card.
                    await asyncio.to_thread(self.runtime.unload)
                    plan = await asyncio.to_thread(self.planner.plan, person, request.garment_kind, request.over_top)
                    stages.update(qwen=plan.seconds["qwen"], sam2=plan.seconds["sam2"])
                except Exception as e:  # noqa: BLE001 - never fail the try-on because the optional layer step failed
                    log.exception("layered planning failed; falling back to segfree")
                    reason = f"layered mode failed: {type(e).__name__}: {e}"
        if should_cancel and should_cancel():
            raise CancelledError()
        images, m = await asyncio.to_thread(
            self.runtime.run,
            person,
            garment,
            request.category,
            request.garment_photo_type,
            request.num_timesteps,
            request.guidance_scale,
            request.seed,
            request.num_samples,
            True,
            plan.label_map if plan else None,
        )
        stages["fashn"] = m.elapsed_seconds
        out, mode, uncovered, edit_fraction, notes = images[0], "segfree", False, None, []
        if plan:
            try:
                out, info = await asyncio.to_thread(self.planner.finish, person, out, plan, m.ca_mask)
                mode, edit_fraction = "layered", info["edit_fraction"]
                reason = "garment layer repainted, everything else is the original photo"
                uncovered = request.garment_kind == "outerwear" and info["uncovered_ratio"] > UNCOVERED_RATIO
                stages["sam2"] += info["sam2"]
                stages["recompose"] = info["recompose"]
                notes += [f"outside_mask_max_diff={info['outside_mask_max_diff']}", f"uncovered_ratio={info['uncovered_ratio']:.2f}"]
            except Exception as e:  # noqa: BLE001 - keep the raw FASHN output
                log.exception("layered recomposition failed; raw FASHN output kept")
                reason = f"recomposition failed, raw output kept: {type(e).__name__}: {e}"
        os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
        await asyncio.to_thread(out.save, request.output_path, format="PNG")
        return TryOnResult(
            output_image_path=request.output_path,
            provider=self.info.name,
            model=self.info.model,
            elapsed_seconds=time.perf_counter() - t0 if plan else m.elapsed_seconds,
            estimated_cost_usd=0.0,
            device=m.device,
            gpu_memory_peak_mb=m.gpu_memory_peak_mb,
            model_load_seconds=m.model_load_seconds,
            notes=[f"parser_mode={self.runtime.parser_mode}", f"dtype={m.dtype}", f"fidelity_mode={mode}", *notes],
            fidelity_mode=mode,
            fidelity_reason=reason,
            uncovered_area=uncovered,
            stage_seconds={k: round(v, 2) for k, v in stages.items()},
            edit_fraction=edit_fraction,
        )


# ---------------------------------------------------------------- remote FASHN
@dataclass
class RemoteGpuConfig:
    url: str = ""
    token: str = ""
    protocol: str = "http"  # "http" (our gpu-worker) | "runpod" (serverless)
    gpu_hourly_usd: float = 0.0
    expected_cold_start_seconds: float = 60.0
    expected_inference_seconds: float = 15.0
    timeout_seconds: float = 600.0


class RemoteFashnProvider:
    """Same open-source model on a self-hosted remote GPU (RunPod, Vast.ai, a
    Hugging Face endpoint...). Photos are sent to that machine, so it needs
    the user's external-processing consent."""

    info = ProviderInfo(
        name="remote_fashn",
        model="fashn-ai/fashn-vton-1.5",
        kind="open_source",
        location="remote_self_hosted",
        fidelity="garment_tryon",
        sends_images_externally=True,
        paid=True,  # GPU time is billed by the host
        description="FASHN VTON v1.5 on your own remote GPU worker (billed by the GPU host per second).",
    )

    def __init__(self, cfg: RemoteGpuConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self._transport = transport

    def available(self) -> tuple[bool, str]:
        if not self.cfg.url:
            return False, "REMOTE_GPU_URL not configured"
        return True, f"{self.cfg.protocol} worker at {self.cfg.url}"

    def estimate_cost_usd(self, request: TryOnRequest) -> float:
        secs = self.cfg.expected_inference_seconds * (request.num_timesteps / 30) * request.num_samples
        secs += self.cfg.expected_cold_start_seconds
        return round(self.cfg.gpu_hourly_usd * secs / 3600, 4)

    def _client(self) -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {self.cfg.token}"} if self.cfg.token else {}
        return httpx.AsyncClient(timeout=self.cfg.timeout_seconds, headers=headers, transport=self._transport)

    async def generate(self, request: TryOnRequest, should_cancel: CancelCheck | None = None) -> TryOnResult:
        ok, why = self.available()
        if not ok:
            raise ProviderUnavailable(why)
        if self.cfg.protocol == "runpod":
            return await self._generate_runpod(request, should_cancel)
        t0 = time.perf_counter()
        async with self._client() as client:
            with open(request.person_image_path, "rb") as pf, open(request.garment_image_path, "rb") as gf:
                files = {"person": ("person.png", pf.read()), "garment": ("garment.png", gf.read())}
            r = await client.post(
                f"{self.cfg.url.rstrip('/')}/v1/tryon",
                files=files,
                data={k: str(v) for k, v in request.params().items()},
            )
            r.raise_for_status()
        wall = time.perf_counter() - t0
        os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
        with open(request.output_path, "wb") as f:
            f.write(r.content)
        h = r.headers
        inference = float(h.get("x-elapsed-seconds", wall))
        load = float(h["x-model-load-seconds"]) if h.get("x-model-load-seconds") else None
        billed = wall  # conservative: the whole request occupied the worker
        return TryOnResult(
            output_image_path=request.output_path,
            provider=self.info.name,
            model=self.info.model,
            elapsed_seconds=inference,
            estimated_cost_usd=round(self.cfg.gpu_hourly_usd * billed / 3600, 5),
            device=h.get("x-device", "remote"),
            gpu_memory_peak_mb=float(h["x-gpu-memory-peak-mb"]) if h.get("x-gpu-memory-peak-mb") else None,
            model_load_seconds=load,
            notes=[f"wall_clock={wall:.2f}s"],
        )

    async def _generate_runpod(self, request: TryOnRequest, should_cancel: CancelCheck | None) -> TryOnResult:
        payload = {
            "input": {
                "person_b64": _jpeg_b64(request.person_image_path),
                "garment_b64": _jpeg_b64(request.garment_image_path),
                **request.params(),
            }
        }
        base = self.cfg.url.rstrip("/")
        t0 = time.perf_counter()
        async with self._client() as client:
            r = await client.post(f"{base}/run", json=payload)
            r.raise_for_status()
            job_id = r.json()["id"]
            while True:
                if should_cancel and should_cancel():
                    await client.post(f"{base}/cancel/{job_id}")
                    raise CancelledError()
                s = (await client.get(f"{base}/status/{job_id}")).json()
                st = s.get("status")
                if st == "COMPLETED":
                    break
                if st in ("FAILED", "CANCELLED", "TIMED_OUT"):
                    raise RuntimeError(f"runpod job {st}: {s.get('error', '')}")
                await asyncio.sleep(2)
        out = s["output"]
        os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
        with open(request.output_path, "wb") as f:
            f.write(base64.b64decode(out["image_b64"]))
        exec_s = (s.get("executionTime") or 0) / 1000
        delay_s = (s.get("delayTime") or 0) / 1000
        return TryOnResult(
            output_image_path=request.output_path,
            provider=self.info.name,
            model=self.info.model,
            elapsed_seconds=float(out.get("elapsed_seconds", exec_s)),
            estimated_cost_usd=round(self.cfg.gpu_hourly_usd * exec_s / 3600, 5),
            device=out.get("device", "runpod"),
            gpu_memory_peak_mb=out.get("gpu_memory_peak_mb"),
            model_load_seconds=out.get("model_load_seconds"),
            notes=[f"queue_delay={delay_s:.1f}s", f"wall_clock={time.perf_counter() - t0:.1f}s"],
        )


# ------------------------------------------------------------------- FASHN API
class FashnApiProvider:
    info = ProviderInfo(
        name="fashn_api",
        model="tryon-v1.6",
        kind="paid_api",
        location="third_party",
        fidelity="garment_tryon",
        sends_images_externally=True,
        paid=True,
        description="FASHN's hosted try-on API (paid per image, separate account and key).",
    )

    def __init__(self, api_key: str, model_name: str = "tryon-v1.6", price_per_image_usd: float = 0.075,
                 base_url: str = "https://api.fashn.ai/v1", transport: httpx.AsyncBaseTransport | None = None):  # fmt: skip
        self.api_key = api_key
        self.model_name = model_name
        self.price = price_per_image_usd
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self.info = ProviderInfo(**{**self.info.__dict__, "model": model_name})

    def available(self) -> tuple[bool, str]:
        return (True, "API key configured") if self.api_key else (False, "FASHN_API_KEY not set")

    def estimate_cost_usd(self, request: TryOnRequest) -> float:
        return round(self.price * request.num_samples, 4)

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=120, headers={"Authorization": f"Bearer {self.api_key}"}, transport=self._transport)

    async def submit(self, request: TryOnRequest) -> str:
        body = {
            "model_name": self.model_name,
            "inputs": {
                # base64 inputs: FASHN documents that base64 image data is not persisted
                "model_image": "data:image/jpeg;base64," + _jpeg_b64(request.person_image_path),
                "garment_image": "data:image/jpeg;base64," + _jpeg_b64(request.garment_image_path),
                "category": request.category,
                "garment_photo_type": request.garment_photo_type,
                "num_samples": request.num_samples,
                "seed": request.seed,
                "return_base64": True,
            },
        }
        async with self._client() as c:
            r = await c.post(f"{self.base_url}/run", json=body)
            r.raise_for_status()
            data = r.json()
        if data.get("error"):
            raise RuntimeError(f"FASHN API error: {data['error']}")
        return data["id"]

    async def poll(self, external_id: str) -> dict:
        async with self._client() as c:
            r = await c.get(f"{self.base_url}/status/{external_id}")
            r.raise_for_status()
            return r.json()

    async def cancel(self, external_id: str) -> None:  # FASHN has no documented cancel endpoint
        return None

    async def generate(self, request: TryOnRequest, should_cancel: CancelCheck | None = None) -> TryOnResult:
        ok, why = self.available()
        if not ok:
            raise ProviderUnavailable(why)
        t0 = time.perf_counter()
        job = await self.submit(request)
        while True:
            s = await self.poll(job)
            st = s.get("status")
            if st == "completed":
                break
            if st in ("failed", "canceled", "time_out"):
                raise RuntimeError(f"FASHN API job {st}: {s.get('error')}")
            if should_cancel and should_cancel():
                # cannot stop a remote job; the image is still billed
                raise CancelledError("cancelled locally; the remote job may still be billed")
            await asyncio.sleep(1.5)
        out = s["output"][0]
        img_bytes = await _fetch_output(out, self._transport)
        os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
        with open(request.output_path, "wb") as f:
            f.write(img_bytes)
        cost = self.estimate_cost_usd(request)
        return TryOnResult(
            output_image_path=request.output_path,
            provider=self.info.name,
            model=self.model_name,
            elapsed_seconds=time.perf_counter() - t0,
            estimated_cost_usd=cost,
            actual_cost_usd=None,  # not returned by the API; see FASHN dashboard
            device="fashn-cloud",
            notes=[f"fashn_job_id={job}"],
        )


async def _fetch_output(out: str, transport=None) -> bytes:
    if out.startswith("data:"):
        return base64.b64decode(out.split(",", 1)[1])
    async with httpx.AsyncClient(timeout=60, transport=transport) as c:
        r = await c.get(out)
        r.raise_for_status()
        return r.content


# ------------------------------------------------------------ OpenAI image edit
class OpenAIImageEditProvider:
    """Generative image editing. It re-renders the photo; it is *not* a
    garment-accurate try-on and is labelled as such in the UI."""

    info = ProviderInfo(
        name="openai_image_edit",
        model="gpt-image-1.5",
        kind="paid_api",
        location="third_party",
        fidelity="generative_edit",
        sends_images_externally=True,
        paid=True,
        description="OpenAI Images API edit (paid, separate API billing; ChatGPT plans do not include API credits).",
    )

    PROMPT = (
        "Dress the person in the first image in the garment shown in the second image. "
        "Keep the person's face, hair, skin tone, body shape and proportions, pose and background exactly as they are. "
        "Reproduce the garment's color, pattern and details faithfully. Photorealistic."
    )

    def __init__(self, api_key: str, model: str = "gpt-image-1.5", price_per_image_usd: float = 0.08,
                 quality: str = "medium", transport: httpx.AsyncBaseTransport | None = None):  # fmt: skip
        self.api_key = api_key
        self.model = model
        self.price = price_per_image_usd
        self.quality = quality
        self._transport = transport
        self.info = ProviderInfo(**{**self.info.__dict__, "model": model})

    def available(self) -> tuple[bool, str]:
        return (True, "API key configured") if self.api_key else (False, "OPENAI_API_KEY not set")

    def estimate_cost_usd(self, request: TryOnRequest) -> float:
        return round(self.price * request.num_samples, 4)

    async def generate(self, request: TryOnRequest, should_cancel: CancelCheck | None = None) -> TryOnResult:
        ok, why = self.available()
        if not ok:
            raise ProviderUnavailable(why)
        t0 = time.perf_counter()
        files = [
            ("image[]", ("person.jpg", base64.b64decode(_jpeg_b64(request.person_image_path)), "image/jpeg")),
            ("image[]", ("garment.jpg", base64.b64decode(_jpeg_b64(request.garment_image_path)), "image/jpeg")),
        ]
        data = {"model": self.model, "prompt": self.PROMPT, "n": str(request.num_samples), "size": "1024x1536", "quality": self.quality}
        async with httpx.AsyncClient(timeout=300, headers={"Authorization": f"Bearer {self.api_key}"}, transport=self._transport) as c:
            r = await c.post("https://api.openai.com/v1/images/edits", data=data, files=files)
            r.raise_for_status()
            body = r.json()
        b64 = body["data"][0]["b64_json"]
        os.makedirs(os.path.dirname(request.output_path) or ".", exist_ok=True)
        with open(request.output_path, "wb") as f:
            f.write(base64.b64decode(b64))
        return TryOnResult(
            output_image_path=request.output_path,
            provider=self.info.name,
            model=self.model,
            elapsed_seconds=time.perf_counter() - t0,
            estimated_cost_usd=self.estimate_cost_usd(request),
            device="openai-cloud",
            notes=["generative edit: garment fidelity is not guaranteed", f"usage={body.get('usage')}"],
        )
