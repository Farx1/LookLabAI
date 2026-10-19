"""Thin runtime around the *official* FASHN VTON v1.5 inference code.

We do not reimplement the network. This module only adds:

* lazy loading and idle unloading (GPU memory is precious);
* a serialising lock (one diffusion job at a time per process);
* timing / peak-VRAM measurement;
* a **parser-free mode** (see LICENSE_AUDIT.md). In the official pipeline the
  FASHN Human Parser (SegFormer-B4, NVIDIA *non-commercial* license) output is
  computed but *discarded* when ``segmentation_free=True`` and the garment is
  a ``flat-lay`` product shot. Parser-free mode skips loading those weights
  for exactly that case and refuses every other case, so results are
  identical to the official pipeline for the inputs it accepts.
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import logging
import sys
import threading
import time
import types
from dataclasses import dataclass
from typing import Literal

from PIL import Image

log = logging.getLogger(__name__)

ParserMode = Literal["parser_free", "full"]


def _install_parser_interface_shim() -> None:
    """Register a minimal ``fashn_human_parser`` module when the real package
    is not installed. It only declares label names (interface constants) so
    ``fashn_vton`` can be imported; it contains no model code or weights."""
    if "fashn_human_parser" in sys.modules or importlib.util.find_spec("fashn_human_parser") is not None:
        return
    mod = types.ModuleType("fashn_human_parser")
    labels = [
        "background", "face", "hair", "top", "dress", "skirt", "pants", "belt", "bag",
        "hat", "scarf", "glasses", "arms", "hands", "legs", "feet", "torso", "jewelry",
    ]  # fmt: skip
    mod.IDS_TO_LABELS = dict(enumerate(labels))
    mod.LABELS_TO_IDS = {v: k for k, v in mod.IDS_TO_LABELS.items()}
    mod.CATEGORY_TO_BODY_COVERAGE = {"tops": "upper", "bottoms": "lower", "one-pieces": "full"}
    mod.BODY_COVERAGE_TO_LABELS = {
        "upper": ["top", "dress", "scarf"],
        "lower": ["skirt", "pants", "belt"],
        "full": ["top", "dress", "scarf", "skirt", "pants", "belt"],
    }
    mod.IDENTITY_LABELS = ["face", "hair", "jewelry", "bag", "glasses", "hat"]

    class FashnHumanParser:  # noqa: D401 - placeholder that must never run
        def __init__(self, *a, **k):
            raise RuntimeError("fashn-human-parser is not installed (parser_free mode)")

    mod.FashnHumanParser = FashnHumanParser
    mod.__looklab_shim__ = True
    sys.modules["fashn_human_parser"] = mod


class _DiscardedParser:
    """Stands in for the human parser in parser-free mode. Its output is only
    consumed by masking steps that parser-free mode disables."""

    def predict(self, image_np):
        import numpy as np

        return np.zeros(image_np.shape[:2], dtype=np.uint8)


@dataclass
class RunMetrics:
    elapsed_seconds: float
    model_load_seconds: float | None
    gpu_memory_peak_mb: float | None
    device: str
    dtype: str


class FashnRuntime:
    MODEL_ID = "fashn-ai/fashn-vton-1.5"

    def __init__(
        self,
        weights_dir: str,
        device: str | None = None,
        parser_mode: ParserMode = "parser_free",
        idle_unload_seconds: float = 600,
        allow_cpu: bool = False,
    ):
        self.weights_dir = weights_dir
        self.device = device
        self.parser_mode = parser_mode
        self.idle_unload_seconds = idle_unload_seconds
        self.allow_cpu = allow_cpu
        self._pipeline = None
        self._lock = threading.Lock()
        self._last_used = 0.0

    # ------------------------------------------------------------- discovery
    @staticmethod
    def torch_status() -> dict:
        try:
            import torch
        except ImportError:
            return {"torch": False, "cuda": False}
        info = {"torch": True, "torch_version": torch.__version__, "cuda": bool(torch.cuda.is_available())}
        if info["cuda"]:
            p = torch.cuda.get_device_properties(0)
            info.update(
                gpu_name=p.name,
                gpu_total_mb=round(p.total_memory / 2**20),
                bf16=bool(torch.cuda.is_bf16_supported()),
            )
        return info

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("fashn_vton") is None:
            return False, "fashn_vton not installed (see docs/TRYON_SETUP.md)"
        if self.parser_mode == "full" and importlib.util.find_spec("fashn_human_parser") is None:
            return False, "parser_mode=full requires the fashn-human-parser package"
        import os

        missing = [
            p
            for p in ("model.safetensors", "dwpose/yolox_l.onnx", "dwpose/dw-ll_ucoco_384.onnx")
            if not os.path.exists(os.path.join(self.weights_dir, p))
        ]
        if missing:
            return False, f"weights missing in {self.weights_dir}: {', '.join(missing)}"
        ts = self.torch_status()
        if not ts.get("cuda") and not self.allow_cpu:
            return False, "no CUDA GPU detected (CPU inference is disabled: see FEASIBILITY.md for measured CPU cost)"
        return True, f"ready on {'cuda:' + ts.get('gpu_name', '') if ts.get('cuda') else 'cpu'}"

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    # ------------------------------------------------------------- lifecycle
    def _load(self) -> float:
        if self._pipeline is not None:
            return 0.0
        t0 = time.perf_counter()
        if self.parser_mode == "parser_free":
            _install_parser_interface_shim()
        fv = importlib.import_module("fashn_vton")
        base_cls = fv.TryOnPipeline
        if self.parser_mode == "parser_free":

            class ParserFreePipeline(base_cls):  # type: ignore[misc, valid-type]
                def _setup_hp_model(self):
                    self.logger.info("parser_free mode: human parser not loaded")
                    self.hp_model = _DiscardedParser()

            cls = ParserFreePipeline
        else:
            cls = base_cls
        self._pipeline = cls(weights_dir=self.weights_dir, device=self.device)
        return time.perf_counter() - t0

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    def _unload_locked(self) -> None:
        if self._pipeline is None:
            return
        self._pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("FASHN pipeline unloaded")

    def maybe_unload_idle(self) -> bool:
        if self._pipeline is not None and time.time() - self._last_used > self.idle_unload_seconds:
            if self._lock.acquire(blocking=False):
                try:
                    self._unload_locked()
                    return True
                finally:
                    self._lock.release()
        return False

    # ------------------------------------------------------------- inference
    def validate(self, garment_photo_type: str, segmentation_free: bool = True) -> None:
        if self.parser_mode == "parser_free" and (garment_photo_type != "flat-lay" or not segmentation_free):
            raise ValueError(
                "parser_free mode supports flat-lay garment photos with segmentation_free=True only. "
                "Garments photographed on a person need the FASHN Human Parser, which is licensed for "
                "non-commercial research/evaluation only; set FASHN_PARSER_MODE=full to accept those terms."
            )

    def run(
        self,
        person: Image.Image,
        garment: Image.Image,
        category: str,
        garment_photo_type: str = "flat-lay",
        num_timesteps: int = 30,
        guidance_scale: float = 1.5,
        seed: int = 42,
        num_samples: int = 1,
        segmentation_free: bool = True,
    ) -> tuple[list[Image.Image], RunMetrics]:
        self.validate(garment_photo_type, segmentation_free)
        with self._lock:
            load_s = self._load()
            import torch

            pipe = self._pipeline
            is_cuda = pipe.device.type == "cuda"
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = pipe(
                person_image=person.convert("RGB"),
                garment_image=garment.convert("RGB"),
                category=category,
                garment_photo_type=garment_photo_type,
                num_samples=num_samples,
                num_timesteps=num_timesteps,
                guidance_scale=guidance_scale,
                seed=seed,
                segmentation_free=segmentation_free,
            )
            if is_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 2**20 if is_cuda else None
            self._last_used = time.time()
            return out.images, RunMetrics(
                elapsed_seconds=elapsed,
                model_load_seconds=load_s or None,
                gpu_memory_peak_mb=peak,
                device=str(pipe.device),
                dtype=str(pipe.inference_dtype),
            )
