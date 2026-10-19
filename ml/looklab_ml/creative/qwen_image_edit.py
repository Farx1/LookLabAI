"""Creative full-outfit visualisation with Qwen-Image-Edit (multi-image).

This is deliberately *separate* from garment try-on: a 20B-parameter
instruction-following editor re-imagines the whole photo from several
reference images. It is good for "how might this whole look feel", not for
checking how a specific garment fits. The UI labels it accordingly.

Status: adapter implemented, disabled by default (``CREATIVE_EDITING_ENABLED``)
and *not validated on hardware in this repository* (see FEASIBILITY.md):
bf16 weights need roughly 40+ GB of VRAM; community 4-bit builds target
~20-24 GB cards.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from dataclasses import dataclass, field

from PIL import Image

PROMPT_TEMPLATE = (
    "Show the person from image 1 wearing this complete outfit: {pieces}. "
    "The garments are shown in the other reference images. Preserve the person's identity, face, "
    "body shape and proportions, skin tone and pose. Keep a natural, realistic photo style."
)


@dataclass
class CreativeRequest:
    person_image_path: str
    garment_image_paths: list[str]
    garment_labels: list[str]
    output_path: str
    seed: int = 0
    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0
    extra_prompt: str = ""


@dataclass
class CreativeResult:
    output_image_path: str
    model: str
    elapsed_seconds: float
    gpu_memory_peak_mb: float | None
    notes: list[str] = field(default_factory=list)


class QwenImageEditProvider:
    name = "local_qwen_image_edit"
    MAX_REFERENCES = 3  # model guidance: up to 3 input images work best

    def __init__(self, model_id: str = "Qwen/Qwen-Image-Edit-2509", min_vram_gb: float = 40.0, cpu_offload: bool = True):
        self.model_id = model_id
        self.min_vram_gb = min_vram_gb
        self.cpu_offload = cpu_offload
        self._pipe = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("diffusers") is None:
            return False, "diffusers not installed"
        try:
            import torch
        except ImportError:
            return False, "torch not installed"
        if not torch.cuda.is_available():
            return False, "requires a CUDA GPU"
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total < self.min_vram_gb and not self.cpu_offload:
            return False, f"GPU has {total:.0f} GB; {self.min_vram_gb:.0f} GB needed without CPU offload"
        return True, f"CUDA GPU with {total:.0f} GB"

    def _load(self):
        if self._pipe is not None:
            return
        import torch
        from diffusers import QwenImageEditPlusPipeline

        pipe = QwenImageEditPlusPipeline.from_pretrained(self.model_id, torch_dtype=torch.bfloat16)
        if self.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self._pipe = pipe

    def unload(self):
        with self._lock:
            self._pipe = None
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass

    def generate(self, req: CreativeRequest) -> CreativeResult:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        import torch

        refs = req.garment_image_paths[: self.MAX_REFERENCES - 1]
        images = [Image.open(req.person_image_path).convert("RGB")] + [Image.open(p).convert("RGB") for p in refs]
        prompt = PROMPT_TEMPLATE.format(pieces=", ".join(req.garment_labels[: len(refs)])) + (
            " " + req.extra_prompt if req.extra_prompt else ""
        )
        with self._lock:
            self._load()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = self._pipe(
                image=images,
                prompt=prompt,
                negative_prompt=" ",
                true_cfg_scale=req.true_cfg_scale,
                num_inference_steps=req.num_inference_steps,
                generator=torch.manual_seed(req.seed),
            )
            elapsed = time.perf_counter() - t0
        out.images[0].save(req.output_path)
        notes = ["generative edit: not an exact virtual fitting"]
        if len(req.garment_image_paths) > len(refs):
            notes.append(f"only the first {len(refs)} garments were used as references")
        return CreativeResult(req.output_path, self.model_id, elapsed, torch.cuda.max_memory_allocated() / 2**20, notes)
