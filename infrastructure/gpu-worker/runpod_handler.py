"""RunPod Serverless entry point (scale-to-zero). Same runtime as worker.py.

Input:  {"person_b64", "garment_b64", "category", "garment_photo_type", "num_timesteps", "guidance_scale", "seed", "num_samples"}
Output: {"image_b64", "elapsed_seconds", "gpu_memory_peak_mb", "model_load_seconds", "device"}
"""

from __future__ import annotations

import base64
import io
import os

import runpod  # provided by the runpod base image / requirements
from PIL import Image, ImageOps

from looklab_ml.tryon.fashn_runtime import FashnRuntime

runtime = FashnRuntime(
    weights_dir=os.environ.get("FASHN_WEIGHTS_DIR", "/weights"), parser_mode=os.environ.get("FASHN_PARSER_MODE", "parser_free")
)


def _img(b64: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(b64)))).convert("RGB")


def handler(job):
    i = job["input"]
    imgs, m = runtime.run(
        _img(i["person_b64"]),
        _img(i["garment_b64"]),
        i["category"],
        i.get("garment_photo_type", "flat-lay"),
        int(i.get("num_timesteps", 30)),
        float(i.get("guidance_scale", 1.5)),
        int(i.get("seed", 42)),
        int(i.get("num_samples", 1)),
    )
    buf = io.BytesIO()
    imgs[0].save(buf, format="PNG")
    return {
        "image_b64": base64.b64encode(buf.getvalue()).decode(),
        "elapsed_seconds": m.elapsed_seconds,
        "gpu_memory_peak_mb": m.gpu_memory_peak_mb,
        "model_load_seconds": m.model_load_seconds,
        "device": m.device,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
