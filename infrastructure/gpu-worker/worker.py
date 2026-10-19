"""LookLab GPU worker: FASHN VTON v1.5 behind a small authenticated HTTP API.

Deploy it on any NVIDIA GPU host you control (RunPod pod, Vast.ai instance,
a Hugging Face Inference Endpoint with a custom container, your own box).
The LookLab API calls it through the ``remote_fashn`` provider with exactly
the same request parameters as local inference, so the browser never knows
where inference runs.

Privacy: images are processed in memory and never written to disk or logged.

    LOOKLAB_WORKER_TOKEN=... uvicorn worker:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import hmac
import io
import os
import threading
import time

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps

from looklab_ml.tryon.fashn_runtime import FashnRuntime

TOKEN = os.environ.get("LOOKLAB_WORKER_TOKEN", "")
WEIGHTS = os.environ.get("FASHN_WEIGHTS_DIR", "/weights")
PARSER_MODE = os.environ.get("FASHN_PARSER_MODE", "parser_free")
MAX_BYTES = 20 * 2**20

runtime = FashnRuntime(
    weights_dir=WEIGHTS, parser_mode=PARSER_MODE, idle_unload_seconds=float(os.environ.get("IDLE_UNLOAD_SECONDS", "900"))
)
app = FastAPI(title="LookLab GPU worker", docs_url=None, redoc_url=None)


def _auth(authorization: str = Header("")) -> None:
    if not TOKEN:
        raise HTTPException(503, "LOOKLAB_WORKER_TOKEN is not set; refusing to serve without auth")
    if not hmac.compare_digest(authorization, f"Bearer {TOKEN}"):
        raise HTTPException(401, "unauthorized")


def _idle_unloader() -> None:
    while True:
        time.sleep(60)
        runtime.maybe_unload_idle()


threading.Thread(target=_idle_unloader, daemon=True).start()


@app.get("/health")
def health():
    ok, why = runtime.available()
    return {"ok": ok, "reason": why, "loaded": runtime.loaded, "parser_mode": PARSER_MODE, **runtime.torch_status()}


async def _image(f: UploadFile) -> Image.Image:
    raw = await f.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image too large")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, "invalid image") from e


@app.post("/v1/tryon", dependencies=[Depends(_auth)])
async def tryon(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    category: str = Form(...),
    garment_photo_type: str = Form("flat-lay"),
    num_timesteps: int = Form(30),
    guidance_scale: float = Form(1.5),
    seed: int = Form(42),
    num_samples: int = Form(1),
):
    if category not in ("tops", "bottoms", "one-pieces"):
        raise HTTPException(422, "bad category")
    ok, why = runtime.available()
    if not ok:
        raise HTTPException(503, why)
    p, g = await _image(person), await _image(garment)
    try:
        imgs, m = runtime.run(
            p, g, category, garment_photo_type, max(10, min(num_timesteps, 50)), guidance_scale, seed, max(1, min(num_samples, 4))
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    buf = io.BytesIO()
    imgs[0].save(buf, format="PNG")
    headers = {
        "X-Elapsed-Seconds": f"{m.elapsed_seconds:.3f}",
        "X-Device": m.device,
        "X-Dtype": m.dtype,
        "Cache-Control": "no-store",
    }
    if m.gpu_memory_peak_mb is not None:
        headers["X-GPU-Memory-Peak-MB"] = f"{m.gpu_memory_peak_mb:.0f}"
    if m.model_load_seconds:
        headers["X-Model-Load-Seconds"] = f"{m.model_load_seconds:.2f}"
    return Response(buf.getvalue(), media_type="image/png", headers=headers)
