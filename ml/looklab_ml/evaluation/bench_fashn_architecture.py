"""Measure FASHN VTON v1.5 denoiser cost on the current device *without*
downloading weights: the official ``TryOnModel`` is instantiated with random
weights (compute cost does not depend on weight values) and one sampling
step (``forward_for_cfg``, conditional + unconditional) is timed.

Usage (needs torch + fashn_vton installed, e.g. pip install --no-deps -e fashn-vton-1.5):
    python -m looklab_ml.evaluation.bench_fashn_architecture --steps 2 --out reports/fashn_cpu_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch

    from looklab_ml.tryon.fashn_runtime import _install_parser_interface_shim

    _install_parser_interface_shim()  # the denoiser does not need the parser
    from fashn_vton import TryOnModel

    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, a.dtype)
    t0 = time.perf_counter()
    model = TryOnModel().to(dev, dtype=dtype).eval()
    build_s = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters())
    h, w = model.input_shape
    b = 1
    x = torch.randn(b, model.channels_in, h, w, device=dev, dtype=dtype)
    kw = dict(
        person_poses=torch.randn(b, 1, h, w, device=dev, dtype=dtype),
        garment_poses=torch.randn(b, 1, h, w, device=dev, dtype=dtype),
        ca_images=torch.randn(b, 3, h, w, device=dev, dtype=dtype),
        garment_images=torch.randn(b, 3, h, w, device=dev, dtype=dtype),
        garment_categories=torch.tensor([1], device=dev),
    )
    times = []
    with torch.inference_mode():
        for _ in range(a.steps):
            t = torch.full((b,), 0.5, device=dev, dtype=dtype)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            s = time.perf_counter()
            model.forward_for_cfg(x, t, **kw)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - s)
    per_step = min(times)
    report = {
        "what": "FASHN VTON v1.5 TryOnModel, random weights, one CFG sampling step (cond+uncond)",
        "device": str(dev),
        "dtype": a.dtype,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "parameters": params,
        "input_shape_hw": [h, w],
        "model_build_seconds": round(build_s, 2),
        "step_seconds": [round(t, 2) for t in times],
        "best_step_seconds": round(per_step, 2),
        "extrapolated_30_steps_seconds": round(per_step * 30, 1),
        "extrapolated_20_steps_seconds": round(per_step * 20, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0),
        "excludes": "DWPose (x2) and human-parser preprocessing; image decode/encode",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if dev.type == "cuda":
        report["gpu_memory_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    print(json.dumps(report, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
