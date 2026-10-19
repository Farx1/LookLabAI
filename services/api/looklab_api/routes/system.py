from __future__ import annotations

import os
import platform

from fastapi import APIRouter, Depends

from looklab_ml.tryon import FashnRuntime

from ..auth import current_user
from ..deps import get_ctx

router = APIRouter(tags=["system"])


def hardware() -> dict:
    mem_gb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_gb = round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    ts = FashnRuntime.torch_status()
    mode = "local_gpu" if ts.get("cuda") else "cpu_only"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gb": mem_gb,
        "torch": ts,
        "execution_mode": mode,
    }


@router.get("/api/health")
def health():
    return {"status": "ok"}


@router.get("/api/system")
def system(ctx=Depends(get_ctx), user=Depends(current_user)):
    p = ctx.policy
    return {
        "hardware": hardware(),
        "auth_mode": ctx.settings.auth_mode,
        "storage_backend": ctx.settings.storage_backend,
        "policy": p.model_dump(),
        "tryon_providers": ctx.registry.tryon_status(),
        "analyzers": ctx.registry.analysis_status(),
        "fashn": {
            "parser_mode": ctx.settings.fashn_parser_mode,
            "weights_dir": str(ctx.settings.fashn_weights_dir),
            "loaded": ctx.registry.fashn_runtime.loaded,
        },
        "creative_editing": {
            "enabled": p.creative_editing.enabled,
            "available": ctx.registry.creative.available() if p.creative_editing.enabled else (False, "disabled in ai.yaml"),
        },
    }
