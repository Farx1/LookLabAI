"""Real FASHN VTON v1.5 inference. Runs only with CUDA + weights:

LOOKLAB_FASHN_WEIGHTS_DIR=./weights/fashn-vton-1.5 uv run pytest -m gpu -s
"""

import os
import time

import pytest

from looklab_ml.tryon import FashnRuntime

WEIGHTS = os.environ.get("LOOKLAB_FASHN_WEIGHTS_DIR", "weights/fashn-vton-1.5")


def _skip_reason():
    rt = FashnRuntime(weights_dir=WEIGHTS)
    ts = rt.torch_status()
    if not ts.get("torch"):
        return "torch not installed"
    if not ts.get("cuda"):
        return "no CUDA GPU"
    ok, why = rt.available()
    return None if ok else why


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(_skip_reason() is not None, reason=f"real GPU try-on: {_skip_reason()}")]


def test_real_tryon_flat_lay_parser_free(tmp_path):
    from looklab_ml.evaluation.synthetic import draw_garment, draw_person

    rt = FashnRuntime(weights_dir=WEIGHTS, parser_mode="parser_free")
    t0 = time.perf_counter()
    imgs, m = rt.run(draw_person(), draw_garment("top", "burgundy"), "tops", "flat-lay", num_timesteps=30)
    print(f"\nload={m.model_load_seconds:.1f}s inference={m.elapsed_seconds:.2f}s peak={m.gpu_memory_peak_mb:.0f}MB "
          f"device={m.device} dtype={m.dtype} wall={time.perf_counter() - t0:.1f}s")  # fmt: skip
    assert imgs[0].size[0] > 0
    imgs[0].save(tmp_path / "out.png")
