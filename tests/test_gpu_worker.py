"""Contract test: RemoteFashnProvider <-> infrastructure/gpu-worker/worker.py.

The diffusion runtime is stubbed; everything else (auth, multipart format,
parameter passing, metric headers, cost accounting) is real.
"""

import asyncio
import importlib.util
import sys

import httpx
import pytest
from PIL import Image

from looklab_ml.evaluation.synthetic import draw_garment, draw_person
from looklab_ml.tryon import RemoteFashnProvider, RemoteGpuConfig, TryOnRequest
from looklab_ml.tryon.fashn_runtime import RunMetrics
from tests.conftest import REPO


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setenv("LOOKLAB_WORKER_TOKEN", "s3cret")
    spec = importlib.util.spec_from_file_location("looklab_worker", REPO / "infrastructure/gpu-worker/worker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["looklab_worker"] = mod
    spec.loader.exec_module(mod)
    calls = []

    def fake_run(person, garment, category, photo_type, steps, cfg, seed, n, segmentation_free=True):
        calls.append(dict(category=category, photo_type=photo_type, steps=steps, seed=seed, size=person.size))
        return [Image.new("RGB", (576, 864), (120, 20, 40))], RunMetrics(4.2, 31.0, 9876.0, "cuda:0", "torch.bfloat16")

    monkeypatch.setattr(mod.runtime, "run", fake_run)
    monkeypatch.setattr(mod.runtime, "available", lambda: (True, "stub"))
    return mod, calls


def _req(tmp_path, **kw):
    p, g = tmp_path / "p.png", tmp_path / "g.png"
    draw_person().save(p)
    draw_garment("top", "cream").save(g)
    return TryOnRequest(str(p), str(g), "tops", output_path=str(tmp_path / "out.png"), **kw)


def test_remote_provider_roundtrip(worker, tmp_path):
    mod, calls = worker
    cfg = RemoteGpuConfig(url="http://worker", token="s3cret", gpu_hourly_usd=0.69)
    prov = RemoteFashnProvider(cfg, transport=httpx.ASGITransport(app=mod.app))
    res = asyncio.run(prov.generate(_req(tmp_path, num_timesteps=20, seed=7)))
    assert calls == [dict(category="tops", photo_type="flat-lay", steps=20, seed=7, size=(768, 1152))]
    assert res.elapsed_seconds == 4.2 and res.gpu_memory_peak_mb == 9876.0 and res.model_load_seconds == 31.0
    assert res.device == "cuda:0" and res.estimated_cost_usd >= 0
    assert Image.open(res.output_image_path).size == (576, 864)
    assert prov.estimate_cost_usd(_req(tmp_path)) == round(0.69 * (15 + 60) / 3600, 4)


def test_worker_requires_token(worker, tmp_path):
    mod, _ = worker
    prov = RemoteFashnProvider(RemoteGpuConfig(url="http://worker", token="wrong"), transport=httpx.ASGITransport(app=mod.app))
    with pytest.raises(httpx.HTTPStatusError) as e:
        asyncio.run(prov.generate(_req(tmp_path)))
    assert e.value.response.status_code == 401
