"""Run a try-on provider over a manifest and record latency, memory, cost
and outputs, plus an HTML contact sheet for side-by-side visual review.

manifest.json: [{"id", "person", "garment", "category", "garment_photo_type", "tags": [...]}]

    python -m looklab_ml.evaluation.bench_tryon --manifest bench/manifest.json --provider local_fashn --out bench/runs/local
    python -m looklab_ml.evaluation.bench_tryon --manifest bench/manifest.json --provider remote_fashn --remote-url ... --out bench/runs/remote

Visual quality (identity, color/pattern fidelity, body proportions, folds,
artifacts) is scored by humans in the generated HTML; see docs/EVALUATION.md.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import time
from pathlib import Path

from ..tryon import FashnRuntime, LocalFashnProvider, MockTryOnProvider, RemoteFashnProvider, RemoteGpuConfig, TryOnRequest

RUBRIC = ["identity", "garment_color", "garment_pattern", "body_proportions", "folds", "artifacts"]


def build(name: str, a) -> object:
    if name == "mock":
        return MockTryOnProvider()
    if name == "local_fashn":
        return LocalFashnProvider(FashnRuntime(a.weights, parser_mode=a.parser_mode))
    if name == "remote_fashn":
        return RemoteFashnProvider(
            RemoteGpuConfig(url=a.remote_url, token=os.environ.get("LOOKLAB_REMOTE_GPU_TOKEN", ""), gpu_hourly_usd=a.gpu_hourly_usd)
        )
    raise SystemExit(f"unsupported provider {name}")


async def run(a) -> dict:
    prov = build(a.provider, a)
    ok, why = prov.available()
    if not ok:
        raise SystemExit(f"{a.provider} unavailable: {why}")
    base = Path(a.manifest).parent
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in json.loads(Path(a.manifest).read_text()):
        req = TryOnRequest(
            person_image_path=str(base / item["person"]),
            garment_image_path=str(base / item["garment"]),
            category=item["category"],
            garment_photo_type=item.get("garment_photo_type", "flat-lay"),
            num_timesteps=a.steps,
            seed=a.seed,
            output_path=str(out / f"{item['id']}.png"),
        )
        t0 = time.perf_counter()
        try:
            r = await prov.generate(req)
            rows.append({**item, "status": "ok", "wall_s": round(time.perf_counter() - t0, 2), "elapsed_s": round(r.elapsed_seconds, 2),
                         "model_load_s": r.model_load_seconds, "gpu_memory_peak_mb": r.gpu_memory_peak_mb, "cost_usd": r.estimated_cost_usd,
                         "device": r.device, "output": Path(r.output_image_path).name})  # fmt: skip
        except Exception as e:  # noqa: BLE001 - failures are data
            rows.append({**item, "status": "failed", "error": f"{type(e).__name__}: {e}"})
    ok_rows = [r for r in rows if r["status"] == "ok"]
    report = {
        "provider": a.provider,
        "steps": a.steps,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n": len(rows),
        "failure_rate": round(1 - len(ok_rows) / max(1, len(rows)), 3),
        "mean_elapsed_s": round(sum(r["elapsed_s"] for r in ok_rows) / max(1, len(ok_rows)), 2),
        "max_gpu_memory_mb": max((r["gpu_memory_peak_mb"] or 0 for r in ok_rows), default=None),
        "total_cost_usd": round(sum(r["cost_usd"] for r in ok_rows), 4),
        "rows": rows,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    cells = "".join(
        f"<tr><td>{html.escape(r['id'])}<br><small>{html.escape(', '.join(r.get('tags', [])))}</small></td>"
        f"<td><img src='{html.escape(os.path.relpath(base / r['person'], out))}'></td><td><img src='{html.escape(os.path.relpath(base / r['garment'], out))}'></td>"
        f"<td>{'<img src=' + chr(39) + html.escape(r['output']) + chr(39) + '>' if r['status'] == 'ok' else html.escape(r.get('error', ''))}</td>"
        f"<td>{''.join(f'<label>{k} <input type=number min=1 max=5></label><br>' for k in RUBRIC)}</td></tr>"
        for r in rows
    )
    (out / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>{a.provider} try-on review</title><style>img{{height:260px}}td{{vertical-align:top;padding:6px}}</style>"
        f"<h1>{a.provider} · {a.steps} steps</h1><table><tr><th>case</th><th>person</th><th>garment</th><th>result</th><th>score 1–5</th></tr>{cells}</table>"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--provider", default="local_fashn", choices=["mock", "local_fashn", "remote_fashn"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="weights/fashn-vton-1.5")
    ap.add_argument("--parser-mode", default="parser_free")
    ap.add_argument("--remote-url", default="")
    ap.add_argument("--gpu-hourly-usd", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    rep = asyncio.run(run(ap.parse_args()))
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
