"""Compare garment analyzers (cv_basic, Qwen3-VL via Ollama, Florence-2) on a
labelled set: category / primary color / pattern accuracy, unknown rate,
latency and process memory. Unavailable analyzers are reported as skipped.

    python -m looklab_ml.evaluation.bench_vision --out ml/looklab_ml/evaluation/reports/vision.json
    python -m looklab_ml.evaluation.bench_vision --images my_labelled_dir/labels.json   # real photos
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

from PIL import Image

from ..vision import build_analyzer
from .synthetic import DEMO_WARDROBE, draw_garment


def synthetic_set():
    for it in DEMO_WARDROBE:
        yield draw_garment(it.kind, it.color, it.pattern), {"category": it.meta["category"], "color": it.color, "pattern": it.pattern}


def file_set(labels: Path):
    for row in json.loads(labels.read_text()):
        yield Image.open(labels.parent / row["file"]).convert("RGB"), row


def evaluate(name: str, items, settings: dict) -> dict:
    a = build_analyzer(name, settings)
    ok, why = a.available()
    if not ok:
        return {"analyzer": name, "skipped": why}
    hits = {"category": [], "color": [], "pattern": []}
    unknown = {"category": 0, "color": 0, "pattern": 0}
    lat = []
    for img, truth in items:
        t0 = time.perf_counter()
        attrs = a.analyze(img).attributes  # no hint text: pixels only
        lat.append(time.perf_counter() - t0)
        pred = {
            "category": attrs["category"].value if "category" in attrs else None,
            "color": attrs["colors"].value[0] if "colors" in attrs else None,
            "pattern": attrs["pattern"].value if "pattern" in attrs else None,
        }
        for k in hits:
            if pred[k] is None:
                unknown[k] += 1
            else:
                hits[k].append(pred[k] == truth[k])
    n = len(lat)
    return {
        "analyzer": name,
        "images": n,
        # accuracy over answered items; unknown answers are counted separately (unknown is not wrong)
        "accuracy_when_answered": {k: round(sum(v) / len(v), 3) if v else None for k, v in hits.items()},
        "unknown_rate": {k: round(v / n, 3) for k, v in unknown.items()},
        "latency_s": {"median": round(statistics.median(lat), 3), "max": round(max(lat), 3)},
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, help="labels.json: [{file, category, color, pattern}]")
    ap.add_argument("--analyzers", default="cv_basic,local_qwen3_vl,local_florence2")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    settings = {"ollama_url": a.ollama_url}
    results = []
    for name in a.analyzers.split(","):
        items = list(file_set(a.images) if a.images else synthetic_set())
        results.append(evaluate(name, items, settings))
    report = {
        "dataset": str(a.images or "synthetic DEMO_WARDROBE (20)"),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    s = json.dumps(report, indent=2)
    print(s)
    if a.out:
        Path(a.out).write_text(s)


if __name__ == "__main__":
    main()
