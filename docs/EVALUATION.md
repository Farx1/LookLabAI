# Evaluation

All reports are JSON files under `ml/looklab_ml/evaluation/reports/` with timestamps and platform. No provider is called "better" without a report backing it.

## 1. Recommendation quality — `bench_recommendation.py`

```bash
uv run python -m looklab_ml.evaluation.bench_recommendation --out ml/looklab_ml/evaluation/reports/recommendation.json
```

Seven scenarios (including the principal use case) on the 20-piece synthetic wardrobe. Hard constraints are re-checked **independently of the engine**: slot completeness, duplicates, budget, ownership, excluded colors, season, required pieces. Also reports diversity (mean pairwise Jaccard distance), unique garments used, preferred-color share and latency.

Latest run (2026-10-19, 4 vCPU): 61 outfits, **100 % hard-constraint satisfaction**, median **27 ms** per request, diversity 0.75–0.91.

**Not yet measured:** human preference ratings. Protocol: 10+ participants, each rates 15 generated outfits for their own wardrobe on a 1–5 scale for "would wear" and "fits the request", with and without learned preferences (A/B).

## 2. Garment color naming

Same script, 231 synthetic garments (33 colors × 7 shapes): **97.0 % exact**, 100 % neutral-vs-chromatic. The only systematic miss is ivory→white. Synthetic flat colors overestimate real-photo accuracy; run `bench_vision.py --images` on labelled real photos.

## 3. Vision-language models — `bench_vision.py`

See `docs/VISION_MODELS.md`. Compares `cv_basic`, Qwen3-VL-2B and Florence-2 on category / color / pattern accuracy (when answered), unknown rate, latency, memory.

## 4. Try-on — `bench_tryon.py`

```bash
uv run python -m looklab_ml.evaluation.bench_tryon --manifest bench/manifest.json --provider local_fashn --out bench/runs/local
```

Outputs `report.json` (latency, model-load time, peak VRAM, cost, failure rate) and `index.html`, a contact sheet with a 1–5 rubric per case: identity preservation, garment color, garment pattern, body-proportion preservation, realistic folds, artifacts.

### Building the benchmark set

Use only photos of people who gave **written consent** for this purpose; keep them in `bench/`, which is git-ignored, or outside the repository. Cover:

* different body proportions and heights;
* loose and structured garments; dresses, skirts, trousers, layered looks;
* indoor/outdoor and low-light conditions; full-body and upper-body framing.

### CPU architecture benchmark — `bench_fashn_architecture.py`

Times the official `TryOnModel` with random weights. Latest CPU run: 96.7 s per CFG step (report `fashn_cpu_bench.json`). Run it on a GPU host with `--device cuda --dtype bfloat16` for a weight-free VRAM/latency estimate.

## 5. Automated tests

```bash
uv run pytest                      # 82 tests (1 GPU test skips without CUDA); no network or API key
uv run pytest -m gpu -s            # real FASHN inference (skips with a reason if unavailable)
pnpm --filter @looklab/web test:e2e   # Playwright journey (needs the stack running + seed_demo.py)
```
