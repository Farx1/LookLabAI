# LookLab AI

An open-source-first personal stylist and virtual fitting room. Build a wardrobe from photos, describe the occasion in plain words, get outfits that respect your taste, budget and body as it is — and, on a GPU, see a garment on your own photo with FASHN VTON v1.5.

**Core rule:** nothing costs money unless you explicitly enable a paid provider *and* approve the specific request. The default monthly paid budget is $0.

| Works on any laptop (CPU, free) | Needs a GPU (yours or one you rent) |
|---|---|
| Wardrobe with automatic color/pattern detection · outfit generation with explanations · free collage previews · AI stylist chat that remembers your constraints · preference learning · saved looks, lookbook · costs dashboard | Photorealistic try-on (FASHN VTON v1.5) · optional creative full-outfit edits (Qwen-Image-Edit) |

## Quick start (no GPU, no API key)

Prerequisites: Python ≥ 3.11 with [uv](https://docs.astral.sh/uv/), Node 22 with pnpm.

```bash
uv sync                                   # Python workspace (API + ML package)
pnpm install                              # web workspace
uv run looklab-api                        # API on http://127.0.0.1:8000
pnpm dev                                  # web on http://localhost:3000
uv run python scripts/seed_demo.py --with-person   # optional: 20-piece synthetic demo wardrobe
```

Or with Docker: `docker compose up --build` → http://localhost:3000 (add `--profile ollama` for local Qwen3-VL analysis, `--profile gpu` for a local FASHN worker).

Open the app, add garments (Wardrobe → Add), then try the principal use case in **Outfit Generator** or **AI Stylist**:

> Find 15 outfits for a romantic anniversary dinner. I prefer loose-fitting, comfortable clothes, neutral colors, Korean-inspired fashion and a budget of €100.

Follow up with *“Make them less fitted”*, *“Use the clothes I already own”*, *“Show me something more sophisticated”*, *“Keep the same shoes but change the dress”* — earlier constraints are kept.

## Real try-on

See **[docs/TRYON_SETUP.md](docs/TRYON_SETUP.md)**: local NVIDIA GPU (free), your own remote GPU worker (same model, billed by your host), ZeroGPU for experiments, or paid APIs (off by default). On a CPU-only machine the Studio offers a clearly labelled **mock preview** to test the workflow; it is never presented as a try-on.

## Repository layout

```
apps/web/                 Next.js 16 PWA (8 pages), Playwright e2e
services/api/             FastAPI backend (looklab_api)
ml/looklab_ml/            vision · recommendation · tryon · creative · evaluation
packages/ui/              shadcn-style UI primitives
packages/shared-types/    domain types + OpenAPI-generated request types
infrastructure/docker/    API & web images        infrastructure/gpu-worker/  FASHN worker (HTTP + RunPod)
config/ai.yaml            routing policy & spending caps (validated at startup)
tests/                    pytest suite             docs/                       guides
FEASIBILITY.md            Phase 0 audit            LICENSE_AUDIT.md            licenses & open questions
```

## Configuration

* **`config/ai.yaml`** — provider routing (`open_source_first`), which paid providers are enabled (none by default), global monthly cap. Unknown provider names or attempts to disable approval fail at startup.
* **Environment** (`LOOKLAB_*`, or `.env`): `DATA_DIR`, `DATABASE_URL` (PostgreSQL supported), `STORAGE_BACKEND=s3` + `S3_*`, `AUTH_MODE=multi_user`, `FASHN_WEIGHTS_DIR`, `FASHN_PARSER_MODE`, `OLLAMA_URL`, `REMOTE_GPU_*`, `FASHN_API_KEY`, `OPENAI_API_KEY`. See `services/api/looklab_api/config.py`.

## Tests

```bash
uv run pytest                          # backend + ML: no GPU, network or keys needed
uv run pytest -m gpu -s                # real try-on, skipped with a reason when unavailable
uv run ruff check .                    # lint
pnpm typecheck && pnpm build           # web
pnpm test:e2e                          # Playwright (stack running + seed_demo.py)
```

## Status and limitations (honest)

* Real FASHN inference has **not yet been run** by this project: the development sandbox had no GPU and could not reach Hugging Face. CPU cost was measured on the real architecture (≈97 s/step ⇒ ≈48 min/image), which is why CPU try-on is off. VRAM/latency on GPUs are to be measured with the provided GPU test.
* **Outfit analysis** (page `/analyze`): photo to detected pieces to top-3 wardrobe matches, with brand, price and link. Measured on 2026-10-20 with `qwen3-vl:2b` on an RTX 3070 Laptop, on 1 photo for timing and VRAM: 3 to 11 s, 3.4 GB VRAM; matching on 6 flat-lay collages (folder `looklab-wardrobe-test`, outside the repo, built from wardrobe product images) gave the exact piece first for 14 of 18 pieces and in the top 3 for 18 of 18. The 2B model still invents or duplicates items on busy photos. The photo is deleted as soon as the analysis ends.
* Qwen3-VL and Florence-2 adapters are implemented but **not benchmarked** for the same reason; deterministic CV analysis is measured.
* Commercial use is **not cleared** — see [LICENSE_AUDIT.md](LICENSE_AUDIT.md) (human-parser license, undisclosed training data).
* Sequential multi-garment try-on and Qwen-Image-Edit creative mode are not enabled until benchmarked.
* Docker images were written but could not be built in the sandbox (Docker Hub rate limit).
* No database migrations yet (SQLAlchemy `create_all`).

More: [architecture](docs/ARCHITECTURE.md) · [privacy](docs/PRIVACY.md) · [vision models](docs/VISION_MODELS.md) · [evaluation](docs/EVALUATION.md).
