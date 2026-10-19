# LookLab AI — Feasibility & Technical Audit (Phase 0)

Audit date: 2026-10-19. Every number below is labelled **measured** (run in this repository, with the report file named) or **sourced** (published by a third party, with the source). Nothing is invented. Where we could not measure something, it says so.

## 1. Development machine (measured)

| Resource | Value |
|---|---|
| CPU | Intel Xeon @ 2.10 GHz, **4 vCPU** |
| RAM | **15 GB**, no swap |
| GPU | **none** (`nvidia-smi` absent, `torch.cuda.is_available() == False`) |
| Disk | ~30 GB free |
| Python / Node | 3.13 / 22 |
| Network | PyPI, npm and GitHub reachable. **Hugging Face, ollama.com, ModelScope and api.fashn.ai are blocked** by the sandbox proxy. Docker Hub rate-limited pulls (HTTP 429). |

Consequences:
* The whole MVP (web app, API, database, wardrobe, recommendations, collages, mock try-on) was built and tested here **without CUDA and without any API key**.
* Model weights (FASHN VTON, DWPose, Qwen3-VL, Florence-2) **could not be downloaded in this environment**, so no real image was generated here and no vision-language model was benchmarked. The code paths are implemented, unit-tested with stubs, and gated by GPU tests that skip with a reason.

## 2. FASHN VTON v1.5 — verified facts

Source code inspected at `fashn-AI/fashn-vton-1.5@7c0f10a` (2026-02-01).

| Item | Finding | How verified |
|---|---|---|
| API | `TryOnPipeline(weights_dir, device)` → `pipeline(person_image, garment_image, category, garment_photo_type, num_samples, num_timesteps, guidance_scale, seed, segmentation_free)` | read `src/fashn_vton/pipeline.py` |
| Categories | `tops`, `bottoms`, `one-pieces` | source |
| Garment input | `model` (worn) or `flat-lay` (product shot) | source |
| Resolution | model works at **864×576** (H×W), aspect-preserving resize + pad | `TryOnModel.input_shape` |
| Precision | bf16 on Ampere+ GPUs, fp32 elsewhere | source |
| Parameters | **971,813,808** | **measured** (instantiated `TryOnModel`) |
| Weights | `model.safetensors` (HF `fashn-ai/fashn-vton-1.5`) + DWPose `yolox_l.onnx`, `dw-ll_ucoco_384.onnx` (HF `fashn-ai/DWPose`) + FASHN Human Parser (~244 MB, auto-download) — ~2 GB total | `scripts/download_weights.py`, README |
| Sampler | rectified-flow Euler, CFG = **2 forward passes per step** (`forward_for_cfg`), default 30 steps | source |
| Human parser usage | always executed on person *and* garment, but with `segmentation_free=True` **and** `garment_photo_type="flat-lay"` its output is discarded | read `pipeline.py` + `preprocessing/agnostic.py` |

The last row matters for licensing (see `LICENSE_AUDIT.md`): LookLab's **parser-free mode** skips loading the non-commercial parser for exactly the inputs where its output is unused, producing results identical to the official pipeline, and refuses other inputs.

## 3. Can try-on run on CPU? (measured: no, not practically)

`ml/looklab_ml/evaluation/bench_fashn_architecture.py` instantiates the **official** `TryOnModel` with random weights (compute cost does not depend on weight values) and times one CFG sampling step at the real input size.

Report: `ml/looklab_ml/evaluation/reports/fashn_cpu_bench.json`

| Metric (4 vCPU Xeon, fp32, torch 2.14) | Value |
|---|---|
| One sampling step (cond + uncond) | **96.7 s** (best of 3: 113.4 / 96.7 / 99.0 s) |
| Extrapolated 20 steps | **~32 min** |
| Extrapolated 30 steps | **~48 min** |
| Peak RAM | **6.1 GB** |
| Excludes | DWPose ×2, image I/O (parser skipped in parser-free mode) |

Conclusion: CPU-only machines are a first-class development mode for everything **except** photorealistic try-on. `FashnRuntime` therefore refuses CPU by default (`LOOKLAB_FASHN_ALLOW_CPU=false`) and the UI says why.

## 4. GPU requirements (partly sourced, VRAM not yet measured)

* **Sourced**: FASHN states ~5 s per image on an H100 ([research page](https://fashn.ai/research/vton-1-5)).
* **Derived**: bf16 weights are ~1.9 GB (972 M × 2 bytes). Activations for ~7 k tokens × 24 blocks × batch 2 come on top. We expect a **12–16 GB consumer GPU (Ampere or newer)** to be sufficient, but this is an *expectation, not a measurement*.
* **To measure**: `uv run pytest -m gpu -s` (tests/test_gpu.py) prints load time, inference time and `torch.cuda.max_memory_allocated()` on real hardware. Every production run also records `gpu_memory_peak_mb` in the `inference_runs` table (Settings → Costs & performance).

## 5. Cheapest viable real-inference paths

| Path | Cost model | Expected cost per image | Notes |
|---|---|---|---|
| **Own NVIDIA GPU (Mode B)** | electricity | ≈ $0 | Best option. Lazy load + idle unload implemented. |
| **Vast.ai RTX 4090 (Mode C)** | marketplace, per hour | interruptible ~$0.10–0.16/h, on-demand ~$0.18–0.48/h ([computeprices](https://computeprices.com/providers/vast/gpus/rtx4090), [getdeploying](https://getdeploying.com/gpus/nvidia-rtx-4090)) | You pay while the instance is up, idle included. |
| **RunPod Pod (Mode C)** | per hour | RTX 4090 ~$0.34–0.69/h ([RunPod](https://www.runpod.io/gpu-cloud/pricing)) | Same container (`infrastructure/gpu-worker`). |
| **RunPod Serverless (Mode C, scale-to-zero)** | per second while a worker runs | flex 16 GB tier $0.00016/s; 4090 ~$0.00044/s flex in an older post ([docs](https://docs.runpod.io/serverless/pricing), [blog](https://blog.runpod.io/runpod-serverless-pricing-update)) | Cold start (container + ~2 GB weights) is billed. `runpod_handler.py` provided. |
| **HF ZeroGPU (Mode D)** | daily quota | free tier a few GPU-minutes/day; PRO more; overage $1 / 10 min ([docs](https://huggingface.co/docs/hub/spaces-zerogpu)) | Proof-of-concept only. **Never** put personal photos in a public Space. |
| FASHN API (Mode E, paid) | per image | $0.075 on-demand ([help](https://help.fashn.ai/plans-and-pricing/api-pricing)) | Disabled by default. |
| OpenAI image edit (Mode E, paid) | tokens | roughly $0.04–0.19 per 1024² image depending on quality ([aggregators](https://pricepertoken.com/gpt-image-pricing)); verify on OpenAI's page | Generative edit, *not* a garment-accurate try-on. Disabled by default. Separate API billing — a ChatGPT plan includes no API credits. |

Illustrative remote estimate (an **assumption** to be replaced by measurement): if a 30-step image takes ~20–40 s on a rented 4090 at $0.40/h, compute is ~$0.002–0.005 per image; a serverless cold start of ~60 s adds ~$0.01–0.03 when the worker was scaled to zero. LookLab's estimator (`RemoteFashnProvider.estimate_cost_usd`) uses your configured `$/h`, the expected inference time and cold start, and the request is charged from the measured wall-clock afterwards.

**Cost-control mechanics implemented**: content-hash result cache (identical photo + garment + parameters ⇒ reused, $0), in-flight request de-duplication, explicit per-request approval with a maximum, monthly cap = min(server cap, user cap) with in-flight reservations counted, no automatic retries of paid jobs, idle model unload, image downscaling before upload.

## 6. Candidate model comparison

| Task | Candidate | License | Size | Status here |
|---|---|---|---|---|
| Try-on | **FASHN VTON v1.5** | Apache-2.0 (code+weights; see audit for deps) | 972 M params, ~2 GB | Integrated (local + remote). Real run pending GPU. |
| Garment analysis | Deterministic CV (k-means in CIELAB, background estimate) | own code | — | **Measured**: 97.0 % exact color name, 100 % neutral/chromatic on 231 synthetic garments (`reports/recommendation.json`) |
| Garment analysis | **Qwen3-VL-2B-Instruct** via Ollama (`qwen3-vl:2b`) | Apache-2.0 (sourced) | ~2 B | Integrated with schema-constrained JSON + taxonomy validation. **Not benchmarked** (Ollama/HF blocked). |
| Garment analysis | **Florence-2-base** via transformers | MIT | 0.23 B | Integrated (caption → keyword mapping). **Not benchmarked** (HF blocked). |
| Outfit reasoning | Deterministic hybrid engine | own code | — | **Measured**: 61 outfits, 100 % hard-constraint satisfaction, median 27 ms |
| Reply phrasing (optional) | small local LLM via Ollama (default `qwen3:1.7b`) | Apache-2.0 | ~1.7 B | Optional, off by default; it never chooses garments. |
| Creative full outfit | **Qwen-Image-Edit-2509** (multi-image) | Apache-2.0 (to re-verify on the model card) | ~20 B | Adapter written, **disabled**; ~40+ GB VRAM bf16, ~20–24 GB with 4-bit community builds (sourced, unverified). Separate from try-on. |

The VLM benchmark harness for Qwen3-VL vs Florence-2 (quality on the labelled synthetic set + latency + memory) is `docs/EVALUATION.md` §3; it runs as soon as either model is reachable.

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Human parser is non-commercial | High (commercial use) | Parser-free mode (flat-lay only) is the default; full mode is opt-in and labelled. Legal review still required — see audit. |
| Training-data provenance of FASHN weights, DWPose (UBody) undisclosed/unverified | Medium | Documented as open questions; not marked commercially cleared. |
| No GPU measurement yet (VRAM, latency) | Medium | GPU test + per-run metrics; numbers will be filled from real runs. |
| Single-garment model; sequential layering may degrade earlier garments | Medium | UI tells users; layering must be benchmarked before being offered; creative editing kept separate. |
| CV analysis on real photos is weaker than on synthetic images | Medium | Everything uncertain stays `unknown`; users can correct; VLMs optional. |
| Remote GPU / paid APIs receive personal photos | High (privacy) | Off by default; require explicit external-processing consent + per-request approval; base64 to FASHN (documented as not persisted). |
| Shared free GPUs (ZeroGPU) have quotas and are public by default | Medium | Documented as experiment-only. |
| Docker images not built in this sandbox (Hub rate limit) | Low | Dockerfiles provided; verify with `docker compose build` on a normal network. |
