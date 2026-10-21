# Real virtual try-on setup (FASHN VTON v1.5)

LookLab never needs a GPU to run, but photorealistic try-on does. Pick one mode. Measured CPU cost (≈97 s per diffusion step, ≈48 min per image on 4 vCPU) is why CPU inference is disabled by default — see `FEASIBILITY.md` §3.

## Mode B — your own NVIDIA GPU (free)

Requirements: NVIDIA GPU with recent drivers (Ampere or newer runs bf16; older cards fall back to fp32 and need more memory). Expected to fit on 12–16 GB; **measure with the GPU test below**.

```bash
# 1. PyTorch with CUDA + the try-on extras, in the project environment
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -e "ml[tryon]"

# 2. Official inference code, pinned. --no-deps keeps the non-commercial human parser out.
git clone https://github.com/fashn-AI/fashn-vton-1.5 third_party/fashn-vton-1.5
git -C third_party/fashn-vton-1.5 checkout 7c0f10af3f91ad4048fe9729c470a13ef905d25a
uv pip install --no-deps -e third_party/fashn-vton-1.5

# 3. Weights (~2 GB): try-on model + DWPose. Does not download the human parser.
uv run python infrastructure/gpu-worker/fetch_weights.py weights/fashn-vton-1.5

# 4. Verify on real hardware: prints load time, inference time, peak VRAM
uv run pytest -m gpu -s tests/test_gpu.py
```

Then restart the API. Settings → AI engines shows **local_fashn: ready**, and the Studio's automatic mode uses it.

Environment knobs: `LOOKLAB_FASHN_WEIGHTS_DIR`, `LOOKLAB_FASHN_IDLE_UNLOAD_SECONDS` (default 600: frees VRAM when idle), `LOOKLAB_FASHN_PARSER_MODE`.

### Parser modes

| Mode | Garment photos accepted | License exposure |
|---|---|---|
| `parser_free` (default) | flat-lay / product shots | no NVIDIA-licensed weights loaded |
| `full` | flat-lay **and** worn-by-a-person photos | requires `pip install fashn-human-parser transformers` and accepting its **non-commercial** license |

### Layered mode: exact garment, untouched photo

FASHN alone redraws the whole frame at about 576x768 and keeps the silhouette of the old garment. The layered mode fixes both. It is on by default when its models are available. Otherwise the Studio falls back to the plain mode and the job result says why (`fidelity_mode: segfree`, `fidelity_reason`). It never switches to a paid provider.

How it works:

1. Qwen3-VL (through your local Ollama) lists the worn items with rough boxes.
2. SAM2 cuts them out. The cut-outs become the label map that FASHN expects, so only the replaced layer is repainted.
3. SAM2 cuts the new garment out of the result. The result is pasted back at full resolution inside the old garment plus the new garment, feathered. Every other pixel is the original photo.

Layer replaced: outerwear replaces the outer layer. A top replaces the visible base layer and keeps the outer layer when one is worn on top. A bottom replaces trousers or skirt. A dress or jumpsuit replaces top and bottom. Scarf, bag, headband, hat and jewelry are always kept.

If the new outer layer is shorter than the old one, the result carries `uncovered_area: true`. The Studio then suggests adding a top.

Install:

```bash
# 1. Vision model (Apache-2.0, 1.9 GB per `ollama list`). Ollama must be running: LookLab never starts it.
ollama pull qwen3-vl:2b

# 2. SAM2 (Apache-2.0) and its weights (184 MB, SAM 2.1 small, sha256 checked by the script), next to the FASHN weights
uv pip install -e "ml[tryon]"
uv run python infrastructure/gpu-worker/fetch_weights.py weights/fashn-vton-1.5 weights/sam2
```

`GET /api/system` (`layered_tryon`) and `GET /api/tryon/providers` (`layered`) report whether Ollama, the model and SAM2 are ready, with the reason when they are not. The SAM2 path can be set with `LOOKLAB_SAM2_WEIGHTS`.

VRAM: Qwen (about 3 GB), FASHN (about 5.5 GB) and SAM2 (under 1 GB) do not fit together on an 8 GB card. The stages therefore run in sequence. FASHN is unloaded first, Qwen is unloaded from Ollama before FASHN starts, and SAM2 is freed after each use. The cost is a FASHN reload on every job (see `FEASIBILITY.md` section 4.2 for measured timings).

Known limits: Qwen3-VL 2B boxes are rough, so hair that falls over a garment and thin straps can be repainted slightly. Belts of the old or new garment can be lost. Use the plain mode if you prefer a whole-image result.

## Mode C — remote GPU running the same model

Build and run the worker on any GPU host you control:

```bash
docker build -f infrastructure/gpu-worker/Dockerfile -t looklab-gpu-worker .
docker run --gpus all -e LOOKLAB_WORKER_TOKEN=$(openssl rand -hex 24) \
  -v looklab-weights:/weights -p 8080:8080 looklab-gpu-worker
curl -H "Authorization: Bearer $TOKEN" https://your-host:8080/health
```

Then configure the API (environment or `.env`):

```
LOOKLAB_REMOTE_GPU_URL=https://your-host:8080
LOOKLAB_REMOTE_GPU_TOKEN=<token>
LOOKLAB_REMOTE_GPU_HOURLY_USD=0.40       # what your host charges, used for estimates
LOOKLAB_REMOTE_GPU_COLD_START_SECONDS=60
```

and opt in, because GPU time is billed and photos leave your machine:

```yaml
# config/ai.yaml
virtual_tryon:
  enabled_paid_providers: [remote_fashn]
spending:
  monthly_limit_usd: 5
```

Users must also enable *external processing* and set their own monthly limit in Settings, and approve each request with its estimated cost.

**RunPod Serverless (scale to zero):** use the same image with `CMD ["python", "runpod_handler.py"]`, mount a network volume at `/weights`, set `LOOKLAB_REMOTE_GPU_PROTOCOL=runpod` and `LOOKLAB_REMOTE_GPU_URL=https://api.runpod.ai/v2/<endpoint-id>` with your RunPod API key as the token. Cold starts (container + weight load) are billed by RunPod.

**Vast.ai / RunPod Pods:** run the HTTP worker; remember the instance bills while idle — stop it when not in use.

Put the worker behind TLS (a reverse proxy or the host's HTTPS endpoint). It refuses all requests if `LOOKLAB_WORKER_TOKEN` is unset and never writes images to disk.

## Mode D — Hugging Face ZeroGPU (experiments only)

ZeroGPU quotas are small and Spaces are public by default. Use the official demo with **non-personal** test images only, or a private Space; never upload real user photos to a public Space.

## Mode E — paid APIs (off by default)

```yaml
virtual_tryon:
  enabled_paid_providers: [fashn_api]        # and/or openai_image_edit
```
```
LOOKLAB_FASHN_API_KEY=...       # FASHN account (separate billing)
LOOKLAB_OPENAI_API_KEY=...      # OpenAI API billing — not included in ChatGPT plans
```

`paid_fallback` is only ever *offered* to the user; LookLab never switches to a paid provider on its own and never retries a paid request.

## Multiple garments

FASHN VTON dresses one garment per run. Applying several garments sequentially is **not** guaranteed to preserve earlier ones and has not been benchmarked yet; the Studio therefore asks the user to pick one piece of an outfit. Use `bench_tryon.py` with layered cases to evaluate before enabling sequential layering.
