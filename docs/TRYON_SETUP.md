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
