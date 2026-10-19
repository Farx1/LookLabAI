# Local garment analysis models

Garment analysis runs in layers, cheapest first:

1. **`cv_basic` (always on, CPU, ~30 ms):** background estimation from the image border, k-means in CIELAB → named colors; solid vs patterned from interior edge density. It *never* guesses category, style or occasion from pixels; category may come from the file name or product title (marked "from title").
2. **Qwen3-VL-2B via Ollama** (`local_qwen3_vl`, optional): structured JSON constrained by a schema; every field is validated against the taxonomy and anything else becomes `unknown`.
3. **Florence-2-base** (`local_florence2`, optional fallback): detailed caption; only words that literally appear are mapped.

Provenance is stored per field: *you* > *AI guess* > *from title* > *seen in photo*. Your edits are never overwritten. Results are cached by image hash.

## Qwen3-VL via Ollama

```bash
# install Ollama from ollama.com, then
ollama pull qwen3-vl:2b
# or with Docker Compose
docker compose --profile ollama up -d && docker compose exec ollama ollama pull qwen3-vl:2b
```

`LOOKLAB_OLLAMA_URL` (default `http://127.0.0.1:11434`), `LOOKLAB_OLLAMA_VISION_MODEL` (default `qwen3-vl:2b`). Set `image_analysis.auto_run_model: true` in `config/ai.yaml` to analyse every upload in the background; otherwise use “Analyze with local AI” on a garment.

## Florence-2

```bash
uv pip install -e "ml[vision-models]"     # torch + transformers
```

Weights download from Hugging Face on first use. Florence-2 uses `trust_remote_code`; pin a model revision for production.

## Benchmark them

```bash
uv run python -m looklab_ml.evaluation.bench_vision --out ml/looklab_ml/evaluation/reports/vision.json
uv run python -m looklab_ml.evaluation.bench_vision --images path/to/labels.json   # your labelled photos
```

Reports accuracy *when answered*, unknown rate, latency and memory per analyzer. In the audit environment only `cv_basic` could run (see `reports/vision.json`); the VLMs were unreachable and are reported as skipped.
