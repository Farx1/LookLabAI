# LookLab AI — License Audit

Audit date: 2026-10-19. **This is an engineering audit, not legal advice.** Nothing in this repository should be treated as commercially cleared until the open items below are reviewed by counsel.

Legend: ✅ verified from the license text or source in the pinned repository · ☑️ stated by the publisher, not re-verified from the artifact (Hugging Face was unreachable from the audit environment) · ⚠️ restriction or open question.

## 1. Virtual try-on pipeline (FASHN VTON v1.5)

| Component | What it is | License | Commercial use | Evidence |
|---|---|---|---|---|
| `fashn-vton` inference code | MMDiT model definition, pipeline, preprocessing | **Apache-2.0** ✅ | Yes | `LICENSE` + `pyproject.toml` at `7c0f10a` |
| `fashn-ai/fashn-vton-1.5` weights | `model.safetensors` (972 M params) | **Apache-2.0** ☑️ | Yes, per publisher | README badge, FASHN [release post](https://fashn.ai/blog/fashn-vton-1-5-open-source-release) / [research page](https://fashn.ai/research/vton-1-5) |
| FASHN VTON training data | 18 M masked try-on pairs + 4 M synthetic triplets (per research page) | **undisclosed** ⚠️ | Unknown | Source and rights of the pairs are not published. The "masked" pairs were presumably produced with a human parser. |
| DWPose code (vendored in `fashn_vton/dwpose`) | pose estimation wrapper | **Apache-2.0** ✅ | Yes | upstream README lists DWPose (Apache-2.0) |
| DWPose weights `dw-ll_ucoco_384.onnx` (HF `fashn-ai/DWPose`) | whole-body keypoints | Apache-2.0 for the release ☑️ | ⚠️ see data | Trained on COCO-WholeBody **and UBody** (DWPose README). UBody is distributed via a request form for research; its terms were not verified. |
| YOLOX weights `yolox_l.onnx` | person detector | **Apache-2.0** ☑️ | Yes | YOLOX repository; trained on COCO |
| COCO / COCO-WholeBody | training data for the above | annotations CC BY 4.0; images under various Flickr licenses | ⚠️ data, not code | standard caveat for COCO-trained models |
| **FASHN Human Parser** (`fashn-human-parser` 0.1.x + HF `fashn-ai/fashn-human-parser`) | SegFormer-B4 fine-tune, 18 classes | **NVIDIA Source Code License for SegFormer** ✅ | **No — §3.3: "may be used or intended for use non-commercially… research or evaluation purposes only"** | package `LICENSE`, README "inherits the NVIDIA Source Code License"; [SegFormer LICENSE](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) |
| `nvidia/mit-b4` | parser's pretrained encoder | NVIDIA SegFormer license ⚠️ | No | same license family |

### How LookLab handles the parser

* **Default `FASHN_PARSER_MODE=parser_free`.** In the official pipeline, `segmentation_free=True` makes `create_clothing_agnostic_image` return the person image untouched, and `garment_photo_type="flat-lay"` makes `create_garment_image` return the garment untouched — the parser's predictions are computed and then **discarded** (`pipeline.py` / `preprocessing/agnostic.py` at `7c0f10a`). Parser-free mode replaces the parser with a stub that is never consulted for those inputs and **rejects** every other input (model-worn garment photos, masked mode) with an explanation. Output is therefore identical to the official pipeline for accepted inputs, and the NVIDIA-licensed weights are neither downloaded nor loaded.
* Because `fashn_vton` imports constants from `fashn_human_parser`, LookLab registers an **interface shim** (label names and category→coverage mapping only, no model code or weights) when the package is absent. The GPU worker image installs `fashn-vton` with `--no-deps` so the parser is not pulled in unless `WITH_HUMAN_PARSER=1`.
* `FASHN_PARSER_MODE=full` enables garments photographed on a person. It requires installing the parser and accepting its **non-commercial** terms. The UI labels the mode in Settings → AI engines.

### Open questions (must be resolved before any commercial launch)

1. Are the FASHN VTON weights a derivative work of the human parser (e.g. trained on parser-generated masks)? If so, does §3.3 propagate? Ask FASHN AI for a written statement.
2. Training-data rights for the 18 M + 4 M try-on pairs.
3. UBody dataset terms as they apply to the released DWPose weights.
4. Whether the interface shim (re-declared label names) needs attribution under the parser package's license. It contains no code from the package beyond label strings; we consider this interoperability, but it is listed for review.

**Status:** personal and research use — fine within these terms. Commercial use — **not cleared**; parser-free mode removes the only *explicit* non-commercial component from the runtime, but items 1–3 remain open.

## 2. Image analysis and generation models (optional)

| Model | Use | License | Notes |
|---|---|---|---|
| Qwen3-VL-2B-Instruct (Ollama `qwen3-vl:2b`) | garment metadata | Apache-2.0 ☑️ | runs locally; outputs are validated against the taxonomy |
| Florence-2-base | captions → metadata | MIT ☑️ | `trust_remote_code=True` executes model-repo code: pin a revision in production |
| Qwen3 1.7B (optional reply phrasing) | text | Apache-2.0 ☑️ | off by default |
| Qwen-Image-Edit-2509 | creative full-outfit edits | Apache-2.0 ☑️ (re-verify on the model card) | disabled; generative, not a fitting tool |
| Ollama runtime | local model server | MIT | |

## 3. Paid providers (disabled by default)

| Provider | Terms to accept | Data notes |
|---|---|---|
| FASHN API | FASHN API terms | base64 inputs: FASHN documents that base64 image data is not persisted ([docs](https://docs.fashn.ai/api-overview/data-retention-privacy)) |
| OpenAI Images API | OpenAI API terms & usage policies | separate API billing; check OpenAI's current API data-retention policy before enabling |
| RunPod / Vast.ai / HF endpoints | host terms | you run your own container; images transit to that host |

## 4. Application dependencies

| Area | Packages | Licenses |
|---|---|---|
| API | FastAPI, Starlette, Uvicorn, SQLAlchemy, Pydantic, pydantic-settings, httpx, python-multipart, PyYAML, Pillow, NumPy | MIT / BSD-3 / MIT-CMU (Pillow) — permissive |
| Optional API | psycopg (LGPL-3.0, dynamic linking), boto3 (Apache-2.0), pillow-heif (BSD-3; bundles libheif LGPL) | LGPL components used as libraries |
| ML runtime | PyTorch (BSD-3), torchvision (BSD-3), safetensors (Apache-2.0), onnxruntime (MIT), OpenCV (Apache-2.0), einops (MIT), matplotlib (PSF-based), huggingface_hub (Apache-2.0), transformers (Apache-2.0), diffusers (Apache-2.0) | permissive |
| Web | Next.js (MIT), React (MIT), Tailwind CSS (MIT), Radix UI (MIT), class-variance-authority (Apache-2.0), clsx (MIT), tailwind-merge (MIT), lucide-react (ISC), SWR (MIT); shadcn/ui pattern (MIT) | permissive |
| Fonts | Cormorant Garamond, Inter (self-hosted at build time by `next/font`) | SIL OFL 1.1 |
| Tests | pytest (MIT), Playwright (Apache-2.0), ruff (MIT) | permissive |

## 5. LookLab's own code

No license has been chosen for this repository yet; that is the owner's decision. All third-party code is consumed as dependencies (not vendored), except the label-name shim described in §1.

## 6. Test and demo images

All images used in tests, benchmarks and the demo wardrobe are **drawn procedurally** (`ml/looklab_ml/evaluation/synthetic.py`): no photographs of real people, no product photography, no consent or copyright exposure. Real-photo evaluation must use images whose subjects have consented in writing (see `docs/EVALUATION.md`).
