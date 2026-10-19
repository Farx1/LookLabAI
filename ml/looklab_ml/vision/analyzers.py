"""Garment analyzers.

Three interchangeable implementations behind one interface:

* ``cv_basic``      – deterministic pixel statistics (always available, CPU).
* ``ollama_qwen3_vl`` – Qwen3-VL-2B-Instruct served by a local Ollama daemon.
* ``florence2``     – Florence-2-base through ``transformers`` (optional extra).

Model output is never trusted blindly: every field is validated against the
taxonomy and anything outside it is dropped to ``unknown``. Each field keeps
its provenance (``source``) so the UI can separate observed facts, model
guesses and user-supplied values.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import httpx
from PIL import Image

from .. import taxonomy as tx
from ..colors import PALETTE
from .color_analysis import analyze_pixels

log = logging.getLogger(__name__)


@dataclass
class Attr:
    value: Any
    source: str  # "cv" | "text" | "model:<name>" | "user"
    confidence: float = 0.0


@dataclass
class GarmentAnalysis:
    analyzer: str
    elapsed_seconds: float
    attributes: dict[str, Attr] = field(default_factory=dict)
    description: str | None = None
    raw: dict | None = None

    def to_dict(self) -> dict:
        return {
            "analyzer": self.analyzer,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "attributes": {k: asdict(v) for k, v in self.attributes.items()},
            "description": self.description,
        }


class GarmentAnalyzer(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...

    def analyze(self, image: Image.Image, hint_text: str | None = None) -> GarmentAnalysis: ...


class CVBasicAnalyzer:
    """Deterministic analysis: colors and solid/patterned from pixels, category
    only from user-supplied text such as a file name or product title."""

    name = "cv_basic"

    def available(self) -> tuple[bool, str]:
        return True, "always available"

    def analyze(self, image: Image.Image, hint_text: str | None = None) -> GarmentAnalysis:
        t0 = time.perf_counter()
        px = analyze_pixels(image)
        attrs: dict[str, Attr] = {}
        if px.colors:
            attrs["colors"] = Attr([c.name for c in px.colors[:3]], "cv", round(px.colors[0].share, 2))
            attrs["color_hex"] = Attr([c.hex for c in px.colors[:3]], "cv", 1.0)
        if px.pattern != tx.UNKNOWN:
            attrs["pattern"] = Attr(px.pattern, "cv", px.pattern_confidence)
        if hint_text:
            guess = tx.guess_category_from_text(hint_text)
            if guess:
                attrs["category"] = Attr(guess[0], "text", 0.7)
                attrs["subcategory"] = Attr(guess[1], "text", 0.7)
        return GarmentAnalysis(self.name, time.perf_counter() - t0, attrs)


# ---------------------------------------------------------------------------
# Vision-language models
# ---------------------------------------------------------------------------

VLM_PROMPT = (
    "You are a fashion cataloguing assistant. Look at the garment in the image and "
    "describe ONLY what is visible. If a property cannot be determined from the image, "
    'answer "unknown". Respond with JSON matching the schema.\n'
    f"category: one of {list(tx.CATEGORIES)}\n"
    "subcategory: short noun such as 'blouse', 'wide-leg trousers', 'midi skirt'\n"
    f"colors: up to 3 from {sorted(PALETTE)}\n"
    f"pattern: one of {list(tx.PATTERNS)}\n"
    f"fit: one of {list(tx.FITS)} (silhouette as shown)\n"
    f"style: up to 3 from {list(tx.STYLES)}\n"
    f"season: any of {list(tx.SEASONS)}\n"
    f"formality: integer 1 (loungewear) to 5 (black tie)\n"
    "details: up to 5 short visible details (e.g. 'puff sleeves', 'pleated', 'v-neck')\n"
    "description: one neutral sentence."
)

VLM_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "subcategory": {"type": "string"},
        "colors": {"type": "array", "items": {"type": "string"}},
        "pattern": {"type": "string"},
        "fit": {"type": "string"},
        "style": {"type": "array", "items": {"type": "string"}},
        "season": {"type": "array", "items": {"type": "string"}},
        "formality": {"type": "integer"},
        "details": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
    },
    "required": ["category", "colors", "pattern", "fit", "style", "formality", "description"],
}


def validate_vlm_output(data: dict, source: str, confidence: float = 0.6) -> tuple[dict[str, Attr], str | None]:
    """Clamp free-form model output to the taxonomy. Unknowns are omitted."""
    attrs: dict[str, Attr] = {}
    cat = tx.clean_choice(data.get("category"), tx.CATEGORIES)
    if cat != tx.UNKNOWN:
        attrs["category"] = Attr(cat, source, confidence)
    sub = data.get("subcategory")
    if isinstance(sub, str) and 0 < len(sub.strip()) <= 40 and sub.strip().lower() != tx.UNKNOWN:
        attrs["subcategory"] = Attr(sub.strip().lower(), source, confidence)
    colors = [c for c in (data.get("colors") or []) if isinstance(c, str)]
    colors = [c.strip().lower().replace("gray", "grey") for c in colors]
    colors = [c for c in colors if c in PALETTE][:3]
    if colors:
        attrs["colors"] = Attr(colors, source, confidence)
    for key, allowed in (("pattern", tx.PATTERNS), ("fit", tx.FITS)):
        v = tx.clean_choice(data.get(key), allowed)
        if v != tx.UNKNOWN:
            attrs[key] = Attr(v, source, confidence)
    styles = tx.clean_list(data.get("style"), tx.STYLES)[:3]
    if styles:
        attrs["style"] = Attr(styles, source, confidence)
    seasons = tx.clean_list(data.get("season"), tx.SEASONS)
    if seasons:
        attrs["season"] = Attr(seasons, source, confidence * 0.8)
    f = data.get("formality")
    if isinstance(f, (int, float)) and tx.FORMALITY_MIN <= int(f) <= tx.FORMALITY_MAX:
        attrs["formality"] = Attr(int(f), source, confidence * 0.8)
    details = [d.strip().lower() for d in (data.get("details") or []) if isinstance(d, str) and 0 < len(d) <= 40]
    if details:
        attrs["details"] = Attr(details[:5], source, confidence)
    desc = data.get("description")
    desc = desc.strip()[:300] if isinstance(desc, str) and desc.strip() else None
    return attrs, desc


def _image_b64(image: Image.Image, max_side: int = 768) -> str:
    img = image.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


class OllamaQwenVLAnalyzer:
    """Qwen3-VL through a local Ollama server (``ollama run qwen3-vl:2b``).

    Runs entirely on the user's machine; no image leaves localhost.
    """

    name = "ollama_qwen3_vl"

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "qwen3-vl:2b", timeout: float = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def available(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=2)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            if any(n == self.model or n.startswith(self.model + ":") or n.split(":")[0] == self.model for n in names):
                return True, f"ollama model {self.model} present"
            return False, f"ollama reachable but model {self.model} not pulled (run: ollama pull {self.model})"
        except Exception as e:  # noqa: BLE001 - availability probe
            return False, f"ollama not reachable at {self.base_url}: {type(e).__name__}"

    def analyze(self, image: Image.Image, hint_text: str | None = None) -> GarmentAnalysis:
        t0 = time.perf_counter()
        prompt = VLM_PROMPT + (f"\nUser-provided title (may help): {hint_text}" if hint_text else "")
        r = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt, "images": [_image_b64(image)]}],
                "format": VLM_SCHEMA,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        data = _parse_json(content)
        attrs, desc = validate_vlm_output(data, f"model:{self.model}")
        return GarmentAnalysis(self.name, time.perf_counter() - t0, attrs, desc, raw=data)


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


class Florence2Analyzer:
    """Florence-2-base caption -> keyword extraction.

    Florence-2 produces captions, not structured output, so only terms that
    literally appear in the caption are mapped onto the taxonomy.
    Requires the ``vision-models`` extra (torch + transformers).
    """

    name = "florence2"

    def __init__(self, model_id: str = "microsoft/Florence-2-base", device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False, "torch/transformers not installed (pip install '.[vision-models]')"
        return True, f"transformers installed; weights for {self.model_id} load lazily"

    def _load(self):
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float16 if dev == "cuda" else torch.float32
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, torch_dtype=dtype).to(dev).eval()
            self.device, self._dtype = dev, dtype

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None

    def caption(self, image: Image.Image) -> str:
        import torch

        self._load()
        task = "<MORE_DETAILED_CAPTION>"
        inputs = self._processor(text=task, images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: (v.to(self.device, self._dtype) if v.dtype.is_floating_point else v.to(self.device)) for k, v in inputs.items()}
        with torch.inference_mode():
            ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=96,
                num_beams=3,
                do_sample=False,
            )
        text = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(text, task=task, image_size=image.size)
        return str(parsed.get(task, text)).strip()

    def analyze(self, image: Image.Image, hint_text: str | None = None) -> GarmentAnalysis:
        t0 = time.perf_counter()
        cap = self.caption(image)
        attrs = caption_to_attributes(cap, f"model:{self.model_id}")
        return GarmentAnalysis(self.name, time.perf_counter() - t0, attrs, cap, raw={"caption": cap})


def caption_to_attributes(caption: str, source: str) -> dict[str, Attr]:
    attrs: dict[str, Attr] = {}
    low = " " + caption.lower().replace(",", " ").replace(".", " ") + " "
    guess = tx.guess_category_from_text(low)
    if guess:
        attrs["category"] = Attr(guess[0], source, 0.5)
        attrs["subcategory"] = Attr(guess[1], source, 0.5)
    colors = [c for c in sorted(PALETTE, key=len, reverse=True) if f" {c} " in low]
    if colors:
        attrs["colors"] = Attr(colors[:3], source, 0.5)
    pattern_words = {
        "striped": "striped",
        "stripes": "striped",
        "floral": "floral",
        "flowers": "floral",
        "plaid": "checked",
        "checkered": "checked",
        "polka": "polka_dot",
        "leopard": "animal",
        "graphic": "graphic",
        "plain": "solid",
        "solid": "solid",
    }
    for w, p in pattern_words.items():
        if f" {w} " in low:
            attrs["pattern"] = Attr(p, source, 0.5)
            break
    fit_words = {"oversized": "oversized", "loose": "relaxed", "relaxed": "relaxed", "fitted": "slim", "slim": "slim"}
    for w, f in fit_words.items():
        if f" {w} " in low:
            attrs["fit"] = Attr(f, source, 0.4)
            break
    return attrs


def merge_analyses(*analyses: GarmentAnalysis) -> dict[str, Attr]:
    """Merge attribute maps; earlier analyses take precedence per field,
    except that model fields override CV-only *absence*. Colors from CV are
    kept when present because pixel statistics are more reliable than a 2B
    model at naming colors."""
    merged: dict[str, Attr] = {}
    for a in analyses:
        for k, v in a.attributes.items():
            if k not in merged:
                merged[k] = v
    return merged


def build_analyzer(name: str, settings: dict | None = None) -> GarmentAnalyzer:
    settings = settings or {}
    if name == "cv_basic":
        return CVBasicAnalyzer()
    if name in ("ollama_qwen3_vl", "local_qwen3_vl"):
        return OllamaQwenVLAnalyzer(
            base_url=settings.get("ollama_url", "http://127.0.0.1:11434"),
            model=settings.get("ollama_vision_model", "qwen3-vl:2b"),
        )
    if name in ("florence2", "local_florence2"):
        return Florence2Analyzer(model_id=settings.get("florence_model_id", "microsoft/Florence-2-base"))
    raise ValueError(f"unknown analyzer {name!r}")


ANALYZER_NAMES = ("cv_basic", "local_qwen3_vl", "local_florence2")
