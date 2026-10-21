"""Layered try-on: change only the garment, keep everything else bit-exact.

Why: FASHN regenerates the whole frame at ~576x768 (background drift) and, without a
human parser, keeps the silhouette of the old garment. This module fixes both with
Apache-2.0 models that run locally, never the non-commercial FASHN Human Parser:

1. Qwen3-VL (Ollama) lists worn items with rough boxes;
2. SAM2 cuts them out; the cut-outs become the label map FASHN expects, so only the
   replaced layer is repainted and accessories (scarf, bag, headband...) are kept;
3. the result is pasted back at full resolution inside a small feathered mask
   (old garment + new garment); every pixel outside it is the original.

Replaced layer by garment kind (accessories are always kept):
outerwear replaces the outer layer; top replaces the visible base/mid layer (the outer
layer is kept when one is detected under it, otherwise the top replaces what is worn);
bottom replaces trousers/skirt; dress and jumpsuit replace top and bottom.

VRAM budget (8 GB card): Qwen 3 GB + FASHN 5.5 GB + SAM2 0.8 GB do not fit together, so the
stages run in sequence and each one is released before the next (see ``LocalFashnProvider``).
"""

from __future__ import annotations

import gc
import importlib.util
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

import httpx
import numpy as np
from PIL import Image

from ..vision.analyzers import OllamaQwenVLAnalyzer, _image_b64, ollama_chat
from .fashn_runtime import LABEL_IDS

log = logging.getLogger(__name__)

PREFILL = '{"items": [{"name": "'
GARMENT_PROMPT = (
    "List every clothing item and accessory worn by the person.\n"
    'Return JSON: {"items":[{"name":str,"layer":"base|mid|outer|bottom|accessory","color":str,"bbox_2d":[x1,y1,x2,y2]}]}\n'
    "bbox_2d in 0-1000 normalized coordinates of the full image. Start with the outer jacket or coat. "
    "Include scarf, bag (and its strap), headband, jewelry, trousers if visible."
)
BODY_PROMPT = (
    "Locate the person's body parts. "
    'Return JSON: {"items":[{"name":"face|hair|left hand|right hand","bbox_2d":[x1,y1,x2,y2]}]}\n'
    "bbox_2d in 0-1000 normalized coordinates of the full image. Only visible parts."
)
HEAD_PROMPT = (
    "List the scarf and the headband worn on the head and neck of the person. Only list items clearly visible. "
    'Return JSON: {"items":[{"name":str,"bbox_2d":[x1,y1,x2,y2]}]}\n'
    "bbox_2d in 0-1000 normalized coordinates of this image."
)

# first match wins; None = ignored item; anything unmatched is an upper-body garment ("top")
_WORDS = (
    ("face", r"face"),
    ("hair", r"hair|ponytail"),
    ("hands", r"hand|finger|gloves?|mittens?"),
    ("scarf", r"scarf|scarves|shawl|bandana"),
    ("bag", r"bag|strap|backpack|purse"),
    ("hat", r"hat|cap|beanie|headband|hairband"),
    ("glasses", r"glasses|sunglasses"),
    ("jewelry", r"necklace|earring|bracelet|watch|ring|jewel\w*"),
    ("skirt", r"skirt"),
    ("dress", r"dress|jumpsuit|romper"),
    ("pants", r"pants|trousers|jeans|leggings|shorts|tights"),
    (None, r"belt|shoe|boot|sneaker|sock|heel"),
)
_WORDS_RE = tuple((lab, re.compile(rf"\b(?:{w})s?\b")) for lab, w in _WORDS)
GARMENT_LABELS = ("top", "dress", "skirt", "pants")
PAINT_ORDER = ("top", "dress", "skirt", "pants", "hair", "face", "scarf", "jewelry", "glasses", "bag", "hat", "hands")


class LayerError(Exception):
    """Layered mode cannot be used for this request; the caller falls back to segfree."""


@dataclass
class LayerPlan:
    label_map: np.ndarray
    old_mask: np.ndarray  # bool: pixels of the garment(s) being replaced
    keep_mask: np.ndarray  # bool: accessories and body parts that must stay untouched
    seconds: dict = field(default_factory=dict)
    person_box: list | None = None  # px box of the whole person: lets ``finish`` tell body from invented background


# ------------------------------------------------------------------ pure helpers
def parse_items(text: str) -> list[dict]:
    """Complete flat ``{...}`` objects with a name and a 4-number box; tolerant of truncated output."""
    out = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        b = d.get("bbox_2d")
        if isinstance(d.get("name"), str) and isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            out.append(d)
    return out


def label_of(name: str) -> str | None:
    low = name.lower()
    for lab, rx in _WORDS_RE:
        if rx.search(low):
            return lab
    return "top"  # shortcut: an unknown word (accessory, shoe variant) is read as an upper-body garment; extend _WORDS when seen


def select_replaced(items: list[dict], kind: str, over_top: bool = False) -> list[dict]:
    """Items (with ``label``/``layer``) that the new garment replaces."""
    up = [i for i in items if i["label"] == "top"]
    if kind == "outerwear":
        r = [i for i in up if i.get("layer") == "outer"] or (up if over_top else [])  # over_top: the top just dressed is the layer under it
    elif kind == "top":
        r = [i for i in up if i.get("layer") != "outer"] or up
    elif kind == "bottom":
        r = [i for i in items if i["label"] in ("pants", "skirt")]
    elif kind in ("dress", "jumpsuit"):
        r = [i for i in items if i["label"] in GARMENT_LABELS]
    else:
        r = []
    if not r:
        raise LayerError(f"no worn {kind or 'garment'} layer found in the photo")
    return r


def recompose(orig: np.ndarray, result: Image.Image, edit: np.ndarray, blur: float = 9) -> tuple[np.ndarray, np.ndarray]:
    """Original pixels everywhere except ``edit`` (feathered inward). Returns (image, support) where
    support = pixels allowed to differ; outside it the output is the input, bit for bit."""
    import cv2

    h, w = orig.shape[:2]
    up = np.asarray(result.convert("RGB").resize((w, h), Image.LANCZOS))
    a = cv2.GaussianBlur(edit.astype(np.float32), (0, 0), blur)
    a[~edit] = 0
    out = (orig * (1 - a[..., None]) + up * a[..., None]).round().astype(np.uint8)
    out[a == 0] = orig[a == 0]
    return out, a > 0


def _center_band(mask: np.ndarray, half: float = 0.09) -> np.ndarray:
    """Vertical strip down the middle of ``mask``: where an open jacket shows the layer under it.
    Only the upper 60% of the mask height: measured, a full-height band makes FASHN stretch the jacket to the knees and leave white patches at the hem.
    shortcut: fixed width (18% of the mask width), a closed jacket loses its centre; read the opening from the garment photo to fix."""
    ys, xs = np.nonzero(mask)
    cx, h, w = (xs.min() + xs.max()) / 2, ys.max() - ys.min(), xs.max() - xs.min()
    band = np.zeros_like(mask)
    band[int(ys.min() + 0.12 * h) : int(ys.min() + 0.6 * h), int(cx - half * w) : int(cx + half * w)] = True
    return band & mask


def _clean(mask: np.ndarray) -> np.ndarray:
    """Drop SAM2 specks (hair dots on a jacket punch holes in FASHN's mask and cause block artefacts)."""
    import cv2

    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    min_area = 0.0006 * mask.size  # 2000 px on 1536x2048
    return np.isin(cc, [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area])


# ------------------------------------------------------------------ SAM2
def sam2_masks(weights: str, image: np.ndarray, specs: list[dict]) -> list[np.ndarray]:
    """One mask per spec ``{box: [x1,y1,x2,y2] px, pos: [(x,y)], neg_from: [index of an earlier spec]}``.
    ``neg_from`` adds the deepest point of those masks as negative points (a scarf box also holds the chin).
    Loads and frees SAM2."""
    import cv2
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", weights, device="cuda")
    try:
        pred = SAM2ImagePredictor(model)
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred.set_image(image)
            for s in specs:
                kw = {}
                neg = []
                for j in s.get("neg_from", []):
                    if out[j].any():
                        d = cv2.distanceTransform(out[j].astype(np.uint8), cv2.DIST_L2, 5)
                        y, x = np.unravel_index(d.argmax(), d.shape)
                        neg.append((x, y))
                pos = list(s.get("pos", []))
                if pos or neg:
                    kw["point_coords"] = np.array(pos + neg, dtype=np.float32)
                    kw["point_labels"] = np.array([1] * len(pos) + [0] * len(neg))
                m, _, _ = pred.predict(box=np.array(s["box"], dtype=np.float32), multimask_output=False, **kw)
                out.append(m[0] > 0)
        return out
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ planner
class LayerPlanner:
    def __init__(self, ollama_url: str, model: str, sam2_weights: str, timeout: float = 180):
        self.url = ollama_url.rstrip("/")
        self.model = model
        self.sam2_weights = sam2_weights
        self.timeout = timeout

    def available(self) -> dict:
        """{"qwen": (ok, why), "sam2": (ok, why)}"""
        qwen = OllamaQwenVLAnalyzer(self.url, self.model).available()
        if importlib.util.find_spec("sam2") is None:
            sam2 = (False, "sam2 not installed (see docs/TRYON_SETUP.md)")
        elif not os.path.exists(self.sam2_weights):
            sam2 = (False, f"SAM2 weights missing: {self.sam2_weights}")
        else:
            sam2 = (True, "sam2.1_hiera_small ready")
        return {"qwen": qwen, "sam2": sam2}

    # -- Qwen3-VL
    def _ask(self, image: Image.Image, prompt: str) -> list[dict]:
        text = ollama_chat(self.url, self.model, prompt, _image_b64(image), PREFILL, self.timeout, keep_alive="2m")
        return parse_items(text)

    def _unload_qwen(self) -> None:
        try:
            httpx.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30)
        except httpx.HTTPError:
            log.warning("could not unload %s from ollama", self.model)

    def detect(self, person: Image.Image) -> list[dict]:
        """Worn items with ``label`` and ``bbox`` (0-1000 of the full image). Qwen is unloaded even on failure."""
        try:
            return self._detect(person)
        finally:
            self._unload_qwen()

    def _detect(self, person: Image.Image) -> list[dict]:
        items = self._ask(person, GARMENT_PROMPT) + self._ask(person, BODY_PROMPT)
        face = next((i for i in items if label_of(i["name"]) == "face"), None)
        if face is not None:  # a 2B model misses small head accessories on the full frame: ask again on a head crop
            x1, y1, x2, y2 = face["bbox_2d"]
            fw, fh = x2 - x1, y2 - y1
            c = [max(0, x1 - 0.6 * fw), max(0, y1 - 0.6 * fh), min(1000, x2 + 0.6 * fw), min(1000, y2 + 1.2 * fh)]
            W, H = person.size
            crop = person.crop((int(c[0] * W / 1000), int(c[1] * H / 1000), int(c[2] * W / 1000), int(c[3] * H / 1000)))
            head = self._ask(crop, HEAD_PROMPT)
            for i in head:  # crop-relative -> full-image 0-1000
                bx = i["bbox_2d"]
                i["bbox_2d"] = [c[0] + bx[0] * (c[2] - c[0]) / 1000, c[1] + bx[1] * (c[3] - c[1]) / 1000,
                                c[0] + bx[2] * (c[2] - c[0]) / 1000, c[1] + bx[3] * (c[3] - c[1]) / 1000]  # fmt: skip
            head_labels = {label_of(i["name"]) for i in head}
            items = [i for i in items if label_of(i["name"]) not in head_labels or label_of(i["name"]) in ("face", "hair", "hands")] + head
        out, seen = [], set()
        for i in items:
            lab = label_of(i["name"])
            box = tuple(round(v) for v in i["bbox_2d"])
            if lab is not None and (lab, box) not in seen:  # a 2B model sometimes repeats one box under several names
                seen.add((lab, box))
                out.append({"name": i["name"], "label": lab, "layer": i.get("layer"), "bbox": [float(v) for v in i["bbox_2d"]]})
        return out

    # -- plan: Qwen + SAM2 -> label map
    def plan(self, person: Image.Image, kind: str, over_top: bool = False) -> LayerPlan:
        t0 = time.perf_counter()
        items = self.detect(person)
        t_qwen = time.perf_counter() - t0
        replaced = select_replaced(items, kind, over_top)
        rep_ids = {id(i) for i in replaced}
        paint = [i for i in items if id(i) in rep_ids or i["label"] not in GARMENT_LABELS]
        W, H = person.size
        body = [i["bbox"] for i in items if i["label"] in ("top", "pants", "hair", "face")]
        person_box = [min(b[0] for b in body) * W / 1000, min(b[1] for b in body) * H / 1000,
                      max(b[2] for b in body) * W / 1000, max(b[3] for b in body) * H / 1000] if body else None  # fmt: skip
        px = lambda b: [max(0, b[0] * W / 1000), max(0, b[1] * H / 1000), min(W - 1, b[2] * W / 1000), min(H - 1, b[3] * H / 1000)]  # noqa: E731
        paint.sort(key=lambda i: PAINT_ORDER.index(i["label"]))  # face before scarf/hat: its mask gives negative points
        chin = next((px(i["bbox"])[3] for i in paint if i["label"] == "face"), 0)
        specs = []
        for i in paint:
            box = px(i["bbox"])
            if id(i) in rep_ids and i["label"] in ("top", "dress") and chin < box[3]:
                box[1] = max(box[1], chin)  # a box that starts above the chin pulls the head and hair into the garment mask
            spec = {"box": box}
            if i["label"] in ("scarf", "hat"):
                spec["pos"] = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)]
                spec["neg_from"] = [j for j, k in enumerate(paint) if k["label"] == "face"]
            specs.append(spec)
        t1 = time.perf_counter()
        masks = [_clean(m) for m in sam2_masks(self.sam2_weights, np.asarray(person.convert("RGB")), specs)]
        t_sam = time.perf_counter() - t1
        # Qwen boxes are rough: a kept item whose mask spills far outside its box is the wrong object
        # (hands in pockets -> the sleeve), and would erase the garment. Drop it.
        ok = [
            id(i) in rep_ids or 0 < m.sum() <= 1.5 * (sp["box"][2] - sp["box"][0]) * (sp["box"][3] - sp["box"][1])
            for i, m, sp in zip(paint, masks, specs, strict=True)
        ]
        paint = [i for i, k in zip(paint, ok, strict=True) if k]
        masks = [m for m, k in zip(masks, ok, strict=True) if k]
        lm = np.zeros((H, W), np.uint8)
        for lab in PAINT_ORDER:
            for i, m in zip(paint, masks, strict=True):
                if i["label"] == lab:
                    lm[m] = LABEL_IDS[lab]
        old = np.zeros((H, W), bool)
        keep = np.zeros((H, W), bool)
        for i, m in zip(paint, masks, strict=True):
            if id(i) in rep_ids:
                old |= m
            else:
                keep |= m
        if over_top and kind == "outerwear" and old.any():
            inner = _center_band(old)
            lm[inner] = LABEL_IDS["jewelry"]  # identity label: FASHN leaves it unmasked, so the layer under stays visible
            keep |= inner
        old &= ~keep
        if not old.any():
            raise LayerError("the replaced garment could not be segmented")
        return LayerPlan(lm, old, keep, {"qwen": t_qwen, "sam2": t_sam}, person_box)

    # -- finish: SAM2 on the result, edit mask, full-resolution paste
    def finish(self, person: Image.Image, result: Image.Image, plan: LayerPlan, ca_mask: np.ndarray, dilate: int = 3) -> tuple[Image.Image, dict]:
        import cv2

        orig = np.asarray(person.convert("RGB"))
        H, W = orig.shape[:2]
        if abs(result.width / result.height - W / H) > 0.01:
            raise LayerError("try-on output aspect ratio differs from the photo")
        ys, xs = np.nonzero(ca_mask)
        if not len(ys):
            raise LayerError("empty repaint mask")
        k = result.width / W
        dist = cv2.distanceTransform(plan.old_mask.astype(np.uint8), cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(dist.argmax(), dist.shape)
        t0 = time.perf_counter()
        spec = {
            "box": [xs.min() * k, ys.min() * k, xs.max() * k, ys.max() * k],
            "pos": [(cx * k, cy * k)],
        }
        specs = [spec] + ([{"box": [v * k for v in plan.person_box], "pos": spec["pos"]}] if plan.person_box else [])
        new, *body = sam2_masks(self.sam2_weights, np.asarray(result.convert("RGB")), specs)
        t_sam = time.perf_counter() - t0
        new = cv2.resize(new.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        new &= ca_mask
        t1 = time.perf_counter()
        union = (plan.old_mask | new).astype(np.uint8)
        edit = (cv2.dilate(union, np.ones((2 * dilate + 1,) * 2, np.uint8)) > 0) & ~plan.keep_mask
        out, support = recompose(orig, result, edit, blur=3)
        if body:  # FASHN invents a blurred background where the old garment overhung the body (puffer -> short jacket): inpaint it instead
            person_m = cv2.resize(body[0].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            far = (edit & (cv2.dilate(person_m, np.ones((5, 5), np.uint8)) == 0)).astype(np.uint8)  # measured: a 9 px margin leaves a dark halo around the jacket, 5 px does not
            fill = cv2.dilate(cv2.morphologyEx(far, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8)) > 0
            fill &= support & ~plan.keep_mask
            out = cv2.inpaint(out, fill.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)  # shortcut: smears busy backgrounds, upgrade = patch-based inpaint
        info = {
            "sam2": t_sam,
            "recompose": time.perf_counter() - t1,
            "edit_fraction": float(support.mean()),
            "outside_mask_max_diff": int(np.abs(out.astype(int) - orig).max(-1)[~support].max(initial=0)),
            # share of the old garment that the new one does not cover: bare skin/lining is imagined there
            "uncovered_ratio": float((plan.old_mask & ~(cv2.dilate(new.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0)).sum() / plan.old_mask.sum()),
        }
        return Image.fromarray(out), info


if __name__ == "__main__":
    # self-check: pixels outside the support are untouched, the edit area is changed
    o = np.random.randint(0, 255, (40, 30, 3), np.uint8)
    r = Image.fromarray(np.zeros((20, 15, 3), np.uint8))
    e = np.zeros((40, 30), bool)
    e[10:20, 10:20] = True
    c, s = recompose(o, r, e, blur=1)
    assert (c[~s] == o[~s]).all() and s.any() and not s.all() and (c[e] != o[e]).any()
    m = np.zeros((20, 40), bool)
    m[2:18, 5:35] = True
    b = _center_band(m)
    assert b.any() and b[:, 20].any() and not b[:, :10].any() and not (b & ~m).any() and not b[12:].any()
    assert [label_of(n) for n in ("Black leather jacket", "red scarf", "shoulder bag strap", "headband", "blue jeans", "ankle boots")] == [
        "top", "scarf", "bag", "hat", "pants", None]  # fmt: skip
    print("ok")
