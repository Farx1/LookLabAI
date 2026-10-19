"""Traditional-CV garment analysis: background removal heuristic, dominant
colors and a coarse solid/patterned estimate.

This runs on any CPU in milliseconds and is the first stage of image
analysis. It only reports what it can observe from pixels; it never guesses
garment category, style or occasion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..colors import nearest_color_name, rgb_to_hex

ANALYSIS_SIZE = 160


@dataclass
class ColorSwatch:
    name: str
    hex: str
    share: float  # fraction of foreground pixels


@dataclass
class PixelAnalysis:
    colors: list[ColorSwatch] = field(default_factory=list)
    pattern: str = "unknown"
    pattern_confidence: float = 0.0
    foreground_ratio: float = 0.0
    background_hex: str | None = None

    @property
    def primary_color(self) -> str:
        return self.colors[0].name if self.colors else "unknown"


def _foreground_mask(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate background from the border and mask pixels close to it."""
    h, w, _ = arr.shape
    b = max(2, min(h, w) // 20)
    border = np.concatenate([arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3), arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    border_spread = float(np.median(np.linalg.norm(border - bg, axis=1)))
    # A busy border (e.g. a person photo) means no uniform background to remove.
    if border_spread > 30:
        return np.ones((h, w), dtype=bool), bg
    dist = np.linalg.norm(arr - bg, axis=2)
    mask = dist > max(28.0, border_spread * 3)
    if mask.mean() < 0.03:  # garment the same color as the background
        return np.ones((h, w), dtype=bool), bg
    return mask, bg


def _kmeans(pixels: np.ndarray, k: int, iters: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = min(k, len(pixels))
    # k-means++ initialisation
    centers = [pixels[rng.integers(len(pixels))]]
    for _ in range(1, k):
        d2 = np.min(((pixels[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else None
        centers.append(pixels[rng.choice(len(pixels), p=probs)])
    c = np.array(centers, dtype=np.float64)
    labels = np.zeros(len(pixels), dtype=int)
    for _ in range(iters):
        d = ((pixels[:, None, :] - c[None]) ** 2).sum(-1)
        labels = d.argmin(1)
        for j in range(k):
            sel = pixels[labels == j]
            if len(sel):
                c[j] = sel.mean(0)
    return c, labels


def analyze_pixels(image: Image.Image, k: int = 4) -> PixelAnalysis:
    img = image.convert("RGB")
    img.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE))
    arr = np.asarray(img, dtype=np.float64)
    mask, bg = _foreground_mask(arr)
    fg = arr[mask]
    if len(fg) < 20:
        return PixelAnalysis()
    centers, labels = _kmeans(fg, k)
    counts = np.bincount(labels, minlength=len(centers)) / len(labels)

    # Merge clusters that map to the same color name.
    merged: dict[str, list] = {}
    for center, share in zip(centers, counts, strict=False):
        if share <= 0:
            continue
        name, _ = nearest_color_name(tuple(center))
        if name in merged:
            merged[name][1] += share
        else:
            merged[name] = [center, share]
    swatches = sorted(
        (ColorSwatch(name=n, hex=rgb_to_hex(tuple(c)), share=round(float(s), 3)) for n, (c, s) in merged.items()),
        key=lambda s: -s.share,
    )
    swatches = [s for s in swatches if s.share >= 0.08] or swatches[:1]

    pattern, conf = _estimate_pattern(arr, mask, swatches)
    return PixelAnalysis(
        colors=swatches,
        pattern=pattern,
        pattern_confidence=conf,
        foreground_ratio=round(float(mask.mean()), 3),
        background_hex=rgb_to_hex(tuple(bg)),
    )


def _estimate_pattern(arr: np.ndarray, mask: np.ndarray, swatches: list[ColorSwatch]) -> tuple[str, float]:
    """Solid vs patterned from color dominance and interior edge density.

    Only "solid" and "patterned" are emitted, and only when the evidence is
    strong; otherwise "unknown". Fine pattern types (floral, check...) need a
    vision-language model or the user.
    """
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))[:-1, :]
    gy = np.abs(np.diff(gray, axis=0))[:, :-1]
    grad = np.hypot(gx, gy)
    inner = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:]
    # erode the mask so garment outline edges do not count as pattern
    er = inner.copy()
    for _ in range(3):
        er[1:-1, 1:-1] = er[1:-1, 1:-1] & er[:-2, 1:-1] & er[2:, 1:-1] & er[1:-1, :-2] & er[1:-1, 2:]
    if er.sum() < 50:
        return "unknown", 0.0
    edge_density = float((grad[er] > 25).mean())
    dominant = swatches[0].share if swatches else 0.0
    if dominant > 0.8 and edge_density < 0.06:
        return "solid", round(min(1.0, dominant), 2)
    if edge_density > 0.18 and dominant < 0.7:
        return "patterned", round(min(1.0, edge_density * 3), 2)
    return "unknown", 0.0
