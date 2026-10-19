"""Deterministic color utilities: naming, families and harmony scoring."""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass

# Fashion color names with representative sRGB values.
PALETTE: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 22),
    "charcoal": (60, 60, 64),
    "grey": (128, 128, 130),
    "light grey": (196, 196, 198),
    "white": (246, 246, 244),
    "cream": (240, 232, 212),
    "ivory": (250, 246, 232),
    "beige": (216, 196, 162),
    "camel": (193, 154, 107),
    "tan": (170, 128, 88),
    "brown": (110, 72, 44),
    "chocolate": (70, 42, 28),
    "khaki": (165, 155, 110),
    "olive": (100, 106, 50),
    "green": (40, 130, 70),
    "sage": (156, 175, 136),
    "mint": (170, 225, 195),
    "teal": (0, 118, 118),
    "navy": (28, 36, 72),
    "blue": (40, 90, 190),
    "light blue": (160, 196, 232),
    "denim": (80, 110, 150),
    "lavender": (190, 170, 220),
    "purple": (100, 50, 140),
    "burgundy": (110, 20, 40),
    "red": (200, 30, 40),
    "rust": (170, 80, 40),
    "orange": (235, 120, 30),
    "mustard": (210, 165, 40),
    "yellow": (245, 215, 60),
    "pink": (235, 130, 170),
    "blush": (232, 188, 180),
    "coral": (240, 120, 100),
}

NEUTRALS = {
    "black",
    "charcoal",
    "grey",
    "light grey",
    "white",
    "cream",
    "ivory",
    "beige",
    "camel",
    "tan",
    "brown",
    "chocolate",
    "khaki",
    "navy",
    "denim",
}
# Soft, muted tones that read as "neutral-adjacent" in most palettes.
SOFT_TONES = {"sage", "blush", "light blue", "lavender", "olive", "mint"}

COLOR_GROUP_KEYWORDS = {
    "neutral": NEUTRALS,
    "neutrals": NEUTRALS,
    "earth": {"camel", "tan", "brown", "chocolate", "khaki", "olive", "rust", "beige", "mustard"},
    "earthy": {"camel", "tan", "brown", "chocolate", "khaki", "olive", "rust", "beige", "mustard"},
    "pastel": {"blush", "lavender", "mint", "light blue", "cream", "pink"},
    "pastels": {"blush", "lavender", "mint", "light blue", "cream", "pink"},
    "dark": {"black", "charcoal", "navy", "chocolate", "burgundy", "brown"},
    "bright": {"red", "orange", "yellow", "blue", "green", "pink", "coral"},
}


def _srgb_to_linear(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def rgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(v) for v in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


_PALETTE_LAB = {name: rgb_to_lab(rgb) for name, rgb in PALETTE.items()}


def nearest_color_name(rgb: tuple[float, float, float]) -> tuple[str, float]:
    """Return the closest palette name and its CIE76 distance."""
    lab = rgb_to_lab(rgb)
    best_name, best_d = "unknown", math.inf
    for name, plab in _PALETTE_LAB.items():
        d = math.dist(lab, plab)
        if d < best_d:
            best_name, best_d = name, d
    return best_name, best_d


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(round(max(0, min(255, v)))) for v in rgb))


def is_neutral(name: str) -> bool:
    return name in NEUTRALS


@dataclass(frozen=True)
class HueInfo:
    hue: float  # degrees 0-360
    saturation: float
    lightness: float


def hue_info(name: str) -> HueInfo:
    r, g, b = (v / 255 for v in PALETTE.get(name, (128, 128, 128)))
    h, light, s = colorsys.rgb_to_hls(r, g, b)
    return HueInfo(h * 360, s, light)


def _hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def expand_color_terms(terms: list[str]) -> set[str]:
    """Turn user words ("neutral", "navy") into concrete palette names."""
    out: set[str] = set()
    for t in terms:
        t = t.lower().strip()
        if t in COLOR_GROUP_KEYWORDS:
            out |= COLOR_GROUP_KEYWORDS[t]
        elif t in PALETTE:
            out.add(t)
        elif t == "gray":
            out.add("grey")
    return out


def harmony_score(colors: list[str]) -> tuple[float, list[str]]:
    """Score a set of garment colors in [0, 1] with human-readable reasons.

    Rules (deterministic, interpretable):
    * neutrals combine with anything;
    * at most two chromatic hue families, ideally analogous (<45 deg) or
      complementary (150-210 deg);
    * clashing mid-distance hues (60-120 deg) between saturated colors are penalised.
    """
    known = [c for c in colors if c in PALETTE]
    if not known:
        return 0.5, ["colors unknown"]
    chromatic = sorted({c for c in known if not is_neutral(c)})
    reasons: list[str] = []
    if not chromatic:
        return 0.95, ["all-neutral palette"]
    score = 0.85
    infos = [hue_info(c) for c in chromatic]
    if len(chromatic) == 1:
        reasons.append(f"{chromatic[0]} accent on neutrals" if len(set(known)) > 1 else f"tonal {chromatic[0]}")
        return 0.9, reasons
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            a, b = infos[i], infos[j]
            d = _hue_distance(a.hue, b.hue)
            soft = chromatic[i] in SOFT_TONES or chromatic[j] in SOFT_TONES
            if d < 45:
                reasons.append(f"analogous {chromatic[i]} + {chromatic[j]}")
                score += 0.05
            elif 150 <= d <= 210:
                reasons.append(f"complementary {chromatic[i]} + {chromatic[j]}")
            elif 60 <= d <= 120 and not soft and min(a.saturation, b.saturation) > 0.45:
                reasons.append(f"{chromatic[i]} and {chromatic[j]} may clash")
                score -= 0.3
            else:
                score -= 0.08
    if len(chromatic) > 2:
        reasons.append(f"{len(chromatic)} accent colors is busy")
        score -= 0.15 * (len(chromatic) - 2)
    return max(0.0, min(1.0, score)), reasons
