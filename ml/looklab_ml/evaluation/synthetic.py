"""Procedurally generated garment and person images.

These are drawn from scratch with Pillow, so they carry no copyright,
likeness or consent concerns. They make tests, the recommendation
benchmark and the demo wardrobe fully reproducible. They are *not* suitable
for judging photorealistic try-on quality; that needs real authorised photos
(see docs/EVALUATION.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

from ..colors import PALETTE

BG = (250, 250, 248)


def _shade(rgb, f):
    return tuple(max(0, min(255, int(c * f))) for c in rgb)


def _pattern(draw: ImageDraw.ImageDraw, box, base, pattern: str, rng: random.Random):
    x0, y0, x1, y1 = box
    alt = _shade(base, 0.55) if sum(base) > 300 else _shade(base, 1.8)
    if pattern == "striped":
        for y in range(y0, y1, 22):
            draw.rectangle((x0, y, x1, y + 9), fill=alt)
    elif pattern == "checked":
        for x in range(x0, x1, 30):
            draw.rectangle((x, y0, x + 10, y1), fill=alt)
        for y in range(y0, y1, 30):
            draw.rectangle((x0, y, x1, y + 10), fill=alt)
    elif pattern == "polka_dot":
        for x in range(x0, x1, 34):
            for y in range(y0, y1, 34):
                draw.ellipse((x + 8, y + 8, x + 20, y + 20), fill=alt)
    elif pattern == "floral":
        for _ in range(60):
            cx, cy = rng.randint(x0, x1), rng.randint(y0, y1)
            for dx, dy in ((0, -7), (7, 0), (0, 7), (-7, 0)):
                draw.ellipse((cx + dx - 6, cy + dy - 6, cx + dx + 6, cy + dy + 6), fill=alt)


def draw_garment(kind: str, color: str, pattern: str = "solid", size: int = 512, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    rgb = PALETTE[color]
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    s = size / 512
    S = lambda pts: [(x * s, y * s) for x, y in pts]  # noqa: E731
    shapes = {
        "top": S(
            [
                (150, 90),
                (210, 70),
                (302, 70),
                (362, 90),
                (450, 170),
                (400, 220),
                (360, 190),
                (360, 440),
                (152, 440),
                (152, 190),
                (112, 220),
                (62, 170),
            ]
        ),
        "sweater": S(
            [
                (150, 80),
                (220, 64),
                (292, 64),
                (362, 80),
                (470, 330),
                (420, 350),
                (362, 210),
                (362, 450),
                (150, 450),
                (150, 210),
                (92, 350),
                (42, 330),
            ]
        ),
        "trousers": S([(170, 60), (342, 60), (372, 460), (290, 460), (256, 200), (222, 460), (140, 460)]),
        "wide_trousers": S([(170, 60), (342, 60), (420, 460), (282, 460), (256, 220), (230, 460), (92, 460)]),
        "skirt": S([(190, 90), (322, 90), (380, 420), (132, 420)]),
        "wide_skirt": S([(200, 80), (312, 80), (450, 440), (62, 440)]),
        "dress": S([(200, 50), (312, 50), (330, 180), (300, 230), (420, 470), (92, 470), (212, 230), (182, 180)]),
        "jacket": S(
            [
                (140, 70),
                (210, 56),
                (302, 56),
                (372, 70),
                (470, 380),
                (420, 400),
                (372, 230),
                (372, 460),
                (140, 460),
                (140, 230),
                (92, 400),
                (42, 380),
            ]
        ),
    }
    if kind in shapes:
        d.polygon(shapes[kind], fill=rgb)
        if pattern != "solid":
            mask = Image.new("L", img.size, 0)
            ImageDraw.Draw(mask).polygon(shapes[kind], fill=255)
            pat = Image.new("RGB", img.size, rgb)
            _pattern(ImageDraw.Draw(pat), (0, 0, size, size), rgb, pattern, rng)
            img.paste(pat, (0, 0), mask)
        if kind == "jacket":
            d.line(S([(256, 60), (256, 460)]), fill=_shade(rgb, 0.6), width=int(4 * s))
    elif kind == "shoes":
        for ox in (0, 190):
            d.rounded_rectangle(S([(70 + ox, 270), (250 + ox, 340)]), radius=int(30 * s), fill=rgb)
            d.ellipse(S([(70 + ox, 220), (150 + ox, 330)]), fill=rgb)
    elif kind == "bag":
        d.rounded_rectangle(S([(120, 200), (392, 420)]), radius=int(24 * s), fill=rgb)
        d.arc(S([(180, 90), (332, 280)]), 180, 360, fill=_shade(rgb, 0.7), width=int(14 * s))
    elif kind == "scarf":
        d.polygon(S([(120, 80), (392, 80), (360, 440), (300, 440), (256, 160), (212, 440), (152, 440)]), fill=rgb)
    else:
        raise ValueError(kind)
    return img


def draw_person(size: tuple[int, int] = (768, 1152), skin=(205, 160, 130), outfit=(120, 120, 125)) -> Image.Image:
    """A flat, stylised mannequin used only to exercise the try-on pipeline plumbing."""
    w, h = size
    img = Image.new("RGB", size, (232, 228, 222))
    d = ImageDraw.Draw(img)
    cx = w // 2
    d.ellipse((cx - 70, 60, cx + 70, 230), fill=skin)
    d.rectangle((cx - 30, 220, cx + 30, 270), fill=skin)
    d.polygon([(cx - 150, 270), (cx + 150, 270), (cx + 120, 640), (cx - 120, 640)], fill=outfit)
    d.polygon([(cx - 150, 270), (cx - 200, 600), (cx - 165, 610), (cx - 120, 330)], fill=skin)
    d.polygon([(cx + 150, 270), (cx + 200, 600), (cx + 165, 610), (cx + 120, 330)], fill=skin)
    d.polygon(
        [(cx - 120, 640), (cx + 120, 640), (cx + 110, 1060), (cx + 20, 1060), (cx, 720), (cx - 20, 1060), (cx - 110, 1060)],
        fill=_shade(outfit, 0.7),
    )
    return img


@dataclass
class DemoItem:
    filename: str
    kind: str
    color: str
    pattern: str = "solid"
    meta: dict = field(default_factory=dict)


# The spec's principal use case: a 20-piece wardrobe.
DEMO_WARDROBE: list[DemoItem] = [
    DemoItem(
        "cream relaxed blouse",
        "top",
        "cream",
        meta=dict(
            category="top",
            subcategory="blouse",
            fit="relaxed",
            style=["romantic", "korean"],
            formality=3,
            season=["spring", "summer", "autumn"],
            price=35,
        ),
    ),
    DemoItem(
        "white oversized shirt",
        "top",
        "white",
        meta=dict(category="top", subcategory="shirt", fit="oversized", style=["minimalist", "korean"], formality=3, price=29),
    ),
    DemoItem(
        "beige knit sweater",
        "sweater",
        "beige",
        meta=dict(
            category="top",
            subcategory="sweater",
            fit="relaxed",
            style=["minimalist", "korean", "casual"],
            formality=2,
            season=["autumn", "winter"],
            price=45,
        ),
    ),
    DemoItem(
        "black fitted top",
        "top",
        "black",
        meta=dict(category="top", subcategory="top", fit="slim", style=["elegant", "chic"], formality=3, price=25),
    ),
    DemoItem(
        "blush camisole",
        "top",
        "blush",
        meta=dict(
            category="top", subcategory="camisole", fit="regular", style=["romantic"], formality=3, season=["spring", "summer"], price=22
        ),
    ),
    DemoItem(
        "navy striped tee",
        "top",
        "navy",
        "striped",
        meta=dict(category="top", subcategory="t-shirt", fit="regular", style=["casual", "classic"], formality=1, price=18),
    ),
    DemoItem(
        "camel wide trousers",
        "wide_trousers",
        "camel",
        meta=dict(
            category="bottom",
            subcategory="trousers",
            fit="relaxed",
            style=["minimalist", "korean", "elegant"],
            formality=3,
            price=49,
            details=["wide-leg"],
        ),
    ),
    DemoItem(
        "black tailored trousers",
        "trousers",
        "black",
        meta=dict(category="bottom", subcategory="trousers", fit="slim", style=["business", "classic"], formality=4, price=55),
    ),
    DemoItem(
        "denim jeans",
        "trousers",
        "denim",
        meta=dict(category="bottom", subcategory="jeans", fit="regular", style=["casual", "streetwear"], formality=1, price=40),
    ),
    DemoItem(
        "ivory wide pleated skirt",
        "wide_skirt",
        "ivory",
        meta=dict(
            category="bottom",
            subcategory="skirt",
            fit="relaxed",
            style=["romantic", "korean"],
            formality=3,
            price=39,
            details=["wide", "pleated", "midi"],
        ),
    ),
    DemoItem(
        "charcoal midi skirt",
        "skirt",
        "charcoal",
        meta=dict(
            category="bottom", subcategory="skirt", fit="regular", style=["minimalist", "elegant"], formality=3, price=32, details=["midi"]
        ),
    ),
    DemoItem(
        "burgundy slip dress",
        "dress",
        "burgundy",
        meta=dict(category="dress", subcategory="dress", fit="regular", style=["romantic", "elegant"], formality=4, price=69),
    ),
    DemoItem(
        "beige linen dress",
        "dress",
        "beige",
        meta=dict(
            category="dress",
            subcategory="dress",
            fit="relaxed",
            style=["minimalist", "korean", "romantic"],
            formality=3,
            season=["spring", "summer"],
            price=59,
        ),
    ),
    DemoItem(
        "floral sundress",
        "dress",
        "sage",
        "floral",
        meta=dict(
            category="dress", subcategory="dress", fit="relaxed", style=["bohemian", "romantic"], formality=2, season=["summer"], price=45
        ),
    ),
    DemoItem(
        "camel trench coat",
        "jacket",
        "camel",
        meta=dict(
            category="outerwear",
            subcategory="trench coat",
            fit="relaxed",
            style=["classic", "elegant", "korean"],
            formality=3,
            season=["autumn", "spring"],
            price=89,
        ),
    ),
    DemoItem(
        "cream cardigan",
        "sweater",
        "cream",
        meta=dict(
            category="outerwear",
            subcategory="cardigan",
            fit="oversized",
            style=["korean", "romantic", "casual"],
            formality=2,
            season=["autumn", "winter", "spring"],
            price=38,
        ),
    ),
    DemoItem(
        "white sneakers",
        "shoes",
        "white",
        meta=dict(category="shoes", subcategory="sneakers", style=["casual", "korean", "minimalist"], formality=1, price=60),
    ),
    DemoItem(
        "black loafers",
        "shoes",
        "black",
        meta=dict(category="shoes", subcategory="loafers", style=["classic", "minimalist", "elegant"], formality=3, price=70),
    ),
    DemoItem(
        "beige mary jane flats",
        "shoes",
        "beige",
        meta=dict(category="shoes", subcategory="flats", style=["romantic", "korean"], formality=3, price=48),
    ),
    DemoItem(
        "chocolate shoulder bag",
        "bag",
        "chocolate",
        meta=dict(category="bag", subcategory="bag", style=["classic", "elegant", "minimalist"], formality=3, price=65),
    ),
]
