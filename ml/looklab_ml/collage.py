"""Editorial outfit collages built from the real garment photos.

This is the zero-cost preview path: no generative model, a few tens of
milliseconds of Pillow work per outfit. Photorealistic try-on only runs for
looks the user explicitly selects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .vision.color_analysis import _foreground_mask

COLLAGE_VERSION = "v4"
CANVAS = (800, 1000)
BG = (247, 241, 232)  # warm cream
SHADOW = (60, 40, 30)

# Cells in canvas coordinates: (x, y, w, h)
LAYOUT = {
    "main": {
        "one_piece": (60, 60, 440, 760),
        "top": (60, 60, 440, 400),
        "bottom": (60, 460, 440, 420),
    },
    "side": ["outerwear", "bag", "accessory", "shoes"],
    "side_box": (530, 60, 230, 880),
}


@dataclass
class CollageItem:
    image: Image.Image
    slot: str


def cutout(image: Image.Image, max_side: int = 640) -> Image.Image:
    """Crop to the garment and make a uniform background transparent."""
    img = image.convert("RGB")
    img.thumbnail((max_side, max_side))
    small = img.copy()
    small.thumbnail((200, 200))
    mask_small, _ = _foreground_mask(np.asarray(small, dtype=np.float64))
    if mask_small.all():
        # No separable background (busy photo, or a white garment on white):
        # present the photo as a rounded card instead of a hard rectangle.
        rgba = img.convert("RGBA")
        rounded = Image.new("L", img.size, 0)
        ImageDraw.Draw(rounded).rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=max(8, min(img.size) // 14), fill=255)
        rgba.putalpha(rounded)
        return rgba
    m = Image.fromarray((mask_small * 255).astype("uint8")).resize(img.size, Image.BILINEAR)
    m = m.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.0))  # erode: no background halo
    rgba = img.convert("RGBA")
    rgba.putalpha(m)
    bbox = m.point(lambda v: 255 if v > 40 else 0).getbbox()
    if bbox:
        pad = 6
        bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(img.width, bbox[2] + pad), min(img.height, bbox[3] + pad))
        rgba = rgba.crop(bbox)
    return rgba


def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    scale = min(w / img.width, h / img.height)
    return img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)


def _paste_with_shadow(canvas: Image.Image, img: Image.Image, cell: tuple[int, int, int, int]) -> None:
    x, y, w, h = cell
    fitted = _fit(img, w, h)
    px = x + (w - fitted.width) // 2
    py = y + (h - fitted.height) // 2
    alpha = fitted.split()[-1]
    shadow = Image.new("RGBA", fitted.size, SHADOW + (0,))
    shadow.putalpha(alpha.point(lambda v: int(v * 0.22)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    canvas.alpha_composite(shadow, (px + 6, py + 10))
    canvas.alpha_composite(fitted, (px, py))


def render_collage(items: list[CollageItem], title: str | None = None) -> Image.Image:
    canvas = Image.new("RGBA", CANVAS, BG + (255,))
    main = [it for it in items if it.slot in ("top", "bottom", "one_piece")]
    side = [it for it in items if it.slot not in ("top", "bottom", "one_piece")]
    for it in main:
        cell = LAYOUT["main"].get(it.slot)
        if it.slot == "one_piece" or len(main) == 1:
            cell = LAYOUT["main"]["one_piece"]
        _paste_with_shadow(canvas, cutout(it.image), cell)
    if side:
        sx, sy, sw, sh = LAYOUT["side_box"]
        order = {s: i for i, s in enumerate(LAYOUT["side"])}
        side.sort(key=lambda it: order.get(it.slot, 9))
        gap = 20
        ch = (sh - gap * (len(side) - 1)) // len(side)
        for i, it in enumerate(side):
            _paste_with_shadow(canvas, cutout(it.image), (sx, sy + i * (ch + gap), sw, ch))
    if title:
        d = ImageDraw.Draw(canvas)
        d.text((60, CANVAS[1] - 50), title[:60], fill=(62, 39, 35, 255))
    return canvas.convert("RGB")
