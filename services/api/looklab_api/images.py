"""Upload validation and sanitisation.

Every uploaded image is decoded, orientation-corrected, re-encoded from raw
pixels (which drops EXIF, GPS, maker notes and any other metadata) and
downscaled to a bounded size before it touches storage.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

try:  # optional iPhone HEIC support
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
    _HEIC = True
except ImportError:  # pragma: no cover - optional
    _HEIC = False

Image.MAX_IMAGE_PIXELS = 50_000_000  # decompression-bomb guard

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "MPO"} | ({"HEIF", "HEIC"} if _HEIC else set())


class ImageValidationError(ValueError):
    pass


@dataclass
class SanitizedImage:
    data: bytes
    thumb: bytes
    ext: str
    content_type: str
    width: int
    height: int
    sha256: str
    image: Image.Image


def sanitize_image(raw: bytes, *, max_bytes: int, max_side: int, min_side: int = 64, keep_alpha: bool = False) -> SanitizedImage:
    if not raw:
        raise ImageValidationError("empty file")
    if len(raw) > max_bytes:
        raise ImageValidationError(f"file too large ({len(raw) / 2**20:.1f} MB > {max_bytes / 2**20:.0f} MB)")
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            fmt = probe.format
            probe.verify()
    except Image.DecompressionBombError as e:
        raise ImageValidationError("image dimensions too large") from e
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ImageValidationError("not a valid image") from e
    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError(f"unsupported format {fmt}; use JPEG, PNG or WEBP")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = ImageOps.exif_transpose(img)
    except Image.DecompressionBombError as e:
        raise ImageValidationError("image dimensions too large") from e
    if min(img.size) < min_side:
        raise ImageValidationError(f"image too small ({img.width}x{img.height}); minimum {min_side}px")

    has_alpha = keep_alpha and (img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info))
    # Re-create from pixel data only: no info dict, no EXIF, no ICC tricks.
    img = img.convert("RGBA" if has_alpha else "RGB")
    clean = Image.new(img.mode, img.size)
    clean.paste(img)
    clean.thumbnail((max_side, max_side), Image.LANCZOS)

    buf = io.BytesIO()
    if has_alpha:
        clean.save(buf, format="PNG", optimize=True)
        ext, ctype = "png", "image/png"
    else:
        clean.save(buf, format="JPEG", quality=92, optimize=True)
        ext, ctype = "jpg", "image/jpeg"
    data = buf.getvalue()

    th = clean.copy()
    th.thumbnail((480, 480), Image.LANCZOS)
    tbuf = io.BytesIO()
    th.convert("RGB").save(tbuf, format="WEBP", quality=82)

    return SanitizedImage(
        data=data,
        thumb=tbuf.getvalue(),
        ext=ext,
        content_type=ctype,
        width=clean.width,
        height=clean.height,
        sha256=hashlib.sha256(data).hexdigest(),
        image=clean,
    )


def has_exif(data: bytes) -> bool:
    with Image.open(io.BytesIO(data)) as im:
        return bool(im.getexif()) or "exif" in im.info
