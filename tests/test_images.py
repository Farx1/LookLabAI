import io

import pytest
from PIL import Image

from looklab_api.images import ImageValidationError, has_exif, sanitize_image
from looklab_ml.evaluation.synthetic import draw_person

MB = 2**20


def jpeg_with_exif() -> bytes:
    img = draw_person((600, 900))
    exif = Image.Exif()
    exif[0x010F] = "SecretCam"  # Make
    exif[0x0132] = "2024:01:01 10:00:00"
    exif[0x8825] = {1: "N", 2: (48.0, 51.0, 24.0), 3: "E", 4: (2.0, 21.0, 3.0)}  # GPS
    exif[0x0112] = 6  # orientation: rotate 90
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_exif_and_gps_stripped_and_orientation_applied():
    raw = jpeg_with_exif()
    assert has_exif(raw)
    out = sanitize_image(raw, max_bytes=15 * MB, max_side=2048)
    assert not has_exif(out.data)
    assert b"SecretCam" not in out.data
    # orientation 6 means the stored landscape-rotated pixels are turned upright
    assert (out.width, out.height) == (900, 600)


def test_downscale_and_hash_stable():
    raw = jpeg_with_exif()
    a = sanitize_image(raw, max_bytes=15 * MB, max_side=512)
    b = sanitize_image(raw, max_bytes=15 * MB, max_side=512)
    assert max(a.width, a.height) == 512 and a.sha256 == b.sha256


@pytest.mark.parametrize(
    "raw,msg",
    [
        (b"", "empty"),
        (b"<html>not an image</html>", "not a valid image"),
        (b"GIF89a" + b"\x00" * 100, "not a valid image"),
    ],
)
def test_rejects_bad_files(raw, msg):
    with pytest.raises(ImageValidationError, match=msg):
        sanitize_image(raw, max_bytes=MB, max_side=1024)


def test_rejects_too_large_and_too_small():
    big = b"\xff" * (2 * MB)
    with pytest.raises(ImageValidationError, match="too large"):
        sanitize_image(big, max_bytes=MB, max_side=1024)
    buf = io.BytesIO()
    Image.new("RGB", (40, 40)).save(buf, "PNG")
    with pytest.raises(ImageValidationError, match="too small"):
        sanitize_image(buf.getvalue(), max_bytes=MB, max_side=1024, min_side=64)


def test_rejects_gif_format():
    buf = io.BytesIO()
    Image.new("RGB", (200, 200)).save(buf, "GIF")
    with pytest.raises(ImageValidationError, match="unsupported format"):
        sanitize_image(buf.getvalue(), max_bytes=MB, max_side=1024)


def test_decompression_bomb_rejected():
    buf = io.BytesIO()
    Image.new("1", (12000, 12000)).save(buf, "PNG")  # 144 MP, tiny file
    with pytest.raises(ImageValidationError, match="dimensions too large"):
        sanitize_image(buf.getvalue(), max_bytes=15 * MB, max_side=1024)


def test_alpha_kept_for_garments_only_when_requested():
    buf = io.BytesIO()
    Image.new("RGBA", (300, 300), (200, 10, 10, 0)).save(buf, "PNG")
    assert sanitize_image(buf.getvalue(), max_bytes=MB, max_side=512, keep_alpha=True).ext == "png"
    assert sanitize_image(buf.getvalue(), max_bytes=MB, max_side=512).ext == "jpg"
