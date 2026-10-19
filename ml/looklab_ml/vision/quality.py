"""Pre-flight quality checks for person photos before try-on.

Cheap, deterministic checks that catch the most common reasons a try-on
result will be poor. They explain the problem in plain language instead of
silently sending an unsuitable image to a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

MIN_SHORT_SIDE = 512
RECOMMENDED_SHORT_SIDE = 768


@dataclass
class QualityIssue:
    code: str
    severity: str  # "error" blocks try-on, "warning" informs
    message: str


@dataclass
class QualityReport:
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness: float
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "width": self.width,
            "height": self.height,
            "brightness": round(self.brightness, 1),
            "contrast": round(self.contrast, 1),
            "sharpness": round(self.sharpness, 1),
            "issues": [i.__dict__ for i in self.issues],
        }


def _laplacian_variance(gray: np.ndarray) -> float:
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def check_person_photo(image: Image.Image) -> QualityReport:
    w, h = image.size
    small = image.convert("L")
    small.thumbnail((512, 512))
    gray = np.asarray(small, dtype=np.float64)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = _laplacian_variance(gray)
    rep = QualityReport(w, h, brightness, contrast, sharpness)
    short = min(w, h)
    if short < MIN_SHORT_SIDE:
        rep.issues.append(
            QualityIssue("low_resolution", "error", f"Photo is {w}x{h}px; at least {MIN_SHORT_SIDE}px on the short side is needed.")
        )
    elif short < RECOMMENDED_SHORT_SIDE:
        rep.issues.append(
            QualityIssue("modest_resolution", "warning", f"{RECOMMENDED_SHORT_SIDE}px+ on the short side gives sharper results.")
        )
    if h < w:
        rep.issues.append(
            QualityIssue(
                "landscape",
                "warning",
                "Landscape photo: the try-on model works at a 2:3 portrait ratio, so the person will appear small.",
            )
        )
    if brightness < 50:
        rep.issues.append(QualityIssue("too_dark", "warning", "Photo is dark; garment colors may render inaccurately."))
    elif brightness > 225:
        rep.issues.append(QualityIssue("overexposed", "warning", "Photo is overexposed; details may be lost."))
    if contrast < 18:
        rep.issues.append(QualityIssue("low_contrast", "warning", "Very low contrast; the body outline may be hard to detect."))
    if sharpness < 20:
        rep.issues.append(QualityIssue("blurry", "warning", "Photo looks blurry; fine garment details will not transfer well."))
    rep.issues.append(
        QualityIssue(
            "framing_unverified",
            "info",
            "Framing is not checked automatically: use a front-facing photo with the body area of the garment visible.",
        )
    )
    return rep
