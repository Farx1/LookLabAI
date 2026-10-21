"""FW2026 trends (trends_fw2026.json) and outfit/trend fit. `materials` and the `gold` color are dropped from the data."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .. import taxonomy as tx
from .models import Garment

_DATA = json.loads((Path(__file__).parent / "trends_fw2026.json").read_text(encoding="utf-8"))
SEASON_LABEL = "FW26"
TREND_MIN_FIT = 0.55  # below this an outfit does not "follow" a trend (surprise floor and label)
TRENDS: dict[str, dict] = {t["id"]: t for t in _DATA["trends"]}
CONFIDENCE_WEIGHT = {"haute": 3, "moyenne": 2, "faible": 1}


def _signals(g: Garment, t: dict) -> list[bool]:
    """One bool per trend signal the garment has data for."""
    s = t["signaux"]
    text = " ".join([g.subcategory or "", g.name or ""]).lower()
    out = [any(x in text or g.category == x for x in s.get("subcategories", []))]
    if g.colors and s.get("colors"):
        out.append(bool(set(g.colors[:2]) & set(s["colors"])))
    if g.style and s.get("styles"):
        out.append(bool(set(g.style) & set(s["styles"])))
    if g.pattern != tx.UNKNOWN and s.get("patterns"):
        out.append(g.pattern in s["patterns"])
    if g.fit in tx.FIT_LOOSENESS and s.get("fits"):
        out.append(g.fit in s["fits"])
    if g.formality is not None and s.get("formality"):
        out.append(s["formality"][0] <= g.formality <= s["formality"][1])
    return out


def touches(garments: list[Garment], t: dict) -> bool:
    """True if at least one garment matches at least one signal."""
    return any(any(_signals(g, t)) for g in garments)


def trend_fit(garments: list[Garment], t: dict) -> float:
    """Share of satisfied signals, 0..1 (mean over garments, plus one term for color pairs)."""
    parts = [statistics.mean(_signals(g, t)) for g in garments]
    pairs = t["signaux"].get("color_pairs")
    if pairs:
        have = {c for g in garments for c in g.colors[:2]}
        parts.append(1.0 if any(set(p) <= have for p in pairs) else 0.0)
    return statistics.mean(parts)


if __name__ == "__main__":
    assert len(TRENDS) == 20 and all("gold" not in t["signaux"]["colors"] for t in TRENDS.values())
    g = Garment(id="a", category="top", subcategory="blazer", colors=["grey"], style=["business"], pattern="solid", fit="slim", formality=3)
    t = TRENDS["skirt-suit-cinched"]
    assert touches([g], t) and trend_fit([g], t) > trend_fit([Garment(id="b", category="top", colors=["red"])], t)
