"""Interpretable preference learning (Layer D).

A sparse linear model over human-readable features (``color:cream``,
``style:korean``, ``fit:relaxed``, ``garment:<id>`` ...). Likes, dislikes and
saves nudge weights with a bounded update; the UI can show exactly why an
outfit was boosted. No neural network, no training job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import Garment

LEARNING_RATES = {"like": 0.25, "save": 0.35, "dislike": -0.3, "tryon": 0.15}
WEIGHT_CLIP = 2.0


def outfit_features(garments: list[Garment]) -> set[str]:
    feats: set[str] = set()
    for g in garments:
        feats.add(f"garment:{g.id}")
        for c in g.colors[:2]:
            feats.add(f"color:{c}")
        for s in g.style:
            feats.add(f"style:{s}")
        if g.fit != "unknown":
            feats.add(f"fit:{g.fit}")
        if g.pattern != "unknown":
            feats.add(f"pattern:{g.pattern}")
        if g.subcategory:
            feats.add(f"piece:{g.subcategory}")
    return feats


@dataclass
class PreferenceModel:
    weights: dict[str, float] = field(default_factory=dict)
    events: int = 0

    def score(self, garments: list[Garment]) -> tuple[float, list[str]]:
        """Return a [0, 1] preference score (0.5 = neutral) and top reasons."""
        feats = outfit_features(garments)
        contribs = [(f, self.weights.get(f, 0.0)) for f in feats]
        total = sum(w for _, w in contribs)
        s = 0.5 + 0.5 * math.tanh(total / 3)
        top = sorted((c for c in contribs if abs(c[1]) >= 0.2), key=lambda c: -abs(c[1]))[:3]
        reasons = [("you like " if w > 0 else "you tend to skip ") + f.split(":", 1)[1] for f, w in top if not f.startswith("garment:")]
        return s, reasons

    def update(self, garments: list[Garment], event: str) -> None:
        lr = LEARNING_RATES.get(event)
        if lr is None:
            raise ValueError(f"unknown feedback event {event!r}")
        feats = outfit_features(garments)
        # Spread the update so large outfits do not move weights more.
        step = lr / max(1, math.sqrt(len(feats)))
        for f in feats:
            w = self.weights.get(f, 0.0) + step
            self.weights[f] = max(-WEIGHT_CLIP, min(WEIGHT_CLIP, w))
        self.events += 1

    def seed_from_profile(self, liked_colors=(), liked_styles=(), disliked_colors=(), disliked_styles=()) -> None:
        """Stated preferences start as moderate weights; feedback can override them."""
        for c in liked_colors:
            self.weights.setdefault(f"color:{c}", 0.6)
        for s in liked_styles:
            self.weights.setdefault(f"style:{s}", 0.6)
        for c in disliked_colors:
            self.weights.setdefault(f"color:{c}", -0.8)
        for s in disliked_styles:
            self.weights.setdefault(f"style:{s}", -0.8)

    def top_features(self, n: int = 12) -> list[tuple[str, float]]:
        return sorted(((f, w) for f, w in self.weights.items() if not f.startswith("garment:")), key=lambda x: -abs(x[1]))[:n]

    def to_dict(self) -> dict:
        return {"weights": {k: round(v, 4) for k, v in self.weights.items()}, "events": self.events}

    @classmethod
    def from_dict(cls, d: dict | None) -> "PreferenceModel":
        d = d or {}
        return cls(weights=dict(d.get("weights", {})), events=int(d.get("events", 0)))
