"""Wardrobe domain helpers: ORM <-> recommendation model, analysis merging."""

from __future__ import annotations

from looklab_ml import taxonomy as tx
from looklab_ml.recommendation import Garment as RecGarment
from looklab_ml.vision import Attr

from .db import Garment, Profile

# Fields that analysis may fill and the user may override.
ANALYZABLE_FIELDS = ("category", "subcategory", "colors", "color_hex", "pattern", "fit", "style", "season", "formality", "details")


def to_rec(g: Garment) -> RecGarment:
    return RecGarment(
        id=g.id,
        category=g.category or tx.UNKNOWN,
        subcategory=g.subcategory,
        name=g.name,
        colors=list(g.colors or []),
        pattern=g.pattern or tx.UNKNOWN,
        fit=g.fit or tx.UNKNOWN,
        style=list(g.style or []),
        season=list(g.season or []),
        occasion=list(g.occasion or []),
        formality=g.formality,
        price=g.price,
        currency=g.currency,
        is_owned=g.is_owned,
        favorite=g.favorite,
        details=list(g.details or []),
    )


def apply_analysis(g: Garment, attrs: dict[str, Attr], description: str | None = None) -> list[str]:
    """Fill fields from analysis without ever overwriting user-supplied values.

    Model output only replaces values that came from a weaker source:
    user > model > text > cv.
    """
    rank = {"user": 4, "model": 3, "text": 2, "cv": 1}
    sources = dict(g.field_sources or {})
    changed = []
    for field, attr in attrs.items():
        if field not in ANALYZABLE_FIELDS:
            continue
        cur = sources.get(field, {}).get("source") if isinstance(sources.get(field), dict) else None
        new_kind = attr.source.split(":")[0]
        if cur and rank.get(cur.split(":")[0], 0) > rank.get(new_kind, 0):
            continue
        if field == "colors" and cur == "cv" and new_kind == "model":
            continue  # pixel statistics name colors more reliably than a 2B VLM
        setattr(g, field, attr.value)
        sources[field] = {"source": attr.source, "confidence": attr.confidence}
        changed.append(field)
    if description and not g.description:
        g.description = description
    g.field_sources = sources
    return changed


def mark_user_fields(g: Garment, fields: list[str]) -> None:
    sources = dict(g.field_sources or {})
    for f in fields:
        sources[f] = {"source": "user", "confidence": 1.0}
    g.field_sources = sources


def bump_wardrobe_version(profile: Profile) -> None:
    profile.wardrobe_version = (profile.wardrobe_version or 0) + 1
