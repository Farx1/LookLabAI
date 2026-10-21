"""Outfit generation service: recommendation + cached collage previews."""

from __future__ import annotations

import hashlib
import io
import json

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from looklab_ml.collage import COLLAGE_VERSION, CollageItem, render_collage
from looklab_ml.recommendation import Constraints, PreferenceModel, RecommendationResult

from .context import AppContext
from .db import Garment, InferenceRun, User
from .wardrobe import to_rec


def preference_model(user: User) -> PreferenceModel:
    prof = user.profile
    pm = PreferenceModel.from_dict(prof.preference_model)
    prefs = prof.preferences or {}
    pm.seed_from_profile(
        prefs.get("liked_colors", []), prefs.get("liked_styles", []), prefs.get("disliked_colors", []), prefs.get("disliked_styles", [])
    )
    return pm


def apply_profile_defaults(c: Constraints, user: User) -> Constraints:
    """Stated profile preferences apply unless the request already set them."""
    prefs = user.profile.preferences or {}
    c = c.copy()
    for col in prefs.get("disliked_colors", []):
        if col not in c.excluded_colors:
            c.excluded_colors = [*c.excluded_colors, col]
    if c.target_looseness is None and prefs.get("fit_preference") in ("slim", "regular", "relaxed", "oversized"):
        c.target_looseness = {"slim": 0.5, "regular": 1.5, "relaxed": 2.5, "oversized": 3.0}[prefs["fit_preference"]]
    if c.budget is None and prefs.get("default_budget"):
        c.budget = float(prefs["default_budget"])
    if prefs.get("currency") and c.budget is None:
        c.currency = prefs["currency"]
    return c


def recommend(
    db: Session, ctx: AppContext, user: User, c: Constraints, avoid_keys: set[str] | None = None
) -> tuple[RecommendationResult, dict]:
    garments = db.scalars(select(Garment).where(Garment.user_id == user.id)).all()
    by_id = {g.id: g for g in garments}
    pm = preference_model(user)
    eff = apply_profile_defaults(c, user)
    key = hashlib.sha256(
        json.dumps(
            [eff.to_dict(), user.profile.wardrobe_version, pm.to_dict(), sorted(avoid_keys or []), len(garments)],
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    user_cache = ctx.rec_cache.setdefault(user.id, {})
    if key in user_cache:
        return user_cache[key], by_id
    res = ctx.engine.recommend([to_rec(g) for g in garments], eff, pm, avoid_keys)
    if len(user_cache) > 32:
        user_cache.pop(next(iter(user_cache)))
    user_cache[key] = res
    db.add(
        InferenceRun(
            user_id=user.id,
            task="recommendation",
            provider="deterministic_hybrid",
            model="looklab-hybrid-v1",
            device="cpu",
            status="succeeded",
        )
    )
    db.commit()
    return res, by_id


def outfit_payload(res: RecommendationResult, by_id: dict[str, Garment]) -> dict:
    out = res.to_dict()
    for o in out["outfits"]:
        o["garments"] = [garment_brief(by_id[i]) for i in o["garment_ids"] if i in by_id]
        o["collage_url"] = "/api/outfits/collage?ids=" + ",".join(o["garment_ids"])
    return out


def garment_brief(g: Garment) -> dict:
    return {
        "id": g.id,
        "name": g.name,
        "category": g.category,
        "subcategory": g.subcategory,
        "colors": g.colors,
        "brand": g.brand,
        "price": g.price,
        "currency": g.currency,
        "is_owned": g.is_owned,
        "thumb_url": f"/api/garments/{g.id}/image?variant=thumb",
        "product_url": g.product_url,
    }


def collage_bytes(ctx: AppContext, user: User, garments: list[Garment]) -> bytes:
    h = hashlib.sha256((COLLAGE_VERSION + "|" + "|".join(f"{g.sha256}:{g.category}" for g in garments)).encode()).hexdigest()[:32]
    key = f"users/{user.id}/collages/{h}.webp"
    if ctx.storage.exists(key):
        return ctx.storage.get(key)
    from looklab_ml.recommendation.models import Garment as RG

    items = []
    for g in garments:
        with Image.open(ctx.storage.local_path(g.image_key)) as im:
            items.append(CollageItem(image=im.convert("RGBA"), slot=RG(id=g.id, category=g.category).slot))
    img = render_collage(items)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=85)
    data = buf.getvalue()
    ctx.storage.put(key, data, "image/webp")
    return data
