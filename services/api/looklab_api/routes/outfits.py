from __future__ import annotations

import random
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from looklab_ml.recommendation import Constraints, parse_request
from looklab_ml.recommendation.preferences import PreferenceModel
from looklab_ml.recommendation.trends import CONFIDENCE_WEIGHT, TREND_MIN_FIT, TRENDS, touches

from ..auth import current_user
from ..db import Garment, SavedOutfit, TryOnResultRow
from ..deps import get_ctx, get_db
from ..outfits import collage_bytes, garment_brief, outfit_payload, preference_model, recommend
from ..serializers import saved_out, tryon_out
from ..wardrobe import to_rec
from .photos import PRIVATE_CACHE

router = APIRouter(prefix="/api", tags=["outfits"])


class GenerateBody(BaseModel):
    prompt: str = Field("", max_length=1000)
    constraints: dict | None = None  # structured overrides from the form
    count: int | None = Field(None, ge=1, le=30)


@router.post("/outfits/generate")
def generate(body: GenerateBody, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    base = Constraints.from_dict(body.constraints) if body.constraints else None
    parsed = parse_request(body.prompt, base) if body.prompt.strip() else None
    c = parsed.constraints if parsed else (base or Constraints())
    if body.count:
        c.count = body.count
    res, by_id = recommend(db, ctx, user, c)
    out = outfit_payload(res, by_id)
    out["understood"] = parsed.understood if parsed else []
    return out


SURPRISE_HISTORY = 30


class SurpriseBody(BaseModel):
    count: int = Field(1, ge=1, le=6)
    seed: int | None = None


@router.post("/outfits/surprise")
def surprise(body: SurpriseBody, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    """New complete outfit from owned pieces only, aligned on a random FW26 trend."""
    seed = body.seed if body.seed is not None else random.randrange(2**31)
    rng = random.Random(seed)
    owned = [to_rec(g) for g in db.scalars(select(Garment).where(Garment.user_id == user.id, Garment.is_owned.is_(True))).all()]
    feasible = [t for t in TRENDS.values() if touches(owned, t)]
    if not feasible:
        raise HTTPException(422, "pas assez de pièces possédées pour une tenue complète")
    model = dict(user.profile.preference_model or {})
    seen = model.get("surprise_seen", [])
    order = []  # weighted random draw order over trends
    while feasible:
        order.append(feasible.pop(rng.choices(range(len(feasible)), [CONFIDENCE_WEIGHT[t["confiance"]] for t in feasible])[0]))

    def run(trend, exclude):
        c = Constraints(count=body.count, owned_only=True, trend_id=trend["id"], min_trend_fit=TREND_MIN_FIT, seed=seed)
        return (trend, *recommend(db, ctx, user, c, exclude))

    reset = False
    best = None
    for t in order:  # a trend short of count: try another one, keep the fullest
        cand = run(t, set(seen))
        if not best or len(cand[1].outfits) > len(best[1].outfits):
            best = cand
        if len(best[1].outfits) >= body.count:
            break
    if not best[1].outfits and seen:  # every combination already shown: start over
        for t in order:
            cand = run(t, None)
            if cand[1].outfits:
                best, reset, seen = cand, True, []
                break
    trend, res, by_id = best
    if not res.outfits:
        raise HTTPException(422, "pas assez de pièces possédées pour une tenue complète")
    model["surprise_seen"] = (seen + [o.key for o in res.outfits])[-SURPRISE_HISTORY:]
    user.profile.preference_model = model
    db.commit()
    out = outfit_payload(res, by_id)
    out["trend"] = {k: trend[k] for k in ("id", "nom", "description", "sources")}
    out["seed"] = seed
    out["history_reset"] = reset
    return out


def _owned_garments(db, user, ids: list[str]) -> list[Garment]:
    out = []
    for i in ids:
        g = db.get(Garment, i)
        if g is None or g.user_id != user.id:
            raise HTTPException(404, f"garment {i} not found")
        out.append(g)
    return out


class VariationsBody(BaseModel):
    garment_ids: list[str] = Field(min_length=1, max_length=5)  # one engine run per piece when no swap id: bounded cost
    count: int = Field(4, ge=1, le=8)
    swap_garment_id: str | None = None


@router.post("/outfits/variations")
def variations(body: VariationsBody, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    """Close outfits where 1-2 pieces change, owned pieces only."""
    _owned_garments(db, user, body.garment_ids)
    if body.swap_garment_id and body.swap_garment_id not in body.garment_ids:
        raise HTTPException(422, "swap_garment_id must be one of garment_ids")
    garments = db.scalars(select(Garment).where(Garment.user_id == user.id)).all()
    res = ctx.engine.variations(
        [to_rec(g) for g in garments], list(dict.fromkeys(body.garment_ids)), body.count, body.swap_garment_id, preference_model(user)
    )
    return outfit_payload(res, {g.id: g for g in garments})


@router.get("/outfits/collage")
def collage(ids: str = Query(..., max_length=400), db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    gids = [i for i in ids.split(",") if i][:8]
    if not gids:
        raise HTTPException(400, "no garments")
    data = collage_bytes(ctx, user, _owned_garments(db, user, gids))
    return Response(data, media_type="image/webp", headers=PRIVATE_CACHE)


class FeedbackBody(BaseModel):
    garment_ids: list[str] = Field(min_length=1, max_length=8)
    event: Literal["like", "dislike", "save", "tryon"]


def _apply_feedback(db, user, garments: list[Garment], event: str) -> dict:
    prof = user.profile
    pm = PreferenceModel.from_dict(prof.preference_model)
    pm.update([to_rec(g) for g in garments], event)
    # keep the surprise history: to_dict() only knows the learned weights
    prof.preference_model = {**pm.to_dict(), "surprise_seen": (prof.preference_model or {}).get("surprise_seen", [])}
    db.commit()
    return {"events": pm.events, "top_features": pm.top_features(8)}


@router.post("/outfits/feedback")
def feedback(body: FeedbackBody, db=Depends(get_db), user=Depends(current_user)):
    return _apply_feedback(db, user, _owned_garments(db, user, body.garment_ids), body.event)


@router.get("/preferences/learned")
def learned(user=Depends(current_user)):
    pm = PreferenceModel.from_dict(user.profile.preference_model)
    return {"events": pm.events, "top_features": pm.top_features(20)}


@router.delete("/preferences/learned")
def reset_learned(db=Depends(get_db), user=Depends(current_user)):
    user.profile.preference_model = {}
    db.commit()
    return {"ok": True}


class SaveBody(BaseModel):
    garment_ids: list[str] = Field(min_length=1, max_length=8)
    title: str = Field("", max_length=120)
    occasion: str | None = Field(None, max_length=40)
    constraints: dict = Field(default_factory=dict)
    score: dict = Field(default_factory=dict)
    explanation: list[str] = Field(default_factory=list)


def _brief_map(db, user) -> dict:
    return {g.id: garment_brief(g) for g in db.scalars(select(Garment).where(Garment.user_id == user.id)).all()}


@router.post("/outfits/saved")
def save_outfit(body: SaveBody, db=Depends(get_db), user=Depends(current_user)):
    garments = _owned_garments(db, user, body.garment_ids)
    o = SavedOutfit(
        user_id=user.id,
        title=body.title or " + ".join((g.name or g.subcategory or g.category) for g in garments)[:120],
        occasion=body.occasion,
        garment_ids=body.garment_ids,
        constraints=body.constraints,
        score=body.score,
        explanation=body.explanation[:8],
    )
    db.add(o)
    db.commit()
    _apply_feedback(db, user, garments, "save")
    return saved_out(o, _brief_map(db, user))


@router.get("/outfits/saved")
def list_saved(db=Depends(get_db), user=Depends(current_user)):
    brief = _brief_map(db, user)
    rows = db.scalars(select(SavedOutfit).where(SavedOutfit.user_id == user.id).order_by(SavedOutfit.created_at.desc())).all()
    return [saved_out(o, brief) for o in rows]


class SavedFeedback(BaseModel):
    event: Literal["like", "dislike"]


@router.post("/outfits/saved/{oid}/feedback")
def saved_feedback(oid: str, body: SavedFeedback, db=Depends(get_db), user=Depends(current_user)):
    o = db.get(SavedOutfit, oid)
    if o is None or o.user_id != user.id:
        raise HTTPException(404, "outfit not found")
    garments = [g for g in (db.get(Garment, i) for i in o.garment_ids) if g is not None and g.user_id == user.id]
    o.feedback = body.event
    db.commit()
    return _apply_feedback(db, user, garments, body.event) if garments else {"events": 0}


@router.delete("/outfits/saved/{oid}")
def delete_saved(oid: str, db=Depends(get_db), user=Depends(current_user)):
    o = db.get(SavedOutfit, oid)
    if o is None or o.user_id != user.id:
        raise HTTPException(404, "outfit not found")
    db.delete(o)
    db.commit()
    return {"deleted": oid}


@router.get("/lookbook")
def lookbook(db=Depends(get_db), user=Depends(current_user)):
    brief = _brief_map(db, user)
    saved = db.scalars(select(SavedOutfit).where(SavedOutfit.user_id == user.id).order_by(SavedOutfit.created_at.desc())).all()
    groups: dict[str, list] = {}
    for o in saved:
        if o.feedback == "dislike":
            continue
        groups.setdefault(o.occasion or "everyday", []).append(saved_out(o, brief))
    results = db.scalars(
        select(TryOnResultRow)
        .where(TryOnResultRow.user_id == user.id)
        .order_by(TryOnResultRow.saved.desc(), TryOnResultRow.created_at.desc())
    ).all()
    return {
        "boards": [{"occasion": k, "outfits": v} for k, v in groups.items()],
        "tryons": [tryon_out(r) for r in results if not r.is_mock][:60],
        "mock_tryons": sum(1 for r in results if r.is_mock),
    }
