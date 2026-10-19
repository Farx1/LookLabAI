from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from looklab_ml.recommendation import Constraints, parse_request
from looklab_ml.recommendation.preferences import PreferenceModel

from ..auth import current_user
from ..db import Garment, SavedOutfit, TryOnResultRow
from ..deps import get_ctx, get_db
from ..outfits import collage_bytes, garment_brief, outfit_payload, recommend
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


def _owned_garments(db, user, ids: list[str]) -> list[Garment]:
    out = []
    for i in ids:
        g = db.get(Garment, i)
        if g is None or g.user_id != user.id:
            raise HTTPException(404, f"garment {i} not found")
        out.append(g)
    return out


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
    prof.preference_model = pm.to_dict()
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
