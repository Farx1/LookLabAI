from __future__ import annotations

import hashlib
import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from looklab_ml import taxonomy as tx
from looklab_ml.tryon import TryOnRequest

from ..auth import current_user
from ..db import Garment, InferenceRun, Job, PersonPhoto, TryOnResultRow
from ..deps import get_ctx, get_db
from ..jobs import enqueue
from ..privacy import delete_tryon_result
from ..providers import ProviderError
from ..serializers import job_out, tryon_out
from ..spending import authorize, effective_limit, month_spend
from .photos import PRIVATE_CACHE

router = APIRouter(prefix="/api", tags=["tryon"])


class TryOnParams(BaseModel):
    num_timesteps: int = Field(30, ge=10, le=50)
    guidance_scale: float = Field(1.5, ge=1.0, le=3.0)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    num_samples: int = Field(1, ge=1, le=4)


class TryOnBody(BaseModel):
    person_photo_id: str
    garment_id: str
    provider: str = "auto"
    category: Literal["tops", "bottoms", "one-pieces"] | None = None
    garment_photo_type: Literal["flat-lay", "model"] | None = None
    params: TryOnParams = TryOnParams()
    approve_paid: bool = False
    max_cost_usd: float | None = Field(None, ge=0)
    force_new: bool = False  # skip the result cache (e.g. "regenerate" with same seed is pointless; change seed instead)


def provider_error(e: ProviderError) -> JSONResponse:
    return JSONResponse(status_code=e.status, content={"detail": {"code": e.code, "message": e.message, **e.extra}})


def _prepare(db, ctx, user, body: TryOnBody):
    photo = db.get(PersonPhoto, body.person_photo_id)
    garment = db.get(Garment, body.garment_id)
    if photo is None or photo.user_id != user.id:
        raise HTTPException(404, "photo not found")
    if garment is None or garment.user_id != user.id:
        raise HTTPException(404, "garment not found")
    if photo.quality and not photo.quality.get("ok", True):
        raise HTTPException(422, {"code": "photo_unsuitable", "issues": photo.quality.get("issues", [])})
    category = body.category or tx.TRYON_CATEGORY.get(garment.category)
    if category is None:
        raise HTTPException(
            422,
            {
                "code": "category_unsupported",
                "message": f"Try-on supports tops, bottoms and one-pieces; this garment is '{garment.category}'. Set its category first.",
            },
        )
    provider = ctx.registry.resolve_tryon(body.provider)
    photo_type = body.garment_photo_type or garment.garment_photo_type or "flat-lay"
    if provider.info.name == "local_fashn":
        try:
            ctx.registry.fashn_runtime.validate(photo_type)
        except ValueError as e:
            raise HTTPException(422, {"code": "parser_required", "message": str(e)}) from e
    req = TryOnRequest(
        person_image_path="", garment_image_path="", category=category, garment_photo_type=photo_type, **body.params.model_dump()
    )
    estimate = provider.estimate_cost_usd(req)
    return photo, garment, provider, req, estimate


@router.get("/tryon/providers")
def providers(ctx=Depends(get_ctx), db=Depends(get_db), user=Depends(current_user)):
    return {
        "providers": ctx.registry.tryon_status(),
        "spending": {
            "month_spend_usd": round(month_spend(db, user.id), 4),
            "effective_limit_usd": effective_limit(ctx.policy.spending.monthly_limit_usd, user),
            "global_limit_usd": ctx.policy.spending.monthly_limit_usd,
        },
        "allow_external_processing": bool((user.profile.privacy or {}).get("allow_external_processing")),
    }


@router.post("/tryon/estimate")
def estimate(body: TryOnBody, ctx=Depends(get_ctx), db=Depends(get_db), user=Depends(current_user)):
    try:
        _, _, provider, req, est = _prepare(db, ctx, user, body)
    except ProviderError as e:
        return provider_error(e)
    return {
        "provider": provider.info.name,
        "model": provider.info.model,
        "paid": provider.info.paid,
        "fidelity": provider.info.fidelity,
        "sends_images_externally": provider.info.sends_images_externally,
        "estimated_cost_usd": est,
        "category": req.category,
        "garment_photo_type": req.garment_photo_type,
    }


@router.post("/tryon/jobs")
def create_job(body: TryOnBody, ctx=Depends(get_ctx), db=Depends(get_db), user=Depends(current_user)):
    try:
        photo, garment, provider, req, est = _prepare(db, ctx, user, body)
        params = req.params()
        dedup = hashlib.sha256(
            json.dumps([provider.info.name, provider.info.model, photo.sha256, garment.sha256, params], sort_keys=True).encode()
        ).hexdigest()
        if not body.force_new:
            # identical inputs + parameters: reuse the stored result at no cost
            cached = db.scalar(select(TryOnResultRow).where(TryOnResultRow.user_id == user.id, TryOnResultRow.dedup_key == dedup).limit(1))
            if cached is not None:
                job = Job(
                    user_id=user.id, kind="tryon", status="succeeded", provider=provider.info.name, dedup_key=dedup,
                    payload={"person_photo_id": photo.id, "garment_id": garment.id, "provider": provider.info.name, "params": params},
                    result={"result_id": cached.id, "cache_hit": True, "provider": cached.provider, "model": cached.model,
                            "is_mock": cached.is_mock, "estimated_cost_usd": 0.0},
                )  # fmt: skip
                db.add(job)
                db.add(
                    InferenceRun(
                        user_id=user.id,
                        task="tryon",
                        provider=provider.info.name,
                        model=provider.info.model,
                        status="succeeded",
                        cache_hit=True,
                    )
                )
                db.commit()
                return job_out(job)
            inflight = db.scalar(
                select(Job).where(Job.user_id == user.id, Job.dedup_key == dedup, Job.status.in_(("queued", "running"))).limit(1)
            )
            if inflight is not None:
                return job_out(inflight)
        auth = authorize(
            db,
            user,
            provider_name=provider.info.name,
            paid=provider.info.paid,
            sends_images_externally=provider.info.sends_images_externally,
            estimate_usd=est,
            approved=body.approve_paid,
            max_cost_usd=body.max_cost_usd,
            global_limit=ctx.policy.spending.monthly_limit_usd,
        )
    except ProviderError as e:
        return provider_error(e)
    job = enqueue(
        db,
        user_id=user.id,
        kind="tryon",
        provider=provider.info.name,
        dedup_key=dedup,
        payload={
            "person_photo_id": photo.id,
            "garment_id": garment.id,
            "provider": provider.info.name,
            "params": params,
            "dedup_key": dedup,
            **auth,
        },
    )
    return job_out(job)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, db=Depends(get_db), user=Depends(current_user)):
    j = db.get(Job, job_id)
    if j is None or j.user_id != user.id:
        raise HTTPException(404, "job not found")
    return job_out(j)


@router.get("/jobs")
def list_jobs(kind: str | None = None, db=Depends(get_db), user=Depends(current_user)):
    stmt = select(Job).where(Job.user_id == user.id)
    if kind:
        stmt = stmt.where(Job.kind == kind)
    return [job_out(j) for j in db.scalars(stmt.order_by(Job.created_at.desc()).limit(50)).all()]


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, db=Depends(get_db), user=Depends(current_user)):
    j = db.get(Job, job_id)
    if j is None or j.user_id != user.id:
        raise HTTPException(404, "job not found")
    if j.status == "queued":
        j.status, j.error = "cancelled", "cancelled before start"
    elif j.status == "running":
        j.cancel_requested = True
    db.commit()
    return job_out(j)


@router.get("/tryon/results")
def list_results(person_photo_id: str | None = None, db=Depends(get_db), user=Depends(current_user)):
    stmt = select(TryOnResultRow).where(TryOnResultRow.user_id == user.id)
    if person_photo_id:
        stmt = stmt.where(TryOnResultRow.person_photo_id == person_photo_id)
    return [tryon_out(r) for r in db.scalars(stmt.order_by(TryOnResultRow.created_at.desc())).all()]


def _owned_result(db, user, rid) -> TryOnResultRow:
    r = db.get(TryOnResultRow, rid)
    if r is None or r.user_id != user.id:
        raise HTTPException(404, "result not found")
    return r


@router.get("/tryon/results/{rid}/image")
def result_image(rid: str, download: bool = False, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    r = _owned_result(db, user, rid)
    headers = dict(PRIVATE_CACHE)
    if download:
        headers["Content-Disposition"] = f'attachment; filename="looklab-{rid[:8]}.png"'
    return Response(ctx.storage.get(r.image_key), media_type="image/png", headers=headers)


class ResultPatch(BaseModel):
    saved: bool


@router.patch("/tryon/results/{rid}")
def patch_result(rid: str, body: ResultPatch, db=Depends(get_db), user=Depends(current_user)):
    r = _owned_result(db, user, rid)
    r.saved = body.saved
    db.commit()
    return tryon_out(r)


@router.delete("/tryon/results/{rid}")
def delete_result(rid: str, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    r = _owned_result(db, user, rid)
    delete_tryon_result(db, ctx, r)
    db.commit()
    return {"deleted": rid}
