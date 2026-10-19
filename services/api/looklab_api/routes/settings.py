from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import case, func, select

from looklab_ml import taxonomy as tx
from looklab_ml.colors import PALETTE

from ..auth import current_user
from ..db import InferenceRun, utcnow
from ..deps import get_ctx, get_db
from ..privacy import delete_all_user_data, export_user_data
from ..serializers import iso
from ..spending import effective_limit, month_spend

router = APIRouter(prefix="/api", tags=["settings"])


class Preferences(BaseModel):
    liked_colors: list[str] = Field(default_factory=list)
    disliked_colors: list[str] = Field(default_factory=list)
    liked_styles: list[str] = Field(default_factory=list)
    disliked_styles: list[str] = Field(default_factory=list)
    fit_preference: str | None = None
    sizes: dict[str, str] = Field(default_factory=dict)
    default_budget: float | None = Field(None, ge=0)
    currency: str = Field("EUR", pattern=r"^[A-Z]{3}$")

    @field_validator("liked_colors", "disliked_colors")
    @classmethod
    def _c(cls, v):
        return [c for c in v if c in PALETTE]

    @field_validator("liked_styles", "disliked_styles")
    @classmethod
    def _s(cls, v):
        return [s for s in v if s in tx.STYLES]

    @field_validator("fit_preference")
    @classmethod
    def _f(cls, v):
        return v if v in tx.FITS else None

    @field_validator("sizes")
    @classmethod
    def _sizes(cls, v):
        return {k[:20]: str(val)[:20] for k, val in list(v.items())[:12]}


class Privacy(BaseModel):
    allow_external_processing: bool = False
    retention_days_generated: int | None = Field(None, ge=1, le=3650)
    retention_days_photos: int | None = Field(None, ge=1, le=3650)


class Spending(BaseModel):
    monthly_limit_usd: float = Field(0.0, ge=0, le=1000)


class SettingsBody(BaseModel):
    preferences: Preferences | None = None
    privacy: Privacy | None = None
    spending: Spending | None = None


def _settings_out(ctx, user):
    prof = user.profile
    return {
        "preferences": Preferences(**(prof.preferences or {})).model_dump(),
        "privacy": Privacy(**(prof.privacy or {})).model_dump(),
        "spending": {
            **Spending(**(prof.spending or {})).model_dump(),
            "global_limit_usd": ctx.policy.spending.monthly_limit_usd,
            "effective_limit_usd": effective_limit(ctx.policy.spending.monthly_limit_usd, user),
        },
    }


@router.get("/settings")
def get_settings(ctx=Depends(get_ctx), user=Depends(current_user)):
    return _settings_out(ctx, user)


@router.put("/settings")
def put_settings(body: SettingsBody, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    prof = user.profile
    if body.preferences is not None:
        prof.preferences = body.preferences.model_dump()
        prof.wardrobe_version = (prof.wardrobe_version or 0) + 1  # invalidates cached recommendations
    if body.privacy is not None:
        prof.privacy = body.privacy.model_dump()
    if body.spending is not None:
        prof.spending = body.spending.model_dump()
    db.commit()
    return _settings_out(ctx, user)


@router.get("/costs")
def costs(days: int = 30, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    since = utcnow() - timedelta(days=max(1, min(days, 365)))
    base = select(InferenceRun).where(InferenceRun.user_id == user.id, InferenceRun.created_at >= since)
    rows = db.scalars(base.order_by(InferenceRun.created_at.desc()).limit(200)).all()
    agg = db.execute(
        select(
            InferenceRun.task,
            InferenceRun.provider,
            func.count(),
            func.sum(case((InferenceRun.cache_hit.is_(True), 1), else_=0)),
            func.avg(InferenceRun.execution_time),
            func.max(InferenceRun.gpu_memory_peak_mb),
            func.sum(InferenceRun.estimated_cost_usd),
        )
        .where(InferenceRun.user_id == user.id, InferenceRun.created_at >= since)
        .group_by(InferenceRun.task, InferenceRun.provider)
    ).all()
    return {
        "month_spend_usd": round(month_spend(db, user.id), 4),
        "effective_limit_usd": effective_limit(ctx.policy.spending.monthly_limit_usd, user),
        "by_provider": [
            {
                "task": t,
                "provider": p,
                "runs": n,
                "cache_hits": int(ch or 0),
                "avg_execution_seconds": round(avg, 3) if avg is not None else None,
                "max_gpu_memory_mb": mx,
                "estimated_cost_usd": round(cost or 0, 4),
            }
            for t, p, n, ch, avg, mx, cost in agg
        ],
        "recent": [
            {
                "id": r.id,
                "task": r.task,
                "provider": r.provider,
                "model": r.model,
                "device": r.device,
                "execution_time": r.execution_time,
                "model_load_seconds": r.model_load_seconds,
                "gpu_memory_peak_mb": r.gpu_memory_peak_mb,
                "estimated_cost_usd": r.estimated_cost_usd,
                "actual_cost_usd": r.actual_cost_usd,
                "paid": r.paid,
                "cache_hit": r.cache_hit,
                "status": r.status,
                "created_at": iso(r.created_at),
            }
            for r in rows
        ],
    }


@router.get("/me/export")
def export(db=Depends(get_db), user=Depends(current_user)):
    return export_user_data(db, user)


@router.delete("/me/data")
def delete_everything(delete_account: bool = False, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    return {"deleted": delete_all_user_data(db, ctx, user, delete_account)}
