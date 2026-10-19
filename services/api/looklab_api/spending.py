"""Hard spending controls for anything that can cost money.

A paid request runs only if *all* of these hold:
1. the provider is enabled in ai.yaml;
2. the user approved this specific request (``approve_paid=true``) and the
   estimate is within the ``max_cost_usd`` they accepted;
3. month-to-date spend + in-flight reservations + estimate stays within
   min(global cap in ai.yaml, the user's own monthly limit);
4. if images leave the machine, the user enabled external processing.
Paid jobs are never retried automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import InferenceRun, Job, User
from .providers import ProviderError


def month_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def month_spend(db: Session, user_id: str) -> float:
    q = select(func.coalesce(func.sum(func.coalesce(InferenceRun.actual_cost_usd, InferenceRun.estimated_cost_usd)), 0.0)).where(
        InferenceRun.user_id == user_id, InferenceRun.paid.is_(True), InferenceRun.created_at >= month_start()
    )
    return float(db.scalar(q) or 0.0)


def reserved_spend(db: Session, user_id: str) -> float:
    jobs = db.scalars(select(Job).where(Job.user_id == user_id, Job.status.in_(("queued", "running")))).all()
    return float(sum((j.payload or {}).get("estimated_cost_usd", 0.0) for j in jobs if (j.payload or {}).get("paid")))


def effective_limit(global_limit: float, user: User) -> float:
    user_limit = float((user.profile.spending or {}).get("monthly_limit_usd", 0.0))
    return min(global_limit, user_limit)


def authorize(
    db: Session,
    user: User,
    *,
    provider_name: str,
    paid: bool,
    sends_images_externally: bool,
    estimate_usd: float,
    approved: bool,
    max_cost_usd: float | None,
    global_limit: float,
) -> dict:
    privacy = user.profile.privacy or {}
    if sends_images_externally and not privacy.get("allow_external_processing", False):
        raise ProviderError(
            403,
            "external_processing_not_allowed",
            f"{provider_name} sends your photos to another machine. Enable external processing in Settings → Privacy first.",
        )
    if not paid:
        return {"paid": False, "estimated_cost_usd": 0.0}
    if not approved:
        raise ProviderError(
            402,
            "approval_required",
            f"{provider_name} costs about ${estimate_usd:.3f} per request. Approve this request explicitly to continue.",
            {"estimated_cost_usd": estimate_usd},
        )
    if max_cost_usd is None or estimate_usd > max_cost_usd + 1e-9:
        raise ProviderError(
            402,
            "estimate_exceeds_approval",
            f"Estimated ${estimate_usd:.3f} exceeds the approved maximum (${max_cost_usd or 0:.3f}).",
            {"estimated_cost_usd": estimate_usd},
        )
    limit = effective_limit(global_limit, user)
    spent = month_spend(db, user.id)
    reserved = reserved_spend(db, user.id)
    if spent + reserved + estimate_usd > limit + 1e-9:
        raise ProviderError(
            402,
            "monthly_limit_reached",
            f"This would exceed your monthly paid-AI limit (${limit:.2f}; spent ${spent:.2f}, in progress ${reserved:.2f}).",
            {"limit_usd": limit, "spent_usd": spent, "reserved_usd": reserved, "estimated_cost_usd": estimate_usd},
        )
    return {"paid": True, "estimated_cost_usd": estimate_usd}
