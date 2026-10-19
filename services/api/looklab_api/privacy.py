"""Deletion, retention and export."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .context import AppContext
from .db import (
    AuthSession,
    Collection,
    Garment,
    InferenceRun,
    Job,
    PersonPhoto,
    Profile,
    SavedOutfit,
    StylistSession,
    TryOnResultRow,
    User,
    utcnow,
)


def delete_tryon_result(db: Session, ctx: AppContext, row: TryOnResultRow) -> None:
    ctx.storage.delete(row.image_key)
    db.delete(row)


def delete_person_photo(db: Session, ctx: AppContext, photo: PersonPhoto, cascade: bool = True) -> int:
    """Delete a photo; generated images of that person go with it by default."""
    n = 0
    if cascade:
        for r in db.scalars(select(TryOnResultRow).where(TryOnResultRow.person_photo_id == photo.id)).all():
            delete_tryon_result(db, ctx, r)
            n += 1
    for j in db.scalars(select(Job).where(Job.user_id == photo.user_id, Job.status.in_(("queued", "running")))).all():
        if (j.payload or {}).get("person_photo_id") == photo.id:
            j.cancel_requested = True
            if j.status == "queued":
                j.status, j.error = "cancelled", "photo deleted"
    ctx.storage.delete(photo.storage_key)
    ctx.storage.delete(photo.thumb_key)
    db.delete(photo)
    return n


def delete_all_user_data(db: Session, ctx: AppContext, user: User, delete_account: bool = False) -> dict:
    uid = user.id
    counts = {}
    for model in (TryOnResultRow, PersonPhoto, Garment, SavedOutfit, Collection, StylistSession, Job, InferenceRun):
        counts[model.__tablename__] = db.execute(delete(model).where(model.user_id == uid)).rowcount
    ctx.storage.delete_prefix(f"users/{uid}/")
    prof = db.get(Profile, uid)
    if prof:
        prof.preferences, prof.privacy, prof.spending, prof.preference_model = {}, {}, {}, {}
        prof.wardrobe_version = (prof.wardrobe_version or 0) + 1
    if delete_account and not user.is_local:
        db.execute(delete(AuthSession).where(AuthSession.user_id == uid))
        db.delete(user)
    db.commit()
    ctx.rec_cache.pop(uid, None)
    return counts


def purge_expired(db: Session, ctx: AppContext) -> int:
    """Apply each user's retention settings to generated images and photos."""
    n = 0
    for prof in db.scalars(select(Profile)).all():
        privacy = prof.privacy or {}
        days = privacy.get("retention_days_generated", ctx.settings.default_generated_retention_days)
        if days:
            cutoff = utcnow() - timedelta(days=int(days))
            rows = db.scalars(
                select(TryOnResultRow).where(
                    TryOnResultRow.user_id == prof.user_id, TryOnResultRow.saved.is_(False), TryOnResultRow.created_at < cutoff
                )
            ).all()
            for r in rows:
                delete_tryon_result(db, ctx, r)
                n += 1
        pdays = privacy.get("retention_days_photos")
        if pdays:
            cutoff = utcnow() - timedelta(days=int(pdays))
            for p in db.scalars(select(PersonPhoto).where(PersonPhoto.user_id == prof.user_id, PersonPhoto.created_at < cutoff)).all():
                n += 1 + delete_person_photo(db, ctx, p)
    db.commit()
    return n


def export_user_data(db: Session, user: User) -> dict:
    def rows(model):
        out = []
        for r in db.scalars(select(model).where(model.user_id == user.id)).all():
            d = {c.name: getattr(r, c.name) for c in model.__table__.columns}
            out.append({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in d.items()})
        return out

    prof = user.profile
    return {
        "user": {"id": user.id, "email": user.email, "display_name": user.display_name},
        "profile": {
            "preferences": prof.preferences,
            "privacy": prof.privacy,
            "spending": prof.spending,
            "preference_model": prof.preference_model,
        },
        "garments": rows(Garment),
        "person_photos": rows(PersonPhoto),
        "saved_outfits": rows(SavedOutfit),
        "tryon_results": rows(TryOnResultRow),
        "inference_runs": rows(InferenceRun),
        "note": "Image files are not embedded; download them from the app before deleting.",
    }
