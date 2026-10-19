from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy import select

from looklab_ml.vision import check_person_photo

from ..auth import current_user
from ..db import PersonPhoto
from ..deps import get_ctx, get_db
from ..images import ImageValidationError, sanitize_image
from ..privacy import delete_person_photo
from ..serializers import photo_out

router = APIRouter(prefix="/api/photos", tags=["photos"])

PRIVATE_CACHE = {"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"}


def owned_photo(db, user, photo_id: str) -> PersonPhoto:
    p = db.get(PersonPhoto, photo_id)
    if p is None or p.user_id != user.id:
        raise HTTPException(404, "photo not found")
    return p


@router.post("")
async def upload_photo(
    file: UploadFile = File(...),
    consent_confirmed: str = Form(""),
    consent_subject: Literal["self", "consenting_person"] = Form("self"),
    label: str = Form(""),
    db=Depends(get_db),
    ctx=Depends(get_ctx),
    user=Depends(current_user),
):
    if consent_confirmed.strip().lower() not in ("1", "true", "yes", "on"):
        raise HTTPException(400, "Confirm that this is you, or a person who has agreed to their photo being used, before uploading.")
    raw = await file.read()
    try:
        img = sanitize_image(raw, max_bytes=ctx.settings.max_upload_mb * 2**20, max_side=ctx.settings.max_image_side, min_side=256)
    except ImageValidationError as e:
        raise HTTPException(422, str(e)) from e
    quality = check_person_photo(img.image).to_dict()
    photo = PersonPhoto(
        user_id=user.id,
        label=label[:80],
        storage_key="",
        thumb_key="",
        sha256=img.sha256,
        width=img.width,
        height=img.height,
        consent_subject=consent_subject,
        quality=quality,
    )
    db.add(photo)
    db.flush()
    photo.storage_key = f"users/{user.id}/photos/{photo.id}.{img.ext}"
    photo.thumb_key = f"users/{user.id}/photos/{photo.id}_thumb.webp"
    ctx.storage.put(photo.storage_key, img.data, img.content_type)
    ctx.storage.put(photo.thumb_key, img.thumb, "image/webp")
    db.commit()
    return photo_out(photo)


@router.get("")
def list_photos(db=Depends(get_db), user=Depends(current_user)):
    rows = db.scalars(select(PersonPhoto).where(PersonPhoto.user_id == user.id).order_by(PersonPhoto.created_at.desc())).all()
    return [photo_out(p) for p in rows]


@router.get("/{photo_id}/image")
def photo_image(
    photo_id: str, variant: Literal["full", "thumb"] = "full", db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)
):
    p = owned_photo(db, user, photo_id)
    key = p.thumb_key if variant == "thumb" else p.storage_key
    media = "image/webp" if key.endswith(".webp") else "image/png" if key.endswith(".png") else "image/jpeg"
    return Response(ctx.storage.get(key), media_type=media, headers=PRIVATE_CACHE)


@router.delete("/{photo_id}")
def delete_photo(photo_id: str, cascade: bool = True, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    p = owned_photo(db, user, photo_id)
    n = delete_person_photo(db, ctx, p, cascade=cascade)
    db.commit()
    return {"deleted": photo_id, "deleted_generated_images": n}
