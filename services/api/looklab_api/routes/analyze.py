"""Analyze an outfit photo. The photo is kept only until its job ends (see docs/PRIVACY.md)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func, select

from ..auth import current_user
from ..db import Job, new_id
from ..deps import get_ctx, get_db
from ..images import ImageValidationError, sanitize_image
from ..jobs import enqueue
from ..serializers import job_out

router = APIRouter(prefix="/api/analyze", tags=["analyze"])

_TRUE = ("1", "true", "yes", "on")
MAX_PENDING = 3  # queued + running analyses per user: each holds a photo on disk and a turn of the 8 GB GPU


@router.post("")
async def analyze_outfit(
    file: UploadFile = File(...),
    person_visible: str = Form("true"),  # str: FastAPI coerces a missing bool form field to False
    consent_confirmed: str = Form(""),
    db=Depends(get_db),
    ctx=Depends(get_ctx),
    user=Depends(current_user),
):
    person = person_visible.strip().lower() in _TRUE
    if person and consent_confirmed.strip().lower() not in _TRUE:
        raise HTTPException(400, "Confirm that this is you, or a person who has agreed to their photo being analyzed, before uploading.")
    pending = db.scalar(
        select(func.count()).select_from(Job).where(Job.user_id == user.id, Job.kind == "outfit_analysis", Job.status.in_(("queued", "running")))
    )
    if pending >= MAX_PENDING:
        raise HTTPException(429, f"{MAX_PENDING} analyses are already pending; wait for one to finish or delete it.")
    raw = await file.read()
    try:
        img = sanitize_image(raw, max_bytes=ctx.settings.max_upload_mb * 2**20, max_side=ctx.settings.max_image_side, min_side=128)
    except ImageValidationError as e:
        raise HTTPException(422, str(e)) from e
    key = f"users/{user.id}/analysis/{new_id()}.{img.ext}"
    ctx.storage.put(key, img.data, img.content_type)
    job = enqueue(db, user_id=user.id, kind="outfit_analysis", payload={"image_key": key, "person_visible": person}, provider="ollama_qwen3_vl")
    return job_out(job)


@router.delete("/{job_id}")
def delete_analysis(job_id: str, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    """Forget an analysis: cancels it if pending, removes the photo if still stored, deletes the result."""
    j = db.get(Job, job_id)
    if j is None or j.user_id != user.id or j.kind != "outfit_analysis":
        raise HTTPException(404, "analysis not found")
    if j.status == "running":
        raise HTTPException(409, "analysis is running; try again in a few seconds")
    if key := j.payload.get("image_key"):  # already swept by purge_expired
        ctx.storage.delete(key)
    if j.status == "queued":  # same transition as POST /jobs/{id}/cancel; the row stays as a cancelled marker
        j.status, j.error = "cancelled", "cancelled before start"
    else:
        db.delete(j)
    db.commit()
    return {"deleted": job_id}
