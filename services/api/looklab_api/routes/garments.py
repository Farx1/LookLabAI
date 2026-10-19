from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field, HttpUrl, field_validator
from sqlalchemy import or_, select

from looklab_ml import taxonomy as tx
from looklab_ml.colors import PALETTE
from looklab_ml.tryon import ProviderUnavailable

from ..analysis import analyze_garment
from ..auth import current_user
from ..db import Collection, Garment, Profile
from ..deps import get_ctx, get_db
from ..images import ImageValidationError, sanitize_image
from ..importer import ImportErrorUser, fetch_product
from ..jobs import enqueue
from ..serializers import garment_out, job_out
from ..wardrobe import bump_wardrobe_version, mark_user_fields
from .photos import PRIVATE_CACHE

router = APIRouter(prefix="/api", tags=["wardrobe"])

MAX_BATCH = 40


def owned_garment(db, user, gid: str) -> Garment:
    g = db.get(Garment, gid)
    if g is None or g.user_id != user.id:
        raise HTTPException(404, "garment not found")
    return g


def _name_from_filename(fn: str | None) -> str | None:
    if not fn:
        return None
    stem = Path(fn).stem
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    if re.fullmatch(r"(img|dsc|pxl|photo|image)?\s*\d+.*", stem, re.I) or len(stem) < 3:
        return None
    return stem[:120]


def _store_garment(db, ctx, user, raw: bytes, name: str | None, **fields) -> Garment:
    img = sanitize_image(
        raw, max_bytes=ctx.settings.max_upload_mb * 2**20, max_side=ctx.settings.max_image_side, min_side=64, keep_alpha=True
    )
    g = Garment(user_id=user.id, image_key="", thumb_key="", sha256=img.sha256, name=name, field_sources={}, analysis={}, **fields)
    db.add(g)
    db.flush()
    g.image_key = f"users/{user.id}/garments/{g.id}.{img.ext}"
    g.thumb_key = f"users/{user.id}/garments/{g.id}_thumb.webp"
    ctx.storage.put(g.image_key, img.data, img.content_type)
    ctx.storage.put(g.thumb_key, img.thumb, "image/webp")
    return g


def _post_upload_analysis(db, ctx, user, g: Garment) -> dict | None:
    """Deterministic CV runs inline (milliseconds). The VLM is queued only if
    configured to auto-run and actually available."""
    analyze_garment(db, ctx, g, "cv_basic")
    ia = ctx.policy.image_analysis
    if ia.auto_run_model:
        name = ctx.registry.model_analyzer()
        if name is not None:
            return job_out(
                enqueue(db, user_id=user.id, kind="garment_analysis", payload={"garment_id": g.id, "analyzer": name}, provider=name)
            )
    return None


@router.post("/garments")
async def upload_garments(
    files: list[UploadFile] = File(...),
    is_owned: str = Form("true"),  # str: FastAPI coerces a *missing* bool form field to False
    garment_photo_type: Literal["flat-lay", "model"] = Form("flat-lay"),
    db=Depends(get_db),
    ctx=Depends(get_ctx),
    user=Depends(current_user),
):
    if len(files) > MAX_BATCH:
        raise HTTPException(413, f"at most {MAX_BATCH} files per batch")
    owned = is_owned.strip().lower() in ("1", "true", "yes", "on")
    created, errors, jobs = [], [], []
    for f in files:
        raw = await f.read()
        try:
            g = _store_garment(db, ctx, user, raw, _name_from_filename(f.filename), is_owned=owned, garment_photo_type=garment_photo_type)
            db.commit()
        except ImageValidationError as e:
            db.rollback()
            errors.append({"filename": f.filename, "error": str(e)})
            continue
        j = _post_upload_analysis(db, ctx, user, g)
        if j:
            jobs.append(j)
        created.append(garment_out(g))
    bump_wardrobe_version(db.get(Profile, user.id))
    db.commit()
    if not created and errors:
        raise HTTPException(422, {"errors": errors})
    return {"created": created, "errors": errors, "analysis_jobs": jobs}


class ImportUrlBody(BaseModel):
    url: HttpUrl
    is_owned: bool = False


@router.post("/garments/import-url")
def import_url(body: ImportUrlBody, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    try:
        info, img_bytes = fetch_product(str(body.url), transport=getattr(ctx, "import_transport", None))
    except ImportErrorUser as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001 - network errors are user-facing
        raise HTTPException(502, f"could not fetch that page: {type(e).__name__}") from e
    try:
        g = _store_garment(
            db, ctx, user, img_bytes, info.title,
            is_owned=body.is_owned, product_url=info.url, price=info.price,
            currency=(info.currency or "EUR"), brand=info.brand, garment_photo_type="flat-lay",
        )  # fmt: skip
    except ImageValidationError as e:
        raise HTTPException(422, f"product image rejected: {e}") from e
    srcs = {}
    for f in ("price", "brand"):
        if getattr(info, f) is not None:
            srcs[f] = {"source": "text", "confidence": 0.9}
    g.field_sources = srcs
    db.commit()
    _post_upload_analysis(db, ctx, user, g)
    bump_wardrobe_version(db.get(Profile, user.id))
    db.commit()
    return garment_out(g)


@router.get("/garments")
def list_garments(
    q: str | None = None,
    category: str | None = None,
    color: str | None = None,
    style: str | None = None,
    season: str | None = None,
    owned: bool | None = None,
    favorite: bool | None = None,
    collection_id: str | None = None,
    db=Depends(get_db),
    user=Depends(current_user),
):
    stmt = select(Garment).where(Garment.user_id == user.id)
    if category:
        stmt = stmt.where(Garment.category == category)
    if owned is not None:
        stmt = stmt.where(Garment.is_owned.is_(owned))
    if favorite is not None:
        stmt = stmt.where(Garment.favorite.is_(favorite))
    if q:
        like = f"%{q.lower()}%"
        stmt = stmt.where(
            or_(Garment.name.ilike(like), Garment.subcategory.ilike(like), Garment.brand.ilike(like), Garment.description.ilike(like))
        )
    rows = db.scalars(stmt.order_by(Garment.created_at.desc())).all()
    # JSON-array filters in Python keep the queries portable across SQLite/Postgres.
    if color:
        rows = [g for g in rows if color in (g.colors or [])]
    if style:
        rows = [g for g in rows if style in (g.style or [])]
    if season:
        rows = [g for g in rows if not g.season or season in g.season]
    if collection_id:
        col = db.get(Collection, collection_id)
        if col is None or col.user_id != user.id:
            raise HTTPException(404, "collection not found")
        ids = set(col.garment_ids)
        rows = [g for g in rows if g.id in ids]
    return [garment_out(g) for g in rows]


@router.get("/garments/vocabulary")
def vocabulary():
    return {
        "categories": tx.CATEGORIES,
        "fits": tx.FITS,
        "patterns": tx.PATTERNS,
        "styles": tx.STYLES,
        "seasons": tx.SEASONS,
        "occasions": tx.OCCASIONS,
        "colors": sorted(PALETTE),
        "color_hex": {k: "#{:02x}{:02x}{:02x}".format(*v) for k, v in PALETTE.items()},
    }


@router.get("/garments/{gid}")
def get_garment(gid: str, db=Depends(get_db), user=Depends(current_user)):
    g = owned_garment(db, user, gid)
    return {**garment_out(g), "analysis": g.analysis}


class GarmentPatch(BaseModel):
    name: str | None = Field(None, max_length=120)
    category: str | None = None
    subcategory: str | None = Field(None, max_length=60)
    colors: list[str] | None = None
    pattern: str | None = None
    fit: str | None = None
    style: list[str] | None = None
    season: list[str] | None = None
    occasion: list[str] | None = None
    formality: int | None = Field(None, ge=1, le=5)
    details: list[str] | None = None
    brand: str | None = Field(None, max_length=80)
    size: str | None = Field(None, max_length=32)
    price: float | None = Field(None, ge=0, le=100000)
    currency: str | None = Field(None, pattern=r"^[A-Z]{3}$")
    product_url: HttpUrl | None = None
    is_owned: bool | None = None
    favorite: bool | None = None
    garment_photo_type: Literal["flat-lay", "model"] | None = None
    notes: str | None = Field(None, max_length=2000)

    @field_validator("category")
    @classmethod
    def _cat(cls, v):
        if v is not None and v not in (*tx.CATEGORIES, tx.UNKNOWN):
            raise ValueError(f"category must be one of {tx.CATEGORIES}")
        return v

    @field_validator("pattern")
    @classmethod
    def _pat(cls, v):
        if v is not None and v not in (*tx.PATTERNS, tx.UNKNOWN):
            raise ValueError(f"pattern must be one of {tx.PATTERNS}")
        return v

    @field_validator("fit")
    @classmethod
    def _fit(cls, v):
        if v is not None and v not in (*tx.FITS, tx.UNKNOWN):
            raise ValueError(f"fit must be one of {tx.FITS}")
        return v

    @field_validator("colors")
    @classmethod
    def _colors(cls, v):
        if v is not None and any(c not in PALETTE for c in v):
            raise ValueError("unknown color name; see /api/garments/vocabulary")
        return v[:3] if v else v

    @field_validator("style")
    @classmethod
    def _style(cls, v):
        if v is not None and any(s not in tx.STYLES for s in v):
            raise ValueError(f"style must be from {tx.STYLES}")
        return v

    @field_validator("season")
    @classmethod
    def _season(cls, v):
        if v is not None and any(s not in tx.SEASONS for s in v):
            raise ValueError(f"season must be from {tx.SEASONS}")
        return v

    @field_validator("occasion")
    @classmethod
    def _occ(cls, v):
        if v is not None and any(s not in tx.OCCASIONS for s in v):
            raise ValueError(f"occasion must be from {tx.OCCASIONS}")
        return v


METADATA_FIELDS = {
    "category",
    "subcategory",
    "colors",
    "pattern",
    "fit",
    "style",
    "season",
    "occasion",
    "formality",
    "details",
    "price",
    "brand",
}


@router.patch("/garments/{gid}")
def patch_garment(gid: str, body: GarmentPatch, db=Depends(get_db), user=Depends(current_user)):
    g = owned_garment(db, user, gid)
    data = body.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(g, k, str(v) if k == "product_url" and v is not None else v)
    if "colors" in data:
        g.color_hex = []
    mark_user_fields(g, [k for k in data if k in METADATA_FIELDS])
    bump_wardrobe_version(db.get(Profile, user.id))
    db.commit()
    return garment_out(g)


@router.delete("/garments/{gid}")
def delete_garment(gid: str, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    g = owned_garment(db, user, gid)
    ctx.storage.delete(g.image_key)
    ctx.storage.delete(g.thumb_key)
    for col in db.scalars(select(Collection).where(Collection.user_id == user.id)).all():
        if gid in col.garment_ids:
            col.garment_ids = [x for x in col.garment_ids if x != gid]
    db.delete(g)
    bump_wardrobe_version(db.get(Profile, user.id))
    db.commit()
    return {"deleted": gid}


@router.get("/garments/{gid}/image")
def garment_image(
    gid: str, variant: Literal["full", "thumb"] = "full", db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)
):
    g = owned_garment(db, user, gid)
    key = g.thumb_key if variant == "thumb" else g.image_key
    media = "image/webp" if key.endswith(".webp") else "image/png" if key.endswith(".png") else "image/jpeg"
    return Response(ctx.storage.get(key), media_type=media, headers=PRIVATE_CACHE)


class AnalyzeBody(BaseModel):
    analyzer: Literal["auto", "cv_basic", "local_qwen3_vl", "local_florence2"] = "auto"


@router.post("/garments/{gid}/analyze")
def analyze(gid: str, body: AnalyzeBody, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    g = owned_garment(db, user, gid)
    name = body.analyzer
    if name == "auto":
        name = ctx.registry.model_analyzer()
        if name is None:
            raise HTTPException(
                409,
                "No local vision-language model is available. Start Ollama with qwen3-vl:2b or install the "
                "vision-models extra, or edit the details manually.",
            )
    if name == "cv_basic":
        try:
            return {"result": analyze_garment(db, ctx, g, "cv_basic"), "garment": garment_out(g)}
        except ProviderUnavailable as e:
            raise HTTPException(409, str(e)) from e
    ok, why = ctx.registry.analyzers[name].available()
    if not ok:
        raise HTTPException(409, f"{name} unavailable: {why}")
    job = enqueue(db, user_id=user.id, kind="garment_analysis", payload={"garment_id": g.id, "analyzer": name}, provider=name)
    return {"job": job_out(job)}


# ------------------------------------------------------------------ collections
class CollectionBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    garment_ids: list[str] = Field(default_factory=list)


@router.get("/collections")
def list_collections(db=Depends(get_db), user=Depends(current_user)):
    rows = db.scalars(select(Collection).where(Collection.user_id == user.id).order_by(Collection.created_at)).all()
    return [{"id": c.id, "name": c.name, "garment_ids": c.garment_ids} for c in rows]


def _validate_ids(db, user, ids: list[str]) -> list[str]:
    for i in ids:
        owned_garment(db, user, i)
    return list(dict.fromkeys(ids))


@router.post("/collections")
def create_collection(body: CollectionBody, db=Depends(get_db), user=Depends(current_user)):
    c = Collection(user_id=user.id, name=body.name, garment_ids=_validate_ids(db, user, body.garment_ids))
    db.add(c)
    db.commit()
    return {"id": c.id, "name": c.name, "garment_ids": c.garment_ids}


@router.put("/collections/{cid}")
def update_collection(cid: str, body: CollectionBody, db=Depends(get_db), user=Depends(current_user)):
    c = db.get(Collection, cid)
    if c is None or c.user_id != user.id:
        raise HTTPException(404, "collection not found")
    c.name, c.garment_ids = body.name, _validate_ids(db, user, body.garment_ids)
    db.commit()
    return {"id": c.id, "name": c.name, "garment_ids": c.garment_ids}


@router.delete("/collections/{cid}")
def delete_collection(cid: str, db=Depends(get_db), user=Depends(current_user)):
    c = db.get(Collection, cid)
    if c is None or c.user_id != user.id:
        raise HTTPException(404, "collection not found")
    db.delete(c)
    db.commit()
    return {"deleted": cid}
