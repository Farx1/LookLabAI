from __future__ import annotations

from .db import Garment, Job, PersonPhoto, SavedOutfit, TryOnResultRow


def iso(dt):
    return dt.isoformat() if dt else None


def garment_out(g: Garment) -> dict:
    return {
        "id": g.id,
        "name": g.name,
        "category": g.category,
        "subcategory": g.subcategory,
        "colors": g.colors or [],
        "color_hex": g.color_hex or [],
        "pattern": g.pattern,
        "fit": g.fit,
        "style": g.style or [],
        "season": g.season or [],
        "occasion": g.occasion or [],
        "formality": g.formality,
        "details": g.details or [],
        "brand": g.brand,
        "size": g.size,
        "price": g.price,
        "currency": g.currency,
        "product_url": g.product_url,
        "is_owned": g.is_owned,
        "favorite": g.favorite,
        "garment_photo_type": g.garment_photo_type,
        "notes": g.notes,
        "description": g.description,
        "field_sources": g.field_sources or {},
        "analyzers_run": sorted((g.analysis or {}).keys()),
        "image_url": f"/api/garments/{g.id}/image",
        "thumb_url": f"/api/garments/{g.id}/image?variant=thumb",
        "created_at": iso(g.created_at),
        "updated_at": iso(g.updated_at),
    }


def photo_out(p: PersonPhoto) -> dict:
    return {
        "id": p.id,
        "label": p.label,
        "width": p.width,
        "height": p.height,
        "consent_subject": p.consent_subject,
        "consent_at": iso(p.consent_at),
        "quality": p.quality,
        "image_url": f"/api/photos/{p.id}/image",
        "thumb_url": f"/api/photos/{p.id}/image?variant=thumb",
        "created_at": iso(p.created_at),
    }


def job_out(j: Job) -> dict:
    return {
        "id": j.id,
        "kind": j.kind,
        "status": j.status,
        "provider": j.provider,
        "result": j.result or {},
        "error": j.error,
        "cancel_requested": j.cancel_requested,
        "estimated_cost_usd": (j.payload or {}).get("estimated_cost_usd", 0.0),
        "paid": bool((j.payload or {}).get("paid")),
        "cache_hit": bool((j.result or {}).get("cache_hit")),
        "created_at": iso(j.created_at),
        "started_at": iso(j.started_at),
        "finished_at": iso(j.finished_at),
    }


def tryon_out(r: TryOnResultRow) -> dict:
    return {
        "id": r.id,
        "kind": r.kind,
        "job_id": r.job_id,
        "person_photo_id": r.person_photo_id,
        "garment_ids": r.garment_ids,
        "provider": r.provider,
        "model": r.model,
        "fidelity": r.fidelity,
        "is_mock": r.is_mock,
        "saved": r.saved,
        "params": r.params,
        "image_url": f"/api/tryon/results/{r.id}/image",
        "created_at": iso(r.created_at),
    }


def saved_out(o: SavedOutfit, garments: dict) -> dict:
    return {
        "id": o.id,
        "title": o.title,
        "occasion": o.occasion,
        "garment_ids": o.garment_ids,
        "garments": [garments[i] for i in o.garment_ids if i in garments],
        "missing_garments": [i for i in o.garment_ids if i not in garments],
        "constraints": o.constraints,
        "score": o.score,
        "explanation": o.explanation,
        "feedback": o.feedback,
        "collage_url": "/api/outfits/collage?ids=" + ",".join(i for i in o.garment_ids if i in garments) if any(i in garments for i in o.garment_ids) else None,
        "created_at": iso(o.created_at),
    }
