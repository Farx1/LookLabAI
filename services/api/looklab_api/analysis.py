"""Garment analysis with content-hash caching (sha256 x analyzer)."""

from __future__ import annotations

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from looklab_ml.tryon import ProviderUnavailable
from looklab_ml.vision import Attr
from looklab_ml.vision.outfit import top_matches

from .context import AppContext
from .db import AnalysisCache, Garment, InferenceRun, Profile
from .serializers import garment_out
from .wardrobe import apply_analysis


def analyze_garment(db: Session, ctx: AppContext, g: Garment, analyzer_name: str, job_id: str | None = None) -> dict:
    cached = db.get(AnalysisCache, (g.sha256, analyzer_name))
    if cached:
        result, cache_hit = cached.result, True
    else:
        analyzer = ctx.registry.analyzers[analyzer_name]
        ok, why = analyzer.available()
        if not ok:
            raise ProviderUnavailable(why)
        with Image.open(ctx.storage.local_path(g.image_key)) as im:
            result = analyzer.analyze(im.convert("RGB"), hint_text=g.name).to_dict()
        db.merge(AnalysisCache(sha256=g.sha256, analyzer=analyzer_name, result=result))
        cache_hit = False
    attrs = {k: Attr(**v) for k, v in result["attributes"].items()}
    changed = apply_analysis(g, attrs, result.get("description"))
    g.analysis = {**(g.analysis or {}), analyzer_name: result}
    db.add(
        InferenceRun(
            user_id=g.user_id,
            job_id=job_id,
            task="garment_analysis",
            provider=analyzer_name,
            model=analyzer_name,
            device="cpu" if analyzer_name == "cv_basic" else "local",
            execution_time=0.0 if cache_hit else result.get("elapsed_seconds"),
            status="succeeded",
            cache_hit=cache_hit,
        )
    )
    if changed:
        prof = db.get(Profile, g.user_id)
        if prof is not None:
            prof.wardrobe_version = (prof.wardrobe_version or 0) + 1
    db.commit()
    return {"garment_id": g.id, "changed": changed, "cache_hit": cache_hit, "analyzer": analyzer_name}


def match_outfit(db: Session, user_id: str, pieces: list[dict], timings: dict) -> dict:
    """Top-3 wardrobe garments (owned or not) per detected piece."""
    garments = [garment_out(g) for g in db.scalars(select(Garment).where(Garment.user_id == user_id)).all()]
    return {
        "pieces": [{**p, "matches": [{"score": s, "garment": g} for g, s in top_matches(p, garments)]} for p in pieces],
        "timings": timings,
    }
