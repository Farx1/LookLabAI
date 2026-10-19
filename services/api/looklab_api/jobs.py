"""Durable job queue on the main database (SQLite or PostgreSQL).

Two lanes so a long GPU job never blocks cheap CPU analysis:
* ``cpu``: garment_analysis
* ``gpu``: tryon, creative
Jobs survive restarts. On startup, interrupted free jobs are re-queued;
interrupted paid jobs are failed, never retried automatically.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
import traceback
from datetime import timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from looklab_ml.tryon import CancelledError, ProviderUnavailable, TryOnRequest

from .context import AppContext
from .db import Garment, InferenceRun, Job, PersonPhoto, TryOnResultRow, utcnow

log = logging.getLogger(__name__)

LANES = {"cpu": ("garment_analysis",), "gpu": ("tryon", "creative")}


def enqueue(db: Session, *, user_id: str, kind: str, payload: dict, provider: str | None = None, dedup_key: str | None = None) -> Job:
    job = Job(user_id=user_id, kind=kind, payload=payload, provider=provider, dedup_key=dedup_key, status="queued")
    db.add(job)
    db.commit()
    return job


def recover_interrupted(db: Session) -> int:
    n = 0
    for job in db.scalars(select(Job).where(Job.status == "running")).all():
        if (job.payload or {}).get("paid"):
            job.status, job.error, job.finished_at = "failed", "interrupted by restart; paid jobs are not retried automatically", utcnow()
        else:
            job.status = "queued"
        n += 1
    db.commit()
    return n


def claim(db: Session, kinds: tuple[str, ...]) -> Job | None:
    for _ in range(3):
        job = db.scalar(select(Job).where(Job.status == "queued", Job.kind.in_(kinds)).order_by(Job.created_at).limit(1))
        if job is None:
            return None
        res = db.execute(
            update(Job)
            .where(Job.id == job.id, Job.status == "queued")
            .values(status="running", started_at=utcnow(), attempts=Job.attempts + 1)
        )
        db.commit()
        if res.rowcount == 1:
            db.refresh(job)
            return job
    return None


class JobRunner:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_lane(self, lane: str) -> None:
        kinds = LANES[lane]
        last_housekeeping = 0.0
        while not self._stop.is_set():
            ran = await self.run_once(kinds)
            if lane == "gpu" and time.time() - last_housekeeping > 60:
                last_housekeeping = time.time()
                self.ctx.registry.fashn_runtime.maybe_unload_idle()
                try:
                    from .privacy import purge_expired

                    with self.ctx.sessionmaker() as db:
                        purge_expired(db, self.ctx)
                except Exception:  # noqa: BLE001
                    log.exception("retention purge failed")
            if not ran:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.ctx.settings.worker_poll_seconds)
                except asyncio.TimeoutError:
                    pass

    async def run_once(self, kinds: tuple[str, ...] = LANES["cpu"] + LANES["gpu"]) -> bool:
        with self.ctx.sessionmaker() as db:
            job = claim(db, kinds)
            if job is None:
                return False
            job_id = job.id
        await self.execute(job_id)
        return True

    async def drain(self, max_jobs: int = 100) -> int:
        """Process queued jobs inline (tests, CLI)."""
        n = 0
        while n < max_jobs and await self.run_once():
            n += 1
        return n

    async def execute(self, job_id: str) -> None:
        handler = {"garment_analysis": self._garment_analysis, "tryon": self._tryon, "creative": self._creative}
        with self.ctx.sessionmaker() as db:
            job = db.get(Job, job_id)
            kind = job.kind
        try:
            result = await handler[kind](job_id)
            with self.ctx.sessionmaker() as db:
                job = db.get(Job, job_id)
                job.status, job.result, job.finished_at = "succeeded", result or {}, utcnow()
                db.commit()
        except CancelledError as e:
            self._finish(job_id, "cancelled", str(e) or "cancelled by user")
        except ProviderUnavailable as e:
            self._finish(job_id, "failed", f"provider unavailable: {e}")
        except Exception as e:  # noqa: BLE001
            log.error("job %s failed: %s", job_id, traceback.format_exc())
            self._finish(job_id, "failed", f"{type(e).__name__}: {e}"[:2000])

    def _finish(self, job_id: str, status: str, error: str) -> None:
        with self.ctx.sessionmaker() as db:
            job = db.get(Job, job_id)
            job.status, job.error, job.finished_at = status, error, utcnow()
            db.commit()

    def _cancel_check(self, job_id: str):
        def check() -> bool:
            with self.ctx.sessionmaker() as db:
                j = db.get(Job, job_id)
                return bool(j and j.cancel_requested)

        return check

    # ------------------------------------------------------------ handlers
    async def _garment_analysis(self, job_id: str) -> dict:
        from .analysis import analyze_garment

        def _run() -> dict:
            with self.ctx.sessionmaker() as db:
                job = db.get(Job, job_id)
                g = db.get(Garment, job.payload["garment_id"])
                if g is None or g.user_id != job.user_id:
                    raise ValueError("garment not found")
                return analyze_garment(db, self.ctx, g, job.payload.get("analyzer", "cv_basic"), job_id)

        return await asyncio.to_thread(_run)

    async def _tryon(self, job_id: str) -> dict:
        ctx = self.ctx
        with ctx.sessionmaker() as db:
            job = db.get(Job, job_id)
            p = job.payload
            photo = db.get(PersonPhoto, p["person_photo_id"])
            garment = db.get(Garment, p["garment_id"])
            if not photo or not garment or photo.user_id != job.user_id or garment.user_id != job.user_id:
                raise ValueError("photo or garment no longer exists")
            person_path = ctx.storage.local_path(photo.storage_key)
            garment_path = ctx.storage.local_path(garment.image_key)
            user_id = job.user_id
        provider = ctx.registry.tryon[p["provider"]]
        tmpdir = ctx.settings.data_dir / "tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)
        fd, out_path = tempfile.mkstemp(suffix=".png", dir=tmpdir)
        os.close(fd)
        req = TryOnRequest(
            person_image_path=person_path,
            garment_image_path=garment_path,
            output_path=out_path,
            **p["params"],
        )
        status = "failed"
        result = None
        try:
            result = await provider.generate(req, should_cancel=self._cancel_check(job_id))
            status = "succeeded"
        finally:
            with ctx.sessionmaker() as db:
                billed = p.get("paid", False) and (status == "succeeded" or p["provider"] == "remote_fashn")
                db.add(
                    InferenceRun(
                        user_id=user_id,
                        job_id=job_id,
                        task="tryon",
                        provider=p["provider"],
                        model=provider.info.model,
                        device=result.device if result else "unknown",
                        execution_time=result.elapsed_seconds if result else None,
                        model_load_seconds=result.model_load_seconds if result else None,
                        gpu_memory_peak_mb=result.gpu_memory_peak_mb if result else None,
                        estimated_cost_usd=(result.estimated_cost_usd if result else p.get("estimated_cost_usd", 0.0)) if billed else 0.0,
                        actual_cost_usd=result.actual_cost_usd if result else None,
                        paid=bool(billed),
                        status=status,
                    )
                )
                db.commit()
        with open(out_path, "rb") as f:
            data = f.read()
        os.unlink(out_path)
        with ctx.sessionmaker() as db:
            row = TryOnResultRow(
                user_id=user_id,
                job_id=job_id,
                person_photo_id=p["person_photo_id"],
                garment_ids=[p["garment_id"]],
                provider=p["provider"],
                model=provider.info.model,
                fidelity=provider.info.fidelity,
                image_key="",
                params=p["params"],
                dedup_key=p.get("dedup_key"),
                is_mock=result.is_mock,
            )
            db.add(row)
            db.flush()
            row.image_key = f"users/{user_id}/tryon/{row.id}.png"
            ctx.storage.put(row.image_key, data, "image/png")
            db.commit()
            return {
                "result_id": row.id,
                "provider": p["provider"],
                "model": provider.info.model,
                "elapsed_seconds": round(result.elapsed_seconds, 3),
                "model_load_seconds": result.model_load_seconds,
                "gpu_memory_peak_mb": result.gpu_memory_peak_mb,
                "estimated_cost_usd": result.estimated_cost_usd,
                "device": result.device,
                "is_mock": result.is_mock,
                "notes": result.notes,
            }

    async def _creative(self, job_id: str) -> dict:
        raise ProviderUnavailable("creative editing is disabled (ai.creative_editing.enabled=false)")


def cleanup_finished(db: Session, older_than_days: int = 30) -> int:
    cutoff = utcnow() - timedelta(days=older_than_days)
    jobs = db.scalars(select(Job).where(Job.finished_at.is_not(None), Job.finished_at < cutoff)).all()
    for j in jobs:
        db.delete(j)
    db.commit()
    return len(jobs)
