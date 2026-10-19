from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, get_settings, load_ai_policy
from .context import AppContext, build_storage
from .db import Base, make_engine, make_sessionmaker
from .jobs import JobRunner, recover_interrupted
from .providers import ProviderError, ProviderRegistry
from .routes import auth_routes, garments, outfits, photos, stylist, system, tryon
from .routes import settings as settings_routes

log = logging.getLogger("looklab")


def create_app(settings: Settings | None = None, provider_overrides: dict | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    policy = load_ai_policy(settings.ai_config_path)
    engine = make_engine(settings.db_url)
    Base.metadata.create_all(engine)
    sm = make_sessionmaker(engine)
    ctx = AppContext(
        settings=settings,
        policy=policy,
        storage=build_storage(settings),
        registry=ProviderRegistry(settings, policy, provider_overrides),
        sessionmaker=sm,
    )
    runner = JobRunner(ctx)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with sm() as db:
            n = recover_interrupted(db)
            if n:
                log.info("recovered %d interrupted jobs", n)
        tasks = []
        if settings.worker_enabled:
            tasks = [asyncio.create_task(runner.run_lane("cpu")), asyncio.create_task(runner.run_lane("gpu"))]
        yield
        runner.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        ctx.registry.fashn_runtime.unload()

    app = FastAPI(title="LookLab AI API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.sessionmaker = sm
    app.state.ctx = ctx
    app.state.runner = runner
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ProviderError)
    async def _provider_error(_: Request, e: ProviderError):
        return JSONResponse(status_code=e.status, content={"detail": {"code": e.code, "message": e.message, **e.extra}})

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        return resp

    for r in (system, auth_routes, photos, garments, outfits, stylist, tryon, settings_routes):
        app.include_router(r.router)

    @app.get("/api/{rest:path}", include_in_schema=False)
    def _not_found(rest: str):
        raise HTTPException(404, "not found")

    return app


def run() -> None:
    import uvicorn

    s = get_settings()
    host = "127.0.0.1" if s.auth_mode == "single_user" else "0.0.0.0"
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("looklab_api.main:create_app", factory=True, host=host, port=8000)
