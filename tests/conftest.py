from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from looklab_api.config import Settings
from looklab_api.main import create_app
from looklab_ml.evaluation.synthetic import DEMO_WARDROBE, draw_garment, draw_person

REPO = Path(__file__).resolve().parents[1]


def png_bytes(img, fmt="PNG", **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        env="test",
        data_dir=tmp_path / "data",
        ai_config_path=REPO / "config" / "ai.yaml",
        worker_enabled=False,
        fashn_weights_dir=tmp_path / "no-weights",
        ollama_url="http://127.0.0.1:9",  # nothing listens here: VLMs unavailable
    )


def make_client(settings, overrides=None):
    app = create_app(settings, provider_overrides=overrides)
    client = TestClient(app)
    client.__enter__()
    return app, client


@pytest.fixture
def app_client(settings):
    app, client = make_client(settings)
    yield app, client
    client.__exit__(None, None, None)


@pytest.fixture
def client(app_client):
    return app_client[1]


@pytest.fixture
def app(app_client):
    return app_client[0]


def drain(app) -> int:
    return asyncio.run(app.state.runner.drain())


def upload_garment(client, item, headers=None) -> dict:
    img = draw_garment(item.kind, item.color, item.pattern)
    r = client.post(
        "/api/garments",
        files=[("files", (f"{item.filename}.png", png_bytes(img), "image/png"))],
        headers=headers or {},
    )
    assert r.status_code == 200, r.text
    g = r.json()["created"][0]
    if item.meta:
        r = client.patch(f"/api/garments/{g['id']}", json=item.meta, headers=headers or {})
        assert r.status_code == 200, r.text
        g = r.json()
    return g


@pytest.fixture
def wardrobe(client):
    return [upload_garment(client, it) for it in DEMO_WARDROBE]


def upload_person(client, headers=None) -> dict:
    r = client.post(
        "/api/photos",
        files={"file": ("me.jpg", png_bytes(draw_person(), "JPEG", quality=90), "image/jpeg")},
        data={"consent_confirmed": "true", "consent_subject": "self"},
        headers=headers or {},
    )
    assert r.status_code == 200, r.text
    return r.json()
