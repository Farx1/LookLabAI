from looklab_api.config import Settings
from looklab_ml.evaluation.synthetic import DEMO_WARDROBE, draw_person
from tests.conftest import REPO, drain, make_client, png_bytes, upload_garment, upload_person


def test_photo_requires_consent_and_strips_exif(client, app):
    raw = png_bytes(draw_person(), "JPEG")
    r = client.post("/api/photos", files={"file": ("me.jpg", raw, "image/jpeg")})
    assert r.status_code == 400
    p = upload_person(client)
    assert p["consent_subject"] == "self" and p["quality"]["ok"]
    stored = app.state.ctx.storage.get(f"users/{client.get('/api/auth/me').json()['id']}/photos/{p['id']}.jpg")
    from looklab_api.images import has_exif

    assert not has_exif(stored)


def test_delete_photo_cascades_generated_images(client, app):
    p = upload_person(client)
    g = upload_garment(client, DEMO_WARDROBE[0])
    client.post("/api/tryon/jobs", json={"person_photo_id": p["id"], "garment_id": g["id"], "provider": "mock"})
    drain(app)
    res = client.get("/api/tryon/results").json()[0]
    r = client.delete(f"/api/photos/{p['id']}")
    assert r.json()["deleted_generated_images"] == 1
    assert client.get(res["image_url"]).status_code == 404
    assert client.get(p["image_url"]).status_code == 404
    root = app.state.ctx.storage.root
    assert not list(root.rglob(f"{p['id']}*")) and not list(root.rglob(f"{res['id']}*"))


def test_delete_all_and_export(client, app):
    upload_person(client)
    upload_garment(client, DEMO_WARDROBE[0])
    exp = client.get("/api/me/export").json()
    assert len(exp["garments"]) == 1 and len(exp["person_photos"]) == 1
    r = client.delete("/api/me/data").json()
    assert r["deleted"]["garments"] == 1 and r["deleted"]["person_photos"] == 1
    assert client.get("/api/garments").json() == [] and client.get("/api/photos").json() == []
    assert not list((app.state.ctx.storage.root / "users").rglob("*.*"))


def test_retention_purges_unsaved_generated_images(client, app):
    from datetime import timedelta

    from looklab_api.db import TryOnResultRow, utcnow
    from looklab_api.privacy import purge_expired

    p = upload_person(client)
    g = upload_garment(client, DEMO_WARDROBE[0])
    for seed in (1, 2):
        client.post(
            "/api/tryon/jobs", json={"person_photo_id": p["id"], "garment_id": g["id"], "provider": "mock", "params": {"seed": seed}}
        )
    drain(app)
    keep = client.get("/api/tryon/results").json()[0]
    client.patch(f"/api/tryon/results/{keep['id']}", json={"saved": True})
    client.put("/api/settings", json={"privacy": {"retention_days_generated": 7}})
    with app.state.sessionmaker() as db:
        for r in db.query(TryOnResultRow).all():
            r.created_at = utcnow() - timedelta(days=10)
        db.commit()
        assert purge_expired(db, app.state.ctx) == 1
    assert [r["id"] for r in client.get("/api/tryon/results").json()] == [keep["id"]]


def _multi(tmp_path):
    s = Settings(env="test", data_dir=tmp_path / "multi", ai_config_path=REPO / "config" / "ai.yaml", worker_enabled=False,
                 auth_mode="multi_user", fashn_weights_dir=tmp_path / "nw", ollama_url="http://127.0.0.1:9")  # fmt: skip
    return make_client(s)


def test_user_data_isolation(tmp_path):
    app, client = _multi(tmp_path)
    assert client.get("/api/garments").status_code == 401
    h = {}
    for who in ("alice", "bob"):
        r = client.post("/api/auth/register", json={"email": f"{who}@example.com", "password": "correct horse battery"})
        assert r.status_code == 200
        h[who] = {"Authorization": f"Bearer {r.json()['token']}"}
    client.cookies.clear()
    ga = upload_garment(client, DEMO_WARDROBE[0], headers=h["alice"])
    pa = upload_person(client, headers=h["alice"])
    client.post("/api/tryon/jobs", json={"person_photo_id": pa["id"], "garment_id": ga["id"], "provider": "mock"}, headers=h["alice"])
    drain(app)
    ra = client.get("/api/tryon/results", headers=h["alice"]).json()[0]
    job = client.get("/api/jobs", headers=h["alice"]).json()[0]
    # Bob sees nothing and cannot address Alice's objects
    assert client.get("/api/garments", headers=h["bob"]).json() == []
    assert client.get("/api/photos", headers=h["bob"]).json() == []
    for url in (f"/api/garments/{ga['id']}", ga["image_url"], pa["image_url"], ra["image_url"], f"/api/jobs/{job['id']}",
                f"/api/outfits/collage?ids={ga['id']}"):  # fmt: skip
        assert client.get(url, headers=h["bob"]).status_code == 404, url
    assert client.patch(f"/api/garments/{ga['id']}", json={"favorite": True}, headers=h["bob"]).status_code == 404
    assert client.delete(f"/api/photos/{pa['id']}", headers=h["bob"]).status_code == 404
    r = client.post("/api/tryon/jobs", json={"person_photo_id": pa["id"], "garment_id": ga["id"], "provider": "mock"}, headers=h["bob"])
    assert r.status_code == 404
    r = client.post("/api/outfits/saved", json={"garment_ids": [ga["id"]]}, headers=h["bob"])
    assert r.status_code == 404
    # Bob deleting all of *his* data leaves Alice untouched
    client.delete("/api/me/data", headers=h["bob"])
    assert len(client.get("/api/garments", headers=h["alice"]).json()) == 1
    client.__exit__(None, None, None)


def test_cookie_auth_requires_csrf_header(tmp_path):
    app, client = _multi(tmp_path)
    r = client.post("/api/auth/register", json={"email": "c@example.com", "password": "long enough pw"})
    assert r.status_code == 200 and "looklab_session" in client.cookies
    assert client.get("/api/garments").status_code == 200
    assert client.put("/api/settings", json={}).status_code == 403
    assert client.put("/api/settings", json={}, headers={"X-Requested-With": "looklab"}).status_code == 200
    client.post("/api/auth/logout")
    client.cookies.clear()
    assert client.get("/api/garments").status_code == 401
    assert client.post("/api/auth/login", json={"email": "c@example.com", "password": "wrong password!"}).status_code == 401
    client.__exit__(None, None, None)
