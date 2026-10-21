from looklab_ml.evaluation.synthetic import DEMO_WARDROBE
from tests.conftest import upload_garment

SPEC = (
    "Find 15 outfits for a romantic anniversary dinner. I prefer loose-fitting, comfortable clothes, "
    "neutral colors, Korean-inspired fashion and a budget of €100."
)


def test_generate_principal_use_case(client, wardrobe):
    r = client.post("/api/outfits/generate", json={"prompt": SPEC})
    assert r.status_code == 200
    d = r.json()
    assert len(d["outfits"]) == 15
    o = d["outfits"][0]
    assert o["garments"] and o["explanation"] and o["collage_url"].startswith("/api/outfits/collage?ids=")
    assert d["constraints"]["budget"] == 100 and "budget 100 EUR" in d["understood"]
    assert {"brand", "price", "product_url"} <= set(o["garments"][0])


def test_collage_is_cached(client, app, wardrobe):
    o = client.post("/api/outfits/generate", json={"prompt": "3 outfits"}).json()["outfits"][0]
    a = client.get(o["collage_url"])
    assert a.status_code == 200 and a.headers["content-type"] == "image/webp"
    storage = app.state.ctx.storage
    files = list((storage.root / "users").rglob("collages/*.webp"))
    assert len(files) == 1
    b = client.get(o["collage_url"])
    assert b.content == a.content and len(list((storage.root / "users").rglob("collages/*.webp"))) == 1


def test_structured_constraints_and_count_override(client, wardrobe):
    r = client.post("/api/outfits/generate", json={"constraints": {"owned_only": True, "excluded_colors": ["black"]}, "count": 4})
    d = r.json()
    assert len(d["outfits"]) == 4
    assert all("black" not in g["colors"] for o in d["outfits"] for g in o["garments"])


def test_save_feedback_and_learned_preferences_persist(client, wardrobe):
    o = client.post("/api/outfits/generate", json={"prompt": "5 outfits"}).json()["outfits"][0]
    saved = client.post("/api/outfits/saved", json={"garment_ids": o["garment_ids"], "occasion": "date"}).json()
    assert saved["title"] and len(saved["garments"]) == len(o["garment_ids"])
    client.post(f"/api/outfits/saved/{saved['id']}/feedback", json={"event": "like"})
    learned = client.get("/api/preferences/learned").json()
    assert learned["events"] == 2 and learned["top_features"]
    assert client.get("/api/outfits/saved").json()[0]["feedback"] == "like"
    lb = client.get("/api/lookbook").json()
    assert lb["boards"][0]["occasion"] == "date"
    client.post("/api/outfits/feedback", json={"garment_ids": o["garment_ids"], "event": "dislike"})
    assert client.get("/api/preferences/learned").json()["events"] == 3
    client.delete("/api/preferences/learned")
    assert client.get("/api/preferences/learned").json()["events"] == 0


def test_saved_outfit_survives_garment_deletion(client, wardrobe):
    o = client.post("/api/outfits/generate", json={"prompt": "2 outfits"}).json()["outfits"][0]
    s = client.post("/api/outfits/saved", json={"garment_ids": o["garment_ids"]}).json()
    client.delete(f"/api/garments/{o['garment_ids'][0]}")
    s2 = client.get("/api/outfits/saved").json()[0]
    assert s2["id"] == s["id"] and s2["missing_garments"] == [o["garment_ids"][0]]


def test_saved_outfit_all_garments_deleted_has_no_collage(client, wardrobe):
    o = client.post("/api/outfits/generate", json={"prompt": "2 outfits"}).json()["outfits"][0]
    client.post("/api/outfits/saved", json={"garment_ids": o["garment_ids"]})
    for gid in o["garment_ids"]:
        client.delete(f"/api/garments/{gid}")
    assert client.get("/api/outfits/saved").json()[0]["collage_url"] is None


def test_profile_preferences_apply(client, wardrobe):
    client.put("/api/settings", json={"preferences": {"disliked_colors": ["black"], "fit_preference": "relaxed"}})
    d = client.post("/api/outfits/generate", json={"prompt": "10 outfits"}).json()
    assert all("black" not in g["colors"] for o in d["outfits"] for g in o["garments"])


def test_empty_wardrobe_is_explained(client):
    d = client.post("/api/outfits/generate", json={"prompt": "3 outfits"}).json()
    assert d["outfits"] == [] and d["notes"]
    upload_garment(client, DEMO_WARDROBE[0])
    d = client.post("/api/outfits/generate", json={"prompt": "3 outfits"}).json()
    assert d["outfits"] == [] and "bottom" in " ".join(d["notes"])


def test_surprise_empty_wardrobe_422(client):
    r = client.post("/api/outfits/surprise", json={})
    assert r.status_code == 422 and "pièces possédées" in r.json()["detail"]


def test_surprise_owned_only_and_novel(client, wardrobe):
    a = client.post("/api/outfits/surprise", json={}).json()
    b = client.post("/api/outfits/surprise", json={}).json()
    assert a["trend"]["nom"] and a["trend"]["sources"] and a["outfits"][0]["explanation"][0].startswith("Tendance FW26")
    assert all(o["score"]["trend"] >= 0.55 for d in (a, b) for o in d["outfits"])
    assert all(g["is_owned"] for d in (a, b) for o in d["outfits"] for g in o["garments"])
    assert a["outfits"][0]["garment_ids"] != b["outfits"][0]["garment_ids"]
    assert len(client.post("/api/outfits/surprise", json={"count": 3}).json()["outfits"]) == 3
    assert client.post("/api/outfits/surprise", json={"count": 7}).status_code == 422


def test_variations_api(client, wardrobe):
    ids = client.post("/api/outfits/generate", json={"prompt": "3 outfits"}).json()["outfits"][0]["garment_ids"]
    r = client.post("/api/outfits/variations", json={"garment_ids": ids, "count": 3})
    assert r.status_code == 200
    outs = r.json()["outfits"]
    assert 1 <= len(outs) <= 3 and all(g["is_owned"] for o in outs for g in o["garments"])
    assert all(o["collage_url"] and o["explanation"][0].startswith(("Remplace", "Ajoute", "Retire")) for o in outs)
    assert client.post("/api/outfits/variations", json={"garment_ids": [*ids[:1], "nope"]}).status_code == 404
    assert client.post("/api/outfits/variations", json={"garment_ids": ids, "count": 9}).status_code == 422
    assert client.post("/api/outfits/variations", json={"garment_ids": [*ids, *ids, *ids]}).status_code == 422
    assert client.post("/api/outfits/variations", json={"garment_ids": ids, "swap_garment_id": "x"}).status_code == 422
