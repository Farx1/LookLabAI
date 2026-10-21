from __future__ import annotations

from PIL import Image, ImageDraw

from looklab_ml.evaluation.synthetic import DEMO_WARDROBE, draw_person
from looklab_ml.vision.outfit import analyze_outfit, describe_piece, drop_phantoms, merge_items, similarity
from tests.conftest import REPO, Settings, drain, make_client, png_bytes, upload_garment


def _post(client, headers=None, **data):
    return client.post(
        "/api/analyze", files={"file": ("look.png", png_bytes(draw_person()), "image/png")}, data=data, headers=headers or {}
    )


def _stored(app, job=None) -> bool:
    return any((app.state.settings.data_dir / "files").glob("users/*/analysis/*"))


def _fake(garment):
    piece = {k: garment[k] for k in ("category", "subcategory", "colors", "color_hex", "pattern")}
    return lambda image, person: ([{"index": 1, "name": "x", "bbox": [0, 0, 500, 500], **piece}], {"detect": 0.1})


def test_consent_required_for_person_photo(client):
    assert _post(client).status_code == 400
    assert _post(client, person_visible="true", consent_confirmed="false").status_code == 400


def test_job_matches_wardrobe_and_deletes_photo(app, client):
    gs = [upload_garment(client, it) for it in DEMO_WARDROBE[:6]]
    target = gs[2]
    app.state.ctx.registry.outfit_analyze = _fake(target)
    job = _post(client, person_visible="false").json()
    assert job["kind"] == "outfit_analysis" and job["status"] == "queued"
    assert _stored(app, job)
    drain(app)
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "succeeded", done
    matches = done["result"]["pieces"][0]["matches"]
    assert matches[0]["garment"]["id"] == target["id"] and matches[0]["score"] > 0.9 and len(matches) <= 3
    assert not _stored(app, job)  # photo gone after the analysis


def test_failed_analysis_still_deletes_photo(app, client):
    def boom(image, person):
        raise RuntimeError("detector crashed")

    app.state.ctx.registry.outfit_analyze = boom
    job = _post(client, person_visible="false").json()
    drain(app)
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "failed"
    assert not _stored(app, job)


def test_delete_pending_analysis_and_foreign_job(tmp_path):
    s = Settings(env="test", data_dir=tmp_path / "m", ai_config_path=REPO / "config" / "ai.yaml", worker_enabled=False,
                 auth_mode="multi_user", fashn_weights_dir=tmp_path / "nw", ollama_url="http://127.0.0.1:9")  # fmt: skip
    app, client = make_client(s)
    h = {}
    for who in ("alice", "bob"):
        r = client.post("/api/auth/register", json={"email": f"{who}@example.com", "password": "correct horse battery"})
        h[who] = {"Authorization": f"Bearer {r.json()['token']}"}
    client.cookies.clear()
    job = _post(client, h["alice"], person_visible="false").json()
    assert client.delete(f"/api/analyze/{job['id']}", headers=h["bob"]).status_code == 404
    assert client.get(f"/api/jobs/{job['id']}", headers=h["bob"]).status_code == 404
    assert _stored(app, job)
    assert client.delete(f"/api/analyze/{job['id']}", headers=h["alice"]).status_code == 200
    assert not _stored(app, job) and client.get(f"/api/jobs/{job['id']}", headers=h["alice"]).json()["status"] == "cancelled"


class _FakePlanner:
    def __init__(self, ask, detect=()):
        self._ask_items, self._detect = ask, list(detect)

    def _ask(self, image, prompt):
        return [dict(i) for i in self._ask_items]

    def detect(self, image):
        return self._detect

    def _unload_qwen(self):
        pass


def test_detection_parsing_and_dedup():
    img = Image.new("RGB", (200, 400), (230, 230, 230))
    img.paste((20, 20, 20), (40, 40, 160, 200))  # black top
    ask = [{"name": "black t-shirt", "bbox_2d": [200, 100, 800, 500]}, {"name": "beige scarf", "bbox_2d": [400, 350, 592, 502]}]
    det = [{"name": "scarf", "layer": None, "bbox": [433.0, 412.0, 607.0, 536.0]}, {"name": "face", "layer": None, "bbox": [0, 0, 5, 5]}]
    pieces, t = analyze_outfit(_FakePlanner(ask, det), img, person=True)
    assert [p["category"] for p in pieces] == ["top", "accessory"]  # duplicate scarf merged
    assert pieces[0]["colors"][0] == "black" and set(t) == {"detect", "describe"}
    assert len(merge_items([{"name": "face", "bbox_2d": [0, 0, 5, 5]}])) == 0
    assert describe_piece(img, {"name": "white shirt", "bbox_2d": [0, 0, 1, 1]})["category"] == "top"  # tiny box: no crash
    assert describe_piece(img, {"name": "Christmas ornament", "bbox_2d": [0, 0, 500, 500]}) is None


def test_duplicate_head_item_and_phantom_box():
    box = lambda x1, y1, x2, y2: {"bbox_2d": [x1, y1, x2, y2]}  # noqa: E731
    a, b = {"name": "black headband", **box(400, 250, 590, 300)}, {"name": "headband", **box(442, 324, 626, 380)}  # no overlap
    assert merge_items([a], [b], worn=True) == [b]  # one headband on a head; the later (head-crop) box wins
    assert len(merge_items([a], [b])) == 2  # a flat-lay may hold two
    pants = {"category": "bottom", "bbox": [300, 600, 600, 990]}
    shoes = {"category": "shoes", "bbox": [302, 601, 598, 988]}  # shoes invented on the trousers
    assert drop_phantoms([shoes, pants]) == [pants] and drop_phantoms([pants, shoes]) == [pants]
    assert drop_phantoms([pants, {"category": "shoes", "bbox": [300, 900, 600, 990]}]) != [pants]  # real shoes stay


def test_subcategory_and_pattern_matching():
    band = {"category": "accessory", "subcategory": "headband", "colors": ["white"], "pattern": "solid"}
    scarf = {"category": "accessory", "subcategory": "echarpe", "colors": ["white"], "pattern": "solid", "name": "Echarpe"}
    assert similarity(band, scarf) == 0.0  # accessories of another kind never match
    knit = {"category": "top", "subcategory": "turtleneck", "colors": ["white"], "pattern": "solid"}
    mine = {"category": "top", "colors": ["white"], "name": "Haut à col roulé", "subcategory": "haut col roule"}  # French catalogue
    polo = {"category": "top", "colors": ["white"], "name": "Pull polo à rayures", "subcategory": "pull polo", "pattern": "patterned"}
    assert similarity(knit, mine) > similarity(knit, polo) + 0.2  # same garment words, and solid vs striped


def test_flat_lay_background_is_masked():
    img = Image.new("RGB", (200, 200), (250, 250, 250))
    ImageDraw.Draw(img).polygon([(85, 0), (115, 0), (200, 200), (0, 200)], fill=(20, 20, 20))  # A-line dress on a white sheet
    item = {"name": "dress", "bbox_2d": [0, 0, 1000, 1000]}
    assert describe_piece(img, item, flat=True)["colors"][0] == "black"  # margin shows the sheet, so it is masked out
    assert describe_piece(img, item)["colors"][0] == "white"  # the plain central crop takes the sheet for the dress


def test_cancelled_analysis_photo_is_swept(app, client):
    from looklab_api.privacy import purge_expired

    job = _post(client, person_visible="false").json()
    client.post(f"/api/jobs/{job['id']}/cancel")
    assert _stored(app)
    with app.state.sessionmaker() as db:
        purge_expired(db, app.state.ctx)
    assert not _stored(app)


def test_pending_analysis_limit_and_failed_photo_sweep(app, client):
    from looklab_api.db import Job
    from looklab_api.privacy import purge_expired

    for _ in range(3):
        assert _post(client, person_visible="false").status_code == 200
    assert _post(client, person_visible="false").status_code == 429
    with app.state.sessionmaker() as db:
        for j in db.query(Job).all():
            j.status = "failed"  # a worker that died without reaching its cleanup
        db.commit()
    assert _stored(app)
    with app.state.sessionmaker() as db:
        purge_expired(db, app.state.ctx)
        assert all("image_key" not in j.payload for j in db.query(Job).all())  # swept once, not again
    assert not _stored(app)
