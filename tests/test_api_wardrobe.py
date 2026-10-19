import httpx

from looklab_ml.evaluation.synthetic import DEMO_WARDROBE, draw_garment
from tests.conftest import png_bytes, upload_garment


def test_upload_analyzes_and_lists(client):
    r = client.post(
        "/api/garments",
        files=[
            ("files", ("cream blouse.png", png_bytes(draw_garment("top", "cream")), "image/png")),
            ("files", ("IMG_2041.png", png_bytes(draw_garment("trousers", "camel")), "image/png")),
            ("files", ("notes.txt", b"hello", "text/plain")),
        ],
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["created"]) == 2 and len(body["errors"]) == 1
    blouse, trousers = body["created"]
    assert blouse["name"] == "cream blouse" and blouse["category"] == "top"
    assert blouse["field_sources"]["category"]["source"] == "text"
    assert blouse["colors"][0] == "cream" and blouse["field_sources"]["colors"]["source"] == "cv"
    assert trousers["name"] is None and trousers["category"] == "unknown"  # no guess from pixels
    assert trousers["is_owned"] is True
    assert len(client.get("/api/garments").json()) == 2


def test_user_edits_are_never_overwritten_by_analysis(client):
    g = upload_garment(client, DEMO_WARDROBE[0])
    r = client.patch(f"/api/garments/{g['id']}", json={"colors": ["blush"]})
    assert r.json()["field_sources"]["colors"]["source"] == "user"
    r = client.post(f"/api/garments/{g['id']}/analyze", json={"analyzer": "cv_basic"})
    assert r.status_code == 200
    assert r.json()["garment"]["colors"] == ["blush"]
    assert r.json()["result"]["cache_hit"] is True  # analysis cached by content hash


def test_metadata_validation(client):
    g = upload_garment(client, DEMO_WARDROBE[0])
    for bad in ({"category": "spaceship"}, {"colors": ["neon unicorn"]}, {"formality": 9}, {"style": ["goth-core"]}, {"currency": "euro"}):
        assert client.patch(f"/api/garments/{g['id']}", json=bad).status_code == 422, bad
    vocab = client.get("/api/garments/vocabulary").json()
    assert "korean" in vocab["styles"] and "cream" in vocab["colors"]


def test_filters_search_favorites_collections(client):
    gs = [upload_garment(client, it) for it in DEMO_WARDROBE[:8]]
    client.patch(f"/api/garments/{gs[0]['id']}", json={"favorite": True})
    assert [g["id"] for g in client.get("/api/garments?favorite=true").json()] == [gs[0]["id"]]
    assert all(g["category"] == "bottom" for g in client.get("/api/garments?category=bottom").json())
    assert client.get("/api/garments?q=blouse").json()[0]["id"] == gs[0]["id"]
    assert all("korean" in g["style"] for g in client.get("/api/garments?style=korean").json())
    col = client.post("/api/collections", json={"name": "Date night", "garment_ids": [gs[0]["id"], gs[6]["id"]]}).json()
    assert {g["id"] for g in client.get(f"/api/garments?collection_id={col['id']}").json()} == {gs[0]["id"], gs[6]["id"]}
    client.delete(f"/api/garments/{gs[0]['id']}")
    assert client.get("/api/collections").json()[0]["garment_ids"] == [gs[6]["id"]]
    assert client.get(f"/api/garments/{gs[0]['id']}").status_code == 404


def test_garment_images_are_private_and_streamed(client):
    g = upload_garment(client, DEMO_WARDROBE[0])
    r = client.get(g["thumb_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert "private" in r.headers["cache-control"]
    assert client.get("/files/users").status_code == 404  # no static file mount


PRODUCT_HTML = """<html><head><title>x</title>
<meta property="og:title" content="Wide Pleated Midi Skirt">
<script type="application/ld+json">{"@context":"https://schema.org","@type":"Product","name":"Wide Pleated Midi Skirt",
"image":"https://shop.example.com/img/skirt.png","brand":{"@type":"Brand","name":"Atelier"},
"offers":{"@type":"Offer","price":"39.90","priceCurrency":"EUR"}}</script></head><body></body></html>"""


def test_import_from_product_page(client, app, monkeypatch):
    import looklab_api.importer as imp

    monkeypatch.setattr(imp, "_check_host", lambda url: None)  # offline test: skip DNS
    img = png_bytes(draw_garment("wide_skirt", "ivory"))

    def handler(request: httpx.Request):
        if request.url.path.endswith(".png"):
            return httpx.Response(200, content=img, headers={"content-type": "image/png"})
        return httpx.Response(200, text=PRODUCT_HTML, headers={"content-type": "text/html"})

    app.state.ctx.import_transport = httpx.MockTransport(handler)
    r = client.post("/api/garments/import-url", json={"url": "https://shop.example.com/p/skirt"})
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["name"] == "Wide Pleated Midi Skirt" and g["price"] == 39.9 and g["brand"] == "Atelier"
    assert g["category"] == "bottom" and g["subcategory"] == "skirt" and g["is_owned"] is False
    assert g["product_url"] == "https://shop.example.com/p/skirt"


def test_import_refuses_private_addresses(client):
    for url in ("http://127.0.0.1:8000/api/garments", "http://localhost/", "http://169.254.169.254/latest/meta-data"):
        r = client.post("/api/garments/import-url", json={"url": url})
        assert r.status_code == 422 and "private or local" in r.json()["detail"], url
