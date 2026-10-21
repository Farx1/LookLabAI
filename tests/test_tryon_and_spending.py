import base64

import httpx
import pytest
import yaml

from looklab_api.db import InferenceRun, Job
from looklab_api.jobs import recover_interrupted
from looklab_ml.evaluation.synthetic import DEMO_WARDROBE
from looklab_ml.tryon import FashnApiProvider, MockTryOnProvider
from tests.conftest import REPO, drain, make_client, png_bytes, upload_garment, upload_person


def setup_pair(client):
    return upload_person(client), upload_garment(client, DEMO_WARDROBE[0])


def test_auto_never_picks_paid_or_mock(client):
    p, g = setup_pair(client)
    r = client.post("/api/tryon/jobs", json={"person_photo_id": p["id"], "garment_id": g["id"]})
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["code"] == "no_free_provider_available"
    local = next(x for x in d["providers"] if x["name"] == "local_fashn")
    assert local["available"] is False and "weights missing" in local["reason"] or "not installed" in local["reason"]


def test_mock_job_lifecycle_and_cache(client, app):
    p, g = setup_pair(client)
    body = {"person_photo_id": p["id"], "garment_id": g["id"], "provider": "mock"}
    j = client.post("/api/tryon/jobs", json=body).json()
    assert j["status"] == "queued" and j["paid"] is False
    dup = client.post("/api/tryon/jobs", json=body).json()
    assert dup["id"] == j["id"]  # in-flight request deduplication
    assert drain(app) == 1
    done = client.get(f"/api/jobs/{j['id']}").json()
    assert done["status"] == "succeeded" and done["result"]["is_mock"] is True
    res = client.get("/api/tryon/results").json()
    assert len(res) == 1 and res[0]["is_mock"] and res[0]["fidelity"] == "mock"
    img = client.get(res[0]["image_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    again = client.post("/api/tryon/jobs", json=body).json()
    assert again["status"] == "succeeded" and again["cache_hit"] is True
    assert drain(app) == 0
    assert len(client.get("/api/tryon/results").json()) == 1
    # new seed => new generation
    j2 = client.post("/api/tryon/jobs", json={**body, "params": {"seed": 7}}).json()
    assert j2["status"] == "queued"


def test_cancel_queued_job(client, app):
    p, g = setup_pair(client)
    j = client.post("/api/tryon/jobs", json={"person_photo_id": p["id"], "garment_id": g["id"], "provider": "mock"}).json()
    assert client.post(f"/api/jobs/{j['id']}/cancel").json()["status"] == "cancelled"
    assert drain(app) == 0


def test_unsupported_category_and_unsuitable_photo(client):
    p = upload_person(client)
    shoes = upload_garment(client, next(i for i in DEMO_WARDROBE if i.kind == "shoes"))
    r = client.post("/api/tryon/jobs", json={"person_photo_id": p["id"], "garment_id": shoes["id"], "provider": "mock"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "category_unsupported"


def test_paid_providers_disabled_by_default(client):
    p, g = setup_pair(client)
    for name in ("fashn_api", "openai_image_edit", "remote_fashn"):
        r = client.post(
            "/api/tryon/jobs",
            json={"person_photo_id": p["id"], "garment_id": g["id"], "provider": name, "approve_paid": True, "max_cost_usd": 5},
        )
        assert r.status_code == 403 and r.json()["detail"]["code"] == "provider_disabled", name


@pytest.fixture
def paid_env(tmp_path, settings):
    """FASHN API enabled in policy and served by a fake transport that counts calls."""
    cfg = yaml.safe_load((REPO / "config" / "ai.yaml").read_text())
    cfg["ai"]["virtual_tryon"]["enabled_paid_providers"] = ["fashn_api"]
    cfg["ai"]["spending"]["monthly_limit_usd"] = 0.2
    f = tmp_path / "ai.yaml"
    f.write_text(yaml.safe_dump(cfg))
    settings.ai_config_path = f
    calls = {"run": 0, "status": 0, "fail": False}
    out_png = png_bytes(__import__("PIL.Image", fromlist=["Image"]).new("RGB", (64, 96), (200, 180, 160)))

    def handler(request: httpx.Request):
        assert request.headers["authorization"] == "Bearer test-key"
        if request.url.path.endswith("/run"):
            calls["run"] += 1
            body = request.read()
            assert b"data:image/jpeg;base64," in body  # base64 inputs, no public URLs
            return httpx.Response(200, json={"id": f"job{calls['run']}", "error": None})
        calls["status"] += 1
        if calls["fail"]:
            return httpx.Response(200, json={"id": "x", "status": "failed", "error": {"name": "ImageLoadError"}})
        return httpx.Response(
            200, json={"id": "x", "status": "completed", "output": ["data:image/png;base64," + base64.b64encode(out_png).decode()]}
        )

    provider = FashnApiProvider("test-key", price_per_image_usd=0.075, transport=httpx.MockTransport(handler))
    app, client = make_client(settings, {"fashn_api": provider})
    yield app, client, calls
    client.__exit__(None, None, None)


def test_paid_requires_consent_approval_and_budget(paid_env):
    app, client, calls = paid_env
    p, g = setup_pair(client)
    body = {"person_photo_id": p["id"], "garment_id": g["id"], "provider": "fashn_api"}
    # 1. external processing consent
    r = client.post("/api/tryon/jobs", json={**body, "approve_paid": True, "max_cost_usd": 1})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "external_processing_not_allowed"
    client.put("/api/settings", json={"privacy": {"allow_external_processing": True}})
    # 2. explicit approval
    r = client.post("/api/tryon/jobs", json=body)
    assert r.status_code == 402 and r.json()["detail"]["code"] == "approval_required"
    assert r.json()["detail"]["estimated_cost_usd"] == 0.075
    r = client.post("/api/tryon/jobs", json={**body, "approve_paid": True, "max_cost_usd": 0.05})
    assert r.json()["detail"]["code"] == "estimate_exceeds_approval"
    # 3. user's own monthly limit defaults to 0
    r = client.post("/api/tryon/jobs", json={**body, "approve_paid": True, "max_cost_usd": 0.1})
    assert r.status_code == 402 and r.json()["detail"]["code"] == "monthly_limit_reached"
    assert calls["run"] == 0  # nothing was ever sent
    # user raises their limit above the global cap: global cap (0.20) still applies
    client.put("/api/settings", json={"spending": {"monthly_limit_usd": 50}})
    assert client.get("/api/settings").json()["spending"]["effective_limit_usd"] == 0.2
    j = client.post("/api/tryon/jobs", json={**body, "approve_paid": True, "max_cost_usd": 0.1}).json()
    assert j["paid"] and j["estimated_cost_usd"] == 0.075
    # in-flight reservation counts toward the cap: a third request would exceed 0.20
    j2 = client.post("/api/tryon/jobs", json={**body, "params": {"seed": 1}, "approve_paid": True, "max_cost_usd": 0.1}).json()
    r3 = client.post("/api/tryon/jobs", json={**body, "params": {"seed": 2}, "approve_paid": True, "max_cost_usd": 0.1})
    assert r3.status_code == 402
    assert drain(app) == 2 and calls["run"] == 2
    costs = client.get("/api/costs").json()
    assert abs(costs["month_spend_usd"] - 0.15) < 1e-9
    assert client.get(f"/api/jobs/{j2['id']}").json()["status"] == "succeeded"


def test_paid_failure_is_not_retried(paid_env):
    app, client, calls = paid_env
    client.put("/api/settings", json={"privacy": {"allow_external_processing": True}, "spending": {"monthly_limit_usd": 1}})
    p, g = setup_pair(client)
    calls["fail"] = True
    j = client.post(
        "/api/tryon/jobs",
        json={"person_photo_id": p["id"], "garment_id": g["id"], "provider": "fashn_api", "approve_paid": True, "max_cost_usd": 0.1},
    ).json()
    drain(app)
    assert client.get(f"/api/jobs/{j['id']}").json()["status"] == "failed"
    assert drain(app) == 0 and calls["run"] == 1
    with app.state.sessionmaker() as db:
        run = db.query(InferenceRun).filter_by(task="tryon").one()
        assert run.status == "failed" and run.paid is False  # FASHN does not bill failed jobs


def test_restart_recovery_requeues_free_but_not_paid(app, client):
    with app.state.sessionmaker() as db:
        uid = client.get("/api/auth/me").json()["id"]
        db.add_all([Job(user_id=uid, kind="garment_analysis", status="running", payload={}),
                    Job(user_id=uid, kind="tryon", status="running", payload={"paid": True})])  # fmt: skip
        db.commit()
        assert recover_interrupted(db) == 2
        statuses = {j.kind: j.status for j in db.query(Job).all()}
    assert statuses == {"garment_analysis": "queued", "tryon": "failed"}


def test_mock_provider_cancellation():
    import asyncio

    from looklab_ml.tryon import CancelledError, TryOnRequest

    req = TryOnRequest("x", "y", "tops", output_path="/tmp/never.png")
    with pytest.raises(CancelledError):
        asyncio.run(MockTryOnProvider(delay_seconds=0.2).generate(req, should_cancel=lambda: True))


class RecordingMock(MockTryOnProvider):
    def __init__(self):
        super().__init__()
        self.categories = []

    async def generate(self, request, should_cancel=None):
        self.categories.append(request.category)
        return await super().generate(request, should_cancel)


def test_multi_garment_layer_order_single_result(settings):
    rec = RecordingMock()
    app, client = make_client(settings, {"mock": rec})
    try:
        p = upload_person(client)
        top = upload_garment(client, next(i for i in DEMO_WARDROBE if i.meta.get("category") == "top"))
        bottom = upload_garment(client, next(i for i in DEMO_WARDROBE if i.meta.get("category") == "bottom"))
        shoes = upload_garment(client, next(i for i in DEMO_WARDROBE if i.kind == "shoes"))
        body = {"person_photo_id": p["id"], "provider": "mock"}
        r = client.post("/api/tryon/jobs", json={**body, "garment_ids": [top["id"], shoes["id"]]})
        assert r.status_code == 422 and r.json()["detail"]["code"] == "category_unsupported"
        j = client.post("/api/tryon/jobs", json={**body, "garment_ids": [top["id"], bottom["id"]]}).json()
        assert drain(app) == 1
        assert client.get(f"/api/jobs/{j['id']}").json()["status"] == "succeeded"
        assert rec.categories == ["bottoms", "tops"]
        res = client.get("/api/tryon/results").json()
        assert len(res) == 1 and res[0]["garment_ids"] == [bottom["id"], top["id"]]
    finally:
        client.__exit__(None, None, None)


def test_multi_garment_foreign_id_rejected(tmp_path):
    from tests.test_privacy_and_isolation import _multi

    app, client = _multi(tmp_path)
    try:
        h = {}
        for who in ("alice", "bob"):
            r = client.post("/api/auth/register", json={"email": f"{who}@example.com", "password": "correct horse battery"})
            h[who] = {"Authorization": f"Bearer {r.json()['token']}"}
        client.cookies.clear()
        mine = upload_garment(client, DEMO_WARDROBE[0], headers=h["alice"])
        theirs = upload_garment(client, DEMO_WARDROBE[1], headers=h["bob"])
        pa = upload_person(client, headers=h["alice"])
        r = client.post(
            "/api/tryon/jobs",
            json={"person_photo_id": pa["id"], "garment_ids": [mine["id"], theirs["id"]], "provider": "mock"},
            headers=h["alice"],
        )
        assert r.status_code == 404
    finally:
        client.__exit__(None, None, None)


def test_both_garment_fields_rejected_and_explicit_category_kept(client):
    p, g = setup_pair(client)
    r = client.post("/api/tryon/jobs", json={"person_photo_id": p["id"], "garment_id": g["id"], "garment_ids": [g["id"]], "provider": "mock"})
    assert r.status_code == 422
    shoes = upload_garment(client, next(i for i in DEMO_WARDROBE if i.meta.get("category") == "shoes"))
    r = client.post("/api/tryon/jobs", json={"person_photo_id": p["id"], "garment_id": shoes["id"], "category": "tops", "provider": "mock"})
    assert r.status_code == 200  # single garment: the explicit category wins over the unsupported one


def test_legacy_job_payload_without_steps(client, app):
    p, g = setup_pair(client)
    j = client.post("/api/tryon/jobs", json={"person_photo_id": p["id"], "garment_id": g["id"], "provider": "mock"}).json()
    with app.state.sessionmaker() as db:
        job = db.get(Job, j["id"])
        old = {k: v for k, v in job.payload.items() if k != "steps"}
        job.payload = {**old, "garment_id": g["id"]}
        db.commit()
    drain(app)
    assert client.get(f"/api/jobs/{j['id']}").json()["status"] == "succeeded"
