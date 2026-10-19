from tests.test_api_outfits import SPEC


def names(msg, wardrobe):
    by = {g["id"]: g["name"] for g in wardrobe}
    return [[by[i] for i in o["garment_ids"]] for o in msg["payload"]["outfits"]]


def test_conversation_keeps_constraints(client, wardrobe):
    s = client.post("/api/stylist/sessions").json()
    m1 = client.post(f"/api/stylist/sessions/{s['id']}/messages", json={"content": SPEC}).json()
    assert len(m1["message"]["payload"]["outfits"]) == 15
    assert "budget 100 EUR" in m1["message"]["content"]
    m2 = client.post(f"/api/stylist/sessions/{s['id']}/messages", json={"content": "Make them less fitted."}).json()
    c = m2["constraints"]
    assert c["budget"] == 100 and c["occasion"] == "date" and c["target_looseness"] == 3.0
    m3 = client.post(f"/api/stylist/sessions/{s['id']}/messages", json={"content": "Show me something more sophisticated."}).json()
    assert m3["constraints"]["formality_shift"] == 1 and m3["constraints"]["budget"] == 100
    hist = client.get(f"/api/stylist/sessions/{s['id']}").json()
    assert [m["role"] for m in hist["messages"]] == ["user", "assistant"] * 3


def test_keep_shoes_change_dress(client, wardrobe):
    s = client.post("/api/stylist/sessions").json()
    sid = s["id"]
    client.post(f"/api/stylist/sessions/{sid}/messages", json={"content": "5 romantic dresses for a date"})
    hist = client.get(f"/api/stylist/sessions/{sid}").json()
    outfits = hist["messages"][-1]["payload"]["outfits"]
    idx = next(i for i, o in enumerate(outfits) if any(g["category"] == "dress" for g in o["garments"]))
    focus = outfits[idx]
    dress = next(g["id"] for g in focus["garments"] if g["category"] == "dress")
    shoes = next(g["id"] for g in focus["garments"] if g["category"] == "shoes")
    client.post(f"/api/stylist/sessions/{sid}/focus", json={"index": idx})
    m = client.post(f"/api/stylist/sessions/{sid}/messages", json={"content": "Keep the same shoes but change the dress."}).json()
    new = m["message"]["payload"]["outfits"]
    assert new, m["message"]["content"]
    for o in new:
        assert shoes in o["garment_ids"] and dress not in o["garment_ids"]
        assert any(g["category"] == "dress" for g in o["garments"])
    assert "Working from outfit" in m["message"]["content"]


def test_sessions_list_and_delete(client, wardrobe):
    s = client.post("/api/stylist/sessions").json()
    client.post(f"/api/stylist/sessions/{s['id']}/messages", json={"content": "2 casual outfits"})
    assert client.get("/api/stylist/sessions").json()[0]["title"] == "2 casual outfits"
    client.delete(f"/api/stylist/sessions/{s['id']}")
    assert client.get(f"/api/stylist/sessions/{s['id']}").status_code == 404
