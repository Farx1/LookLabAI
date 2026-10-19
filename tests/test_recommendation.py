import pytest

from looklab_ml.evaluation.synthetic import DEMO_WARDROBE
from looklab_ml.recommendation import Constraints, Garment, OutfitEngine, PreferenceModel, filter_garments, parse_request


@pytest.fixture
def wardrobe():
    return [Garment(id=f"g{i}", name=it.filename, colors=[it.color], pattern=it.pattern, **it.meta) for i, it in enumerate(DEMO_WARDROBE)]


def by_name(w, name):
    return next(g for g in w if g.name == name)


def test_every_outfit_is_complete_and_unique(wardrobe):
    res = OutfitEngine().recommend(wardrobe, Constraints(count=15))
    assert len(res.outfits) == 15
    keys = {o.key for o in res.outfits}
    assert len(keys) == 15
    for o in res.outfits:
        slots = [g.slot for g in o.garments]
        assert ("one_piece" in slots) != ("top" in slots and "bottom" in slots)
        assert slots.count("shoes") <= 1


def test_principal_use_case(wardrobe):
    c = parse_request(
        "Find 15 outfits for a romantic anniversary dinner. I prefer loose-fitting, comfortable clothes, neutral colors, "
        "Korean-inspired fashion and a budget of €100."
    ).constraints
    res = OutfitEngine().recommend(wardrobe, c)
    assert len(res.outfits) == 15
    # sporty / casual-only pieces (formality 1) are filtered out for a date
    assert all(g.formality is None or g.formality >= 2 for o in res.outfits for g in o.garments)
    top = res.outfits[0]
    assert top.score.total > 0.8 and top.explanation


def test_budget_is_hard_for_purchases(wardrobe):
    for g in wardrobe:
        g.is_owned = g.name not in ("camel trench coat", "burgundy slip dress")
    res = OutfitEngine().recommend(wardrobe, Constraints(count=30, budget=80))
    assert res.outfits
    for o in res.outfits:
        assert o.known_cost <= 80
        assert "burgundy slip dress" not in [g.name for g in o.garments] or o.known_cost <= 80
    assert all("camel trench coat" != g.name for o in res.outfits for g in o.garments)  # 89 > 80
    assert res.rejected.get("over budget") == 1


def test_unknown_price_rejected_when_budget_set(wardrobe):
    g = by_name(wardrobe, "black loafers")
    g.is_owned, g.price = False, None
    pool, rejected = filter_garments(wardrobe, Constraints(budget=200))
    assert g not in pool and rejected["price unknown (budget set)"] == 1


def test_owned_only(wardrobe):
    by_name(wardrobe, "beige linen dress").is_owned = False
    res = OutfitEngine().recommend(wardrobe, Constraints(count=30, owned_only=True))
    assert all(g.is_owned for o in res.outfits for g in o.garments)


def test_excluded_colors_and_dislikes(wardrobe):
    res = OutfitEngine().recommend(wardrobe, Constraints(count=30, excluded_colors=["black"]))
    assert all("black" not in g.colors for o in res.outfits for g in o.garments)


def test_required_piece_and_detail(wardrobe):
    c = parse_request("Generate 3 Korean-inspired outfits with a wide skirt.").constraints
    res = OutfitEngine().recommend(wardrobe, c)
    assert res.outfits
    for o in res.outfits:
        assert any(g.name == "ivory wide pleated skirt" for g in o.garments)


def test_missing_required_piece_becomes_suggestion(wardrobe):
    c = parse_request("2 outfits with a maxi skirt").constraints
    res = OutfitEngine().recommend(wardrobe, c)
    assert res.suggestions and "skirt" in res.suggestions[0].description
    assert any("No maxi skirt" in n for n in res.notes)


def test_locked_and_excluded_items(wardrobe):
    shoes = by_name(wardrobe, "white sneakers")
    dress = by_name(wardrobe, "beige linen dress")
    c = Constraints(count=5, locked_garment_ids=[shoes.id], excluded_garment_ids=[dress.id])
    res = OutfitEngine().recommend(wardrobe, c)
    assert res.outfits
    for o in res.outfits:
        assert shoes.id in o.garment_ids and dress.id not in o.garment_ids


def test_season_filter(wardrobe):
    res = OutfitEngine().recommend(wardrobe, Constraints(count=30, season="summer"))
    assert all(not g.season or "summer" in g.season for o in res.outfits for g in o.garments)
    assert all(g.slot != "outerwear" for o in res.outfits for g in o.garments)


def test_scoring_prefers_harmony_and_coherence(wardrobe):
    e = OutfitEngine()
    good = [by_name(wardrobe, "cream relaxed blouse"), by_name(wardrobe, "camel wide trousers"), by_name(wardrobe, "beige mary jane flats")]
    clash = [by_name(wardrobe, "navy striped tee"), by_name(wardrobe, "black tailored trousers"), by_name(wardrobe, "white sneakers")]
    c = Constraints(occasion="date", preferred_styles=["romantic"])
    assert e.score(good, c)[0].total > e.score(clash, c)[0].total


def test_diversity(wardrobe):
    res = OutfitEngine().recommend(wardrobe, Constraints(count=10))
    from collections import Counter

    use = Counter(g.id for o in res.outfits for g in o.garments if g.slot in ("top", "bottom", "one_piece"))
    assert max(use.values()) <= 5


def test_preference_learning_changes_ranking(wardrobe):
    e = OutfitEngine()
    c = Constraints(count=10)
    pm = PreferenceModel()
    base = e.recommend(wardrobe, c, pm)
    target = [g for g in wardrobe if g.name in ("navy striped tee", "denim jeans", "white sneakers")]
    for _ in range(6):
        pm.update(target, "like")
    after = e.recommend(wardrobe, c, pm)

    def rank(res):
        pair = {"navy striped tee", "denim jeans"}
        return next((i for i, o in enumerate(res.outfits) if pair <= {g.name for g in o.garments}), 99)

    assert rank(after) < rank(base) or rank(after) == 0
    s_like, reasons = pm.score(target)
    assert s_like > 0.6 and any("you like" in r for r in reasons)
    pm2 = PreferenceModel.from_dict(pm.to_dict())
    assert pm2.to_dict() == pm.to_dict()


def test_unknown_metadata_never_crashes():
    w = [Garment(id="t", category="top"), Garment(id="b", category="bottom"), Garment(id="x")]
    res = OutfitEngine().recommend(w, Constraints(count=3))
    assert len(res.outfits) == 1 and res.rejected == {"category unknown": 1}
    assert res.outfits[0].warnings


def test_not_enough_garments_message():
    res = OutfitEngine().recommend([Garment(id="t", category="top")], Constraints())
    assert not res.outfits and res.notes and res.suggestions
