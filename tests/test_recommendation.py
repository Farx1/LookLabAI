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


def test_trend_id_steers_choice(wardrobe):
    from looklab_ml.recommendation.trends import TREND_MIN_FIT, TRENDS, trend_fit

    t = TRENDS["burgundy-neutral"]
    plain = OutfitEngine().recommend(wardrobe, Constraints(count=1)).outfits[0]
    steered = OutfitEngine().recommend(wardrobe, Constraints(count=1, trend_id=t["id"])).outfits[0]
    assert trend_fit(steered.garments, t) >= trend_fit(plain.garments, t)
    assert (steered.explanation[0] == f"Tendance FW26 : {t['nom']}") == (trend_fit(steered.garments, t) >= TREND_MIN_FIT)


def test_no_trend_score_unchanged_and_seed_reproducible(wardrobe):
    e = OutfitEngine()
    a = e.recommend(wardrobe, Constraints(count=5)).outfits
    assert all(o.score.trend == 0.5 for o in a)
    assert [o.key for o in a] == [o.key for o in e.recommend(wardrobe, Constraints(count=5)).outfits]
    s1 = [o.key for o in e.recommend(wardrobe, Constraints(count=5, seed=7)).outfits]
    assert s1 == [o.key for o in e.recommend(wardrobe, Constraints(count=5, seed=7)).outfits]


def test_shoe_formality_matches_occasion():
    w = [
        Garment(id="t", category="top", formality=3),
        Garment(id="b", category="bottom", formality=3),
        Garment(id="s", category="shoes", subcategory="baskets"),
        Garment(id="h", category="shoes", subcategory="escarpins à talons"),
    ]
    pool, rej = filter_garments(w, Constraints(occasion="dinner"))
    assert {g.id for g in pool} == {"t", "b", "h"} and rej == {"shoes do not suit the occasion": 1}
    assert "s" in {g.id for g in filter_garments(w, Constraints(occasion="casual"))[0]}


def test_trend_floor_and_label(wardrobe):
    from looklab_ml.recommendation.trends import TREND_MIN_FIT, TRENDS, trend_fit

    t = TRENDS["burgundy-neutral"]
    res = OutfitEngine().recommend(wardrobe, Constraints(count=5, trend_id=t["id"], min_trend_fit=TREND_MIN_FIT))
    for o in res.outfits:
        assert trend_fit(o.garments, t) >= TREND_MIN_FIT and o.explanation[0].startswith("Tendance FW26")
    low = OutfitEngine().recommend(wardrobe, Constraints(count=50, trend_id=t["id"])).outfits
    assert all(o.explanation[0].startswith("Tendance") == (o.score.trend >= TREND_MIN_FIT) for o in low)


def test_variations_keep_swap_and_differ(wardrobe):
    eng = OutfitEngine()
    base = eng.recommend(wardrobe, Constraints(count=1, owned_only=True)).outfits[0]
    orig = set(base.garment_ids)
    res = eng.variations(wardrobe, base.garment_ids, count=4)
    assert res.outfits and len({o.key for o in res.outfits}) == len(res.outfits) and base.key not in {o.key for o in res.outfits}
    for o in res.outfits:
        ids = set(o.garment_ids)
        slots = [g.slot for g in o.garments]
        assert 1 <= len(ids - orig) <= 2 and ("one_piece" in slots) != ("top" in slots and "bottom" in slots)
        assert o.explanation[0].split(" ")[0] in ("Remplace", "Ajoute", "Retire") and all(g.is_owned for g in o.garments)
    swap = base.garment_ids[0]
    res = eng.variations(wardrobe, base.garment_ids, count=8, swap_id=swap)
    assert res.outfits and all(swap not in o.garment_ids for o in res.outfits)
    # kept items are guaranteed even when two share a slot
    two_tops = [g.id for g in wardrobe if g.slot == "top"][:2]
    assert all(set(two_tops) <= set(o.garment_ids) for o in eng.recommend(wardrobe, Constraints(locked_garment_ids=two_tops)).outfits)


def test_swap_text_pairs_by_slot():
    from looklab_ml.recommendation.engine import _swap_text

    shirt, oxford = Garment(id="a", category="top", name="Chemise X"), Garment(id="b", category="top", name="Chemise oxford Z")
    coat, belt = Garment(id="c", category="outerwear", name="Manteau Y"), Garment(id="d", category="accessory", name="Ceinture W")
    assert _swap_text([shirt], [coat, oxford]) == "Remplace Chemise X par Chemise oxford Z ; ajoute Manteau Y"
    assert _swap_text([shirt, belt], [oxford]) == "Remplace Chemise X par Chemise oxford Z ; retire Ceinture W"
    assert _swap_text([belt], [coat]) == "Ajoute Manteau Y ; retire Ceinture W"
