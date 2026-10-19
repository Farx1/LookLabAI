from looklab_ml.recommendation import parse_request

SPEC = (
    "Find 15 outfits for a romantic anniversary dinner. I prefer loose-fitting, comfortable clothes, "
    "neutral colors, Korean-inspired fashion and a budget of €100."
)


def test_principal_request():
    p = parse_request(SPEC)
    c = p.constraints
    assert c.count == 15
    assert c.occasion == "date"
    assert c.budget == 100 and c.currency == "EUR"
    assert c.target_looseness >= 2
    assert "neutral" in c.preferred_colors
    assert {"romantic", "korean"} <= set(c.preferred_styles)


def test_followups_keep_previous_constraints():
    c = parse_request(SPEC).constraints
    c2 = parse_request("Make them less fitted.", c).constraints
    assert c2.budget == 100 and c2.occasion == "date" and c2.count == 15
    assert c2.target_looseness > c.target_looseness or c2.target_looseness == 3.0
    c3 = parse_request("Use the clothes I already own.", c2).constraints
    assert c3.owned_only and c3.budget == 100
    c4 = parse_request("Show me something more sophisticated.", c3).constraints
    assert c4.formality_shift == 1 and "elegant" in c4.preferred_styles and c4.owned_only
    # the original object is not mutated
    assert c.formality_shift == 0 and not c.owned_only


def test_keep_and_change_slots():
    p = parse_request("Keep the same shoes but change the dress.", parse_request(SPEC).constraints)
    assert p.keep_slots == ["shoes"]
    assert p.change_slots == ["one_piece"]


def test_required_piece_with_detail():
    c = parse_request("Generate a Korean-inspired outfit with a wide skirt.").constraints
    assert c.count == 1
    assert "korean" in c.preferred_styles
    assert c.required_subcategories == ["skirt"]
    assert "wide" in c.required_details


def test_budget_variants_and_exclusions():
    c = parse_request("Create 10 looks under €80, no black and without heels").constraints
    assert c.count == 10 and c.budget == 80
    assert "black" in c.excluded_colors and "black" not in c.preferred_colors
    assert "heels" in c.excluded_subcategories
    assert parse_request("max 50 dollars").constraints.budget == 50
    assert parse_request("max 50 dollars").constraints.currency == "USD"


def test_reset():
    c = parse_request(SPEC).constraints
    c2 = parse_request("start over: 3 outfits for work", c).constraints
    assert c2.budget is None and c2.occasion == "work" and c2.count == 3


def test_unrelated_text_changes_nothing():
    c = parse_request(SPEC).constraints
    c2 = parse_request("thanks!", c).constraints
    assert c2 == c
