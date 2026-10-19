import io

from PIL import Image

from looklab_ml.evaluation.synthetic import draw_garment, draw_person
from looklab_ml.vision import CVBasicAnalyzer, analyze_pixels, check_person_photo
from looklab_ml.vision.analyzers import caption_to_attributes, validate_vlm_output


def test_dominant_color_names():
    for color in ("cream", "burgundy", "navy", "camel", "black", "sage"):
        px = analyze_pixels(draw_garment("top", color))
        assert px.primary_color == color, (color, px.colors)


def test_background_removed_from_colors():
    px = analyze_pixels(draw_garment("shoes", "burgundy"))
    assert px.colors[0].name == "burgundy"
    assert all(c.name not in ("white", "ivory") for c in px.colors)


def test_solid_vs_patterned():
    assert analyze_pixels(draw_garment("dress", "beige")).pattern == "solid"
    assert analyze_pixels(draw_garment("dress", "navy", "checked")).pattern in ("patterned", "unknown")
    assert analyze_pixels(draw_garment("dress", "navy", "checked")).pattern != "solid"


def test_cv_never_guesses_category_from_pixels():
    a = CVBasicAnalyzer().analyze(draw_garment("dress", "beige"))
    assert "category" not in a.attributes
    a2 = CVBasicAnalyzer().analyze(draw_garment("dress", "beige"), hint_text="Linen midi DRESS")
    assert a2.attributes["category"].value == "dress" and a2.attributes["category"].source == "text"


def test_vlm_output_is_clamped_to_taxonomy():
    raw = {
        "category": "spacesuit",
        "subcategory": "blouse",
        "colors": ["cream", "unicorn-glitter", "Gray"],
        "pattern": "polka dot",
        "fit": "very baggy",
        "style": ["korean", "cyberpunk"],
        "season": ["autumn", "monsoon"],
        "formality": 9,
        "details": ["puff sleeves"],
        "description": "A cream blouse.",
    }
    attrs, desc = validate_vlm_output(raw, "model:qwen3-vl:2b")
    assert "category" not in attrs  # hallucinated value dropped -> unknown
    assert attrs["colors"].value == ["cream", "grey"]
    assert attrs["pattern"].value == "polka_dot"
    assert "fit" not in attrs and "formality" not in attrs
    assert attrs["style"].value == ["korean"] and attrs["season"].value == ["autumn"]
    assert desc == "A cream blouse." and attrs["subcategory"].source.startswith("model:")


def test_caption_keyword_mapping():
    attrs = caption_to_attributes("A woman's navy striped oversized shirt with long sleeves.", "model:florence2")
    assert attrs["category"].value == "top" and attrs["pattern"].value == "striped"
    assert attrs["colors"].value == ["navy"] and attrs["fit"].value == "oversized"


def test_person_photo_quality():
    rep = check_person_photo(draw_person())
    assert rep.ok
    small = check_person_photo(draw_person((300, 450)))
    assert not small.ok and small.issues[0].code == "low_resolution"
    dark = Image.new("RGB", (800, 1200), (10, 10, 10))
    codes = {i.code for i in check_person_photo(dark).issues}
    assert {"too_dark", "low_contrast", "blurry"} <= codes


def test_collage_renders():
    from looklab_ml.collage import CollageItem, render_collage

    img = render_collage([CollageItem(draw_garment("top", "cream"), "top"), CollageItem(draw_garment("trousers", "camel"), "bottom"),
                          CollageItem(draw_garment("shoes", "black"), "shoes")])  # fmt: skip
    assert img.size == (800, 1000)
    buf = io.BytesIO()
    img.save(buf, "WEBP")
    assert buf.tell() > 1000
