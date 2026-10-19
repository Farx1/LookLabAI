"""Controlled fashion vocabulary shared by vision, recommendation and the API.

Everything a model or a parser produces is validated against these sets.
Values outside the vocabulary become ``"unknown"`` instead of being kept as
free text, so hallucinated labels never leak into scoring.
"""

from __future__ import annotations

from typing import Iterable

UNKNOWN = "unknown"

# Outfit slots. "one_piece" garments (dress, jumpsuit) replace top + bottom.
CATEGORIES = (
    "top",
    "bottom",
    "dress",
    "jumpsuit",
    "outerwear",
    "shoes",
    "bag",
    "accessory",
)
ONE_PIECE_CATEGORIES = ("dress", "jumpsuit")

# Mapping to the FASHN VTON category argument.
TRYON_CATEGORY = {
    "top": "tops",
    "outerwear": "tops",
    "bottom": "bottoms",
    "dress": "one-pieces",
    "jumpsuit": "one-pieces",
}

SUBCATEGORY_HINTS = {
    # keyword -> (category, subcategory)
    "t-shirt": ("top", "t-shirt"),
    "tshirt": ("top", "t-shirt"),
    "tee": ("top", "t-shirt"),
    "shirt": ("top", "shirt"),
    "blouse": ("top", "blouse"),
    "sweater": ("top", "sweater"),
    "jumper": ("top", "sweater"),
    "pullover": ("top", "sweater"),
    "cardigan": ("outerwear", "cardigan"),
    "hoodie": ("top", "hoodie"),
    "sweatshirt": ("top", "sweatshirt"),
    "tank": ("top", "tank top"),
    "camisole": ("top", "camisole"),
    "crop top": ("top", "crop top"),
    "top": ("top", "top"),
    "knit": ("top", "knit"),
    "jeans": ("bottom", "jeans"),
    "denim": ("bottom", "jeans"),
    "trousers": ("bottom", "trousers"),
    "pants": ("bottom", "trousers"),
    "chinos": ("bottom", "chinos"),
    "shorts": ("bottom", "shorts"),
    "skirt": ("bottom", "skirt"),
    "leggings": ("bottom", "leggings"),
    "joggers": ("bottom", "joggers"),
    "dress": ("dress", "dress"),
    "gown": ("dress", "gown"),
    "jumpsuit": ("jumpsuit", "jumpsuit"),
    "overalls": ("jumpsuit", "overalls"),
    "romper": ("jumpsuit", "romper"),
    "playsuit": ("jumpsuit", "romper"),
    "jacket": ("outerwear", "jacket"),
    "blazer": ("outerwear", "blazer"),
    "coat": ("outerwear", "coat"),
    "trench": ("outerwear", "trench coat"),
    "parka": ("outerwear", "parka"),
    "vest": ("outerwear", "vest"),
    "gilet": ("outerwear", "vest"),
    "sneakers": ("shoes", "sneakers"),
    "trainers": ("shoes", "sneakers"),
    "boots": ("shoes", "boots"),
    "heels": ("shoes", "heels"),
    "pumps": ("shoes", "heels"),
    "loafers": ("shoes", "loafers"),
    "sandals": ("shoes", "sandals"),
    "flats": ("shoes", "flats"),
    "mules": ("shoes", "mules"),
    "shoes": ("shoes", "shoes"),
    "bag": ("bag", "bag"),
    "handbag": ("bag", "handbag"),
    "tote": ("bag", "tote"),
    "clutch": ("bag", "clutch"),
    "backpack": ("bag", "backpack"),
    "scarf": ("accessory", "scarf"),
    "belt": ("accessory", "belt"),
    "hat": ("accessory", "hat"),
    "cap": ("accessory", "cap"),
    "necklace": ("accessory", "necklace"),
    "earrings": ("accessory", "earrings"),
    "jewelry": ("accessory", "jewelry"),
    "jewellery": ("accessory", "jewelry"),
    "sunglasses": ("accessory", "sunglasses"),
}

FITS = ("slim", "regular", "relaxed", "oversized")
# Ordinal looseness used when the user asks for "less fitted" / "more fitted".
FIT_LOOSENESS = {"slim": 0, "regular": 1, "relaxed": 2, "oversized": 3}

PATTERNS = (
    "solid",
    "striped",
    "checked",
    "floral",
    "polka_dot",
    "animal",
    "graphic",
    "patterned",
)
BOLD_PATTERNS = ("floral", "animal", "graphic", "checked", "patterned")

STYLES = (
    "minimalist",
    "romantic",
    "elegant",
    "classic",
    "casual",
    "sporty",
    "streetwear",
    "bohemian",
    "korean",
    "preppy",
    "edgy",
    "business",
    "vintage",
    "chic",
)

SEASONS = ("spring", "summer", "autumn", "winter")

OCCASIONS = (
    "date",
    "dinner",
    "work",
    "casual",
    "party",
    "wedding_guest",
    "travel",
    "sport",
    "formal",
    "brunch",
    "interview",
)

# Formality scale: 1 = loungewear, 5 = black tie.
FORMALITY_MIN, FORMALITY_MAX = 1, 5

# Occasion knowledge base: target formality range and affine styles.
OCCASION_PROFILES: dict[str, dict] = {
    "date": {"formality": (3, 4), "styles": ("romantic", "elegant", "chic", "minimalist")},
    "dinner": {"formality": (3, 4), "styles": ("elegant", "chic", "classic", "romantic")},
    "work": {"formality": (3, 4), "styles": ("business", "classic", "minimalist", "elegant")},
    "interview": {"formality": (4, 5), "styles": ("business", "classic", "minimalist")},
    "casual": {"formality": (1, 3), "styles": ("casual", "minimalist", "streetwear", "korean")},
    "brunch": {"formality": (2, 3), "styles": ("casual", "chic", "romantic", "korean")},
    "party": {"formality": (3, 5), "styles": ("edgy", "chic", "elegant", "romantic")},
    "wedding_guest": {"formality": (4, 5), "styles": ("elegant", "romantic", "classic")},
    "formal": {"formality": (5, 5), "styles": ("elegant", "classic")},
    "travel": {"formality": (1, 3), "styles": ("casual", "sporty", "minimalist")},
    "sport": {"formality": (1, 1), "styles": ("sporty",)},
}

# Pairwise style affinity in [-1, 1]. Unlisted pairs are neutral (0).
STYLE_AFFINITY: dict[frozenset, float] = {
    frozenset({"minimalist", "korean"}): 0.8,
    frozenset({"minimalist", "classic"}): 0.7,
    frozenset({"minimalist", "elegant"}): 0.6,
    frozenset({"minimalist", "chic"}): 0.6,
    frozenset({"romantic", "elegant"}): 0.7,
    frozenset({"romantic", "korean"}): 0.5,
    frozenset({"romantic", "vintage"}): 0.6,
    frozenset({"romantic", "bohemian"}): 0.5,
    frozenset({"elegant", "classic"}): 0.8,
    frozenset({"elegant", "chic"}): 0.8,
    frozenset({"elegant", "business"}): 0.6,
    frozenset({"classic", "preppy"}): 0.6,
    frozenset({"classic", "business"}): 0.7,
    frozenset({"casual", "streetwear"}): 0.6,
    frozenset({"casual", "sporty"}): 0.5,
    frozenset({"casual", "korean"}): 0.5,
    frozenset({"streetwear", "korean"}): 0.5,
    frozenset({"streetwear", "edgy"}): 0.6,
    frozenset({"sporty", "streetwear"}): 0.6,
    frozenset({"bohemian", "vintage"}): 0.6,
    frozenset({"sporty", "elegant"}): -0.7,
    frozenset({"sporty", "business"}): -0.7,
    frozenset({"sporty", "romantic"}): -0.5,
    frozenset({"streetwear", "business"}): -0.5,
    frozenset({"edgy", "preppy"}): -0.4,
    frozenset({"bohemian", "business"}): -0.5,
}


def style_affinity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return STYLE_AFFINITY.get(frozenset({a, b}), 0.0)


def clean_choice(value: object, allowed: Iterable[str]) -> str:
    """Return ``value`` normalised if it is in ``allowed``, else ``UNKNOWN``."""
    if not isinstance(value, str):
        return UNKNOWN
    v = value.strip().lower().replace("-", "_").replace(" ", "_")
    allowed_set = set(allowed)
    if v in allowed_set:
        return v
    # tolerate "polka dot" vs "polka_dot" and "t_shirt" style variants
    v2 = v.replace("_", " ")
    for a in allowed_set:
        if a.replace("_", " ") == v2:
            return a
    return UNKNOWN


def clean_list(values: object, allowed: Iterable[str]) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    out: list[str] = []
    for v in values:
        c = clean_choice(v, allowed)
        if c != UNKNOWN and c not in out:
            out.append(c)
    return out


def guess_category_from_text(text: str) -> tuple[str, str] | None:
    """Keyword lookup on user-supplied text (file name, product title).

    Longest keyword wins so that "crop top" beats "top". Returns ``None``
    when nothing matches; it never guesses from pixels.
    """
    t = " " + text.lower().replace("_", " ").replace("-", " ") + " "
    best: tuple[int, tuple[str, str]] | None = None
    for kw, val in SUBCATEGORY_HINTS.items():
        k = kw.replace("-", " ")
        if f" {k} " in t or f" {k}s " in t:
            if best is None or len(k) > best[0]:
                best = (len(k), val)
    return best[1] if best else None
