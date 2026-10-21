"""Outfit photo -> detected pieces -> closest wardrobe garments.

Qwen3-VL (local Ollama, see ``tryon.layers``) boxes the pieces; each crop is read with the
deterministic CV analyzer (colors, pattern) plus the item name (category, subcategory).
Matching is a plain distance, no embeddings: hard category gate, then subcategory,
CIE76 color distance in Lab and pattern.
"""

from __future__ import annotations

import math
import re
import time

from PIL import Image

from .. import taxonomy as tx
from ..colors import PALETTE, hex_to_rgb, rgb_to_hex, rgb_to_lab
from ..tryon.layers import PREFILL, LayerPlanner, label_of, parse_items
from .analyzers import CVBasicAnalyzer

# "from left to right / each item once / at most 6" stops a 2B model from looping on one item in a multi-item photo
OUTFIT_PROMPT = (
    "List every clothing item, pair of shoes and accessory visible in the image, from left to right. "
    "Each distinct item exactly once, at most 6 items.\n"
    'Return JSON: {"items":[{"name":str,"layer":"base|mid|outer|bottom|accessory","bbox_2d":[x1,y1,x2,y2]}]}\n'
    "bbox_2d in 0-1000 normalized coordinates of the full image. "
    "Name each item precisely (color and type, e.g. green wool coat, yellow rain boots)."
)
_CATEGORY_OF_LABEL = {"pants": "bottom", "skirt": "bottom", "dress": "dress", "bag": "bag", "scarf": "accessory", "hat": "accessory",
                      "glasses": "accessory", "jewelry": "accessory", "top": "top"}  # fmt: skip
_SHOES = re.compile(r"shoe|boot|sneaker|sandal|heel|loafer|mule|pump|trainer|slipper")
_NEAR = {frozenset({"top", "outerwear"}), frozenset({"dress", "jumpsuit"})}  # taxonomy puts cardigan/vest in outerwear, shops say top
NEAR_FACTOR = 0.6
_BODY = {"face", "hair", "hands"}
_SINGLE = {"hat", "scarf", "glasses"}  # shortcut: worn twice at once is rare; a flat-lay (worn=False) is not deduped this way
_SMALL = {"shoes", "accessory", "bag"}  # categories a 2B model hallucinates on the box of a garment
W_SUB, W_COLOR, W_PATTERN = 0.35, 0.5, 0.15
LAB_SCALE = 40.0  # CIE76 distance at which two colors count as unrelated
# (canonical subcategory, category when the taxonomy has none, regex). First match wins, so specific words come first.
# Wardrobes are catalogued in French ("haut col roule", "echarpe") and Qwen answers in English: both land on one word.
_SUBS = tuple((c, k, re.compile(rf"\b(?:{w})")) for c, k, w in (
    ("trench", "outerwear", "trench"), ("puffer", "outerwear", "puffer|doudoune|down jacket|padded"),
    ("biker", "outerwear", "biker|perfecto|leather jacket"), ("blazer", "outerwear", "blazer"),
    ("parka", "outerwear", "parka"), ("coat", "outerwear", "coat|manteau"), ("jacket", "outerwear", "jacket|veste"),
    ("trousers", "bottom", "culotte"), ("skirt", "bottom", "skirt|jupe"), ("jeans", "bottom", "jeans?|denim"), ("shorts", "bottom", r"shorts\b"),
    ("trousers", "bottom", "trousers|pants|pantalon|chinos|leggings|joggers"),
    ("dress", "dress", "dress|robe"), ("jumpsuit", "jumpsuit", "jumpsuit|romper|combinaison"),
    ("turtleneck", "top", r"turtleneck|roll[ -]?neck|mock[ -]?neck|col roul"), ("polo", "top", "polo"),
    ("cardigan", "outerwear", "cardigan|gilet"), ("sweater", "top", "sweater|jumper|pullover|pull|knit"),
    ("sweatshirt", "top", "sweatshirt|hoodie"), ("t-shirt", "top", r"t-?shirt|tee\b"),
    ("blouse", "top", "blouse|chemisier"), ("shirt", "top", "shirt|chemise|button[ -]?up"), ("tank", "top", "tank|camisole"),
    ("boots", "shoes", "boots?|bottes?|bottines?"), ("sneakers", "shoes", "sneakers?|trainers?|baskets?"),
    ("mules", "shoes", "mules?"), ("heels", "shoes", "heels?|pumps?|talons?"), ("sandals", "shoes", "sandals?"),
    ("loafers", "shoes", "loafers?|flats|ballerin"),
    ("bag", "bag", r"bag|sac\b|purse|tote|clutch|backpack"),
    ("scarf", "accessory", "scarf|scarves|[eé]charpe|foulard|shawl|snood"),
    ("headband", "accessory", r"headband|hairband|serre[ -]?t[eê]te"), ("hat", "accessory", r"hat|cap\b|beanie|bonnet"),
    ("belt", "accessory", "belt|ceinture"),
))  # fmt: skip
# near subcategories (a coat is closer to a trench than to a jean): half credit
_FAMILY = {"turtleneck": "knit", "polo": "knit", "cardigan": "knit", "sweater": "knit", "sweatshirt": "knit",
           "trench": "coat", "puffer": "coat", "parka": "coat", "coat": "coat",
           "jacket": "jacket", "blazer": "jacket", "biker": "jacket", "jeans": "legs", "trousers": "legs",
           "blouse": "shirt", "shirt": "shirt"}  # fmt: skip
_PATTERN_WORDS = tuple((p, re.compile(w)) for p, w in (
    ("striped", r"stripe|rayure|ray[ée]e?\b"), ("checked", r"check|plaid|gingham|tartan|carreaux"),
    ("floral", r"floral|flower|fleur"), ("polka_dot", r"polka|\bdots?\b|\bpois\b"), ("animal", r"leopard|zebra|animal")))  # fmt: skip


# ------------------------------------------------------------------ detection
def _overlap(a: list[float], b: list[float]) -> float:
    """Intersection over the smaller box: a 2B model boxes one scarf twice with different extents."""
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def _iou(a: list[float], b: list[float]) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_items(*groups: list[dict], worn: bool = False) -> list[dict]:
    """Concatenate item lists; an item that overlaps (> 40% of the smaller box) an earlier one with the same label is dropped.
    Worn: a person wears one hat/scarf/glasses, so a same-label item elsewhere is the same piece boxed twice
    (``black headband`` + ``headband``); the later box (head-crop pass, tighter on head items) replaces the earlier."""
    out: list[dict] = []
    for g in groups:
        for i in g:
            lab = label_of(i["name"])
            if lab in _BODY:
                continue
            twin = next(
                (
                    k
                    for k, o in enumerate(out)
                    if label_of(o["name"]) == lab and (_overlap(i["bbox_2d"], o["bbox_2d"]) > 0.4 or (worn and lab in _SINGLE))
                ),
                None,
            )
            if twin is None:
                out.append(i)
            elif worn and lab in _SINGLE and _overlap(i["bbox_2d"], out[twin]["bbox_2d"]) <= 0.4:
                out[twin] = i
    return out


def detect_items(planner: LayerPlanner, image: Image.Image, person: bool) -> list[dict]:
    """Raw ``{name, layer, bbox_2d}`` items. A worn outfit also goes through the 3-pass layer detector
    (it finds small head accessories and bags); a flat-lay needs one pass."""
    try:
        items = planner._ask(image, OUTFIT_PROMPT)
        if person:
            items += [{"name": d["name"], "layer": d["layer"], "bbox_2d": d["bbox"]} for d in planner.detect(image)]
    finally:
        planner._unload_qwen()
    return merge_items(items, worn=person)


# ------------------------------------------------------------------ pieces
def canon_sub(text: str) -> tuple[str | None, str | None]:
    """(canonical subcategory, its category) of a free-text name in English or French; (None, None) when unknown."""
    low = text.lower()
    return next(((c, k) for c, k, rx in _SUBS if rx.search(low)), (None, None))


def name_pattern(text: str) -> str | None:
    """Pattern stated in a name ("striped polo", "rayure"): the CV only tells solid from patterned."""
    low = text.lower()
    return next((p for p, rx in _PATTERN_WORDS if rx.search(low)), None)


def describe_piece(image: Image.Image, item: dict, analyzer: CVBasicAnalyzer | None = None, flat: bool = False) -> dict | None:
    """Category/subcategory from the item name, colors and pattern from the central 70% of the crop.
    Flat-lay: the crop keeps a 10% margin of background so the CV analyzer masks it out (a 70% core of a trouser
    box is half background). None when the name is not clothing (a 2B model also boxes ornaments and signs)."""
    W, H = image.size
    x1, y1, x2, y2 = (max(0.0, min(1000.0, v)) for v in item["bbox_2d"])
    box = (
        int(x1 * W / 1000),
        int(y1 * H / 1000),
        max(int(x2 * W / 1000), int(x1 * W / 1000) + 2),
        max(int(y2 * H / 1000), int(y1 * H / 1000) + 2),
    )
    if flat:
        mx, my = int((box[2] - box[0]) * 0.1), int((box[3] - box[1]) * 0.1)
        core = image.crop((max(0, box[0] - mx), max(0, box[1] - my), min(W, box[2] + mx), min(H, box[3] + my)))
    else:
        crop = image.crop(box)
        cw, ch = crop.size
        core = crop.crop((int(cw * 0.15), int(ch * 0.15), max(int(cw * 0.85), int(cw * 0.15) + 2), max(int(ch * 0.85), int(ch * 0.15) + 2)))
    attrs = (analyzer or CVBasicAnalyzer()).analyze(core, item["name"]).attributes
    get = lambda k, d=None: attrs[k].value if k in attrs else d  # noqa: E731
    cat, sub = get("category"), get("subcategory")
    csub, ccat = canon_sub(item["name"])
    sub, cat = csub or sub, cat or ccat
    if cat is None:
        lab = label_of(item["name"])
        if _SHOES.search(item["name"].lower()):
            cat = "shoes"
        elif lab == "top":  # label_of falls back to "top" for any unknown word
            return None
        else:
            cat = _CATEGORY_OF_LABEL.get(lab or "", "accessory")
    colors, hexes = get("colors", []), get("color_hex", [])
    if not flat:  # a worn-outfit box is full of hair, floor and shelves: the color word of the name is the better main color
        named = re.search(r"\b(%s)\b" % "|".join(sorted(PALETTE, key=len, reverse=True)), item["name"].lower())
        if named:
            n = named.group(1)
            colors, hexes = (
                [n] + [c for c in colors if c != n],
                [rgb_to_hex(PALETTE[n])] + [h for c, h in zip(colors, hexes, strict=False) if c != n],
            )
    return {
        "name": item["name"],
        "category": cat,
        "subcategory": sub,
        "colors": colors,
        "color_hex": hexes,
        "pattern": name_pattern(item["name"]) or get("pattern", tx.UNKNOWN),
        "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
    }


# ------------------------------------------------------------------ similarity
def _labs(colors: list[str], hexes: list[str]) -> list[tuple[float, float, float]]:
    rgbs = [hex_to_rgb(h) for h in hexes] or [PALETTE[c] for c in colors if c in PALETTE]
    return [rgb_to_lab(c) for c in rgbs[:2]]


def color_sim(a: dict, b: dict) -> float:
    """Main color counts 0.7, second color 0.3; each is matched with the closest color of the other piece."""
    la, lb = _labs(a.get("colors") or [], a.get("color_hex") or []), _labs(b.get("colors") or [], b.get("color_hex") or [])
    if not la or not lb:
        return 0.5
    sims = [max(0.0, 1 - min(math.dist(x, y) for y in lb) / LAB_SCALE) for x in la]
    return sims[0] if len(sims) == 1 else 0.7 * sims[0] + 0.3 * sims[1]


def sub_sim(a: str | None, b: str | None, cat_a: str = "", cat_b: str = "") -> float:
    """A subcategory equal to its category ("top", "bag") says nothing: treated as unknown."""
    if not a or not b or a == cat_a or b == cat_b:
        return 0.5
    ca, cb = canon_sub(a)[0], canon_sub(b)[0]
    if not ca or not cb:
        return 1.0 if a.lower() == b.lower() else 0.5
    if ca == cb:
        return 1.0
    return 0.5 if _FAMILY.get(ca) and _FAMILY.get(ca) == _FAMILY.get(cb) else 0.1


def pattern_sim(a: str, b: str) -> float:
    if tx.UNKNOWN in (a, b):
        return 0.5
    if a == b:
        return 1.0
    if "solid" in (a, b):
        return 0.0  # solid vs anything else: no
    return 0.7 if "patterned" in (a, b) else 0.3  # CV "patterned" is generic; striped vs checked is not close


def similarity(piece: dict, g: dict) -> float:
    """0-1; 0 when the category differs (hard gate), except near categories which are discounted."""
    cats = {piece["category"], g.get("category")}
    if len(cats) == 2 and frozenset(cats) not in _NEAR:
        return 0.0
    sub = sub_sim(piece.get("subcategory"), g.get("subcategory"), piece["category"], g.get("category", ""))
    if piece["category"] == "accessory" and sub <= 0.1:
        return 0.0  # a headband is not a scarf
    # a cardigan filed under "top" by the shop and "outerwear" by the taxonomy is still the same garment
    near = len(cats) == 2 and sub < 1.0
    gpat = name_pattern(g.get("name") or "") or g.get("pattern") or tx.UNKNOWN
    return (NEAR_FACTOR if near else 1.0) * (
        W_SUB * sub + W_COLOR * color_sim(piece, g) + W_PATTERN * pattern_sim(piece.get("pattern") or tx.UNKNOWN, gpat)
    )


def top_matches(piece: dict, garments: list[dict], k: int = 3) -> list[tuple[dict, float]]:
    scored = sorted(((g, similarity(piece, g)) for g in garments), key=lambda t: -t[1])
    return [(g, round(s, 3)) for g, s in scored[:k] if s > 0]


def drop_phantoms(pieces: list[dict]) -> list[dict]:
    """Two pieces of different categories on almost the same box: the model invented one ("brown shoes" on the trousers).
    The garment is kept over the shoe/accessory/bag, else the first."""
    out: list[dict] = []
    for p in pieces:
        twin = next((q for q in out if q["category"] != p["category"] and _iou(p["bbox"], q["bbox"]) > 0.85), None)
        if twin is None:
            out.append(p)
        elif twin["category"] in _SMALL and p["category"] not in _SMALL:
            out[out.index(twin)] = p
    return out


def analyze_outfit(planner: LayerPlanner, image: Image.Image, person: bool) -> tuple[list[dict], dict]:
    """Pieces (without matches) and per-stage seconds."""
    t0 = time.perf_counter()
    items = detect_items(planner, image, person)
    t1 = time.perf_counter()
    cv = CVBasicAnalyzer()
    pieces = drop_phantoms(list(filter(None, (describe_piece(image, i, cv, flat=not person) for i in items))))
    pieces = [{"index": n, **p} for n, p in enumerate(pieces, 1)]
    return pieces, {"detect": round(t1 - t0, 2), "describe": round(time.perf_counter() - t1, 2)}


if __name__ == "__main__":
    # self-check: parsing is tolerant, the same piece beats a different color, a different category scores 0
    assert parse_items(PREFILL + 'beige scarf", "bbox_2d": [1, 2, 30, 40]}, {"name": "tru')[0]["name"] == "beige scarf"
    assert _overlap([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    p = {"category": "top", "subcategory": "t-shirt", "colors": ["white"], "color_hex": ["#f6f6f4"], "pattern": "solid"}
    same = dict(p)
    dark = {**p, "colors": ["black"], "color_hex": ["#141416"]}
    assert similarity(p, same) > similarity(p, dark) > 0 and similarity(p, {**same, "category": "bottom"}) == 0
    assert round(similarity(p, same), 3) == 1.0 and 0 < similarity(p, {**same, "category": "outerwear", "subcategory": "jacket"}) < 0.7
    assert canon_sub("Haut à col roulé")[0] == canon_sub("black turtleneck")[0] == "turtleneck" and name_pattern("polo rayé") == "striped"
    print("ok")
