"""Deterministic natural-language request parser.

Turns stylist requests into :class:`Constraints` updates. Follow-up
messages are applied on top of the previous constraints so nothing the user
said earlier is lost ("make them less fitted" keeps the budget, occasion,
colors...). An optional local LLM may *add* to this, but the deterministic
parse is always the baseline and is what the tests pin down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import taxonomy as tx
from ..colors import COLOR_GROUP_KEYWORDS, PALETTE
from .models import Constraints

NUMBER_WORDS = {
    "one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}  # fmt: skip

OCCASION_KEYWORDS = [
    (r"\b(anniversary|romantic dinner|date night|date|valentine)", "date"),
    (r"\bwedding\b", "wedding_guest"),
    (r"\b(interview)\b", "interview"),
    (r"\b(office|work|business meeting|meeting)\b", "work"),
    (r"\b(gala|black tie|formal event|ceremony)\b", "formal"),
    (r"\b(party|club|night out|cocktail)\b", "party"),
    (r"\bbrunch\b", "brunch"),
    (r"\b(travel|trip|flight|airport)\b", "travel"),
    (r"\b(gym|workout|running|sport)\b", "sport"),
    (r"\bdinner\b", "dinner"),
    (r"\b(casual|weekend|everyday|errands)\b", "casual"),
]

STYLE_KEYWORDS = {
    "minimalist": r"minimal(ist)?|clean lines|understated",
    "romantic": r"romantic|feminine|soft",
    "elegant": r"elegant|sophisticated|refined|polished|classy",
    "classic": r"classic|timeless",
    "casual": r"\bcasual\b|relaxed vibe|laid[- ]back",
    "sporty": r"sporty|athleisure|athletic",
    "streetwear": r"street ?wear|urban",
    "bohemian": r"boho|bohemian",
    "korean": r"korean|k-?fashion|seoul",
    "preppy": r"preppy",
    "edgy": r"edgy|rock|grunge",
    "business": r"business|corporate|professional",
    "vintage": r"vintage|retro",
    "chic": r"\bchic\b|parisian|french girl",
}

SEASON_KEYWORDS = {
    "spring": r"\bspring\b",
    "summer": r"\bsummer|hot weather|heatwave\b",
    "autumn": r"\b(autumn|fall)\b",
    "winter": r"\bwinter|cold weather|snow\b",
}

CURRENCY_SYMBOLS = {"€": "EUR", "$": "USD", "£": "GBP"}


@dataclass
class ParsedRequest:
    constraints: Constraints
    understood: list[str] = field(default_factory=list)
    keep_slots: list[str] = field(default_factory=list)
    change_slots: list[str] = field(default_factory=list)
    outfit_ref: int | None = None  # 1-based index the user referred to
    reset: bool = False
    wants_tryon: bool = False


SLOT_WORDS = {
    "shoes": "shoes", "shoe": "shoes", "sneakers": "shoes", "boots": "shoes", "heels": "shoes",
    "dress": "one_piece", "jumpsuit": "one_piece",
    "top": "top", "shirt": "top", "blouse": "top", "sweater": "top",
    "bottom": "bottom", "skirt": "bottom", "pants": "bottom", "trousers": "bottom", "jeans": "bottom",
    "jacket": "outerwear", "coat": "outerwear", "blazer": "outerwear", "cardigan": "outerwear",
    "bag": "bag", "accessory": "accessory", "accessories": "accessory",
}  # fmt: skip


def _words_to_count(text: str) -> int | None:
    noun = r"(?:[\w'-]+\s+){0,3}?(?:outfits?|looks?|options?|ideas?|combinations?)\b"
    if re.search(r"\ba few\s+" + noun, text):
        return 3
    if re.search(r"\ba couple(?: of)?\s+" + noun, text):
        return 2
    m = re.search(r"\b(\d{1,2})\s+" + noun, text)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(" + "|".join(NUMBER_WORDS) + r")\s+" + noun, text)
    if m:
        return NUMBER_WORDS[m.group(1)]
    return None


def _budget(text: str) -> tuple[float, str] | None:
    pat = [
        r"(?:budget(?: of)?|under|below|less than|max(?:imum)?|up to|within|no more than)\s*([€$£])\s?(\d+(?:[.,]\d+)?)",
        r"(?:budget(?: of)?|under|below|less than|max(?:imum)?|up to|within|no more than)\s*(\d+(?:[.,]\d+)?)\s?([€$£]|eur|euros?|usd|dollars?|gbp|pounds?)",
        r"([€$£])\s?(\d+(?:[.,]\d+)?)\s*(?:budget|max|or less)",
    ]
    for p in pat:
        m = re.search(p, text)
        if m:
            a, b = m.group(1), m.group(2)
            if a in CURRENCY_SYMBOLS:
                sym, amt = a, b
            else:
                amt, sym = a, b
            cur = CURRENCY_SYMBOLS.get(sym)
            if cur is None:
                cur = "EUR" if sym.startswith("eur") else "USD" if sym.startswith(("usd", "dollar")) else "GBP"
            return float(amt.replace(",", ".")), cur
    return None


def _color_terms(fragment: str) -> list[str]:
    terms = []
    for name in sorted(list(PALETTE) + list(COLOR_GROUP_KEYWORDS) + ["gray"], key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", fragment):
            terms.append("grey" if name == "gray" else name)
            fragment = re.sub(rf"\b{re.escape(name)}\b", " ", fragment)
    return terms


def _subcategory_terms(fragment: str) -> list[tuple[str, str]]:
    found = []
    for kw, (cat, sub) in sorted(tx.SUBCATEGORY_HINTS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(kw)}s?\b", fragment):
            found.append((cat, sub))
            fragment = re.sub(rf"\b{re.escape(kw)}s?\b", " ", fragment)
    return found


def parse_request(text: str, previous: Constraints | None = None) -> ParsedRequest:
    t = " " + text.lower().strip() + " "
    t = t.replace("’", "'")
    reset = bool(re.search(r"\b(start over|reset|new request|forget (that|everything))\b", t))
    c = Constraints() if (previous is None or reset) else previous.copy()
    # list fields must not alias the previous object's lists
    for f in ("preferred_colors", "preferred_styles", "required_colors", "excluded_colors",
              "required_subcategories", "required_categories", "excluded_subcategories",
              "excluded_categories", "locked_garment_ids", "excluded_garment_ids", "required_details"):  # fmt: skip
        setattr(c, f, list(getattr(c, f)))
    out = ParsedRequest(constraints=c, reset=reset)
    u = out.understood

    n = _words_to_count(t)
    if n:
        c.count = max(1, min(n, 30))
        u.append(f"{c.count} outfit{'s' if c.count > 1 else ''}")

    b = _budget(t)
    if b:
        c.budget, c.currency = b
        u.append(f"budget {c.budget:g} {c.currency}")

    for pat, occ in OCCASION_KEYWORDS:
        if re.search(pat, t):
            c.occasion = occ
            u.append(f"occasion: {occ.replace('_', ' ')}")
            break

    for season, pat in SEASON_KEYWORDS.items():
        if re.search(pat, t):
            c.season = season
            u.append(f"season: {season}")
            break

    # --- exclusions first so "no black" does not become a preference
    neg_spans = re.findall(
        r"\b(?:no|not|without|avoid|except|hate|dislike|don't like|never)\s+([a-z ,\-]+?)(?=[.;!?]|\bbut\b|\band\b|$)", t
    )
    neg_text = " ".join(neg_spans)
    for col in _color_terms(neg_text):
        if col not in c.excluded_colors:
            c.excluded_colors.append(col)
            u.append(f"no {col}")
    for _cat, sub in _subcategory_terms(neg_text):
        if sub not in c.excluded_subcategories:
            c.excluded_subcategories.append(sub)
            u.append(f"no {sub}")
    t_pos = t
    for span in neg_spans:
        t_pos = t_pos.replace(span, " ")

    # --- fit
    if re.search(r"\b(less fitted|looser|more relaxed|less tight|more room|less clingy)\b", t_pos):
        c.target_looseness = min(3.0, (c.target_looseness if c.target_looseness is not None else 1.5) + 1.0)
        u.append("looser fit")
    elif re.search(r"\b(more fitted|tighter|slimmer|more tailored|less baggy)\b", t_pos):
        c.target_looseness = max(0.0, (c.target_looseness if c.target_looseness is not None else 1.5) - 1.0)
        u.append("more fitted")
    elif re.search(r"\b(oversized|baggy)\b", t_pos):
        c.target_looseness = 3.0
        u.append("oversized fit")
    elif re.search(r"\b(loose|loose-fitting|relaxed|comfortable|comfy|flowy|roomy)\b", t_pos):
        c.target_looseness = 2.5
        u.append("loose, comfortable fit")
    elif re.search(r"\b(fitted|slim|bodycon|tailored)\b", t_pos):
        c.target_looseness = 0.5
        u.append("fitted silhouette")

    # --- formality
    if re.search(r"\b(more (sophisticated|elegant|formal|dressy|polished|refined)|dressier|dress (it|them) up|classier)\b", t_pos):
        c.formality_shift = min(2, c.formality_shift + 1)
        if "elegant" not in c.preferred_styles:
            c.preferred_styles.append("elegant")
        u.append("more sophisticated")
    elif re.search(r"\b(more casual|less formal|dress (it|them) down|more relaxed look)\b", t_pos):
        c.formality_shift = max(-2, c.formality_shift - 1)
        u.append("more casual")

    # --- styles
    for style, pat in STYLE_KEYWORDS.items():
        if style == "romantic" and re.search(r"\bsoft (pink|blue|tones?|colou?rs?)\b", t_pos):
            continue
        if re.search(pat, t_pos) and style not in c.preferred_styles:
            if style == "casual" and c.occasion == "casual":
                continue
            c.preferred_styles.append(style)
            u.append(f"{style} style")

    # --- ownership
    if re.search(r"\b(i (already )?own|my (own )?(clothes|wardrobe|closet)|already have|only (my|owned)|from my wardrobe)\b", t_pos):
        c.owned_only = True
        u.append("only clothes you own")
    elif re.search(r"\b(can buy|new pieces|shopping|allow purchases|buy something)\b", t_pos):
        c.owned_only = False
        u.append("purchases allowed")

    # --- slot keep / change ("keep the same shoes but change the dress")
    for m in re.finditer(r"\bkeep (?:the )?(?:same )?(\w+)", t_pos):
        slot = SLOT_WORDS.get(m.group(1))
        if slot:
            out.keep_slots.append(slot)
            u.append(f"keep the {m.group(1)}")
    for m in re.finditer(r"\b(?:change|swap|replace|different|another|new) (?:the )?(\w+)", t_pos):
        slot = SLOT_WORDS.get(m.group(1))
        if slot:
            out.change_slots.append(slot)
            u.append(f"change the {m.group(1)}")
    m = re.search(r"\b(?:outfit|look|option|number|#)\s*#?(\d{1,2})\b", t_pos)
    if m:
        out.outfit_ref = int(m.group(1))

    # --- colors (positive): preferences unless phrased as a requirement
    pos_colors = _color_terms(re.sub(r"\b(keep|change|swap|replace)\b.*", " ", t_pos))
    must = re.search(r"\b(must|has to|have to|needs? to|required?|only)\b[^.]*?(colou?r|in )", t_pos)
    for col in pos_colors:
        if col in c.excluded_colors:
            continue
        if must and col in PALETTE:
            if col not in c.required_colors:
                c.required_colors.append(col)
                u.append(f"must include {col}")
        elif col not in c.preferred_colors:
            c.preferred_colors.append(col)
            u.append(f"prefers {col}")

    # --- required pieces ("with a wide skirt", "include a blazer")
    req_spans = re.findall(
        r"\b(?:with|include|including|featuring|wearing|using|around|built around) (?:an? |the |my )?([a-z \-]+?)(?=[.,;!?]|\band\b|\bbut\b|$)",
        t_pos,
    )
    for span in req_spans:
        for _cat, sub in _subcategory_terms(span):
            if any(k in out.change_slots for k in (SLOT_WORDS.get(sub, ""),)):
                continue
            if sub not in c.required_subcategories:
                c.required_subcategories.append(sub)
                u.append(f"includes a {sub}")
            for adj in ("wide", "midi", "maxi", "mini", "pleated", "long", "short", "cropped", "high-waisted"):
                if re.search(rf"\b{adj}\b", span) and adj not in c.required_details:
                    c.required_details.append(adj)
                    u.append(f"{adj} cut")
    if re.search(r"\b(no dress(es)?|not a dress)\b", t):
        c.required_categories = [x for x in c.required_categories if x != "dress"]

    out.wants_tryon = bool(re.search(r"\b(try (it|this|them) on|virtual try|show (it )?on me)\b", t))
    return out
