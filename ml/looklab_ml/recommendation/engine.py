"""Hybrid outfit recommendation engine.

Layer A  hard filtering (budget, ownership, exclusions, season, occasion...)
Layer B  interpretable compatibility scoring
Layer D  learned preference weights (see ``preferences.py``)
Then     MMR re-ranking for outfit diversity.

Layer C (local LLM) is optional and lives outside this module: it only
rewrites explanations, it never decides what is in an outfit.
"""

from __future__ import annotations

import itertools
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field

from .. import taxonomy as tx
from ..colors import NEUTRALS, expand_color_terms, harmony_score
from .models import Constraints, Garment, Outfit, ScoreBreakdown, ShoppingSuggestion
from .preferences import PreferenceModel
from .trends import SEASON_LABEL, TREND_MIN_FIT, TRENDS, trend_fit

WEIGHTS = {
    "color": 0.20,
    "style": 0.15,
    "formality": 0.12,
    "layering": 0.06,
    "fit": 0.12,
    "preference": 0.10,
    "request": 0.20,
    "pattern": 0.05,
    "trend": 0.50,  # strong: only used by "surprise me". Ignored (neutral, weight dropped) without Constraints.trend_id
}
MAX_BASES = 160
SHOE_OPTIONS = 3
OUTER_OPTIONS = 2
MMR_LAMBDA = 0.72
# Shoe dressiness inferred from the name (FR/EN) when the garment has no formality; first match wins.
SHOE_FORMALITY = (
    (("talon", "heel", "escarpin", "pump", "loafer", "mocassin", "derby", "ballerine"), 4),
    (("basket", "sneaker", "trainer", "tennis", "sandal", "claquette", "tong"), 2),
    (("boot", "botte", "bottine"), 3),
)
DETAIL_FIT_EQUIVALENTS = {"wide": ("relaxed", "oversized")}


@dataclass
class RecommendationResult:
    outfits: list[Outfit]
    constraints: Constraints
    pool_size: int
    rejected: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    suggestions: list[ShoppingSuggestion] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "outfits": [o.to_dict() for o in self.outfits],
            "constraints": self.constraints.to_dict(),
            "pool_size": self.pool_size,
            "rejected": self.rejected,
            "notes": self.notes,
            "suggestions": [s.__dict__ for s in self.suggestions],
        }


def _target_formality(c: Constraints) -> tuple[float, float] | None:
    if not c.occasion or c.occasion not in tx.OCCASION_PROFILES:
        if c.formality_shift:
            base = 3 + c.formality_shift
            return (base, base)
        return None
    lo, hi = tx.OCCASION_PROFILES[c.occasion]["formality"]
    lo = max(tx.FORMALITY_MIN, min(tx.FORMALITY_MAX, lo + c.formality_shift))
    hi = max(tx.FORMALITY_MIN, min(tx.FORMALITY_MAX, hi + c.formality_shift))
    return lo, hi


def shoe_formality(g: Garment) -> float | None:
    if g.formality is not None:
        return g.formality
    text = " ".join([g.subcategory or "", g.name or ""]).lower()
    return next((f for keys, f in SHOE_FORMALITY if any(k in text for k in keys)), None)


def _matches_details(g: Garment, details: list[str]) -> bool:
    text = " ".join([g.name or "", g.subcategory or "", " ".join(g.details)]).lower()
    for d in details:
        if d in text:
            continue
        if g.fit in DETAIL_FIT_EQUIVALENTS.get(d, ()):
            continue
        return False
    return True


def _matches_sub(g: Garment, sub: str) -> bool:
    text = " ".join([g.subcategory or "", g.name or ""]).lower()
    return sub in text or g.category == sub


def _swap_text(removed: list[Garment], added: list[Garment]) -> str:
    """'Remplace X par Y ; ajoute Z ; retire W': a removed piece pairs with an added one of the same slot."""
    rest, swaps, dropped = list(added), [], []
    for r in removed:
        a = next((a for a in rest if a.slot == r.slot), None)
        if a:
            rest.remove(a)
            swaps.append(f"{r.label} par {a.label}")
        else:
            dropped.append(r.label)
    parts = ["remplace " + " et ".join(swaps)] if swaps else []
    parts += ["ajoute " + " et ".join(g.label for g in rest)] if rest else []
    parts += ["retire " + " et ".join(dropped)] if dropped else []
    text = " ; ".join(parts)
    return text[:1].upper() + text[1:]


def filter_garments(wardrobe: list[Garment], c: Constraints) -> tuple[list[Garment], dict[str, int]]:
    """Layer A: hard constraints. Returns the pool and rejection counts."""
    excluded_colors = expand_color_terms(c.excluded_colors)
    target = _target_formality(c)
    occ_styles = tx.OCCASION_PROFILES.get(c.occasion or "", {}).get("styles", ())
    pool: list[Garment] = []
    rejected: Counter[str] = Counter()
    for g in wardrobe:
        locked = g.id in c.locked_garment_ids
        reason = None
        if g.category not in tx.CATEGORIES:
            reason = "category unknown"
        elif g.id in c.excluded_garment_ids and not locked:
            reason = "excluded by you"
        elif not locked and any(col in excluded_colors for col in g.colors[:2]):
            reason = "excluded color"
        elif not locked and (g.category in c.excluded_categories or any(_matches_sub(g, s) for s in c.excluded_subcategories)):
            reason = "excluded piece type"
        elif c.owned_only and not g.is_owned and not locked:
            reason = "not owned"
        elif c.season and g.season and c.season not in g.season and not locked:
            reason = "wrong season"
        elif (
            g.category == "shoes"
            and c.occasion in tx.OCCASION_PROFILES
            and not locked
            and (sf := shoe_formality(g)) is not None
            and not (target[0] <= sf <= target[1] + 1)
        ):
            reason = "shoes do not suit the occasion"
        elif target and g.formality is not None and not locked and not (target[0] - 1 <= g.formality <= target[1] + 1):
            reason = "formality does not suit the occasion"
        elif occ_styles and g.style and not locked and all(max(tx.style_affinity(s, o) for o in occ_styles) <= -0.5 for s in g.style):
            reason = "style does not suit the occasion"
        elif c.budget is not None and not g.is_owned and not locked:
            if g.price is None:
                reason = "price unknown (budget set)"
            elif g.price > c.budget:
                reason = "over budget"
        if reason:
            rejected[reason] += 1
        else:
            pool.append(g)
    return pool, dict(rejected)


class OutfitEngine:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(WEIGHTS)
        if weights:
            self.weights.update(weights)

    # ------------------------------------------------------------------ scoring
    def score(self, garments: list[Garment], c: Constraints, prefs: PreferenceModel | None = None) -> tuple[ScoreBreakdown, list[str]]:
        reasons: list[str] = []
        colors = [col for g in garments for col in g.colors[:2]]
        color_s, color_r = harmony_score(colors)
        reasons += color_r[:1]

        # style coherence: best affinity between each pair of garments' style sets
        styled = [g for g in garments if g.style]
        if len(styled) >= 2:
            pair_scores = [max(tx.style_affinity(a, b) for a in g1.style for b in g2.style) for g1, g2 in itertools.combinations(styled, 2)]
            style_s = (statistics.mean(pair_scores) + 1) / 2
            if style_s >= 0.75:
                common = Counter(s for g in styled for s in g.style).most_common(2)
                reasons.append("cohesive " + " / ".join(s for s, _ in common) + " styling")
            elif style_s < 0.4:
                reasons.append("mixes contrasting styles")
        else:
            style_s = 0.55

        # formality
        forms = [g.formality for g in garments if g.formality is not None]
        target = _target_formality(c)
        if forms:
            consistency = max(0.0, 1 - statistics.pstdev(forms) / 1.5)
            mean_f = statistics.mean(forms)
            if target:
                dist = max(0.0, target[0] - mean_f, mean_f - target[1])
                closeness = max(0.0, 1 - dist / 2)
                form_s = 0.5 * consistency + 0.5 * closeness
                if closeness > 0.9 and c.occasion:
                    reasons.append(f"right level of dressiness for {c.occasion.replace('_', ' ')}")
            else:
                form_s = 0.4 + 0.6 * consistency
        else:
            form_s = 0.5

        # layering vs season
        has_outer = any(g.slot == "outerwear" for g in garments)
        if c.season in ("autumn", "winter"):
            layer_s = 0.9 if has_outer else 0.45
            if has_outer:
                reasons.append(f"layered for {c.season}")
        elif c.season == "summer":
            layer_s = 0.4 if has_outer else 0.85
        else:
            layer_s = 0.65
        if not any(g.slot == "shoes" for g in garments):
            layer_s -= 0.1

        # fit
        fits = [
            tx.FIT_LOOSENESS[g.fit] for g in garments if g.fit in tx.FIT_LOOSENESS and g.slot in ("top", "bottom", "one_piece", "outerwear")
        ]
        if fits and c.target_looseness is not None:
            fit_s = statistics.mean(1 - abs(f - c.target_looseness) / 3 for f in fits)
            if fit_s > 0.75:
                reasons.append("relaxed silhouettes as requested" if c.target_looseness >= 2 else "fitted silhouette as requested")
        elif fits:
            fit_s = 0.75 if (max(fits) - min(fits)) >= 1 else 0.6  # volume balance
        else:
            fit_s = 0.5

        # learned preferences
        if prefs:
            pref_s, pref_r = prefs.score(garments)
            reasons += pref_r[:1]
        else:
            pref_s = 0.5

        # alignment with the explicit request
        parts = []
        pref_colors = expand_color_terms(c.preferred_colors)
        if pref_colors:
            share = sum(1 for g in garments if g.colors and g.colors[0] in pref_colors) / len(garments)
            parts.append(share)
            if share >= 0.75:
                reasons.append("stays in your " + (", ".join(c.preferred_colors[:2])) + " palette")
        want_styles = list(c.preferred_styles) or list(tx.OCCASION_PROFILES.get(c.occasion or "", {}).get("styles", ()))
        if want_styles:
            st = [g for g in garments if g.style]
            if st:
                match = statistics.mean(max(max(tx.style_affinity(s, w) for w in want_styles) for s in g.style) for g in st)
                parts.append((match + 1) / 2)
                if match > 0.7 and c.preferred_styles:
                    reasons.append(" + ".join(c.preferred_styles[:2]) + " feel")
        if c.occasion:
            tagged = [g for g in garments if g.occasion]
            if tagged:
                parts.append(sum(1 for g in tagged if c.occasion in g.occasion) / len(tagged))
        fav = sum(1 for g in garments if g.favorite)
        req_s = statistics.mean(parts) if parts else 0.6
        req_s = min(1.0, req_s + 0.03 * fav)

        # pattern mixing
        bold = sum(1 for g in garments if g.pattern in tx.BOLD_PATTERNS)
        pattern_s = 1.0 if bold <= 1 else 0.3
        if bold > 1:
            reasons.append("two bold patterns compete")

        # trend alignment
        trend = TRENDS.get(c.trend_id or "")
        if trend:
            trend_s = trend_fit(garments, trend)
            if trend_s >= TREND_MIN_FIT:  # never claim a trend the outfit does not follow
                reasons.insert(0, f"Tendance {SEASON_LABEL} : {trend['nom']}")
        else:
            trend_s = 0.5

        comps = {
            "color": color_s, "style": style_s, "formality": form_s, "layering": layer_s,
            "fit": fit_s, "preference": pref_s, "request": req_s, "pattern": pattern_s, "trend": trend_s,
        }  # fmt: skip
        w = {k: v for k, v in self.weights.items() if trend or k != "trend"}
        total = sum(w[k] * comps[k] for k in w) / sum(w.values())
        return ScoreBreakdown(**comps, total=total), reasons

    # ---------------------------------------------------------------- assembly
    def recommend(
        self,
        wardrobe: list[Garment],
        c: Constraints,
        prefs: PreferenceModel | None = None,
        avoid_keys: set[str] | None = None,
    ) -> RecommendationResult:
        pool, rejected = filter_garments(wardrobe, c)
        rng = random.Random(c.seed) if c.seed is not None else None
        res = RecommendationResult(outfits=[], constraints=c, pool_size=len(pool), rejected=rejected)
        by_slot: dict[str, list[Garment]] = {}
        for g in pool:
            by_slot.setdefault(g.slot, []).append(g)
        locked = [g for g in pool if g.id in c.locked_garment_ids]
        locked_slots = {g.slot: g for g in locked}
        for missing in set(c.locked_garment_ids) - {g.id for g in locked}:
            res.notes.append(f"a kept item ({missing[:8]}) is no longer available")

        # --- required pieces
        req_subs = list(c.required_subcategories)
        unmet: list[str] = []
        for sub in req_subs:
            cands = [g for g in pool if _matches_sub(g, sub) and _matches_details(g, c.required_details)]
            if not cands:
                unmet.append(sub)
        for sub in unmet:
            detail = " ".join(c.required_details)
            desc = " ".join(x for x in [self._palette_word(c), detail, sub] if x)
            res.suggestions.append(
                ShoppingSuggestion(
                    category=tx.SUBCATEGORY_HINTS.get(sub, (sub, sub))[0],
                    description=desc,
                    reason=f"no {detail + ' ' if detail else ''}{sub} in your wardrobe matches",
                    search_query=" ".join(x for x in [desc, *(c.preferred_styles[:1])] if x),
                )
            )
            res.notes.append(
                f"No {(' '.join(c.required_details) + ' ') if c.required_details else ''}{sub} available — showing the closest alternatives."
            )
        active_subs = [s for s in req_subs if s not in unmet]

        def slot_cands(slot: str) -> list[Garment]:
            if slot in locked_slots:
                return [locked_slots[slot]]
            cands = by_slot.get(slot, [])
            for sub in active_subs:
                sub_slot = tx.SUBCATEGORY_HINTS.get(sub, (None,))[0]
                sub_slot = "one_piece" if sub_slot in tx.ONE_PIECE_CATEGORIES else sub_slot
                if sub_slot == slot:
                    cands = [g for g in cands if _matches_sub(g, sub) and _matches_details(g, c.required_details)]
            return cands

        need_one_piece = (
            any(tx.SUBCATEGORY_HINTS.get(s, ("",))[0] in tx.ONE_PIECE_CATEGORIES for s in active_subs) or "one_piece" in locked_slots
        )
        need_separates = (
            any(tx.SUBCATEGORY_HINTS.get(s, ("",))[0] in ("top", "bottom") for s in active_subs)
            or "top" in locked_slots
            or "bottom" in locked_slots
        )

        bases: list[list[Garment]] = []
        if not need_one_piece:
            for top, bottom in itertools.product(slot_cands("top"), slot_cands("bottom")):
                bases.append([top, bottom])
        if not need_separates:
            bases += [[op] for op in slot_cands("one_piece")]
        if not bases:
            missing = []
            if not by_slot.get("top") and not by_slot.get("one_piece"):
                missing.append("top or dress")
            if not by_slot.get("bottom") and not by_slot.get("one_piece"):
                missing.append("bottom")
            res.notes.append(
                "Not enough garments to build an outfit"
                + (f" (missing: {', '.join(missing)})" if missing else " with these constraints")
                + ". Add garments or relax a constraint."
            )
            for m in missing:
                res.suggestions.append(
                    ShoppingSuggestion(
                        m, f"{self._palette_word(c)} {m}".strip(), "missing from wardrobe", f"{self._palette_word(c)} {m}".strip()
                    )
                )
            return res

        # Rank bases by partial score to bound the search.
        bases.sort(key=lambda b: -self.score(b, c, prefs)[0].total)
        bases = bases[:MAX_BASES]

        shoes = slot_cands("shoes")
        outers = slot_cands("outerwear")
        extras = slot_cands("bag") + slot_cands("accessory")
        if not shoes:
            res.suggestions.append(
                ShoppingSuggestion(
                    "shoes",
                    f"{self._palette_word(c)} shoes".strip(),
                    "no shoes available for this request",
                    f"{self._palette_word(c)} {self._fit_word(c)} shoes".strip(),
                )
            )
        candidates: dict[str, tuple[list[Garment], ScoreBreakdown, list[str]]] = {}
        trend = TRENDS.get(c.trend_id or "")
        for base in bases:
            shoe_opts: list[Garment | None] = (
                sorted(shoes, key=lambda s: -self.score(base + [s], c, prefs)[0].total)[:SHOE_OPTIONS] if shoes else [None]
            )
            if "outerwear" in locked_slots:
                outer_opts: list[Garment | None] = [locked_slots["outerwear"]]
            elif c.season == "summer":
                outer_opts = [None]
            else:
                outer_opts = [None] + sorted(outers, key=lambda o: -self.score(base + [o], c, prefs)[0].total)[:OUTER_OPTIONS]
            for shoe, outer in itertools.product(shoe_opts, outer_opts):
                outfit = base + [x for x in (outer, shoe) if x is not None]
                # an accessory/bag only if it improves the score
                if extras:
                    best = max(extras, key=lambda e: self.score(outfit + [e], c, prefs)[0].total)
                    for slot in ("bag", "accessory"):
                        if slot in locked_slots and locked_slots[slot] not in outfit:
                            outfit = outfit + [locked_slots[slot]]
                    if best not in outfit and self.score(outfit + [best], c, prefs)[0].total > self.score(outfit, c, prefs)[0].total:
                        outfit = outfit + [best]
                if not self._satisfies(outfit, c, active_subs, {g.id for g in locked}):
                    continue
                if trend and trend_fit(outfit, trend) < c.min_trend_fit:
                    continue
                key = "|".join(sorted(g.id for g in outfit))
                if avoid_keys and key in avoid_keys:
                    continue
                if key not in candidates:
                    sb, reasons = self.score(outfit, c, prefs)
                    if rng:
                        sb.total += rng.uniform(-0.04, 0.04)
                    candidates[key] = (outfit, sb, reasons)

        if not candidates:
            res.notes.append("No combination satisfies every constraint (budget, colors or kept items). Try relaxing one.")
            return res

        selected = self._mmr(list(candidates.values()), c.count)
        for garments, sb, reasons in selected:
            res.outfits.append(self._build_outfit(garments, sb, reasons, c, res.suggestions))
        if len(res.outfits) < c.count:
            res.notes.append(f"Only {len(res.outfits)} distinct outfit(s) fit these constraints with the current wardrobe.")
        return res

    def variations(
        self,
        wardrobe: list[Garment],
        garment_ids: list[str],
        count: int = 4,
        swap_id: str | None = None,
        prefs: PreferenceModel | None = None,
    ) -> RecommendationResult:
        """Owned-only outfits close to ``garment_ids``: 1-2 pieces change, the rest is kept.

        One run per swapped slot (the whole outfit if ``swap_id`` is None): other pieces locked, the
        swapped one excluded, the original avoided. Runs are interleaved so every slot gets a turn.
        """
        by_id = {g.id: g for g in wardrobe}
        orig = [by_id[i] for i in garment_ids]
        orig_ids = {g.id for g in orig}
        targets = [g for g in orig if swap_id in (None, g.id)]
        c0 = Constraints(count=count, owned_only=True)
        res = RecommendationResult(outfits=[], constraints=c0, pool_size=0)
        runs = []
        for t in targets:
            c = c0.copy(locked_garment_ids=[g.id for g in orig if g is not t], excluded_garment_ids=[t.id])
            r = self.recommend(wardrobe, c, prefs, {"|".join(sorted(orig_ids))})
            res.pool_size = max(res.pool_size, r.pool_size)
            runs.append(r.outfits)
        seen: set[str] = set()
        for row in itertools.zip_longest(*runs):
            for o in filter(None, row):
                new = set(o.garment_ids)
                removed, added = [by_id[i] for i in orig_ids - new], [by_id[i] for i in new - orig_ids]
                if o.key in seen or not 1 <= max(len(removed), len(added)) <= 2 or len(res.outfits) >= count:
                    continue
                seen.add(o.key)
                o.explanation.insert(0, _swap_text(removed, added))
                res.outfits.append(o)
        if not res.outfits:
            res.notes.append("Aucune variation possible avec les pièces possédées.")
        return res

    def _satisfies(self, outfit: list[Garment], c: Constraints, active_subs: list[str], locked_ids: set[str]) -> bool:
        if not locked_ids <= {g.id for g in outfit}:  # kept items must all be present (two kept items can share a slot)
            return False
        for sub in active_subs:
            if not any(_matches_sub(g, sub) for g in outfit):
                return False
        for col in c.required_colors:
            wanted = expand_color_terms([col])
            if not any(set(g.colors[:2]) & wanted for g in outfit):
                return False
        if c.budget is not None:
            cost = sum(g.price or 0 for g in outfit if not g.is_owned)
            if cost > c.budget + 1e-9:
                return False
        return True

    def _mmr(self, cands: list[tuple[list[Garment], ScoreBreakdown, list[str]]], k: int):
        remaining = sorted(cands, key=lambda x: -x[1].total)
        chosen: list[tuple[list[Garment], ScoreBreakdown, list[str]]] = []
        usage: Counter[str] = Counter()
        while remaining and len(chosen) < k:
            best_i, best_v = 0, -1e9
            for i, cand in enumerate(remaining[:400]):
                ids = {g.id for g in cand[0]}
                sim = 0.0
                for ch in chosen:
                    cids = {g.id for g in ch[0]}
                    sim = max(sim, len(ids & cids) / len(ids | cids))
                overuse = sum(usage[g.id] for g in cand[0] if g.slot not in ("shoes", "bag", "accessory")) * 0.02
                v = MMR_LAMBDA * cand[1].total - (1 - MMR_LAMBDA) * sim - overuse
                if v > best_v:
                    best_i, best_v = i, v
            pick = remaining.pop(best_i)
            chosen.append(pick)
            usage.update(g.id for g in pick[0])
        return chosen

    def _build_outfit(self, garments, sb, reasons, c: Constraints, global_suggestions) -> Outfit:
        order = {"outerwear": 0, "top": 1, "one_piece": 1, "bottom": 2, "shoes": 3, "bag": 4, "accessory": 5}
        garments = sorted(garments, key=lambda g: order.get(g.slot, 9))
        to_buy = [g for g in garments if not g.is_owned]
        known_cost = sum(g.price for g in to_buy if g.price is not None)
        unknown = [g.id for g in to_buy if g.price is None]
        explanation = list(dict.fromkeys(r for r in reasons if r))[:4]
        if to_buy:
            cur = c.currency
            explanation.append(
                "to buy: "
                + ", ".join(f"{g.label} ({g.price:g} {cur})" if g.price is not None else f"{g.label} (price unknown)" for g in to_buy)
            )
        else:
            explanation.append("uses only pieces you own")
        warnings = []
        clothing = ("top", "bottom", "one_piece", "outerwear")
        unknown_attrs = sum(1 for g in garments if not g.colors or not g.style or (g.slot in clothing and g.fit == tx.UNKNOWN))
        if unknown_attrs >= max(2, len(garments) // 2):
            warnings.append("several pieces have incomplete metadata; scores are less certain")
        return Outfit(
            garment_ids=[g.id for g in garments],
            garments=garments,
            score=sb,
            explanation=explanation,
            known_cost=known_cost,
            unknown_price_items=unknown,
            required_purchases=[g.id for g in to_buy],
            suggestions=list(global_suggestions),
            warnings=warnings,
        )

    @staticmethod
    def _palette_word(c: Constraints) -> str:
        if c.preferred_colors:
            return c.preferred_colors[0]
        return ""

    @staticmethod
    def _fit_word(c: Constraints) -> str:
        if c.target_looseness is None:
            return ""
        return "relaxed" if c.target_looseness >= 2 else "fitted" if c.target_looseness <= 1 else ""


def neutral_share(garments: list[Garment]) -> float:
    cols = [g.colors[0] for g in garments if g.colors]
    return sum(1 for x in cols if x in NEUTRALS) / len(cols) if cols else 0.0
