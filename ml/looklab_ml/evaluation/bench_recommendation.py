"""Reproducible recommendation + garment-color benchmark.

Checks every returned outfit against the *hard* constraints independently
of the engine (so an engine bug cannot grade itself), and measures diversity,
preference alignment and latency. Also measures the CV color namer on
procedurally drawn garments with known ground truth.

    python -m looklab_ml.evaluation.bench_recommendation --out ml/looklab_ml/evaluation/reports/recommendation.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import statistics
import time

from ..colors import NEUTRALS, PALETTE, expand_color_terms
from ..recommendation import Garment, OutfitEngine, parse_request
from ..vision import analyze_pixels
from .synthetic import DEMO_WARDROBE, draw_garment

SCENARIOS = [
    "Find 15 outfits for a romantic anniversary dinner. I prefer loose-fitting, comfortable clothes, neutral colors, Korean-inspired fashion and a budget of €100.",
    "Create 10 looks under €80.",
    "10 outfits for work, no black",
    "Generate a Korean-inspired outfit with a wide skirt.",
    "8 casual summer outfits using only clothes I own",
    "5 elegant winter outfits for a party",
    "12 outfits with a dress",
]


def wardrobe(wishlist: tuple[str, ...] = ("camel trench coat", "burgundy slip dress", "black loafers")) -> list[Garment]:
    out = []
    for i, it in enumerate(DEMO_WARDROBE):
        g = Garment(id=f"g{i:02d}", name=it.filename, colors=[it.color], pattern=it.pattern, **it.meta)
        g.is_owned = it.filename not in wishlist
        out.append(g)
    return out


def check_hard(o, c) -> list[str]:
    errs = []
    slots = [g.slot for g in o.garments]
    if not (("one_piece" in slots) ^ ("top" in slots and "bottom" in slots)):
        errs.append("incomplete or double base")
    if any(slots.count(s) > 1 for s in ("top", "bottom", "one_piece", "shoes", "outerwear")):
        errs.append("duplicate slot")
    if c.budget is not None and sum(g.price or 0 for g in o.garments if not g.is_owned) > c.budget:
        errs.append("over budget")
    if c.owned_only and any(not g.is_owned for g in o.garments):
        errs.append("not owned")
    excl = expand_color_terms(c.excluded_colors)
    if any(set(g.colors[:2]) & excl for g in o.garments):
        errs.append("excluded color")
    if c.season and any(g.season and c.season not in g.season for g in o.garments):
        errs.append("wrong season")
    for sub in c.required_subcategories:
        if not any(sub in (g.subcategory or "") for g in o.garments):
            errs.append(f"missing {sub}")
    return errs


def run_recommendation() -> dict:
    w = wardrobe()
    engine = OutfitEngine()
    rows = []
    for prompt in SCENARIOS:
        c = parse_request(prompt).constraints
        t0 = time.perf_counter()
        res = engine.recommend(w, c)
        ms = (time.perf_counter() - t0) * 1000
        violations = [e for o in res.outfits for e in check_hard(o, c)]
        sets = [set(o.garment_ids) for o in res.outfits]
        jd = [1 - len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
        pref_colors = expand_color_terms(c.preferred_colors)
        color_align = statistics.mean(
            sum(1 for g in o.garments if g.colors and g.colors[0] in pref_colors) / len(o.garments) for o in res.outfits
        ) if pref_colors and res.outfits else None  # fmt: skip
        rows.append(
            {
                "prompt": prompt,
                "requested": c.count,
                "returned": len(res.outfits),
                "hard_constraint_violations": len(violations),
                "violation_details": sorted(set(violations)),
                "mean_pairwise_jaccard_distance": round(statistics.mean(jd), 3) if jd else None,
                "unique_garments_used": len(set().union(*sets)) if sets else 0,
                "preferred_color_share": round(color_align, 3) if color_align is not None else None,
                "mean_score": round(statistics.mean(o.score.total for o in res.outfits), 3) if res.outfits else None,
                "notes": res.notes,
                "latency_ms": round(ms, 1),
            }
        )
    total = sum(r["returned"] for r in rows)
    return {
        "wardrobe_size": len(w),
        "scenarios": rows,
        "summary": {
            "outfits_checked": total,
            "constraint_satisfaction_rate": round(1 - sum(r["hard_constraint_violations"] for r in rows) / max(1, total), 4),
            "median_latency_ms": statistics.median(r["latency_ms"] for r in rows),
        },
        "not_measured": "human preference ratings (needs a user study; see docs/EVALUATION.md)",
    }


def run_color_naming() -> dict:
    kinds = ["top", "trousers", "dress", "skirt", "jacket", "shoes", "bag"]
    total = correct = family = 0
    misses = []
    for color in PALETTE:
        for k in kinds:
            pred = analyze_pixels(draw_garment(k, color)).primary_color
            total += 1
            correct += pred == color
            # a "family" hit: both neutral, or both chromatic within the same named group
            family += pred == color or (pred in NEUTRALS) == (color in NEUTRALS)
            if pred != color:
                misses.append(f"{color}->{pred} ({k})")
    return {
        "images": total,
        "exact_accuracy": round(correct / total, 4),
        "neutral_vs_chromatic_accuracy": round(family / total, 4),
        "sample_misses": misses[:15],
        "caveat": "synthetic flat colors on a plain background; real photos with lighting variation will score lower",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    report = {
        "benchmark": "looklab-recommendation-v1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recommendation": run_recommendation(),
        "color_naming": run_color_naming(),
    }
    s = json.dumps(report, indent=2)
    print(s)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s)


if __name__ == "__main__":
    main()
