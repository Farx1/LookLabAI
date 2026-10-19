"""Seed a running LookLab API with the 20-piece synthetic demo wardrobe.

    uv run python scripts/seed_demo.py --api http://127.0.0.1:8000 [--with-person]

All images are drawn procedurally (no real people, no copyrighted photos).
"""

from __future__ import annotations

import argparse
import io

import httpx

from looklab_ml.evaluation.synthetic import DEMO_WARDROBE, draw_garment, draw_person


def png(img) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--with-person", action="store_true", help="also add a synthetic mannequin photo")
    a = ap.parse_args()
    h = {"X-Requested-With": "looklab"}
    with httpx.Client(base_url=a.api, timeout=60, headers=h) as c:
        existing = {g["name"] for g in c.get("/api/garments").raise_for_status().json()}
        for it in DEMO_WARDROBE:
            if it.filename in existing:
                continue
            r = c.post(
                "/api/garments", files=[("files", (f"{it.filename}.png", png(draw_garment(it.kind, it.color, it.pattern)), "image/png"))]
            )
            r.raise_for_status()
            gid = r.json()["created"][0]["id"]
            c.patch(f"/api/garments/{gid}", json=it.meta).raise_for_status()
            print("added", it.filename)
        if a.with_person:
            c.post(
                "/api/photos",
                files={"file": ("mannequin.png", png(draw_person()), "image/png")},
                data={"consent_confirmed": "true", "consent_subject": "self", "label": "synthetic mannequin"},
            ).raise_for_status()
            print("added synthetic person photo")


if __name__ == "__main__":
    main()
