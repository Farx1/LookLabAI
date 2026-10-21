# Architecture

```
Browser (Next.js PWA) ──/api/*──▶ Next.js rewrite ──▶ FastAPI (services/api)
                                                     │
                     ┌───────────────────────────────┼─────────────────────────────┐
                     ▼                               ▼                             ▼
              SQLite/PostgreSQL              Private storage                 Job runner (2 lanes)
              (SQLAlchemy)                   local FS / S3 (no public URLs)  cpu: garment analysis
                                                                             gpu: try-on, creative
                                                     │
                                         looklab_ml (ml/looklab_ml)
          ┌───────────────┬───────────────┬──────────────────┬────────────────────────┐
          vision          recommendation   collage            tryon providers
          cv_basic        parser (NL)      Pillow editorial   mock · local_fashn · remote_fashn
          Qwen3-VL/Ollama engine (A,B,D)   previews ($0)      fashn_api · openai_image_edit
          Florence-2      preferences                         (+ creative: qwen_image_edit)
```

The browser only talks to the Next.js origin; whether inference runs locally, on a remote GPU or at a paid API is invisible to it.

## Backend modules (`services/api/looklab_api`)

| Module | Responsibility |
|---|---|
| `config.py` | env settings + `config/ai.yaml` policy, validated against implemented provider names (typos fail at startup; approval cannot be disabled; paid fallback must be explicitly enabled) |
| `providers.py` | provider registry and routing. `auto` only picks free local open-source execution |
| `spending.py` | per-request approval, max-cost check, monthly cap = min(server, user), in-flight reservations, consent for external processing |
| `jobs.py` | durable DB-backed queue; restart recovery re-queues free jobs and fails paid ones (never auto-retried); cancellation; retention purge; idle model unload |
| `images.py` | decode, orientation, re-encode from pixels (drops EXIF/GPS), size + decompression-bomb limits |
| `analysis.py` / `wardrobe.py` | analysis cache by image hash; provenance-aware merging (user > model > text > cv) |
| `outfits.py` / `stylist.py` | recommendation with result cache; lazily rendered + cached collages; conversational constraint tracking |
| `privacy.py` | cascade deletion, delete-all, export, retention |
| `importer.py` | product-page import (JSON-LD/Open Graph) with SSRF protection |
| `auth.py` | `single_user` (local MVP) or `multi_user` (scrypt passwords, hashed session tokens, HttpOnly cookie + CSRF header) |

Every query is scoped by `user_id`; foreign objects return 404 (tests in `test_privacy_and_isolation.py`).

## Recommendation engine (`ml/looklab_ml/recommendation`)

* **Parser** — deterministic regex/lexicon parse of requests into `Constraints`; follow-ups update the previous constraints instead of replacing them.
* **Layer A** — hard filters: category known, exclusions, ownership, season, occasion formality window, style clashes with the occasion, budget (unknown price of a purchase is rejected when a budget is set).
* **Layer B** — interpretable score: color harmony (CIELAB palette, neutrals, analogous/complementary, clash penalty), style affinity matrix, formality consistency + occasion target, layering vs season, fit vs requested looseness, request alignment, pattern mixing.
* **Layer C** — optional local LLM rephrases replies; never selects garments.
* **Layer D** — sparse linear preference model over readable features (`color:cream`, `style:korean`, …) updated by like / dislike / save.
* **Assembly** — base (top+bottom or one-piece) beam of 160 → shoes/outerwear/accessory options → hard re-check → MMR diversity re-ranking.

## Outfit analysis (`POST /api/analyze`)

Photo to pieces to wardrobe matches. Everything runs on this machine: Qwen3-VL through the local Ollama server, plus deterministic CV. No embedding model, no remote call.

1. **Upload** (`routes/analyze.py`): consent check when a person is visible (`consent_confirmed`, same rule as person photos), `sanitize_image` (EXIF/GPS dropped), storage at `users/<id>/analysis/<id>.<ext>`, job `outfit_analysis` on the `gpu` lane (Qwen shares the card with FASHN, so the two never run together; FASHN is unloaded before Qwen starts).
2. **Detection** (`ml/looklab_ml/vision/outfit.py`): one Qwen pass lists every item with a 0-1000 box ("left to right, each item once, at most 6" prevents the 2B model from looping on one item). A worn outfit also goes through `LayerPlanner.detect` (scarf, bag, headband). Boxes of the same kind that overlap by more than 40% are merged; on a worn outfit a second hat, scarf or glasses item is the same piece boxed twice (`black headband` + `headband`) and the later box wins. Body parts and non-clothing names are dropped.
3. **Description**: each box is cropped. Category and subcategory come from the item name (`taxonomy.guess_category_from_text`), colors and pattern from the central 70% of the crop (`analyze_pixels`). `canon_sub` maps English and French names to one subcategory (`col roulé` and `turtleneck`). On a worn outfit the color word of the name is the main color (the box is full of hair and floor). On a flat-lay the crop keeps a 10% margin so the sheet is masked out, and a pattern word in the name (`striped`, `rayures`) beats the CV `patterned`. Two pieces of different categories on almost the same box (IoU above 0.85) are one invention: the shoe, accessory or bag is dropped (`drop_phantoms`).
4. **Matching** (`similarity`): category is a hard gate (top and outerwear, dress and jumpsuit are discounted by 0.6 because the taxonomy and shops disagree on cardigans and vests). Score = 0.35 subcategory + 0.5 color (CIE76 distance in Lab, main color 0.7, second 0.3) + 0.15 pattern. Subcategory: same canonical word 1.0, same family (turtleneck, polo, sweater) 0.5, different 0.1, unknown 0.5; an accessory of another kind (headband vs scarf) scores 0. A subcategory equal to its category ("top") counts as unknown. The near-category discount is skipped when both subcategories are the same canonical word. Pattern: solid vs patterned 0, generic `patterned` vs a named pattern 0.7, two different named patterns 0.3. Measured on 12 flat-lay collages (36 garments, 40-item wardrobe): top-1 23/36 before, 35/36 after; top-3 33/36 before, 36/36 after (weights unchanged; the rules were written against the first 6 collages, the other 6 are held out: 9/18 to 17/18). Top 3 per piece, owned or not, with brand, price and product link.
5. **Result**: stored on the job (pieces, boxes, matches, timings), read with `GET /api/jobs/{id}`. The photo is deleted when the job ends. `DELETE /api/analyze/{id}` removes the result.
6. **Web** (`apps/web/app/analyze`): numbered boxes over the local copy of the photo, one choice per piece, then `POST /api/outfits/saved`, `/studio?garments=` and `/api/outfits/variations`.

Measured on 2026-10-20 on an RTX 3070 Laptop (8 GB) with `qwen3-vl:2b`, on 1 photo: 3 to 11 s (first call includes model load), VRAM peak 3.4 GB. Limits: a 2B model sometimes boxes a non-existent item or gives two items the same box, and it names colors poorly (colors come from pixels instead). Pieces with no match in the wardrobe cannot be saved as an outfit.

## Scaling path

SQLite → PostgreSQL (`LOOKLAB_DATABASE_URL=postgresql+psycopg://…`, `pip install looklab-api[postgres]`); local files → S3-compatible private bucket (`LOOKLAB_STORAGE_BACKEND=s3`); the DB job queue can be replaced by Redis/RQ when concurrency demands it. Schema migrations (Alembic) are not set up yet — `create_all` is used for the MVP.
