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

## Scaling path

SQLite → PostgreSQL (`LOOKLAB_DATABASE_URL=postgresql+psycopg://…`, `pip install looklab-api[postgres]`); local files → S3-compatible private bucket (`LOOKLAB_STORAGE_BACKEND=s3`); the DB job queue can be replaced by Redis/RQ when concurrency demands it. Schema migrations (Alembic) are not set up yet — `create_all` is used for the MVP.
