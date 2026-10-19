"""Conversational stylist: deterministic constraint tracking + recommendation.

Each user message is parsed on top of the session's accumulated constraints,
so follow-ups ("make them less fitted", "keep the same shoes but change the
dress") never lose the earlier budget, occasion or palette. A small local
LLM (Ollama) may optionally rephrase the reply; it never chooses garments.
"""

from __future__ import annotations

import logging

import httpx
from sqlalchemy.orm import Session

from looklab_ml.recommendation import Constraints, parse_request

from .context import AppContext
from .db import Garment, StylistMessage, StylistSession, User
from .outfits import outfit_payload, recommend

log = logging.getLogger(__name__)
DEFAULT_FOLLOWUP_COUNT = 5


def _slot_of(g: Garment) -> str:
    return "one_piece" if g.category in ("dress", "jumpsuit") else g.category


def handle_message(db: Session, ctx: AppContext, user: User, session: StylistSession, text: str) -> StylistMessage:
    db.add(StylistMessage(session_id=session.id, role="user", content=text))
    previous = Constraints.from_dict(session.constraints) if session.constraints else None
    parsed = parse_request(text, previous)
    c = parsed.constraints
    notes: list[str] = []
    turn_required: list[str] = []  # requirements for this turn only

    # Resolve references to a previous outfit for keep/change requests.
    focus_idx = (parsed.outfit_ref - 1) if parsed.outfit_ref else session.focus_index
    last = session.last_outfits or []
    if (parsed.keep_slots or parsed.change_slots) and not last:
        notes.append("I don't have a previous outfit to modify yet, so I built new ones.")
    if last and (parsed.keep_slots or parsed.change_slots):
        focus_idx = max(0, min(focus_idx, len(last) - 1))
        focus_ids = last[focus_idx]
        focus = [g for g in (db.get(Garment, i) for i in focus_ids) if g is not None and g.user_id == user.id]
        for g in focus:
            slot = _slot_of(g)
            if slot in parsed.keep_slots and g.id not in c.locked_garment_ids:
                c.locked_garment_ids.append(g.id)
            if slot in parsed.change_slots:
                if g.id not in c.excluded_garment_ids:
                    c.excluded_garment_ids.append(g.id)
                c.locked_garment_ids = [x for x in c.locked_garment_ids if x != g.id]
        # "change the dress" means a different one-piece, not separates
        if "one_piece" in parsed.change_slots and not any(s in c.required_subcategories for s in ("dress", "jumpsuit")):
            if any(_slot_of(g) == "one_piece" for g in focus):
                turn_required.append("dress")
        if not parsed.understood or all(u.startswith(("keep", "change")) for u in parsed.understood):
            c.count = min(c.count, DEFAULT_FOLLOWUP_COUNT)
        notes.append(f"Working from outfit {focus_idx + 1}.")

    avoid = {"|".join(sorted(ids)) for ids in last} if (parsed.change_slots or "looser fit" in parsed.understood) else None
    turn = c.copy(required_subcategories=[*c.required_subcategories, *turn_required])
    res, by_id = recommend(db, ctx, user, turn, avoid)
    payload = outfit_payload(res, by_id)

    session.constraints = c.to_dict()
    session.last_outfits = [o.garment_ids for o in res.outfits]
    session.focus_index = 0
    if session.title == "New conversation":
        session.title = text[:60]

    reply = compose_reply(parsed.understood, res, notes)
    if ctx.policy.outfit_recommendation.llm_explanations == "ollama":
        reply = _llm_polish(ctx, reply) or reply
    msg = StylistMessage(
        session_id=session.id,
        role="assistant",
        content=reply,
        payload={**payload, "understood": parsed.understood, "wants_tryon": parsed.wants_tryon},
    )
    db.add(msg)
    db.commit()
    return msg


def compose_reply(understood: list[str], res, notes: list[str]) -> str:
    parts = []
    if understood:
        parts.append("Noted: " + ", ".join(understood) + ".")
    parts += notes
    n = len(res.outfits)
    if n:
        parts.append(f"Here {'is' if n == 1 else 'are'} {n} look{'s' if n != 1 else ''} from your wardrobe, best match first.")
        top = res.outfits[0]
        if top.explanation:
            parts.append("Top pick: " + "; ".join(top.explanation[:3]) + ".")
    parts += res.notes
    if res.suggestions:
        parts.append("Worth adding: " + "; ".join(s.description for s in res.suggestions[:3]) + ".")
    if n:
        parts.append("Tap a look to save it or open it in the Try-On Studio.")
    return " ".join(parts)


def _llm_polish(ctx: AppContext, draft: str) -> str | None:
    """Optional: let a small local model rephrase. Facts must not change."""
    try:
        r = httpx.post(
            f"{ctx.settings.ollama_url}/api/generate",
            json={
                "model": ctx.policy.outfit_recommendation.ollama_text_model,
                "prompt": "Rewrite this stylist reply warmly in at most 4 sentences. Keep every fact, number and garment exactly; add nothing.\n\n"
                + draft,
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=60,
        )
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        return text or None
    except Exception as e:  # noqa: BLE001
        log.info("LLM polish unavailable: %s", e)
        return None
