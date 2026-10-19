from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..auth import current_user
from ..db import StylistMessage, StylistSession
from ..deps import get_ctx, get_db
from ..serializers import iso
from ..stylist import handle_message

router = APIRouter(prefix="/api/stylist", tags=["stylist"])


def _owned(db, user, sid: str) -> StylistSession:
    s = db.get(StylistSession, sid)
    if s is None or s.user_id != user.id:
        raise HTTPException(404, "conversation not found")
    return s


def _msg(m: StylistMessage) -> dict:
    return {"id": m.id, "role": m.role, "content": m.content, "payload": m.payload, "created_at": iso(m.created_at)}


def _session(s: StylistSession, messages=None) -> dict:
    d = {"id": s.id, "title": s.title, "constraints": s.constraints, "updated_at": iso(s.updated_at)}
    if messages is not None:
        d["messages"] = [_msg(m) for m in messages]
    return d


@router.post("/sessions")
def create(db=Depends(get_db), user=Depends(current_user)):
    s = StylistSession(user_id=user.id, constraints={}, last_outfits=[])
    db.add(s)
    db.commit()
    return _session(s, [])


@router.get("/sessions")
def list_sessions(db=Depends(get_db), user=Depends(current_user)):
    rows = db.scalars(select(StylistSession).where(StylistSession.user_id == user.id).order_by(StylistSession.updated_at.desc())).all()
    return [_session(s) for s in rows]


@router.get("/sessions/{sid}")
def get_session(sid: str, db=Depends(get_db), user=Depends(current_user)):
    s = _owned(db, user, sid)
    msgs = db.scalars(select(StylistMessage).where(StylistMessage.session_id == s.id).order_by(StylistMessage.created_at)).all()
    return _session(s, msgs)


class MessageBody(BaseModel):
    content: str = Field(min_length=1, max_length=1000)


@router.post("/sessions/{sid}/messages")
def post_message(sid: str, body: MessageBody, db=Depends(get_db), ctx=Depends(get_ctx), user=Depends(current_user)):
    s = _owned(db, user, sid)
    msg = handle_message(db, ctx, user, s, body.content)
    return {"message": _msg(msg), "constraints": s.constraints}


class FocusBody(BaseModel):
    index: int = Field(ge=0, le=29)


@router.post("/sessions/{sid}/focus")
def focus(sid: str, body: FocusBody, db=Depends(get_db), user=Depends(current_user)):
    s = _owned(db, user, sid)
    s.focus_index = min(body.index, max(0, len(s.last_outfits or []) - 1))
    db.commit()
    return {"focus_index": s.focus_index}


@router.delete("/sessions/{sid}")
def delete(sid: str, db=Depends(get_db), user=Depends(current_user)):
    s = _owned(db, user, sid)
    db.delete(s)
    db.commit()
    return {"deleted": sid}
