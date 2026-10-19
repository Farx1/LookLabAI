from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..auth import COOKIE, create_session, current_user, ensure_profile, hash_password, revoke_session, verify_password
from ..db import User
from ..deps import get_db, get_settings_dep

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=255, pattern=r"^[^@\s]+@[^@\s]+$")
    password: str = Field(min_length=8, max_length=200)
    display_name: str = Field("", max_length=80)


@router.post("/register")
def register(body: Credentials, response: Response, db=Depends(get_db), settings=Depends(get_settings_dep)):
    if settings.auth_mode != "multi_user":
        raise HTTPException(400, "registration is only used in multi_user mode")
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "email already registered")
    user = User(email=email, password_hash=hash_password(body.password), display_name=body.display_name or email.split("@")[0])
    db.add(user)
    db.flush()
    ensure_profile(db, user)
    db.commit()
    token = create_session(db, user, settings, response)
    return {"id": user.id, "email": user.email, "token": token}


@router.post("/login")
def login(body: Credentials, response: Response, db=Depends(get_db), settings=Depends(get_settings_dep)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "invalid email or password")
    token = create_session(db, user, settings, response)
    return {"id": user.id, "email": user.email, "token": token}


@router.post("/logout")
def logout(request: Request, response: Response, db=Depends(get_db)):
    token = request.cookies.get(COOKIE) or request.headers.get("authorization", "")[7:]
    if token:
        revoke_session(db, token)
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@router.get("/me")
def me(user=Depends(current_user), settings=Depends(get_settings_dep)):
    return {
        "id": user.id,
        "email": None if user.is_local else user.email,
        "display_name": user.display_name,
        "auth_mode": settings.auth_mode,
    }
