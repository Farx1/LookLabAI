"""Authentication.

* ``single_user`` (default, local MVP): every request acts as the local user.
  Bind the API to localhost in this mode.
* ``multi_user``: e-mail + password accounts, opaque session tokens stored
  hashed, delivered as an HttpOnly SameSite=Lax cookie or Bearer header.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import timedelta

from fastapi import Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import Settings
from .db import AuthSession, Profile, User, utcnow
from .deps import get_db, get_settings_dep

COOKIE = "looklab_session"
LOCAL_USER_EMAIL = "local@looklab.invalid"


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored or not stored.startswith("scrypt$"):
        return False
    _, salt_hex, dk_hex = stored.split("$")
    dk = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1, dklen=32)
    return hmac.compare_digest(dk.hex(), dk_hex)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def ensure_profile(db: Session, user: User) -> Profile:
    if user.profile is None:
        user.profile = Profile(user_id=user.id, preferences={}, privacy={}, spending={}, preference_model={})
        db.flush()
    return user.profile


def get_or_create_local_user(db: Session) -> User:
    user = db.scalar(select(User).where(User.email == LOCAL_USER_EMAIL))
    if user is None:
        user = User(email=LOCAL_USER_EMAIL, display_name="You", is_local=True)
        db.add(user)
        db.flush()
        ensure_profile(db, user)
        db.commit()
    return user


def create_session(db: Session, user: User, settings: Settings, response: Response | None = None) -> str:
    token = secrets.token_urlsafe(32)
    db.add(AuthSession(token_hash=_token_hash(token), user_id=user.id, expires_at=utcnow() + timedelta(days=settings.session_days)))
    db.commit()
    if response is not None:
        response.set_cookie(
            COOKIE, token, httponly=True, samesite="lax", secure=settings.cookie_secure, max_age=settings.session_days * 86400, path="/"
        )
    return token


def revoke_session(db: Session, token: str) -> None:
    s = db.get(AuthSession, _token_hash(token))
    if s:
        db.delete(s)
        db.commit()


def _token_from_request(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.cookies.get(COOKIE)


def current_user(request: Request, db: Session = Depends(get_db), settings: Settings = Depends(get_settings_dep)) -> User:
    if settings.auth_mode == "single_user":
        user = get_or_create_local_user(db)
        ensure_profile(db, user)
        return user
    token = _token_from_request(request)
    if not token:
        raise HTTPException(401, "not authenticated")
    s = db.get(AuthSession, _token_hash(token))
    if s is None or s.expires_at.replace(tzinfo=s.expires_at.tzinfo or utcnow().tzinfo) < utcnow():
        raise HTTPException(401, "session expired")
    # Cookie-authenticated unsafe requests must carry a custom header (CSRF guard).
    if request.method not in ("GET", "HEAD", "OPTIONS") and not request.headers.get("authorization"):
        if request.headers.get("x-requested-with") != "looklab":
            raise HTTPException(403, "missing X-Requested-With header")
    user = db.get(User, s.user_id)
    if user is None:
        raise HTTPException(401, "user not found")
    ensure_profile(db, user)
    return user
