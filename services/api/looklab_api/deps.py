from __future__ import annotations

from typing import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from .config import Settings


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> Iterator[Session]:
    db = request.app.state.sessionmaker()
    try:
        yield db
    finally:
        db.close()


def get_ctx(request: Request):
    return request.app.state.ctx
