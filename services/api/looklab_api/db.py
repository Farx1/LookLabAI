from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON, list: JSON}


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(80), default="")
    is_local: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    profile: Mapped["Profile"] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Profile(Base):
    __tablename__ = "profiles"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    privacy: Mapped[dict] = mapped_column(JSON, default=dict)
    spending: Mapped[dict] = mapped_column(JSON, default=dict)
    preference_model: Mapped[dict] = mapped_column(JSON, default=dict)
    wardrobe_version: Mapped[int] = mapped_column(Integer, default=0)
    user: Mapped[User] = relationship(back_populates="profile")


class PersonPhoto(Base):
    __tablename__ = "person_photos"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(80), default="")
    storage_key: Mapped[str] = mapped_column(String(255))
    thumb_key: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    consent_subject: Mapped[str] = mapped_column(String(32))  # self | consenting_person
    consent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    quality: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Garment(Base):
    __tablename__ = "garments"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    image_key: Mapped[str] = mapped_column(String(255))
    thumb_key: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str | None] = mapped_column(String(120))
    category: Mapped[str] = mapped_column(String(32), default="unknown")
    subcategory: Mapped[str | None] = mapped_column(String(60))
    colors: Mapped[list] = mapped_column(JSON, default=list)
    color_hex: Mapped[list] = mapped_column(JSON, default=list)
    pattern: Mapped[str] = mapped_column(String(32), default="unknown")
    fit: Mapped[str] = mapped_column(String(32), default="unknown")
    style: Mapped[list] = mapped_column(JSON, default=list)
    season: Mapped[list] = mapped_column(JSON, default=list)
    occasion: Mapped[list] = mapped_column(JSON, default=list)
    formality: Mapped[int | None] = mapped_column(Integer)
    details: Mapped[list] = mapped_column(JSON, default=list)
    brand: Mapped[str | None] = mapped_column(String(80))
    size: Mapped[str | None] = mapped_column(String(32))
    price: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3), default="EUR")
    product_url: Mapped[str | None] = mapped_column(String(1000))
    is_owned: Mapped[bool] = mapped_column(Boolean, default=True)
    favorite: Mapped[bool] = mapped_column(Boolean, default=False)
    garment_photo_type: Mapped[str] = mapped_column(String(16), default="flat-lay")
    notes: Mapped[str | None] = mapped_column(Text)
    field_sources: Mapped[dict] = mapped_column(JSON, default=dict)
    analysis: Mapped[dict] = mapped_column(JSON, default=dict)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Collection(Base):
    __tablename__ = "collections"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(80))
    garment_ids: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SavedOutfit(Base):
    __tablename__ = "saved_outfits"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(120), default="")
    occasion: Mapped[str | None] = mapped_column(String(40))
    garment_ids: Mapped[list] = mapped_column(JSON, default=list)
    constraints: Mapped[dict] = mapped_column(JSON, default=dict)
    score: Mapped[dict] = mapped_column(JSON, default=dict)
    explanation: Mapped[list] = mapped_column(JSON, default=list)
    feedback: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class StylistSession(Base):
    __tablename__ = "stylist_sessions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(120), default="New conversation")
    constraints: Mapped[dict] = mapped_column(JSON, default=dict)
    last_outfits: Mapped[list] = mapped_column(JSON, default=list)
    focus_index: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class StylistMessage(Base):
    __tablename__ = "stylist_messages"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(ForeignKey("stylist_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    provider: Mapped[str | None] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    dedup_key: Mapped[str | None] = mapped_column(String(64), index=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TryOnResultRow(Base):
    __tablename__ = "tryon_results"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str | None] = mapped_column(String(32))
    person_photo_id: Mapped[str | None] = mapped_column(String(32), index=True)
    garment_ids: Mapped[list] = mapped_column(JSON, default=list)
    kind: Mapped[str] = mapped_column(String(16), default="tryon")  # tryon | creative
    provider: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(80))
    fidelity: Mapped[str] = mapped_column(String(32))
    image_key: Mapped[str] = mapped_column(String(255))
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    dedup_key: Mapped[str | None] = mapped_column(String(64), index=True)
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False)
    saved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class InferenceRun(Base):
    __tablename__ = "inference_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str | None] = mapped_column(String(32))
    task: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(80))
    device: Mapped[str] = mapped_column(String(60), default="unknown")
    execution_time: Mapped[float | None] = mapped_column(Float)
    model_load_seconds: Mapped[float | None] = mapped_column(Float)
    gpu_memory_peak_mb: Mapped[float | None] = mapped_column(Float)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    actual_cost_usd: Mapped[float | None] = mapped_column(Float)
    paid: Mapped[bool] = mapped_column(Boolean, default=False)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AnalysisCache(Base):
    __tablename__ = "analysis_cache"
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    analyzer: Mapped[str] = mapped_column(String(40), primary_key=True)
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def make_engine(url: str):
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, future=True, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def make_sessionmaker(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
