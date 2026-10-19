from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from .. import taxonomy as tx


@dataclass
class Garment:
    id: str
    category: str = tx.UNKNOWN
    subcategory: str | None = None
    name: str | None = None
    colors: list[str] = field(default_factory=list)
    pattern: str = tx.UNKNOWN
    fit: str = tx.UNKNOWN
    style: list[str] = field(default_factory=list)
    season: list[str] = field(default_factory=list)
    occasion: list[str] = field(default_factory=list)
    formality: int | None = None
    price: float | None = None
    currency: str = "EUR"
    is_owned: bool = True
    favorite: bool = False
    details: list[str] = field(default_factory=list)

    @property
    def slot(self) -> str:
        if self.category in tx.ONE_PIECE_CATEGORIES:
            return "one_piece"
        return self.category

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        parts = [self.colors[0] if self.colors else None, self.subcategory or self.category]
        return " ".join(p for p in parts if p and p != tx.UNKNOWN) or "garment"


@dataclass
class Constraints:
    """Accumulated styling constraints. Hard constraints filter, soft ones score."""

    count: int = 10
    occasion: str | None = None
    season: str | None = None
    budget: float | None = None
    currency: str = "EUR"
    owned_only: bool = False
    # soft preferences
    target_looseness: float | None = None  # 0 slim .. 3 oversized
    preferred_colors: list[str] = field(default_factory=list)
    preferred_styles: list[str] = field(default_factory=list)
    formality_shift: int = 0  # "more sophisticated" = +1
    # hard constraints
    required_colors: list[str] = field(default_factory=list)
    excluded_colors: list[str] = field(default_factory=list)
    required_subcategories: list[str] = field(default_factory=list)
    required_categories: list[str] = field(default_factory=list)
    excluded_subcategories: list[str] = field(default_factory=list)
    excluded_categories: list[str] = field(default_factory=list)
    locked_garment_ids: list[str] = field(default_factory=list)
    excluded_garment_ids: list[str] = field(default_factory=list)
    required_details: list[str] = field(default_factory=list)  # e.g. "wide" for "a wide skirt"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Constraints":
        if not d:
            return cls()
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def copy(self, **changes) -> "Constraints":
        return replace(self, **changes)


@dataclass
class ScoreBreakdown:
    color: float
    style: float
    formality: float
    layering: float
    fit: float
    preference: float
    request: float
    pattern: float
    total: float

    def to_dict(self) -> dict:
        return {k: round(v, 3) for k, v in asdict(self).items()}


@dataclass
class ShoppingSuggestion:
    category: str
    description: str
    reason: str
    search_query: str


@dataclass
class Outfit:
    garment_ids: list[str]
    garments: list[Garment]
    score: ScoreBreakdown
    explanation: list[str]
    known_cost: float
    unknown_price_items: list[str]
    required_purchases: list[str]
    suggestions: list[ShoppingSuggestion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return "|".join(sorted(self.garment_ids))

    def to_dict(self) -> dict:
        return {
            "garment_ids": self.garment_ids,
            "score": self.score.to_dict(),
            "explanation": self.explanation,
            "known_cost": round(self.known_cost, 2),
            "unknown_price_items": self.unknown_price_items,
            "required_purchases": self.required_purchases,
            "suggestions": [asdict(s) for s in self.suggestions],
            "warnings": self.warnings,
        }
