from __future__ import annotations

from dataclasses import dataclass, field

from looklab_ml.recommendation import OutfitEngine

from .config import AIPolicy, Settings
from .providers import ProviderRegistry
from .storage import LocalStorage, S3Storage, Storage


@dataclass
class AppContext:
    settings: Settings
    policy: AIPolicy
    storage: Storage
    registry: ProviderRegistry
    sessionmaker: object
    engine: OutfitEngine = field(default_factory=OutfitEngine)
    rec_cache: dict = field(default_factory=dict)


def build_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "s3":
        return S3Storage(settings.s3_bucket, settings.s3_endpoint_url, settings.s3_region, settings.data_dir / "s3-cache")
    return LocalStorage(settings.data_dir / "files")
