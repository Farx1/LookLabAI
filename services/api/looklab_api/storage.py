"""Private object storage. Files are never exposed through public URLs:
the API streams them after an ownership check."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def delete_prefix(self, prefix: str) -> None: ...
    def local_path(self, key: str) -> str: ...


def _safe_key(key: str) -> str:
    if key.startswith("/") or ".." in key.split("/"):
        raise ValueError("invalid storage key")
    return key


class LocalStorage:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def _p(self, key: str) -> Path:
        return self.root / _safe_key(key)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        os.chmod(tmp, 0o600)
        tmp.replace(p)

    def get(self, key: str) -> bytes:
        return self._p(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def delete(self, key: str) -> None:
        try:
            self._p(key).unlink()
        except FileNotFoundError:
            pass

    def delete_prefix(self, prefix: str) -> None:
        p = self._p(prefix.rstrip("/"))
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)

    def local_path(self, key: str) -> str:
        return str(self._p(key))


class S3Storage:
    """S3-compatible private bucket (AWS S3, MinIO, R2...). Objects are put
    without ACLs; the bucket must block public access."""

    def __init__(self, bucket: str, endpoint_url: str = "", region: str = "", cache_dir: Path | None = None):
        import boto3

        self.bucket = bucket
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url or None, region_name=region or None)
        self.cache = LocalStorage(cache_dir or Path("/tmp/looklab-s3-cache"))

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self.s3.put_object(Bucket=self.bucket, Key=_safe_key(key), Body=data, ContentType=content_type, ServerSideEncryption="AES256")

    def get(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=_safe_key(key))["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=_safe_key(key))
            return True
        except Exception:  # noqa: BLE001
            return False

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=_safe_key(key))
        self.cache.delete(key)

    def delete_prefix(self, prefix: str) -> None:
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=_safe_key(prefix)):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})
        self.cache.delete_prefix(prefix)

    def local_path(self, key: str) -> str:
        """Materialise to a private local cache for model inference."""
        if not self.cache.exists(key):
            self.cache.put(key, self.get(key))
        return self.cache.local_path(key)
