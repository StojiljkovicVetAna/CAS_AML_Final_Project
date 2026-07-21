"""ChromaDB client helpers for local and cloud deployments."""

from __future__ import annotations

from functools import lru_cache
import inspect
import logging
from pathlib import Path
from typing import Any

from .config import BackendSettings, load_settings


LOGGER = logging.getLogger(__name__)


def _load_chromadb():
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("chromadb is required for vector database access") from exc
    return chromadb


@lru_cache(maxsize=1)
def get_cached_settings() -> BackendSettings:
    return load_settings()


@lru_cache(maxsize=1)
def get_chroma_client() -> Any:
    settings = get_cached_settings()
    chromadb = _load_chromadb()
    LOGGER.info("Using chromadb client version %s.", getattr(chromadb, "__version__", "unknown"))

    if settings.chroma_mode == "cloud":
        missing = [
            name
            for name, value in {
                "CHROMA_API_KEY": settings.chroma_api_key,
                "CHROMA_TENANT": settings.chroma_tenant,
                "CHROMA_DATABASE": settings.chroma_database,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing Chroma Cloud environment variables: {', '.join(missing)}")
        kwargs: dict[str, Any] = {
            "tenant": settings.chroma_tenant,
            "database": settings.chroma_database,
            "api_key": settings.chroma_api_key,
        }
        signature = inspect.signature(chromadb.CloudClient)
        if "cloud_host" in signature.parameters:
            kwargs["cloud_host"] = settings.chroma_host
        elif "host" in signature.parameters:
            kwargs["host"] = settings.chroma_host
        if "cloud_port" in signature.parameters:
            kwargs["cloud_port"] = settings.chroma_port
        elif "port" in signature.parameters:
            kwargs["port"] = settings.chroma_port
        if "enable_ssl" in signature.parameters:
            kwargs["enable_ssl"] = settings.chroma_ssl
        elif "ssl" in signature.parameters:
            kwargs["ssl"] = settings.chroma_ssl
        return chromadb.CloudClient(**kwargs)

    if settings.chroma_mode in {"local", "persistent"}:
        path = Path(settings.chroma_persist_path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(path))

    raise ValueError(f"Unsupported CHROMA_MODE: {settings.chroma_mode}")


def get_collection(name: str | None = None, *, create: bool = True) -> Any:
    settings = get_cached_settings()
    collection_name = name or settings.chroma_collection
    client = get_chroma_client()
    if create:
        return client.get_or_create_collection(name=collection_name)
    return client.get_collection(name=collection_name)


def reset_collection(name: str | None = None) -> Any:
    """Delete and recreate a Chroma collection.

    This is needed when replacing vectors from one embedding model with another,
    because existing collections can have a fixed embedding dimensionality.
    """

    settings = get_cached_settings()
    collection_name = name or settings.chroma_collection
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
        LOGGER.warning("Deleted Chroma collection %s before upload.", collection_name)
    except Exception as exc:
        message = str(exc).lower()
        if "not found" not in message and "does not exist" not in message:
            raise
        LOGGER.info("Chroma collection %s did not exist; creating it.", collection_name)
    return client.get_or_create_collection(name=collection_name)
